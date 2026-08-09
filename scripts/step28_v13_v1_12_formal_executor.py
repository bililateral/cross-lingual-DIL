#!/usr/bin/env python3
"""Execute one authorized v1.12 formal split inside private staging.

Core generation, each M1 rewire, finalization, quality audit, and publication
are deliberately separate resumable states.  No state permits regeneration.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import step28_v13_v1_12_formal_common as formal
import step28_v13_v1_12_generate_split as generator
import step28_v13_v1_12_preceremony as preceremony


ROOT = Path(__file__).resolve().parents[1]


class FormalExecutorError(ValueError):
    """Raised when an authorized formal stage cannot advance safely."""


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_json_no_replace(path: Path, value: Mapping[str, Any]) -> None:
    preceremony.write_bytes_no_replace_long_path(path, _json_bytes(value))


def _repo_pin(path: Path, *, include_self_hash: bool = False) -> dict[str, Any]:
    try:
        relative = path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise FormalExecutorError("Executor path escapes repository") from exc
    result: dict[str, Any] = {
        "path": relative,
        "size_bytes": preceremony.stat_long_path(path).st_size,
        "sha256": preceremony.sha256_file(path),
    }
    if include_self_hash:
        document = preceremony.load_json_strict(path)
        preceremony.validate_canonical_self_hash(
            document, label=f"formal executor pin {path.name}"
        )
        result["canonical_self_hash"] = document["canonical_self_hash"]
    return result


def _paths(draft: Mapping[str, Any], split: str) -> dict[str, Path]:
    public_root = preceremony._repo_path(str(draft["release"]["public_root"]))
    private_root = preceremony._repo_path(str(draft["release"]["private_root"]))
    return {
        "stage": private_root / "_staging" / split,
        "private_final": private_root / "splits" / split,
        "public_final": public_root / split,
        "core_marker": private_root / "_staging" / split / "CORE_COMPLETE.json",
        "finalized_marker": private_root / "_staging" / split / "STAGE_FINALIZED.json",
        "quality_marker": private_root / "_staging" / split / "QUALITY_PASS.json",
    }


def _authority_reference(
    execution_lock_path: Path, lock: Mapping[str, Any]
) -> dict[str, Any]:
    pin = _repo_pin(execution_lock_path, include_self_hash=True)
    kinds = {
        "READY_FOR_TRAIN_DEVELOPMENT_GENERATION": (
            "train_development_execution_lock"
        ),
        "READY_FOR_AUDIT_A_GENERATION_ONLY": "audit_a_generation_lock",
        "READY_FOR_AUDIT_B_GENERATION_ONLY": "audit_b_generation_lock",
    }
    status = str(lock.get("status", ""))
    if status not in kinds:
        raise FormalExecutorError("Unknown execution authority status")
    return {
        "kind": kinds[status],
        **pin,
        "status": status,
    }


def _load_split_authority(
    *, split: str, lock_path: Path
) -> tuple[dict[str, str], dict[str, str], dict[str, Any], dict[str, Any]]:
    if split in {"train", "development"}:
        capabilities, commitments, validated = (
            formal.load_split_generator_capabilities(
                split=split, execution_lock_path=lock_path
            )
        )
        return (
            capabilities,
            commitments,
            validated,
            validated["execution_lock"],
        )
    if split in {"audit_a", "audit_b"}:
        capabilities, commitments, validated = (
            formal.load_audit_generator_capabilities(
                split=split, audit_lock_path=lock_path
            )
        )
        return capabilities, commitments, validated, validated["audit_lock"]
    raise FormalExecutorError("Unknown formal split authority")


def _validate_marker(path: Path, *, status: str, split: str) -> dict[str, Any]:
    document = preceremony.load_json_strict(path)
    preceremony.validate_canonical_self_hash(
        document, label=f"formal stage marker {path.name}"
    )
    if document.get("status") != status or document.get("split") != split:
        raise FormalExecutorError(f"Formal stage marker drift: {path.name}")
    return document


def _load_public_split_authority(
    *, split: str, lock_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate an authority without parsing any raw split capability."""

    if split in {"train", "development"}:
        validated = formal.load_and_validate_execution_lock(lock_path)
        lock = validated["execution_lock"]
        if lock["authorizations"][f"formal_{split}_generation"] is not True:
            raise FormalExecutorError(f"Formal split is not authorized: {split}")
        return validated, lock
    if split in {"audit_a", "audit_b"}:
        validated = formal.load_and_validate_audit_lock(lock_path)
        lock = validated["audit_lock"]
        if lock["authorizations"][f"formal_{split}_generation"] is not True:
            raise FormalExecutorError(f"Formal split is not authorized: {split}")
        return validated, lock
    raise FormalExecutorError("Unknown formal split authority")


def _core_start_path(lock_path: Path, split: str) -> Path:
    return lock_path.parent / f"{split}_core_generation_start.json"


def _expected_core_start(
    *, split: str, lock_path: Path, lock: Mapping[str, Any]
) -> dict[str, Any]:
    return preceremony.with_canonical_self_hash(
        {
            "version": "2026-08-03-step28-v13-v1-12-core-start-v1",
            "status": "FORMAL_CORE_GENERATION_STARTED_NO_RESTART",
            "run_id": lock["run_id"],
            "split": split,
            "authority": _authority_reference(lock_path, lock),
            "generator_capability_commitments": lock[
                "generator_capability_commitments"
            ][split],
            "world_count": 500,
            "pair_count": 189000,
            "raw_master_or_capability_present": False,
            "missing_stage_or_marker_permanently_closes_run": True,
            "regeneration_or_seed_replacement_forbidden": True,
        }
    )


def _m1_start_path(lock_path: Path, role: str) -> Path:
    return lock_path.parent / f"train_{role}_materialization_start.json"


def _expected_m1_start(
    *, role: str, lock_path: Path, lock: Mapping[str, Any]
) -> dict[str, Any]:
    return preceremony.with_canonical_self_hash(
        {
            "version": "2026-08-03-step28-v13-v1-12-m1-start-v1",
            "status": "FORMAL_M1_MATERIALIZATION_STARTED_NO_RESTART",
            "run_id": lock["run_id"],
            "split": "train",
            "role": role,
            "authority": _authority_reference(lock_path, lock),
            "rewire_key_commitment": lock["m1_capability_commitments"][role],
            "world_count": 500,
            "pair_count": 189000,
            "raw_rewire_key_present": False,
            "missing_receipt_permanently_closes_run": True,
            "rematerialization_or_key_replacement_forbidden": True,
        }
    )


def _publish_or_validate_exact_marker(
    path: Path, expected: Mapping[str, Any], *, label: str
) -> bool:
    """Return True for an existing exact marker, False for a new write."""

    if preceremony.exists_long_path(path):
        _validate_exact_marker(path, expected, label=label)
        return True
    _write_json_no_replace(path, expected)
    return False


def _validate_exact_marker(
    path: Path, expected: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    if not preceremony.exists_long_path(path):
        raise FormalExecutorError(f"Required {label} is missing")
    observed = preceremony.load_json_strict(path)
    preceremony.validate_canonical_self_hash(observed, label=label)
    if dict(observed) != dict(expected):
        raise FormalExecutorError(f"Existing {label} differs from replay")
    return observed


def generate_core(
    *,
    split: str,
    execution_lock_path: Path = formal.DEFAULT_EXECUTION_LOCK_PATH,
    progress_every: int = 25,
) -> dict[str, Any]:
    validated_public, active_lock = _load_public_split_authority(
        split=split, lock_path=execution_lock_path
    )
    draft = validated_public["draft"]
    paths = _paths(draft, split)
    stage = paths["stage"]
    start_path = _core_start_path(execution_lock_path, split)
    expected_start = _expected_core_start(
        split=split,
        lock_path=execution_lock_path,
        lock=active_lock,
    )
    publication_receipt = execution_lock_path.parent / (
        f"{split}_publication_receipt.json"
    )
    if preceremony.exists_long_path(publication_receipt):
        raise FormalExecutorError(
            "Formal split publication receipt exists; generation is closed"
        )
    if preceremony.exists_long_path(
        paths["public_final"]
    ) or preceremony.exists_long_path(paths["private_final"]):
        raise FormalExecutorError(
            "Formal final split already exists; regeneration is forbidden"
        )
    stage_exists = preceremony.exists_long_path(stage)
    if stage_exists and not preceremony.exists_long_path(start_path):
        raise FormalExecutorError(
            "Formal core stage exists without its earlier start receipt"
        )
    start_existed = _publish_or_validate_exact_marker(
        start_path,
        expected_start,
        label=f"formal {split} core start receipt",
    )
    if start_existed and not stage_exists:
        raise FormalExecutorError(
            "Formal core start exists but its stage is missing; regeneration is forbidden"
        )
    if stage_exists:
        marker = _validate_marker(
            paths["core_marker"],
            status="PASS_FORMAL_CORE_STAGE_COMPLETE",
            split=split,
        )
        if marker.get("core_start_receipt") != _repo_pin(
            start_path, include_self_hash=True
        ):
            raise FormalExecutorError("Recovered core/start receipt lineage drift")
        return {
            "status": "PASS_RECOVERED_EXISTING_FORMAL_CORE_STAGE",
            "split": split,
            "core_marker_sha256": preceremony.sha256_file(
                paths["core_marker"]
            ),
            "pair_count": marker["pair_count"],
        }
    capabilities, structure_commitments, validated, selected_lock = (
        _load_split_authority(split=split, lock_path=execution_lock_path)
    )
    if selected_lock != active_lock or validated["draft"] != draft:
        raise FormalExecutorError("Split authority changed after core start")
    os.makedirs(
        preceremony._filesystem_path(stage.parent), exist_ok=True
    )
    built = generator._build_core_stage_impl(
        output_root=stage,
        split=split,
        world_count=500,
        design_only=False,
        draft=draft,
        validated_baseline=validated["baseline"],
        capabilities={
            "split": split,
            "generator": capabilities,
            "m1": {},
        },
        structure_commitments=structure_commitments,
        authority_reference=_authority_reference(
            execution_lock_path, active_lock
        ),
        force_first_candidate_collision=False,
        progress_every=progress_every,
    )
    receipt_path = (
        built["public_root"] / "audit" / "generation_integrity_receipt.json"
    )
    receipt = preceremony.load_json_strict(receipt_path)
    preceremony.validate_canonical_self_hash(
        receipt, label=f"formal {split} generation receipt"
    )
    expected_counts = {
        "world_count": 500,
        "pair_count": 189000,
        "positive_count": 10000,
        "negative_count": 179000,
        "seller_count": 14000,
        "identity33_row_count": 189000,
        "identity_asset_count": formal.expected_identity_asset_count(
            draft, split=split, world_count=500
        ),
        "retrieval_query_count": 14000,
        "retrieval_qrel_count": 20000,
    }
    if (
        receipt.get("status") != "PASS_FORMAL_CORE_STAGE"
        or receipt.get("aggregate_counts") != expected_counts
        or receipt.get("formal_authorization_used") is not True
        or receipt.get("master_seed_access") is not False
    ):
        raise FormalExecutorError("Formal core generation receipt drift")
    marker = preceremony.with_canonical_self_hash(
        {
            "version": "2026-08-03-step28-v13-v1-12-core-marker-v1",
            "status": "PASS_FORMAL_CORE_STAGE_COMPLETE",
            "split": split,
            "world_count": 500,
            "pair_count": 189000,
            "generation_receipt": _repo_pin(
                receipt_path, include_self_hash=True
            ),
            "core_start_receipt": _repo_pin(
                start_path, include_self_hash=True
            ),
            "master_seed_access": False,
            "m1_capability_access": False,
            "stage_finalized": False,
            "stage_published": False,
        }
    )
    _write_json_no_replace(paths["core_marker"], marker)
    return {
        "status": "PASS_NEW_FORMAL_CORE_STAGE",
        "split": split,
        "pair_count": 189000,
        "core_marker_sha256": preceremony.sha256_file(paths["core_marker"]),
    }


def materialize_train_m1(
    *,
    replicate: int,
    execution_lock_path: Path = formal.DEFAULT_EXECUTION_LOCK_PATH,
) -> dict[str, Any]:
    if not 1 <= replicate <= 5:
        raise FormalExecutorError("M1 replicate must be 1..5")
    role = f"m1_r{replicate:02d}"
    validated = formal.load_and_validate_execution_lock(execution_lock_path)
    lock = validated["execution_lock"]
    paths = _paths(validated["draft"], "train")
    core_marker = _validate_marker(
        paths["core_marker"],
        status="PASS_FORMAL_CORE_STAGE_COMPLETE",
        split="train",
    )
    core_start_path = _core_start_path(execution_lock_path, "train")
    _validate_exact_marker(
        core_start_path,
        _expected_core_start(
            split="train",
            lock_path=execution_lock_path,
            lock=lock,
        ),
        label="formal train core start receipt",
    )
    if core_marker.get("core_start_receipt") != _repo_pin(
        core_start_path, include_self_hash=True
    ):
        raise FormalExecutorError("Formal train core/start lineage drift")
    if preceremony.exists_long_path(paths["finalized_marker"]):
        raise FormalExecutorError("Cannot add M1 after formal stage finalization")
    target = (
        paths["stage"]
        / "private"
        / "m1"
        / f"r{replicate:02d}"
        / "structural_receipt.json"
    )
    start_path = _m1_start_path(execution_lock_path, role)
    expected_start = _expected_m1_start(
        role=role, lock_path=execution_lock_path, lock=lock
    )
    if preceremony.exists_long_path(target):
        _validate_exact_marker(
            start_path,
            expected_start,
            label=f"formal {role} materialization start receipt",
        )
        receipt = _validate_marker(
            target,
            status="PASS_FORMAL_M1_STRUCTURAL_REPLAY",
            split="train",
        )
        if (
            receipt.get("replicate") != f"r{replicate:02d}"
            or receipt.get("authority_reference", {}).get("m1_start_receipt")
            != _repo_pin(start_path, include_self_hash=True)
        ):
            raise FormalExecutorError("Existing formal M1 receipt replicate drift")
        return {
            "status": "PASS_RECOVERED_EXISTING_FORMAL_M1_RECEIPT",
            "replicate": replicate,
            "receipt_sha256": preceremony.sha256_file(target),
        }
    if preceremony.exists_long_path(start_path):
        _validate_exact_marker(
            start_path,
            expected_start,
            label=f"formal {role} materialization start receipt",
        )
        raise FormalExecutorError(
            f"Formal {role} start exists without its receipt; rematerialization is forbidden"
        )
    _publish_or_validate_exact_marker(
        start_path,
        expected_start,
        label=f"formal {role} materialization start receipt",
    )
    rewire_key, selected = formal.load_train_m1_capability(
        role=role, execution_lock_path=execution_lock_path
    )
    if selected["execution_lock"] != lock or selected["draft"] != validated["draft"]:
        raise FormalExecutorError("M1 authority changed after start receipt")
    receipt = generator.write_m1_structural_receipt(
        private_root=paths["stage"] / "private",
        public_root=paths["stage"] / "public",
        replicate=replicate,
        rewire_key_hex=rewire_key,
        design_only=False,
        authority_reference={
            "execution_lock": _authority_reference(
                execution_lock_path, lock
            ),
            "m1_start_receipt": _repo_pin(
                start_path, include_self_hash=True
            ),
        },
    )
    if (
        receipt.get("status") != "PASS_FORMAL_M1_STRUCTURAL_REPLAY"
        or receipt.get("raw_rewire_key_persisted") is not False
        or int(receipt.get("mapping_count", -1)) != 189000
    ):
        raise FormalExecutorError("Formal M1 structural receipt drift")
    return {
        "status": "PASS_NEW_FORMAL_M1_RECEIPT",
        "replicate": replicate,
        "receipt_sha256": preceremony.sha256_file(target),
    }


def finalize_formal_stage(
    *,
    split: str,
    execution_lock_path: Path = formal.DEFAULT_EXECUTION_LOCK_PATH,
) -> dict[str, Any]:
    if split in {"train", "development"}:
        validated = formal.load_and_validate_execution_lock(execution_lock_path)
        active_lock = validated["execution_lock"]
    elif split in {"audit_a", "audit_b"}:
        validated = formal.load_and_validate_audit_lock(execution_lock_path)
        active_lock = validated["audit_lock"]
    else:
        raise FormalExecutorError("Unknown formal split finalizer scope")
    if active_lock["authorizations"][f"formal_{split}_generation"] is not True:
        raise FormalExecutorError(f"Formal split finalization is not authorized: {split}")
    paths = _paths(validated["draft"], split)
    core_marker = _validate_marker(
        paths["core_marker"],
        status="PASS_FORMAL_CORE_STAGE_COMPLETE",
        split=split,
    )
    start_path = _core_start_path(execution_lock_path, split)
    _validate_exact_marker(
        start_path,
        _expected_core_start(
            split=split,
            lock_path=execution_lock_path,
            lock=active_lock,
        ),
        label=f"formal {split} core start receipt",
    )
    if core_marker.get("core_start_receipt") != _repo_pin(
        start_path, include_self_hash=True
    ):
        raise FormalExecutorError("Formal core/start receipt lineage drift")
    if split == "train":
        for replicate in range(1, 6):
            role = f"m1_r{replicate:02d}"
            m1_start_path = _m1_start_path(execution_lock_path, role)
            _validate_exact_marker(
                m1_start_path,
                _expected_m1_start(
                    role=role,
                    lock_path=execution_lock_path,
                    lock=active_lock,
                ),
                label=f"formal {role} materialization start receipt",
            )
            receipt_path = (
                paths["stage"]
                / "private"
                / "m1"
                / f"r{replicate:02d}"
                / "structural_receipt.json"
            )
            receipt = preceremony.load_json_strict(receipt_path)
            preceremony.validate_canonical_self_hash(
                receipt, label=f"formal train M1 r{replicate:02d}"
            )
            if (
                receipt.get("status")
                != "PASS_FORMAL_M1_STRUCTURAL_REPLAY"
                or receipt.get("replicate") != f"r{replicate:02d}"
                or receipt.get("authority_reference", {}).get(
                    "m1_start_receipt"
                )
                != _repo_pin(m1_start_path, include_self_hash=True)
            ):
                raise FormalExecutorError("Formal train M1 receipt closure failed")
    if preceremony.exists_long_path(paths["finalized_marker"]):
        marker = _validate_marker(
            paths["finalized_marker"],
            status="PASS_FORMAL_STAGE_FINALIZED",
            split=split,
        )
        generator.validate_stage(
            output_root=paths["stage"],
            split=split,
            world_count=500,
            design_only=False,
        )
        return {
            "status": "PASS_RECOVERED_EXISTING_FORMAL_FINALIZED_STAGE",
            "split": split,
            "split_manifest_sha256": marker["split_manifest_sha256"],
        }
    manifest = generator.finalize_stage(
        output_root=paths["stage"],
        split=split,
        world_count=500,
        design_only=False,
    )
    marker = preceremony.with_canonical_self_hash(
        {
            "version": "2026-08-03-step28-v13-v1-12-finalized-marker-v1",
            "status": "PASS_FORMAL_STAGE_FINALIZED",
            "split": split,
            "world_count": 500,
            "pair_count": 189000,
            "split_manifest_sha256": preceremony.sha256_file(
                paths["stage"] / "public" / "split_manifest.json"
            ),
            "split_manifest_canonical_self_hash": manifest[
                "canonical_self_hash"
            ],
            "private_manifest_sha256": preceremony.sha256_file(
                paths["stage"] / "private" / "private_manifest.json"
            ),
            "quality_passed": False,
            "stage_published": False,
        }
    )
    _write_json_no_replace(paths["finalized_marker"], marker)
    return {
        "status": "PASS_NEW_FORMAL_FINALIZED_STAGE",
        "split": split,
        "split_manifest_sha256": marker["split_manifest_sha256"],
    }


def _validate_published_split(
    *,
    public_root: Path,
    private_root: Path,
    split: str,
    draft: Mapping[str, Any],
) -> dict[str, Any]:
    split_manifest = preceremony.load_json_strict(
        public_root / "split_manifest.json"
    )
    private_manifest = preceremony.load_json_strict(
        private_root / "private_manifest.json"
    )
    preceremony.validate_canonical_self_hash(
        split_manifest, label=f"published formal {split} split manifest"
    )
    preceremony.validate_canonical_self_hash(
        private_manifest, label=f"published formal {split} private manifest"
    )
    public_records = split_manifest.get("public_files")
    private_records = private_manifest.get("files")
    expected_public = set(draft["release"]["public_common_members"])
    expected_public.update(
        draft["release"]["public_supervision_members"][split]
    )
    expected_private = set(draft["release"]["private_common_members"])
    if split == "train":
        expected_private.update(
            draft["release"]["train_private_additional_members"]
        )
    if (
        split_manifest.get("version")
        != draft["release"]["split_manifest_version"]
        or private_manifest.get("version")
        != "2026-08-03-step28-v13-v1-12-private-manifest-v1"
        or split_manifest.get("status") != "PASS_FORMAL_PERSISTED_STAGE"
        or private_manifest.get("status") != "PASS_FORMAL_PRIVATE_STAGE"
        or split_manifest.get("split") != split
        or private_manifest.get("split") != split
        or split_manifest.get("run_id") != draft["run_id"]
        or split_manifest.get("design_only") is not False
        or private_manifest.get("design_only") is not False
        or split_manifest.get("formal_authorization_used") is not True
        or int(split_manifest.get("world_count", -1)) != 500
        or int(private_manifest.get("world_count", -1)) != 500
        or int(split_manifest.get("pair_count", -1)) != 189000
        or int(split_manifest.get("positive_count", -1)) != 10000
        or int(split_manifest.get("negative_count", -1)) != 179000
        or int(split_manifest.get("c40_member_count", -1)) != 0
        or split_manifest.get("scientific_metrics_produced") is not False
        or split_manifest.get("publication_occurred_before_manifest_freeze")
        is not False
        or not isinstance(public_records, list)
        or not isinstance(private_records, list)
        or not all(isinstance(record, Mapping) for record in public_records)
        or not all(isinstance(record, Mapping) for record in private_records)
        or int(split_manifest.get("public_file_count", -1))
        != len(public_records or [])
        or int(private_manifest.get("file_count", -1))
        != len(private_records or [])
        or {str(record.get("path", "")) for record in public_records or []}
        != expected_public - {"split_manifest.json"}
        or {str(record.get("path", "")) for record in private_records or []}
        != expected_private - {"private_manifest.json"}
        or preceremony.sha256_file(private_root / "private_manifest.json")
        != split_manifest.get("private_manifest_sha256")
        or private_manifest.get("canonical_self_hash")
        != split_manifest.get("private_manifest_canonical_self_hash")
    ):
        raise FormalExecutorError("Published formal split manifest drift")
    generator._validate_records(
        public_root,
        public_records,
        allowed_unlisted={"split_manifest.json"},
    )
    generator._validate_records(
        private_root,
        private_records,
        allowed_unlisted={"private_manifest.json"},
    )
    return {
        "public_manifest": split_manifest,
        "private_manifest": private_manifest,
    }


def _cleanup_published_stage(
    *, paths: Mapping[str, Path], split: str
) -> None:
    """Remove only validated state markers left after an interrupted publish."""

    stage = paths["stage"]
    if not preceremony.exists_long_path(stage):
        return
    if preceremony.exists_long_path(
        stage / "public"
    ) or preceremony.exists_long_path(stage / "private"):
        raise FormalExecutorError(
            "Published finals coexist with staged payload directories"
        )
    expected = {
        "CORE_COMPLETE.json": (
            paths["core_marker"],
            "PASS_FORMAL_CORE_STAGE_COMPLETE",
        ),
        "STAGE_FINALIZED.json": (
            paths["finalized_marker"],
            "PASS_FORMAL_STAGE_FINALIZED",
        ),
        "QUALITY_PASS.json": (
            paths["quality_marker"],
            "PASS_FORMAL_STAGE_QUALITY",
        ),
    }
    with os.scandir(preceremony._filesystem_path(stage)) as entries:
        observed = {entry.name for entry in entries}
    extras = observed - set(expected)
    if extras:
        raise FormalExecutorError(
            f"Unexpected residual publication-stage members: {sorted(extras)}"
        )
    for name in sorted(observed):
        marker_path, status = expected[name]
        _validate_marker(marker_path, status=status, split=split)
    for name in sorted(observed):
        marker_path, _status = expected[name]
        os.unlink(preceremony._filesystem_path(marker_path))
    os.rmdir(preceremony._filesystem_path(stage))
    try:
        os.rmdir(preceremony._filesystem_path(stage.parent))
    except OSError:
        # Other split stages may still be intentionally present.
        pass


def publish_formal_stage(
    *,
    split: str,
    execution_lock_path: Path = formal.DEFAULT_EXECUTION_LOCK_PATH,
) -> dict[str, Any]:
    if split in {"train", "development"}:
        validated = formal.load_and_validate_execution_lock(execution_lock_path)
        active_lock = validated["execution_lock"]
        expected_quality_status = (
            "PASS_FORMAL_TRAIN_DEVELOPMENT_QUALITY_GATE"
        )
    elif split in {"audit_a", "audit_b"}:
        validated = formal.load_and_validate_audit_lock(execution_lock_path)
        active_lock = validated["audit_lock"]
        expected_quality_status = "PASS_FORMAL_SEALED_AUDIT_SPLIT_QUALITY"
    else:
        raise FormalExecutorError("Unknown formal publisher scope")
    if active_lock["authorizations"][f"formal_{split}_generation"] is not True:
        raise FormalExecutorError(f"Formal split publication is not authorized: {split}")
    draft = validated["draft"]
    paths = _paths(draft, split)
    stage = paths["stage"]
    public_final = paths["public_final"]
    private_final = paths["private_final"]
    receipt_path = execution_lock_path.parent / f"{split}_publication_receipt.json"
    core_start_path = _core_start_path(execution_lock_path, split)
    _validate_exact_marker(
        core_start_path,
        _expected_core_start(
            split=split,
            lock_path=execution_lock_path,
            lock=active_lock,
        ),
        label=f"formal {split} core start receipt",
    )
    m1_start_pins: dict[str, dict[str, Any]] = {}
    if split == "train":
        for role in formal.M1_ROLES:
            start_path = _m1_start_path(execution_lock_path, role)
            _validate_exact_marker(
                start_path,
                _expected_m1_start(
                    role=role,
                    lock_path=execution_lock_path,
                    lock=active_lock,
                ),
                label=f"formal {role} materialization start receipt",
            )
            m1_start_pins[role] = _repo_pin(
                start_path, include_self_hash=True
            )

    if preceremony.exists_long_path(receipt_path):
        receipt = preceremony.load_json_strict(receipt_path)
        preceremony.validate_canonical_self_hash(
            receipt, label=f"formal {split} publication receipt"
        )
        published = _validate_published_split(
            public_root=public_final,
            private_root=private_final,
            split=split,
            draft=draft,
        )
        quality_spec = receipt.get("quality_receipt", {})
        if not isinstance(quality_spec, Mapping) or not quality_spec.get("path"):
            raise FormalExecutorError("Existing publication quality pin is malformed")
        quality_path = preceremony._repo_path(str(quality_spec["path"]))
        quality_document = preceremony.load_json_strict(quality_path)
        preceremony.validate_canonical_self_hash(
            quality_document,
            label=f"recovered formal {split} quality receipt",
        )
        if (
            receipt.get("status")
            != "PASS_FORMAL_SPLIT_PUBLISHED_NO_REPLACE"
            or receipt.get("run_id") != draft["run_id"]
            or receipt.get("split") != split
            or receipt.get("public_root")
            != public_final.relative_to(ROOT).as_posix()
            or receipt.get("private_root")
            != private_final.relative_to(ROOT).as_posix()
            or receipt.get("public_manifest_sha256")
            != preceremony.sha256_file(public_final / "split_manifest.json")
            or receipt.get("public_manifest_canonical_self_hash")
            != published["public_manifest"]["canonical_self_hash"]
            or receipt.get("private_manifest_sha256")
            != preceremony.sha256_file(private_final / "private_manifest.json")
            or receipt.get("private_manifest_canonical_self_hash")
            != published["private_manifest"]["canonical_self_hash"]
            or preceremony.sha256_file(quality_path) != quality_spec.get("sha256")
            or quality_document.get("canonical_self_hash")
            != quality_spec.get("canonical_self_hash")
            or quality_document.get("status") != expected_quality_status
            or (
                split in {"audit_a", "audit_b"}
                and quality_document.get("split") != split
            )
            or receipt.get("no_replace_publish") is not True
            or receipt.get("formal_seed_or_capability_persisted_publicly")
            is not False
            or receipt.get("core_start_receipt")
            != _repo_pin(core_start_path, include_self_hash=True)
            or receipt.get("m1_start_receipts") != m1_start_pins
        ):
            raise FormalExecutorError("Existing publication receipt status drift")
        for marker_name, marker_path in (
            ("finalized_marker_sha256", paths["finalized_marker"]),
            ("quality_marker_sha256", paths["quality_marker"]),
        ):
            if preceremony.exists_long_path(marker_path) and preceremony.sha256_file(
                marker_path
            ) != receipt.get(marker_name):
                raise FormalExecutorError(
                    f"Existing publication {marker_name} pin drift"
                )
        _cleanup_published_stage(paths=paths, split=split)
        return {
            "status": "PASS_RECOVERED_EXISTING_FORMAL_SPLIT_PUBLICATION",
            "split": split,
            "publication_receipt_sha256": preceremony.sha256_file(receipt_path),
        }
    if not preceremony.exists_long_path(stage):
        raise FormalExecutorError(
            "Formal stage is missing and has no publication receipt"
        )
    finalized = _validate_marker(
        paths["finalized_marker"],
        status="PASS_FORMAL_STAGE_FINALIZED",
        split=split,
    )
    quality = _validate_marker(
        paths["quality_marker"],
        status="PASS_FORMAL_STAGE_QUALITY",
        split=split,
    )
    quality_receipt_path = preceremony._repo_path(
        str(quality["quality_receipt"]["path"])
    )
    quality_receipt = preceremony.load_json_strict(quality_receipt_path)
    preceremony.validate_canonical_self_hash(
        quality_receipt, label="formal train/development quality receipt"
    )
    if (
        quality_receipt.get("status") != expected_quality_status
        or (
            split in {"audit_a", "audit_b"}
            and quality_receipt.get("split") != split
        )
        or preceremony.sha256_file(quality_receipt_path)
        != quality["quality_receipt"]["sha256"]
    ):
        raise FormalExecutorError("Formal quality receipt pin drift")

    os.makedirs(
        preceremony._filesystem_path(private_final.parent), exist_ok=True
    )
    os.makedirs(
        preceremony._filesystem_path(public_final.parent), exist_ok=True
    )
    staged_private = stage / "private"
    staged_public = stage / "public"
    if not preceremony.exists_long_path(private_final):
        if not preceremony.exists_long_path(staged_private):
            raise FormalExecutorError("Private publish source disappeared")
        os.rename(
            preceremony._filesystem_path(staged_private),
            preceremony._filesystem_path(private_final),
        )
    elif preceremony.exists_long_path(staged_private):
        raise FormalExecutorError("Both staged and final private split exist")
    if not preceremony.exists_long_path(public_final):
        if not preceremony.exists_long_path(staged_public):
            raise FormalExecutorError("Public publish source disappeared")
        os.rename(
            preceremony._filesystem_path(staged_public),
            preceremony._filesystem_path(public_final),
        )
    elif preceremony.exists_long_path(staged_public):
        raise FormalExecutorError("Both staged and final public split exist")
    published = _validate_published_split(
        public_root=public_final,
        private_root=private_final,
        split=split,
        draft=draft,
    )
    receipt = preceremony.with_canonical_self_hash(
        {
            "version": "2026-08-03-step28-v13-v1-12-publication-v1",
            "status": "PASS_FORMAL_SPLIT_PUBLISHED_NO_REPLACE",
            "run_id": draft["run_id"],
            "split": split,
            "world_count": 500,
            "pair_count": 189000,
            "public_root": public_final.relative_to(ROOT).as_posix(),
            "private_root": private_final.relative_to(ROOT).as_posix(),
            "public_manifest_sha256": preceremony.sha256_file(
                public_final / "split_manifest.json"
            ),
            "public_manifest_canonical_self_hash": published[
                "public_manifest"
            ]["canonical_self_hash"],
            "private_manifest_sha256": preceremony.sha256_file(
                private_final / "private_manifest.json"
            ),
            "private_manifest_canonical_self_hash": published[
                "private_manifest"
            ]["canonical_self_hash"],
            "quality_receipt": dict(quality["quality_receipt"]),
            "core_start_receipt": _repo_pin(
                core_start_path, include_self_hash=True
            ),
            "m1_start_receipts": m1_start_pins,
            "finalized_marker_sha256": preceremony.sha256_file(
                paths["finalized_marker"]
            ),
            "quality_marker_sha256": preceremony.sha256_file(
                paths["quality_marker"]
            ),
            "no_replace_publish": True,
            "formal_seed_or_capability_persisted_publicly": False,
        }
    )
    _write_json_no_replace(receipt_path, receipt)
    _cleanup_published_stage(paths=paths, split=split)
    return {
        "status": "PASS_NEW_FORMAL_SPLIT_PUBLICATION",
        "split": split,
        "publication_receipt_sha256": preceremony.sha256_file(receipt_path),
    }


def authorize_audit_a_generation(
    *,
    execution_lock_path: Path = formal.DEFAULT_EXECUTION_LOCK_PATH,
    output: Path = formal.DEFAULT_AUDIT_A_LOCK_PATH,
) -> dict[str, Any]:
    formal.require_canonical_path(
        output,
        formal.DEFAULT_AUDIT_A_LOCK_PATH,
        label="v1.12 Audit A generation lock",
    )
    validated = formal.load_and_validate_execution_lock(execution_lock_path)
    draft = validated["draft"]
    parent_lock = validated["execution_lock"]
    quality_path = preceremony._repo_path(
        str(
            validated["prelock"]["custody"][
                "train_development_quality_receipt_path"
            ]
        )
    )
    quality = preceremony.load_json_strict(quality_path)
    preceremony.validate_canonical_self_hash(
        quality, label="formal train/development quality receipt"
    )
    if (
        quality.get("status")
        != "PASS_FORMAL_TRAIN_DEVELOPMENT_QUALITY_GATE"
        or quality.get("c40_generated_or_read") is not False
        or quality.get("audit_a_or_b_truth_read") is not False
    ):
        raise FormalExecutorError("Train/development quality does not authorize audits")
    publication_pins: dict[str, dict[str, Any]] = {}
    for split in ("train", "development"):
        paths = _paths(draft, split)
        _validate_published_split(
            public_root=paths["public_final"],
            private_root=paths["private_final"],
            split=split,
            draft=draft,
        )
        receipt_path = execution_lock_path.parent / f"{split}_publication_receipt.json"
        receipt = preceremony.load_json_strict(receipt_path)
        preceremony.validate_canonical_self_hash(
            receipt, label=f"formal {split} publication receipt"
        )
        if (
            receipt.get("status")
            != "PASS_FORMAL_SPLIT_PUBLISHED_NO_REPLACE"
            or receipt.get("quality_receipt")["sha256"]
            != preceremony.sha256_file(quality_path)
        ):
            raise FormalExecutorError("Train/development publication evidence drift")
        core_start_path = _core_start_path(execution_lock_path, split)
        _validate_exact_marker(
            core_start_path,
            _expected_core_start(
                split=split,
                lock_path=execution_lock_path,
                lock=parent_lock,
            ),
            label=f"formal {split} core start receipt",
        )
        if receipt.get("core_start_receipt") != _repo_pin(
            core_start_path, include_self_hash=True
        ):
            raise FormalExecutorError("Publication/core start lineage drift")
        expected_m1_starts: dict[str, dict[str, Any]] = {}
        if split == "train":
            for role in formal.M1_ROLES:
                start_path = _m1_start_path(execution_lock_path, role)
                _validate_exact_marker(
                    start_path,
                    _expected_m1_start(
                        role=role,
                        lock_path=execution_lock_path,
                        lock=parent_lock,
                    ),
                    label=f"formal {role} start receipt",
                )
                expected_m1_starts[role] = _repo_pin(
                    start_path, include_self_hash=True
                )
        if receipt.get("m1_start_receipts") != expected_m1_starts:
            raise FormalExecutorError("Publication/M1 start lineage drift")
        publication_pins[split] = _repo_pin(
            receipt_path, include_self_hash=True
        )
    audit_lock = preceremony.with_canonical_self_hash(
        {
            "version": "2026-08-03-step28-v13-v1-12-audit-a-generation-lock-v1",
            "status": "READY_FOR_AUDIT_A_GENERATION_ONLY",
            "run_id": draft["run_id"],
            "authorizations": {
                "formal_seed_ceremony": False,
                "formal_train_generation": False,
                "formal_development_generation": False,
                "formal_audit_a_generation": True,
                "formal_audit_b_generation": False,
                "model_training": False,
                "audit_truth_unsealing": False,
            },
            "parent_execution_lock": _repo_pin(
                execution_lock_path, include_self_hash=True
            ),
            "train_development_quality_receipt": _repo_pin(
                quality_path, include_self_hash=True
            ),
            "train_development_publication_receipts": publication_pins,
            "source_closure_canonical_sha256": parent_lock[
                "source_closure_canonical_sha256"
            ],
            "master_commitments": parent_lock["master_commitments"],
            "generator_capability_commitments": parent_lock[
                "generator_capability_commitments"
            ],
            "private_generator_capability_files": parent_lock[
                "private_generator_capability_files"
            ],
            "formal_split_order": ["audit_a"],
            "audit_b_requires_published_audit_a": True,
            "audit_truth_or_qrels_read_before_prediction": False,
            "model_training_authorized": False,
            "seed_or_capability_replacement_forbidden": True,
        }
    )
    if preceremony.exists_long_path(output):
        observed = preceremony.load_json_strict(output)
        if preceremony.canonical_json_bytes(observed) != preceremony.canonical_json_bytes(
            audit_lock
        ):
            raise FormalExecutorError("Existing Audit A lock differs")
    else:
        _write_json_no_replace(output, audit_lock)
    formal.load_and_validate_audit_lock(output)
    return {
        "status": "PASS_AUDIT_A_GENERATION_AUTHORIZED_ONLY",
        "audit_a_lock_sha256": preceremony.sha256_file(output),
        "audit_truth_or_qrels_read": False,
    }


def authorize_audit_b_generation(
    *,
    audit_a_lock_path: Path = formal.DEFAULT_AUDIT_A_LOCK_PATH,
    output: Path = formal.DEFAULT_AUDIT_B_LOCK_PATH,
) -> dict[str, Any]:
    formal.require_canonical_path(
        output,
        formal.DEFAULT_AUDIT_B_LOCK_PATH,
        label="v1.12 Audit B generation lock",
    )
    validated = formal.load_and_validate_audit_lock(audit_a_lock_path)
    draft = validated["draft"]
    parent_lock = validated["audit_a_lock"]
    paths = _paths(draft, "audit_a")
    published = _validate_published_split(
        public_root=paths["public_final"],
        private_root=paths["private_final"],
        split="audit_a",
        draft=draft,
    )
    publication_path = (
        audit_a_lock_path.parent / "audit_a_publication_receipt.json"
    )
    publication = preceremony.load_json_strict(publication_path)
    preceremony.validate_canonical_self_hash(
        publication, label="formal Audit A publication receipt"
    )
    quality_spec = publication.get("quality_receipt", {})
    quality_path = preceremony._repo_path(str(quality_spec.get("path", "")))
    quality = preceremony.load_json_strict(quality_path)
    preceremony.validate_canonical_self_hash(
        quality, label="formal Audit A sealed quality receipt"
    )
    if (
        publication.get("status")
        != "PASS_FORMAL_SPLIT_PUBLISHED_NO_REPLACE"
        or publication.get("run_id") != draft["run_id"]
        or publication.get("split") != "audit_a"
        or publication.get("public_manifest_sha256")
        != preceremony.sha256_file(paths["public_final"] / "split_manifest.json")
        or publication.get("public_manifest_canonical_self_hash")
        != published["public_manifest"]["canonical_self_hash"]
        or publication.get("private_manifest_sha256")
        != preceremony.sha256_file(paths["private_final"] / "private_manifest.json")
        or publication.get("private_manifest_canonical_self_hash")
        != published["private_manifest"]["canonical_self_hash"]
        or quality_path
        != audit_a_lock_path.parent / "audit_a_quality_gate.json"
        or preceremony.sha256_file(quality_path) != quality_spec.get("sha256")
        or quality.get("canonical_self_hash")
        != quality_spec.get("canonical_self_hash")
        or quality.get("status")
        != "PASS_FORMAL_SEALED_AUDIT_SPLIT_QUALITY"
        or quality.get("split") != "audit_a"
        or quality.get("classification_labels_parsed") is not False
        or quality.get("retrieval_qrels_parsed") is not False
        or quality.get("controller_membership_parsed") is not False
        or quality.get("model_training_or_prediction_started") is not False
    ):
        raise FormalExecutorError(
            "Audit A publication does not authorize Audit B"
        )
    audit_b_lock = preceremony.with_canonical_self_hash(
        {
            "version": "2026-08-03-step28-v13-v1-12-audit-b-generation-lock-v1",
            "status": "READY_FOR_AUDIT_B_GENERATION_ONLY",
            "run_id": draft["run_id"],
            "authorizations": {
                "formal_seed_ceremony": False,
                "formal_train_generation": False,
                "formal_development_generation": False,
                "formal_audit_a_generation": False,
                "formal_audit_b_generation": True,
                "model_training": False,
                "audit_truth_unsealing": False,
            },
            "parent_audit_a_lock": _repo_pin(
                audit_a_lock_path, include_self_hash=True
            ),
            "audit_a_publication_receipt": _repo_pin(
                publication_path, include_self_hash=True
            ),
            "audit_a_quality_receipt": _repo_pin(
                quality_path, include_self_hash=True
            ),
            "source_closure_canonical_sha256": parent_lock[
                "source_closure_canonical_sha256"
            ],
            "master_commitments": parent_lock["master_commitments"],
            "generator_capability_commitments": parent_lock[
                "generator_capability_commitments"
            ],
            "private_generator_capability_files": parent_lock[
                "private_generator_capability_files"
            ],
            "formal_split_order": ["audit_b"],
            "audit_a_published_before_authorization": True,
            "audit_truth_or_qrels_read_before_prediction": False,
            "model_training_authorized": False,
            "seed_or_capability_replacement_forbidden": True,
        }
    )
    if preceremony.exists_long_path(output):
        observed = preceremony.load_json_strict(output)
        if preceremony.canonical_json_bytes(
            observed
        ) != preceremony.canonical_json_bytes(audit_b_lock):
            raise FormalExecutorError("Existing Audit B lock differs")
    else:
        _write_json_no_replace(output, audit_b_lock)
    formal.load_and_validate_audit_lock(output)
    return {
        "status": "PASS_AUDIT_B_GENERATION_AUTHORIZED_AFTER_AUDIT_A",
        "audit_b_lock_sha256": preceremony.sha256_file(output),
        "audit_truth_or_qrels_read": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--generate-core", action="store_true")
    action.add_argument("--materialize-m1", action="store_true")
    action.add_argument("--finalize-stage", action="store_true")
    action.add_argument("--publish-stage", action="store_true")
    action.add_argument("--authorize-audit-a", action="store_true")
    action.add_argument("--authorize-audit-b", action="store_true")
    parser.add_argument("--split", choices=formal.SPLITS)
    parser.add_argument("--replicate", type=int)
    parser.add_argument(
        "--execution-lock",
        type=Path,
        default=None,
    )
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.execution_lock is not None:
        lock_path = args.execution_lock.resolve()
    elif args.split == "audit_a":
        lock_path = formal.DEFAULT_AUDIT_A_LOCK_PATH
    elif args.split == "audit_b":
        lock_path = formal.DEFAULT_AUDIT_B_LOCK_PATH
    else:
        lock_path = formal.DEFAULT_EXECUTION_LOCK_PATH
    if args.generate_core:
        if args.split is None:
            raise FormalExecutorError("--generate-core requires --split")
        result = generate_core(
            split=args.split,
            execution_lock_path=lock_path,
            progress_every=args.progress_every,
        )
    elif args.materialize_m1:
        if args.replicate is None or args.split not in {None, "train"}:
            raise FormalExecutorError(
                "--materialize-m1 requires --replicate and train scope"
            )
        result = materialize_train_m1(
            replicate=args.replicate, execution_lock_path=lock_path
        )
    elif args.finalize_stage:
        if args.split is None:
            raise FormalExecutorError("--finalize-stage requires --split")
        result = finalize_formal_stage(
            split=args.split, execution_lock_path=lock_path
        )
    elif args.publish_stage:
        if args.split is None:
            raise FormalExecutorError("--publish-stage requires --split")
        result = publish_formal_stage(
            split=args.split, execution_lock_path=lock_path
        )
    elif args.authorize_audit_a:
        if args.split is not None or args.replicate is not None:
            raise FormalExecutorError(
                "--authorize-audit-a takes no split/replicate"
            )
        result = authorize_audit_a_generation(
            execution_lock_path=lock_path,
            output=formal.DEFAULT_AUDIT_A_LOCK_PATH,
        )
    else:
        if args.split is not None or args.replicate is not None:
            raise FormalExecutorError(
                "--authorize-audit-b takes no split/replicate"
            )
        if args.execution_lock is None:
            lock_path = formal.DEFAULT_AUDIT_A_LOCK_PATH
        result = authorize_audit_b_generation(
            audit_a_lock_path=lock_path,
            output=formal.DEFAULT_AUDIT_B_LOCK_PATH,
        )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
