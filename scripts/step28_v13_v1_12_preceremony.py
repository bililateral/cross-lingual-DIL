#!/usr/bin/env python3
"""Validate the Step28-v13 v1.12 clean-room pre-ceremony boundary.

This module is deliberately incapable of creating formal seeds or starting
formal generation/training.  It validates the compact failed-run exclusions,
the reusable successful-v1.2 source closure, the nine historical failure
regressions, and a two-world design-only end-to-end replay.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import hmac
import itertools
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter, deque
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_PATH = (
    ROOT / "schema" / "step28_v13_v1_12_cleanroom_preceremony_policy.json"
)
SPLITS = ("train", "development", "audit_a", "audit_b")
FAILED_RUNS = tuple(f"v1.{value}" for value in range(3, 12))
SPLIT_MANIFEST_VERSION = (
    "2026-08-03-step28-v13-v1-12-cleanroom-split-manifest-v1"
)
HEX_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
UID_PATTERNS = {
    "world_uid": re.compile(r"^w_[0-9a-f]{64}$"),
    "seller_uid": re.compile(r"^sel_[0-9a-f]{64}$"),
    "item_uid": re.compile(r"^itm_[0-9a-f]{64}$"),
}


class PreceremonyError(ValueError):
    """Raised when a frozen pre-ceremony boundary fails closed."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def with_canonical_self_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    if "canonical_self_hash" in payload:
        raise PreceremonyError("Self-hash field must be the final derived field")
    output = dict(payload)
    output["canonical_self_hash"] = canonical_sha256(output)
    return output


def validate_canonical_self_hash(
    document: Mapping[str, Any], *, label: str
) -> None:
    claimed = document.get("canonical_self_hash")
    if not isinstance(claimed, str) or HEX_SHA256_RE.fullmatch(claimed) is None:
        raise PreceremonyError(f"{label} lacks a canonical self-hash")
    payload = dict(document)
    payload.pop("canonical_self_hash")
    observed = canonical_sha256(payload)
    if observed != claimed:
        raise PreceremonyError(
            f"{label} canonical self-hash drift: expected={claimed} "
            f"observed={observed}"
        )


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise PreceremonyError(f"Duplicate JSON key: {key}")
        output[key] = value
    return output


def _repo_path(relative_path: str) -> Path:
    candidate = Path(str(relative_path))
    if candidate.is_absolute():
        raise PreceremonyError("Repository artifact path must be relative")
    root = ROOT.resolve()
    resolved = (ROOT / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PreceremonyError("Repository artifact path escapes the root") from exc
    return resolved


def _filesystem_path(path: Path) -> str:
    """Return one path representation for Windows write/read/stat/walk."""

    absolute = str(path.resolve())
    if os.name != "nt" or absolute.startswith("\\\\?\\"):
        return absolute
    if absolute.startswith("\\\\"):
        return "\\\\?\\UNC\\" + absolute[2:]
    return "\\\\?\\" + absolute


def read_bytes_long_path(path: Path) -> bytes:
    with open(_filesystem_path(path), "rb") as handle:
        return handle.read()


def stat_long_path(path: Path) -> os.stat_result:
    return os.stat(_filesystem_path(path))


def exists_long_path(path: Path) -> bool:
    try:
        stat_long_path(path)
    except FileNotFoundError:
        return False
    return True


def write_bytes_no_replace_long_path(path: Path, payload: bytes) -> None:
    if not isinstance(payload, bytes):
        raise TypeError("Long-path writer accepts bytes only")
    os.makedirs(_filesystem_path(path.parent), exist_ok=True)
    with open(_filesystem_path(path), "xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def walk_files_long_path(root: Path) -> list[Path]:
    filesystem_root = _filesystem_path(root)
    output: list[Path] = []
    for directory, names, filenames in os.walk(filesystem_root):
        names.sort()
        filenames.sort()
        for filename in filenames:
            relative = os.path.relpath(os.path.join(directory, filename), filesystem_root)
            output.append(root / Path(relative))
    return sorted(output, key=lambda value: value.as_posix().encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(_filesystem_path(path), "rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_json_strict(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            read_bytes_long_path(path).decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreceremonyError(f"Invalid UTF-8 JSON: {path}") from exc
    if not isinstance(value, dict):
        raise PreceremonyError(f"Expected a JSON object: {path}")
    return value


def verify_file_pin(spec: Mapping[str, Any], *, label: str) -> Path:
    if not {"path", "sha256", "size_bytes"}.issubset(spec):
        raise PreceremonyError(f"Malformed file pin for {label}")
    path = _repo_path(str(spec["path"]))
    try:
        size = stat_long_path(path).st_size
    except FileNotFoundError as exc:
        raise PreceremonyError(f"Missing pinned file for {label}: {path}") from exc
    expected_size = int(spec["size_bytes"])
    expected_sha = str(spec["sha256"])
    if size != expected_size or sha256_file(path) != expected_sha:
        raise PreceremonyError(f"Pinned bytes drift for {label}: {spec['path']}")
    return path


def _validate_strictly_sorted_hashes(
    values: Any, *, expected_count: int, label: str
) -> None:
    if not isinstance(values, list) or len(values) != expected_count:
        raise PreceremonyError(f"{label} count drift")
    previous: str | None = None
    for value in values:
        if not isinstance(value, str) or HEX_SHA256_RE.fullmatch(value) is None:
            raise PreceremonyError(f"{label} contains a malformed SHA-256")
        if previous is not None and value <= previous:
            raise PreceremonyError(f"{label} is not strictly sorted and unique")
        previous = value


def load_failed_exclusions(
    spec: Mapping[str, Any],
) -> tuple[frozenset[str], frozenset[str]]:
    path = verify_file_pin(spec, label="failed identity exclusion archive")
    archive = load_json_strict(path)
    validate_canonical_self_hash(archive, label="failed identity exclusion archive")
    if (
        archive.get("version")
        != "2026-08-03-step28-v13-failed-identity-exclusion-archive-v1"
        or archive.get("status")
        != "PASS_FAILED_RUN_HASH_ONLY_EXCLUSION_ARCHIVE"
        or archive.get("scope_through_failed_run")
        != "v13_training_ready_v1_11_20260803"
        or archive.get("raw_identity_values_persisted") is not False
        or archive.get("raw_private_keys_persisted") is not False
        or archive.get("scientific_metrics_produced") is not False
        or archive.get("future_registry_must_not_require_deleted_failed_payloads")
        is not True
        or archive.get("canonical_self_hash") != spec["canonical_self_hash"]
    ):
        raise PreceremonyError("Failed identity exclusion archive boundary drift")
    combined_count = int(spec["combined_unique_value_hash_count"])
    commitment_count = int(spec["forbidden_master_seed_commitment_count"])
    if (
        int(archive.get("combined_unique_value_hash_count", -1)) != combined_count
        or int(archive.get("base_unique_value_hash_count", -1))
        + int(archive.get("v1_11_added_unique_value_hash_count", -1))
        != combined_count
    ):
        raise PreceremonyError("Failed identity exclusion count arithmetic drift")
    hashes = archive.get("combined_value_hashes")
    commitments = archive.get("forbidden_master_seed_commitments")
    _validate_strictly_sorted_hashes(
        hashes, expected_count=combined_count, label="failed identity hashes"
    )
    _validate_strictly_sorted_hashes(
        commitments,
        expected_count=commitment_count,
        label="forbidden master commitments",
    )
    return frozenset(hashes), frozenset(commitments)


def validate_failed_cleanup(spec: Mapping[str, Any]) -> dict[str, Any]:
    path = verify_file_pin(spec, label="failed file cleanup manifest")
    manifest = load_json_strict(path)
    validate_canonical_self_hash(manifest, label="failed file cleanup manifest")
    files = manifest.get("failed_code_policy_and_test_files")
    if (
        manifest.get("version")
        != "2026-08-03-step28-v13-failed-file-cleanup-manifest-v1"
        or manifest.get("status") != "PASS_PREDELETE_HASH_ONLY_FILE_MANIFEST"
        or manifest.get("canonical_self_hash") != spec["canonical_self_hash"]
        or manifest.get("failed_runs") != list(FAILED_RUNS)
        or manifest.get("successful_v1_2_release_deleted") is not False
        or manifest.get("git_tracked_step28_v13_release_and_audit_code_deleted")
        is not False
        or not isinstance(files, list)
        or len(files) != int(spec["failed_file_count"])
        or int(manifest.get("failed_code_policy_and_test_file_count", -1))
        != int(spec["failed_file_count"])
    ):
        raise PreceremonyError("Failed cleanup manifest boundary drift")
    for record in files:
        if set(record) != {"disposition", "path", "sha256", "size_bytes"}:
            raise PreceremonyError("Failed cleanup file-record schema drift")
        if (
            record["disposition"]
            != "DELETE_FAILED_RUN_SPECIFIC_OR_UNCOMMITTED_CHAIN_CODE"
        ):
            raise PreceremonyError("Unexpected cleanup disposition")
        if exists_long_path(_repo_path(str(record["path"]))):
            raise PreceremonyError(
                f"Deleted failed-version file was restored: {record['path']}"
            )
    retained = manifest.get("retained_records")
    if (
        not isinstance(retained, Mapping)
        or int(retained.get("failed_identity_hash_count", -1)) != 915996
        or retained.get("raw_identity_values_or_private_keys_retained") is not False
        or int(retained.get("small_failure_receipt_count", -1)) != 7
    ):
        raise PreceremonyError("Failed cleanup retained-record boundary drift")
    removed_roots: list[Path] = []
    removed_roots.extend(
        _repo_path(
            "reports/step28_synthetic_chinese_dataset/" + str(run_name)
        )
        for run_name in manifest.get("deleted_public_run_roots", [])
    )
    removed_roots.extend(
        _repo_path(
            "reports/step28_synthetic_chinese_dataset/private_custody/"
            + str(record["name"])
        )
        for record in manifest.get("deleted_private_custody_roots", [])
    )
    removed_roots.extend(
        _repo_path("reports/step28_v13_identity_transfer/" + str(run_name))
        for run_name in manifest.get("deleted_preexecution_roots", [])
    )
    restored_roots = [
        path.relative_to(ROOT).as_posix()
        for path in removed_roots
        if exists_long_path(path)
    ]
    if restored_roots:
        raise PreceremonyError(
            f"Deleted failed-run roots were restored: {restored_roots}"
        )
    return manifest


def _walk_mapping_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key)
            yield from _walk_mapping_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_mapping_keys(child)


def validate_policy(policy_path: Path = DEFAULT_POLICY_PATH) -> dict[str, Any]:
    policy = load_json_strict(policy_path)
    validate_canonical_self_hash(policy, label="v1.12 pre-ceremony policy")
    if (
        policy.get("version")
        != "2026-08-03-step28-v13-v1-12-cleanroom-preceremony-policy-v1"
        or policy.get("status")
        != "DESIGN_VALIDATION_ONLY_NO_FORMAL_AUTHORIZATION"
        or policy.get("run_id")
        != "v13_training_ready_v1_12_cleanroom_20260803"
        or set(policy.get("authorizations", {}))
        != {
            "formal_seed_generation",
            "formal_dataset_generation",
            "model_training",
            "audit_truth_unsealing",
        }
        or set(policy.get("authorizations", {}).values()) != {False}
        or policy.get("lineage", {}).get("failed_runs_permanently_closed")
        != list(FAILED_RUNS)
    ):
        raise PreceremonyError("v1.12 policy identity or authorization drift")
    scope = policy.get("implementation_scope", {})
    if (
        scope.get("preceremony_validator_and_design_preflight") is not True
        or scope.get("formal_seed_ceremony_implemented") is not False
        or scope.get("formal_split_generator_implemented") is not False
        or scope.get("model_training_implemented") is not False
    ):
        raise PreceremonyError("Current implementation-scope claim drift")
    forbidden_secret_keys = {
        key
        for key in _walk_mapping_keys(policy)
        if key.endswith("_key_hex")
        or key.endswith("_seed_hex")
        or key in {"master_seed", "master_seed_hex"}
    }
    if forbidden_secret_keys:
        raise PreceremonyError(
            f"Public pre-ceremony policy contains secret fields: {sorted(forbidden_secret_keys)}"
        )
    design = policy["dataset_design"]
    if (
        design["split_order"] != list(SPLITS)
        or int(design["worlds_per_split"]) != 500
        or int(design["sellers_per_world"]) != 28
        or int(design["controllers_per_world"]) != 12
        or int(design["mechanism_slots_per_world"]) != 12
        or design["mechanism_slot_unique_key"]
        != ["world_uid", "mechanism_slot_uid"]
        or int(design["complete_pairs_per_world"]) != math.comb(28, 2)
        or int(design["positive_pairs_per_world"]) != 20
        or int(design["negative_pairs_per_world"]) != 358
        or int(design["pairs_per_split"]) != 500 * 378
        or int(design["positive_pairs_per_split"]) != 500 * 20
        or int(design["negative_pairs_per_split"]) != 500 * 358
        or design["random_ap_baseline"] != "20/378"
        or design["classification_universe"] != "all_unordered_pairs_in_K28"
        or design["c40_role"]
        != "private_post_prediction_mechanism_diagnostic_only"
        or int(design["retrieval_queries_per_world"]) != 28
        or int(design["retrieval_gallery_per_query"]) != 27
    ):
        raise PreceremonyError("Full-378 dataset design drift")
    models = policy["models"]
    if (
        models["m0"] != "frozen_real_english_LightGBM_legacy18_LaBSE"
        or int(models["m1_replicates"]) != 5
        or models["m1_mapping"]
        != "per_world_full378_endpoint_disjoint_whole_identity33_row_bijection"
        or models["m2"]
        != "same_as_m1_except_correct_identity33_alignment"
        or models["m3"] != ["M3_base", "M3_joint"]
        or models["primary_identity_contrasts"]
        != ["M2_minus_mean_M1", "M2_minus_each_M1"]
    ):
        raise PreceremonyError("M0/M1/M2/M3 scientific role drift")
    expected_classification_metrics = [
        "roc_auc",
        "average_precision",
        "trapezoidal_pr_auc",
        "precision",
        "recall",
        "f1",
        "specificity",
        "balanced_accuracy",
        "mcc",
        "brier",
        "log_loss",
        "recall_at_fpr_0_01",
    ]
    expected_retrieval_metrics = [
        "mrr",
        "map",
        "recall_at_1",
        "recall_at_3",
        "recall_at_5",
        "recall_at_10",
        "ndcg_at_1",
        "ndcg_at_3",
        "ndcg_at_5",
        "ndcg_at_10",
    ]
    metrics = policy["metrics"]
    if (
        metrics["classification"] != expected_classification_metrics
        or metrics["retrieval"] != expected_retrieval_metrics
        or metrics["resampling_unit"] != "world"
        or metrics["development_freezes_threshold_only"] is not True
    ):
        raise PreceremonyError("Required reporting metric contract drift")
    custody = policy["custody"]
    if (
        custody["private_root_must_be_git_ignored"] is not True
        or custody["master_seed_mounted_to_generator"] is not False
        or custody["master_seed_mounted_to_model"] is not False
        or custody["capability_scoped_derived_keys_only"] is not True
        or custody["audit_private_assets_in_public_release"] is not False
        or custody["audit_truth_in_model_mount"] is not False
        or int(custody["m1_replicates_visible_per_fit_process"]) != 1
    ):
        raise PreceremonyError("Private/model custody boundary drift")
    projection = policy["visible_projection"]
    if (
        projection["title_rule"] != "normalize_only_never_identity_replace"
        or projection["description_rule"]
        != "remove_only_registered_context_guard_suffix_then_normalize"
        or projection["visible_natural_language_fields"]
        != ["title", "description"]
        or projection["join_only_uid_scanned_as_natural_language"] is not False
    ):
        raise PreceremonyError("Visible/join-only projection boundary drift")
    verify_file_pin(policy["contract"], label="v1.12 construction contract")
    for index, spec in enumerate(policy["design_only_base_inputs"]):
        verify_file_pin(spec, label=f"design-only base input {index}")
    candidate_implementation = policy["candidate_preceremony_implementation"]
    if set(candidate_implementation) != {"validator", "contract_tests"}:
        raise PreceremonyError("Candidate pre-ceremony implementation schema drift")
    for role in ("validator", "contract_tests"):
        verify_file_pin(
            candidate_implementation[role],
            label=f"candidate pre-ceremony {role}",
        )
    closure = policy["reusable_historical_source_closure"]
    members = closure["members"]
    member_paths = [str(record["path"]) for record in members]
    if (
        len(members) != int(closure["member_count"])
        or member_paths != sorted(member_paths, key=lambda value: value.encode("utf-8"))
        or len(set(member_paths)) != len(member_paths)
        or canonical_sha256(members) != closure["canonical_sha256"]
        or int(closure["c40_member_count"]) != 0
        or int(closure["failed_v1_3_through_v1_11_member_count"]) != 0
        or any("c40" in path.casefold() or "full378" in path.casefold() for path in member_paths)
    ):
        raise PreceremonyError("Reusable successful-v1.2 source closure drift")
    for index, spec in enumerate(members):
        verify_file_pin(spec, label=f"reusable source member {index}")
    regressions = policy["history_failure_regressions"]
    expected_regression_ids = [
        "v1_3_world_scoped_mechanism_slot",
        "v1_4_long_path_and_single_manifest_version",
        "v1_5_per_asset_collision_resolution",
        "v1_6_final_body_self_hash",
        "v1_7_shortcut_solver_preceremony_convergence",
        "v1_8_registered_suffix_only_redaction",
        "v1_9_visible_text_join_uid_scope_split",
        "v1_10_single_premodel_member_contract",
        "v1_11_shared_world_scoped_auditor",
    ]
    if [record.get("id") for record in regressions] != expected_regression_ids:
        raise PreceremonyError("Historical failure regression registry drift")
    collision = policy["identity_collision_resolution"]
    if (
        collision["strategy"]
        != "fixed_asset_order_first_admissible_per_asset_counter"
        or int(collision["maximum_counter"]) != 1024
        or collision["single_global_salt_for_all_candidates"] is not False
        or collision["master_seed_retry_on_collision"] is not False
        or collision["historical_hash_archive_is_read_only"] is not True
        or collision["same_run_allocated_hash_set_is_monotonic"] is not True
    ):
        raise PreceremonyError("Identity collision-resolution strategy drift")
    preflight = policy["design_preflight"]
    if (
        preflight["formal_key_or_seed_access"] is not False
        or preflight["base_mode"] != "development_smoke"
        or preflight["split_order"] != ["train", "development"]
        or int(preflight["worlds_per_split"]) != 1
        or int(preflight["expected_world_count"]) != 2
        or int(preflight["expected_pair_count"]) != 756
        or int(preflight["expected_positive_count"]) != 40
        or int(preflight["expected_negative_count"]) != 716
        or int(preflight["expected_identity33_rows"]) != 756
        or int(preflight["expected_identity_asset_count"]) != 168
        or int(preflight["expected_forced_identity_collision_count"]) != 1
        or int(preflight["expected_m1_mapping_count"]) != 10
        or preflight["scientific_metrics_produced"] is not False
    ):
        raise PreceremonyError("Design-only preflight contract drift")
    if (
        policy["next_lock_requirements"][
            "formal_seed_generation_may_be_authorized_by_this_policy"
        ]
        is not False
    ):
        raise PreceremonyError("Draft policy attempted to authorize formal seeds")
    cleanup = validate_failed_cleanup(policy["failed_file_cleanup_manifest"])
    hashes, commitments = load_failed_exclusions(
        policy["failed_identity_exclusion_archive"]
    )
    private_root = _repo_path(policy["custody"]["private_root"])
    ignored = subprocess.run(
        ["git", "check-ignore", "--quiet", str(private_root)],
        cwd=ROOT,
        check=False,
    )
    if policy["custody"]["private_root_must_be_git_ignored"] is not True or ignored.returncode != 0:
        raise PreceremonyError("v1.12 private custody root is not Git ignored")
    return {
        "policy": policy,
        "cleanup_manifest": cleanup,
        "failed_identity_hashes": hashes,
        "forbidden_master_commitments": commitments,
    }


def validate_world_scoped_mechanism_slots(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_world_count: int | None = None,
    expected_rows_per_world: int = 12,
) -> dict[str, int]:
    """Validate the one authoritative v1.3/v1.11 composite scope."""

    required = {
        "world_uid",
        "controller_uid",
        "mechanism",
        "mechanism_slot_uid",
    }
    composite_keys: set[tuple[str, str]] = set()
    global_slots: set[str] = set()
    world_counts: Counter[str] = Counter()
    for row in rows:
        if set(row) != required:
            raise PreceremonyError("Mechanism assignment schema drift")
        world_uid = str(row["world_uid"])
        slot_uid = str(row["mechanism_slot_uid"])
        if (
            UID_PATTERNS["world_uid"].fullmatch(world_uid) is None
            or not str(row["controller_uid"])
            or not str(row["mechanism"])
            or not slot_uid
        ):
            raise PreceremonyError("Mechanism assignment value-domain drift")
        key = (world_uid, slot_uid)
        if key in composite_keys:
            raise PreceremonyError(
                "Mechanism slot collides inside one world-scoped namespace"
            )
        composite_keys.add(key)
        global_slots.add(slot_uid)
        world_counts[world_uid] += 1
    if (
        not rows
        or set(world_counts.values()) != {expected_rows_per_world}
        or (
            expected_world_count is not None
            and len(world_counts) != expected_world_count
        )
    ):
        raise PreceremonyError("Mechanism assignment world cardinality drift")
    return {
        "row_count": len(rows),
        "world_count": len(world_counts),
        "world_scoped_unique_key_count": len(composite_keys),
        "global_template_slot_count": len(global_slots),
        "expected_cross_world_template_reuse_row_count": (
            len(rows) - len(global_slots)
        ),
        "within_world_duplicate_row_count": 0,
    }


def validate_exact_member_contract(
    expected_members: Sequence[str], observed_members: Sequence[str]
) -> None:
    if (
        list(expected_members) != list(observed_members)
        or len(set(expected_members)) != len(expected_members)
    ):
        raise PreceremonyError(
            "Producer/consumer premodel member contract is not exact"
        )


def validate_optimizer_audit(audit: Mapping[str, Any]) -> None:
    required = {
        "solver_success",
        "convergence_warning_count",
        "iteration_count",
        "maximum_iterations",
        "normalized_gradient",
        "gradient_tolerance",
        "objective_finite",
        "preceremony_exact_configuration",
    }
    if set(audit) != required:
        raise PreceremonyError("Shortcut optimizer audit schema drift")
    gradient = float(audit["normalized_gradient"])
    tolerance = float(audit["gradient_tolerance"])
    if (
        audit["solver_success"] is not True
        or int(audit["convergence_warning_count"]) != 0
        or int(audit["iteration_count"]) >= int(audit["maximum_iterations"])
        or not math.isfinite(gradient)
        or not math.isfinite(tolerance)
        or tolerance <= 0.0
        or gradient > tolerance
        or audit["objective_finite"] is not True
        or audit["preceremony_exact_configuration"] is not True
    ):
        raise PreceremonyError(
            "Exact shortcut optimizer did not converge before seed ceremony"
        )


def select_first_admissible_per_asset_candidate(
    candidate_hashes: Iterable[str],
    *,
    historical_forbidden: frozenset[str] | set[str],
    allocated_in_current_run: set[str],
    additional_forbidden: frozenset[str] | set[str] = frozenset(),
) -> tuple[int, str]:
    """Resolve one asset collision without changing or screening its master seed."""

    for counter, value_hash in enumerate(candidate_hashes):
        if HEX_SHA256_RE.fullmatch(value_hash) is None:
            raise PreceremonyError("Identity candidate hash is malformed")
        if (
            value_hash in historical_forbidden
            or value_hash in additional_forbidden
            or value_hash in allocated_in_current_run
        ):
            continue
        allocated_in_current_run.add(value_hash)
        return counter, value_hash
    raise PreceremonyError("Per-asset deterministic candidate domain exhausted")


def _candidate_identity_value(
    *,
    key_hex: str,
    world_uid: str,
    identity_asset_uid: str,
    identity_type: str,
    counter: int,
) -> str:
    import step28_v13_identity_values as identity_values

    if counter < 0 or HEX_SHA256_RE.fullmatch(key_hex) is None:
        raise PreceremonyError("Identity candidate counter/key is malformed")
    digest = hmac.new(
        bytes.fromhex(key_hex),
        b"step28-v13-v1.12-identity-value"
        + b"\x1f"
        + world_uid.encode("utf-8")
        + b"\x1f"
        + identity_asset_uid.encode("utf-8")
        + b"\x1f"
        + identity_type.encode("ascii")
        + b"\x1f"
        + str(counter).encode("ascii"),
        hashlib.sha256,
    ).digest()
    modulus = identity_values.domain_size(
        identity_type, "parser_safe_hex_v2"
    )
    return identity_values.encode_identity_value(
        identity_type,
        int.from_bytes(digest, "big", signed=False) % modulus,
        handle_encoding="parser_safe_hex_v2",
    )


def identity_value_collides_with_visible_text(
    value: str,
    *,
    visible_texts: Sequence[str],
    visible_compacts: Sequence[str],
) -> bool:
    import step7_v3_1_source_data as source

    normalized = str(value).strip().casefold()
    compact = source.compact_identifier(value)
    if not normalized or not compact:
        raise PreceremonyError("Identity candidate has an empty visible projection")
    return any(normalized in text for text in visible_texts) or any(
        compact in text for text in visible_compacts
    )


def remap_world_identity_values(
    world: Mapping[str, Any],
    *,
    template: Mapping[str, Any],
    key_hex: str,
    historical_forbidden: frozenset[str] | set[str],
    allocated_in_current_run: set[str],
    maximum_counter: int,
    force_first_candidate_collision: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply per-asset collision resolution to every linked world field."""

    import step28_v13_common as common
    import step28_v13_identity_values as identity_values
    import step28_v13_text_renderer as renderer
    import step7_v3_1_source_data as source

    if maximum_counter < 1:
        raise PreceremonyError("Identity collision counter budget is invalid")
    output = copy.deepcopy(dict(world))
    public = output.get("public")
    private = output.get("private")
    if not isinstance(public, dict) or not isinstance(private, dict):
        raise PreceremonyError("World identity remap input boundary is malformed")
    assets = private.get("identity_assets")
    slot_audit = private.get("identity_slots_audit")
    slot_edit = private.get("identity_slots_edit")
    items = public.get("items")
    if not all(isinstance(value, list) for value in (assets, slot_audit, slot_edit, items)):
        raise PreceremonyError("World identity remap tables are missing")
    world_uids = {
        str(row["world_uid"])
        for row in [*items, *private.get("mechanism_assignments", [])]
    }
    if len(world_uids) != 1:
        raise PreceremonyError("World identity remap received mixed worlds")
    world_uid = next(iter(world_uids))
    guards = renderer.context_guard_pool(template)
    visible_texts: list[str] = []
    visible_compacts: list[str] = []
    for item in items:
        raw_description = str(item["description"])
        boundary, _count = _earliest_context_guard_boundary(
            raw_description, guards=guards
        )
        for text in (
            str(item["title"]),
            raw_description if boundary is None else raw_description[:boundary],
        ):
            normalized = source.normalize_redacted_text(text).casefold()
            visible_texts.append(normalized)
            visible_compacts.append(source.compact_identifier(normalized))
    ordered_assets = sorted(
        assets, key=lambda row: str(row["identity_asset_uid"]).encode("utf-8")
    )
    if (
        not ordered_assets
        or len({str(row["identity_asset_uid"]) for row in ordered_assets})
        != len(ordered_assets)
        or len({str(row["identity_uid"]) for row in ordered_assets})
        != len(ordered_assets)
    ):
        raise PreceremonyError("Identity asset UID/value lineage is not one-to-one")

    forced_forbidden: set[str] = set()
    forced_collision_count = 0
    old_uid_to_new: dict[str, tuple[str, str, str]] = {}
    selected_counters: list[int] = []
    visible_text_candidate_rejection_count = 0
    allocation_start_count = len(allocated_in_current_run)
    for asset_index, asset in enumerate(ordered_assets):
        identity_type = str(asset["identity_type"])
        asset_uid = str(asset["identity_asset_uid"])
        old_identity_uid = str(asset["identity_uid"])
        old_value = str(asset["identity_value"])

        def candidates() -> Iterable[tuple[str, str]]:
            for counter in range(maximum_counter + 1):
                value = _candidate_identity_value(
                    key_hex=key_hex,
                    world_uid=world_uid,
                    identity_asset_uid=asset_uid,
                    identity_type=identity_type,
                    counter=counter,
                )
                yield value, identity_values.value_hash(value)

        candidate_rows = candidates()
        if force_first_candidate_collision and asset_index == 0:
            first_value, first_hash = next(candidate_rows)
            del first_value
            forced_forbidden.add(first_hash)
            forced_collision_count = 1
            candidate_rows = itertools.chain(
                [("", first_hash)], candidate_rows
            )
        values_by_hash: dict[str, str] = {}

        def hashes() -> Iterable[str]:
            nonlocal visible_text_candidate_rejection_count
            for value, value_hash in candidate_rows:
                if value:
                    values_by_hash[value_hash] = value
                    if identity_value_collides_with_visible_text(
                        value,
                        visible_texts=visible_texts,
                        visible_compacts=visible_compacts,
                    ):
                        forced_forbidden.add(value_hash)
                        visible_text_candidate_rejection_count += 1
                yield value_hash

        counter, selected_hash = select_first_admissible_per_asset_candidate(
            hashes(),
            historical_forbidden=historical_forbidden,
            allocated_in_current_run=allocated_in_current_run,
            additional_forbidden=forced_forbidden,
        )
        new_value = values_by_hash.get(selected_hash)
        if new_value is None:
            raise PreceremonyError("Selected identity candidate lacks its value")
        if len(new_value) != len(old_value):
            raise PreceremonyError("Identity remap changed fixed-width surface length")
        new_identity_uid = "id_" + common.canonical_sha256(
            {
                "contact_type": identity_type.strip().lower(),
                "normalized_value": new_value.strip().lower(),
            }
        )
        old_uid_to_new[old_identity_uid] = (
            new_identity_uid,
            new_value,
            selected_hash,
        )
        selected_counters.append(counter)
        asset["identity_value"] = new_value
        asset["identity_uid"] = new_identity_uid

    slot_by_uid: dict[str, Mapping[str, Any]] = {}
    edits_by_item: dict[str, list[tuple[int, int, str, str]]] = {}
    for row in slot_audit:
        slot_uid = str(row["slot_uid"])
        old_identity_uid = str(row["identity_uid"])
        replacement = old_uid_to_new.get(old_identity_uid)
        if replacement is None or slot_uid in slot_by_uid:
            raise PreceremonyError("Identity slot references an unknown/duplicate asset")
        new_identity_uid, new_value, _value_hash = replacement
        old_surface = str(row["raw_surface"])
        if len(old_surface) != len(new_value):
            raise PreceremonyError("Identity slot replacement length drift")
        row["identity_uid"] = new_identity_uid
        row["raw_surface"] = new_value
        row["downstream_canonical_value"] = new_value.strip().lower()
        slot_by_uid[slot_uid] = row
        edits_by_item.setdefault(str(row["item_uid"]), []).append(
            (int(row["start"]), int(row["end"]), old_surface, new_value)
        )
    if len(slot_by_uid) != len(slot_audit):
        raise PreceremonyError("Identity slot audit keyset drift during remap")
    for row in slot_edit:
        source = slot_by_uid.get(str(row["slot_uid"]))
        if source is None:
            raise PreceremonyError("Identity edit row lacks its authoritative slot")
        for field in (
            "item_uid",
            "seller_uid",
            "field_name",
            "start",
            "end",
            "identity_type",
            "time_bucket",
        ):
            if row[field] != source[field]:
                raise PreceremonyError("Identity edit/slot immutable lineage drift")
        row["raw_surface"] = source["raw_surface"]
        row["downstream_canonical_value"] = source[
            "downstream_canonical_value"
        ]

    changed_item_count = 0
    item_uids = {str(row["item_uid"]) for row in items}
    if not set(edits_by_item).issubset(item_uids):
        raise PreceremonyError("Identity slot edit references an unknown item")
    for item in items:
        item_uid = str(item["item_uid"])
        description = str(item["description"])
        edits = sorted(edits_by_item.get(item_uid, []), reverse=True)
        for start, end, old_surface, new_surface in edits:
            if description[start:end] != old_surface:
                raise PreceremonyError("Identity remap offset does not round-trip")
            description = description[:start] + new_surface + description[end:]
        if edits:
            changed_item_count += 1
            item["description"] = description

    if len(allocated_in_current_run) - allocation_start_count != len(assets):
        raise PreceremonyError("Identity remap allocation count drift")
    selected_hashes = {value[2] for value in old_uid_to_new.values()}
    if (
        len(selected_hashes) != len(assets)
        or selected_hashes & set(historical_forbidden)
    ):
        raise PreceremonyError("Identity remap collision closure failed")
    return output, {
        "world_uid": world_uid,
        "identity_asset_count": len(assets),
        "identity_slot_count": len(slot_audit),
        "changed_item_count": changed_item_count,
        "maximum_counter": maximum_counter,
        "maximum_selected_counter": max(selected_counters),
        "nonzero_counter_count": sum(value > 0 for value in selected_counters),
        "forced_design_collision_count": forced_collision_count,
        "visible_text_candidate_rejection_count": (
            visible_text_candidate_rejection_count
        ),
        "historical_intersection_count": 0,
        "same_run_intersection_count": 0,
        "selected_value_hashes_sha256": canonical_sha256(
            sorted(selected_hashes)
        ),
    }


def build_split_manifest_receipt(
    *, split: str, file_records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if split not in SPLITS:
        raise PreceremonyError("Unknown split manifest split")
    payload = {
        "version": SPLIT_MANIFEST_VERSION,
        "split": split,
        "file_records": [dict(row) for row in file_records],
        "file_count": len(file_records),
    }
    return with_canonical_self_hash(payload)


def validate_split_manifest_receipt(document: Mapping[str, Any]) -> None:
    validate_canonical_self_hash(document, label="v1.12 split manifest")
    records = document.get("file_records")
    if (
        document.get("version") != SPLIT_MANIFEST_VERSION
        or document.get("split") not in SPLITS
        or not isinstance(records, list)
        or document.get("file_count") != len(records)
    ):
        raise PreceremonyError("v1.12 split manifest contract drift")


def run_windows_long_path_replay() -> dict[str, Any]:
    """Exercise the same helpers for write, read, stat, and tree traversal."""

    root = Path(tempfile.mkdtemp(prefix="step28-v1-12-long-path-"))
    try:
        nested = root.joinpath(*(("segment_" + "x" * 24,) * 9))
        target = nested / "manifest_payload.json"
        if os.name == "nt" and len(str(target.resolve())) <= 260:
            raise PreceremonyError("Long-path replay did not cross MAX_PATH")
        payload = b'{"long_path_replay":true}\n'
        write_bytes_no_replace_long_path(target, payload)
        observed = read_bytes_long_path(target)
        size = stat_long_path(target).st_size
        files = walk_files_long_path(root)
        if observed != payload or size != len(payload) or files != [target]:
            raise PreceremonyError("Long-path write/read/stat/walk replay drift")
        file_record = {
            "path": target.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(observed).hexdigest(),
            "size_bytes": size,
        }
        manifest = build_split_manifest_receipt(
            split="train", file_records=[file_record]
        )
        validate_split_manifest_receipt(manifest)
        return {
            "status": "PASS_LONG_PATH_SINGLE_IMPLEMENTATION_REPLAY",
            "path_length": len(str(target.resolve())),
            "file_count": 1,
            "payload_sha256": file_record["sha256"],
            "manifest_version": SPLIT_MANIFEST_VERSION,
        }
    finally:
        shutil.rmtree(_filesystem_path(root), ignore_errors=False)


def _pair_endpoint_index(
    pair_rows: Sequence[Mapping[str, Any]],
) -> dict[str, frozenset[str]]:
    output: dict[str, frozenset[str]] = {}
    for row in pair_rows:
        pair_uid = str(row["canonical_pair_uid"])
        endpoints = frozenset(
            (str(row["seller_uid_left"]), str(row["seller_uid_right"]))
        )
        if len(endpoints) != 2 or pair_uid in output:
            raise PreceremonyError("Pair endpoint input is malformed or duplicated")
        output[pair_uid] = endpoints
    return output


def build_endpoint_disjoint_derangement(
    pair_rows: Sequence[Mapping[str, Any]],
    *,
    world_uid: str,
    key_hex: str,
) -> list[dict[str, str]]:
    """Build a deterministic perfect matching in the disjoint-edge graph."""

    if HEX_SHA256_RE.fullmatch(key_hex) is None:
        raise PreceremonyError("M1 design-only key is not canonical 32-byte hex")
    endpoints = _pair_endpoint_index(pair_rows)
    pair_uids = sorted(endpoints, key=lambda value: value.encode("utf-8"))
    if len(pair_uids) != 378:
        raise PreceremonyError("M1 derangement requires the complete 378-pair universe")
    key = bytes.fromhex(key_hex)
    adjacency: list[list[int]] = []
    for source_uid in pair_uids:
        candidates = [
            index
            for index, destination_uid in enumerate(pair_uids)
            if endpoints[source_uid].isdisjoint(endpoints[destination_uid])
        ]
        candidates.sort(
            key=lambda index: (
                hmac.new(
                    key,
                    b"step28-v13-v1.12-m1-edge"
                    + b"\x1f"
                    + world_uid.encode("utf-8")
                    + b"\x1f"
                    + source_uid.encode("utf-8")
                    + b"\x1f"
                    + pair_uids[index].encode("utf-8"),
                    hashlib.sha256,
                ).digest(),
                pair_uids[index].encode("utf-8"),
            )
        )
        if len(candidates) != math.comb(26, 2):
            raise PreceremonyError("Disjoint-edge candidate degree is not 325")
        adjacency.append(candidates)

    left_match = [-1] * len(pair_uids)
    right_match = [-1] * len(pair_uids)
    distance = [0] * len(pair_uids)

    def bfs() -> bool:
        queue: deque[int] = deque()
        found = False
        for left in range(len(pair_uids)):
            if left_match[left] < 0:
                distance[left] = 0
                queue.append(left)
            else:
                distance[left] = -1
        while queue:
            left = queue.popleft()
            for right in adjacency[left]:
                paired_left = right_match[right]
                if paired_left < 0:
                    found = True
                elif distance[paired_left] < 0:
                    distance[paired_left] = distance[left] + 1
                    queue.append(paired_left)
        return found

    def dfs(left: int) -> bool:
        for right in adjacency[left]:
            paired_left = right_match[right]
            if paired_left < 0 or (
                distance[paired_left] == distance[left] + 1 and dfs(paired_left)
            ):
                left_match[left] = right
                right_match[right] = left
                return True
        distance[left] = -1
        return False

    matching_size = 0
    while bfs():
        for left in range(len(pair_uids)):
            if left_match[left] < 0 and dfs(left):
                matching_size += 1
    if matching_size != len(pair_uids) or -1 in left_match or -1 in right_match:
        raise PreceremonyError("Endpoint-disjoint bipartite matching is incomplete")

    mapping = [
        {
            "world_uid": world_uid,
            "destination_pair_uid": pair_uids[right],
            "source_pair_uid": pair_uids[left],
        }
        for right, left in enumerate(right_match)
    ]
    if (
        len({row["source_pair_uid"] for row in mapping}) != 378
        or len({row["destination_pair_uid"] for row in mapping}) != 378
        or any(
            row["source_pair_uid"] == row["destination_pair_uid"]
            or not endpoints[row["source_pair_uid"]].isdisjoint(
                endpoints[row["destination_pair_uid"]]
            )
            for row in mapping
        )
    ):
        raise PreceremonyError("M1 mapping is not an endpoint-disjoint bijection")
    return mapping


def rewire_identity33_rows(
    identity_rows: Sequence[Mapping[str, Any]],
    mapping: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    if not identity_rows:
        raise PreceremonyError("M1 identity33 input is empty")
    source_index = {
        str(row["canonical_pair_uid"]): dict(row) for row in identity_rows
    }
    identity_keyset = set(source_index)
    source_keyset = {str(row["source_pair_uid"]) for row in mapping}
    destination_keyset = {
        str(row["destination_pair_uid"]) for row in mapping
    }
    world_uids = {str(row["world_uid"]) for row in identity_rows}
    mapping_world_uids = {str(row["world_uid"]) for row in mapping}
    if (
        len(source_index) != len(identity_rows)
        or len(mapping) != len(identity_rows)
        or source_keyset != identity_keyset
        or destination_keyset != identity_keyset
        or len(world_uids) != 1
        or mapping_world_uids != world_uids
    ):
        raise PreceremonyError("M1 identity33 row or mapping keyset drift")
    feature_names = [
        name
        for name in identity_rows[0]
        if name not in {"canonical_pair_uid", "world_uid"}
    ]
    output: list[dict[str, Any]] = []
    for link in mapping:
        source = source_index.get(str(link["source_pair_uid"]))
        if source is None:
            raise PreceremonyError("M1 mapping references an unknown source row")
        output.append(
            {
                "canonical_pair_uid": str(link["destination_pair_uid"]),
                "world_uid": str(link["world_uid"]),
                **{name: source[name] for name in feature_names},
            }
        )
    output.sort(key=lambda row: str(row["canonical_pair_uid"]).encode("utf-8"))
    original_vectors = sorted(
        (tuple(str(row[name]) for name in feature_names) for row in identity_rows)
    )
    rewired_vectors = sorted(
        (tuple(str(row[name]) for name in feature_names) for row in output)
    )
    if (
        original_vectors != rewired_vectors
        or len({str(row["canonical_pair_uid"]) for row in output}) != len(output)
    ):
        raise PreceremonyError("M1 changed the identity33 joint-row multiset")
    return output


def validate_join_only_uid_lineage(
    rows: Sequence[Mapping[str, Any]], *, fields: Sequence[str]
) -> None:
    """Validate join identifiers without scanning their substrings as text."""

    for row in rows:
        for field in fields:
            pattern = UID_PATTERNS.get(field)
            if pattern is None:
                raise PreceremonyError(f"No join-only UID contract for {field}")
            if pattern.fullmatch(str(row.get(field, ""))) is None:
                raise PreceremonyError(f"Malformed join-only UID field: {field}")


def _earliest_context_guard_boundary(
    value: str, *, guards: Sequence[str]
) -> tuple[int | None, int]:
    positions = [position for guard in guards if (position := value.find(guard)) >= 0]
    return (min(positions) if positions else None), sum(
        value.count(guard) for guard in guards
    )


def project_registered_visible_text(
    *,
    policy: Mapping[str, Any],
    template: Mapping[str, Any],
    sellers: Sequence[Mapping[str, Any]],
    items: Sequence[Mapping[str, Any]],
    parsed_rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Project synthetic visible text without a global title redactor."""

    import step28_v13_text_renderer as renderer
    import step7_v3_1_source_data as source

    seller_uids = {str(row["seller_uid"]) for row in sellers}
    if len(seller_uids) != 28:
        raise PreceremonyError("Visible projection requires 28 sellers")
    validate_join_only_uid_lineage(
        sellers, fields=("world_uid", "seller_uid")
    )
    validate_join_only_uid_lineage(
        items, fields=("world_uid", "seller_uid", "item_uid")
    )
    item_uids = [str(row["item_uid"]) for row in items]
    world_uids = {str(row["world_uid"]) for row in sellers} | {
        str(row["world_uid"]) for row in items
    }
    if (
        len(item_uids) != len(set(item_uids))
        or len(world_uids) != 1
        or any(str(row["seller_uid"]) not in seller_uids for row in items)
    ):
        raise PreceremonyError("Visible projection item/seller/world lineage drift")
    parsed_item_uids = {str(row["item_uid"]) for row in parsed_rows}
    if not parsed_item_uids.issubset(item_uids):
        raise PreceremonyError("Parsed identity row references an unknown item")
    guards = renderer.context_guard_pool(template)
    if not guards or len(guards) != len(set(guards)):
        raise PreceremonyError("Context-guard registry is empty or duplicated")

    redacted_items: list[dict[str, Any]] = []
    profile_safe_items: list[dict[str, Any]] = []
    for item in sorted(
        items,
        key=lambda row: (
            str(row["world_uid"]).encode("utf-8"),
            str(row["seller_uid"]).encode("utf-8"),
            str(row["item_uid"]).encode("utf-8"),
        ),
    ):
        item_uid = str(item["item_uid"])
        raw_title = str(item["title"])
        raw_description = str(item["description"])
        boundary, guard_count = _earliest_context_guard_boundary(
            raw_description, guards=guards
        )
        has_registered_identity = item_uid in parsed_item_uids
        if has_registered_identity != (boundary is not None):
            raise PreceremonyError(
                "Parser identity rows and registered description boundary disagree"
            )
        clean_title = source.normalize_redacted_text(raw_title)
        description_prefix = (
            raw_description if boundary is None else raw_description[:boundary]
        )
        clean_description = source.normalize_redacted_text(description_prefix)
        if any(guard in clean_title or guard in clean_description for guard in guards):
            raise PreceremonyError("Context guard survived visible projection")
        if clean_title != source.normalize_redacted_text(raw_title):
            raise PreceremonyError("Identity-free title changed")
        public_row = {
            "world_uid": str(item["world_uid"]),
            "seller_uid": str(item["seller_uid"]),
            "item_uid": item_uid,
            "title": clean_title,
            "description": clean_description,
        }
        redacted_items.append(public_row)
        profile_safe_items.append(
            {
                "world_uid": str(item["world_uid"]),
                "seller_uid": str(item["seller_uid"]),
                "item_uid": item_uid,
                "time_bucket": int(item["time_bucket"]),
                "category": str(item["category"]),
                "title": clean_title,
                "description": clean_description,
            }
        )
        if has_registered_identity and guard_count <= 0:
            raise PreceremonyError("Registered identity suffix was not removed")

    visible_text = "\n".join(
        str(row[field]).casefold()
        for row in redacted_items
        for field in ("title", "description")
    )
    identity_surfaces = {
        str(row[field]).strip().casefold()
        for row in parsed_rows
        for field in ("raw_value", "normalized_value")
        if str(row[field]).strip()
    }
    if any(surface in visible_text for surface in identity_surfaces):
        raise PreceremonyError("Parsed identity surface survived in visible text")
    if len(redacted_items) != len(items) or len(profile_safe_items) != len(items):
        raise PreceremonyError("Visible projection item keyset drift")
    return {
        "redacted_items": redacted_items,
        "profile_safe_items": profile_safe_items,
    }


def validate_full_pair_labels(
    *,
    pair_rows: Sequence[Mapping[str, Any]],
    controller_membership: Sequence[Mapping[str, Any]],
    expected_world_uid: str,
) -> list[dict[str, Any]]:
    """Independently derive all 378 labels after label-free inputs freeze."""

    import step28_v13_common as common

    seller_to_controller: dict[str, str] = {}
    controller_counts: Counter[str] = Counter()
    for row in controller_membership:
        if set(row) != {"world_uid", "controller_uid", "seller_uid"}:
            raise PreceremonyError("Controller membership schema drift")
        if str(row["world_uid"]) != expected_world_uid:
            raise PreceremonyError("Controller membership world drift")
        seller_uid = str(row["seller_uid"])
        controller_uid = str(row["controller_uid"])
        if seller_uid in seller_to_controller:
            raise PreceremonyError("Seller has duplicate controller membership")
        seller_to_controller[seller_uid] = controller_uid
        controller_counts[controller_uid] += 1
    if (
        len(seller_to_controller) != 28
        or len(controller_counts) != 12
        or Counter(controller_counts.values()) != Counter({2: 8, 3: 4})
    ):
        raise PreceremonyError("Controller topology is not eight dyads plus four triads")
    expected_pair_uids = {
        common.canonical_pair_uid(left, right)
        for index, left in enumerate(common.utf8_sort(seller_to_controller))
        for right in common.utf8_sort(seller_to_controller)[index + 1 :]
    }
    labels: list[dict[str, Any]] = []
    observed_pair_uids: set[str] = set()
    for row in pair_rows:
        if set(row) != {
            "canonical_pair_uid",
            "world_uid",
            "seller_uid_left",
            "seller_uid_right",
        }:
            raise PreceremonyError("Complete pair endpoint schema drift")
        left = str(row["seller_uid_left"])
        right = str(row["seller_uid_right"])
        pair_uid = str(row["canonical_pair_uid"])
        if (
            str(row["world_uid"]) != expected_world_uid
            or left not in seller_to_controller
            or right not in seller_to_controller
            or left == right
            or pair_uid != common.canonical_pair_uid(left, right)
            or pair_uid in observed_pair_uids
        ):
            raise PreceremonyError("Complete pair endpoint value/key drift")
        observed_pair_uids.add(pair_uid)
        labels.append(
            {
                "world_uid": expected_world_uid,
                "canonical_pair_uid": pair_uid,
                "label": int(
                    seller_to_controller[left] == seller_to_controller[right]
                ),
            }
        )
    if (
        observed_pair_uids != expected_pair_uids
        or len(labels) != 378
        or sum(int(row["label"]) for row in labels) != 20
    ):
        raise PreceremonyError("Full-378 label universe or class count drift")
    labels.sort(key=lambda row: str(row["canonical_pair_uid"]).encode("utf-8"))
    return labels


def _one_design_world(
    *,
    base_policy: dict[str, Any],
    template: dict[str, Any],
    fixture: dict[str, Any],
    style_profile: dict[str, Any],
    split: str,
    identity_remap_key_hex: str,
    historical_forbidden: frozenset[str],
    allocated_identity_hashes: set[str],
    maximum_identity_counter: int,
    force_first_candidate_collision: bool,
) -> dict[str, Any]:
    import step28_v13_common as common
    import step28_v13_history_features as history_features
    import step28_v13_production_chain as production
    import step28_v13_profiles as profiles
    import step28_v13_structure as structure
    import step28_v13_world_builder as world_builder

    records = [
        row
        for row in structure.build_mode_world_pool(
            base_policy, mode="development_smoke"
        )
        if row["split"] == split
    ]
    if not records:
        raise PreceremonyError(f"No design-only world exists for {split}")
    record = records[0]
    world_uid = str(record["world_uid"])
    structure_key = common.structure_key_for_split(
        base_policy, mode="development_smoke", split=split
    )
    world = world_builder.build_world(
        policy=base_policy,
        template=template,
        fixture=fixture,
        style_profile=style_profile,
        mode="development_smoke",
        world_record=record,
        structure_key_hex=structure_key,
    )
    world, identity_allocation_audit = remap_world_identity_values(
        world,
        template=template,
        key_hex=identity_remap_key_hex,
        historical_forbidden=historical_forbidden,
        allocated_in_current_run=allocated_identity_hashes,
        maximum_counter=maximum_identity_counter,
        force_first_candidate_collision=force_first_candidate_collision,
    )
    public = world["public"]
    private = world["private"]
    parsed = production.parse_observed_world(
        base_policy,
        mode="development_smoke",
        split=split,
        sellers=public["sellers"],
        items=public["items"],
    )
    parser_audit = production.validate_parser_against_private_plan(
        base_policy,
        mode="development_smoke",
        split=split,
        sellers=public["sellers"],
        items=public["items"],
        parsed_rows=parsed,
        identity_slots_audit=private["identity_slots_audit"],
        noise_slots_audit=private["noise_slots_audit"],
        render_asts=private["render_asts"],
    )
    projection = project_registered_visible_text(
        policy=base_policy,
        template=template,
        sellers=public["sellers"],
        items=public["items"],
        parsed_rows=parsed,
    )
    history_rows = production.project_history_safe_occurrences(
        base_policy,
        mode="development_smoke",
        split=split,
        sellers=public["sellers"],
        items=public["items"],
        parsed_rows=parsed,
    )
    history_item_index = [
        {
            "world_uid": str(row["world_uid"]),
            "seller_uid": str(row["seller_uid"]),
            "item_uid": str(row["item_uid"]),
            "time_bucket": int(row["time_bucket"]),
        }
        for row in public["items"]
    ]
    history_item_index.sort(
        key=lambda row: (
            row["world_uid"].encode("utf-8"),
            row["seller_uid"].encode("utf-8"),
            row["item_uid"].encode("utf-8"),
        )
    )
    attestation = production.build_history_projection_attestation(
        base_policy,
        mode="development_smoke",
        split=split,
        world_uid=world_uid,
        sellers=public["sellers"],
        items=public["items"],
        history_safe_occurrences=history_rows,
        history_item_index=history_item_index,
        parsed_rows=parsed,
        identity_slots_audit=private["identity_slots_audit"],
        noise_slots_audit=private["noise_slots_audit"],
        render_asts=private["render_asts"],
    )
    identity33, identity33_audit = history_features.build_identity33_all_pairs(
        base_policy,
        mode="development_smoke",
        split=split,
        history_safe_occurrences=history_rows,
        history_item_index=history_item_index,
        projection_attestations=[attestation],
        complete_model_pair_endpoints=public["complete_model_pair_endpoints"],
    )
    seller_profiles, profile_audit = profiles.build_world_profiles(
        base_policy,
        mode="development_smoke",
        split=split,
        sellers=public["sellers"],
        items=projection["profile_safe_items"],
    )

    # M1 construction intentionally occurs before the private label formula.
    mapping_hashes: list[str] = []
    for key_hex in base_policy["randomness"]["development_smoke"][
        "rewire_key_hexes"
    ]:
        mapping = build_endpoint_disjoint_derangement(
            public["complete_model_pair_endpoints"],
            world_uid=world_uid,
            key_hex=str(key_hex),
        )
        rewire_identity33_rows(identity33, mapping)
        mapping_hashes.append(canonical_sha256(mapping))
    if len(mapping_hashes) != 5 or len(set(mapping_hashes)) != 5:
        raise PreceremonyError("Five design-only M1 mappings are not independent")

    labels = validate_full_pair_labels(
        pair_rows=public["complete_model_pair_endpoints"],
        controller_membership=private["controller_membership"],
        expected_world_uid=world_uid,
    )
    if parser_audit.get("exact_rows_and_flags") is not True:
        raise PreceremonyError("Design-only parser replay is not exact")
    if (
        len(identity33) != 378
        or int(identity33_audit["feature_count"]) != 33
        or len(seller_profiles) != 28
        or int(profile_audit["seller_count"]) != 28
    ):
        raise PreceremonyError("Design-only model input cardinality drift")
    return {
        "split": split,
        "world_uid": world_uid,
        "pair_count": 378,
        "positive_count": sum(int(row["label"]) for row in labels),
        "negative_count": sum(1 - int(row["label"]) for row in labels),
        "item_count": len(public["items"]),
        "seller_profile_count": len(seller_profiles),
        "parsed_identity_occurrence_count": len(parsed),
        "history_safe_occurrence_count": len(history_rows),
        "identity33_row_count": len(identity33),
        "identity33_feature_count": 33,
        "visible_projection_sha256": canonical_sha256(projection),
        "identity33_sha256": canonical_sha256(identity33),
        "seller_profiles_sha256": canonical_sha256(seller_profiles),
        "m1_mapping_sha256": mapping_hashes,
        "identity_allocation_audit": identity_allocation_audit,
        "mechanism_assignments": [dict(row) for row in private["mechanism_assignments"]],
    }


def run_design_preflight(
    policy_path: Path = DEFAULT_POLICY_PATH,
) -> dict[str, Any]:
    validated = validate_policy(policy_path)
    policy = validated["policy"]

    import step28_v13_common as common

    base_policy_path = _repo_path(
        next(
            record["path"]
            for record in policy["design_only_base_inputs"]
            if record["path"].endswith("synthetic_chinese_dataset_policy.json")
        )
    )
    base_policy = common.load_policy(
        base_policy_path, mode="development_smoke"
    )
    template, fixture = common.validate_policy_release_documents(
        base_policy, mode="development_smoke"
    )
    style_profile = common.load_json(
        common.verify_file_pin(
            base_policy["style_reference_boundary"]["generator_release_inputs"][
                "profile"
            ],
            label="design-only style reference",
        )
    )
    smoke_identity_key = bytes.fromhex(
        str(base_policy["randomness"]["development_smoke"]["identity_value_key_hex"])
    )
    allocated_identity_hashes: set[str] = set()
    world_results: list[dict[str, Any]] = []
    for split_index, split in enumerate(
        policy["design_preflight"]["split_order"]
    ):
        remap_key = hmac.new(
            smoke_identity_key,
            b"step28-v13-v1.12-design-only-identity-remap"
            + b"\x1f"
            + str(split).encode("ascii"),
            hashlib.sha256,
        ).hexdigest()
        world_results.append(
            _one_design_world(
            base_policy=base_policy,
            template=template,
            fixture=fixture,
            style_profile=style_profile,
            split=split,
            identity_remap_key_hex=remap_key,
            historical_forbidden=validated["failed_identity_hashes"],
            allocated_identity_hashes=allocated_identity_hashes,
            maximum_identity_counter=int(
                policy["identity_collision_resolution"]["maximum_counter"]
            ),
            force_first_candidate_collision=(split_index == 0),
        )
        )
    mechanism_rows = [
        row
        for result in world_results
        for row in result.pop("mechanism_assignments")
    ]
    mechanism_scope = validate_world_scoped_mechanism_slots(
        mechanism_rows,
        expected_world_count=int(policy["design_preflight"]["expected_world_count"]),
    )
    if (
        mechanism_scope["row_count"] != 24
        or mechanism_scope["world_scoped_unique_key_count"] != 24
        or mechanism_scope["global_template_slot_count"] != 12
        or mechanism_scope["expected_cross_world_template_reuse_row_count"]
        != 12
        or mechanism_scope["within_world_duplicate_row_count"] != 0
    ):
        raise PreceremonyError(
            "Design worlds do not demonstrate world-scoped mechanism-slot reuse"
        )
    long_path = run_windows_long_path_replay()
    total_pairs = sum(int(row["pair_count"]) for row in world_results)
    total_positive = sum(int(row["positive_count"]) for row in world_results)
    total_negative = sum(int(row["negative_count"]) for row in world_results)
    total_identity33 = sum(int(row["identity33_row_count"]) for row in world_results)
    total_identity_assets = sum(
        int(row["identity_allocation_audit"]["identity_asset_count"])
        for row in world_results
    )
    forced_collision_count = sum(
        int(row["identity_allocation_audit"]["forced_design_collision_count"])
        for row in world_results
    )
    expected = policy["design_preflight"]
    if (
        len(world_results) != int(expected["expected_world_count"])
        or total_pairs != int(expected["expected_pair_count"])
        or total_positive != int(expected["expected_positive_count"])
        or total_negative != int(expected["expected_negative_count"])
        or total_identity33 != int(expected["expected_identity33_rows"])
        or total_identity_assets != int(expected["expected_identity_asset_count"])
        or len(allocated_identity_hashes) != total_identity_assets
        or forced_collision_count
        != int(expected["expected_forced_identity_collision_count"])
        or sum(
            int(row["identity_allocation_audit"]["nonzero_counter_count"])
            for row in world_results
        )
        < forced_collision_count
        or sum(len(row["m1_mapping_sha256"]) for row in world_results)
        != int(expected["expected_m1_mapping_count"])
    ):
        raise PreceremonyError("Two-world design preflight aggregate drift")
    receipt = with_canonical_self_hash(
        {
            "version": "2026-08-03-step28-v13-v1-12-two-world-preceremony-receipt-v1",
            "status": "PASS_DESIGN_ONLY_NO_FORMAL_AUTHORIZATION",
            "run_id": policy["run_id"],
            "policy_sha256": sha256_file(policy_path),
            "policy_canonical_self_hash": policy["canonical_self_hash"],
            "formal_seed_or_key_access": False,
            "formal_dataset_rows_produced": 0,
            "scientific_metrics_produced": False,
            "model_training_performed": False,
            "world_count": len(world_results),
            "pair_count": total_pairs,
            "positive_count": total_positive,
            "negative_count": total_negative,
            "identity33_row_count": total_identity33,
            "identity_asset_count": total_identity_assets,
            "forced_identity_collision_count": forced_collision_count,
            "m1_mapping_count": sum(
                len(row["m1_mapping_sha256"]) for row in world_results
            ),
            "world_results": world_results,
            "mechanism_scope_audit": mechanism_scope,
            "long_path_replay": long_path,
            "failed_identity_hash_count_validated": len(
                validated["failed_identity_hashes"]
            ),
            "forbidden_master_commitment_count_validated": len(
                validated["forbidden_master_commitments"]
            ),
            "formal_authorizations_after_preflight": dict(
                policy["authorizations"]
            ),
        }
    )
    return receipt


def publish_receipt_no_replace(path: Path, receipt: Mapping[str, Any]) -> str:
    payload = json.dumps(
        receipt, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ).encode("utf-8") + b"\n"
    if path.exists():
        if read_bytes_long_path(path) != payload:
            raise PreceremonyError(
                "Existing design-preflight receipt has different frozen bytes"
            )
        return "EXACT_EXISTING_RECEIPT_REUSED"
    write_bytes_no_replace_long_path(path, payload)
    return "NEW_RECEIPT_PUBLISHED_NO_REPLACE"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--run-design-preflight", action="store_true")
    parser.add_argument("--write-receipt", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.write_receipt and not args.run_design_preflight:
        raise PreceremonyError(
            "--write-receipt requires --run-design-preflight"
        )
    if not args.run_design_preflight:
        validated = validate_policy(args.policy)
        print(
            "PASS_STEP28_V13_V1_12_PRECEREMONY_POLICY",
            len(validated["failed_identity_hashes"]),
            len(validated["forbidden_master_commitments"]),
            "FORMAL_AUTHORIZATION_FALSE",
        )
        return
    receipt = run_design_preflight(args.policy)
    disposition = "RECEIPT_NOT_WRITTEN"
    if args.write_receipt:
        policy = load_json_strict(args.policy)
        disposition = publish_receipt_no_replace(
            _repo_path(policy["design_preflight"]["receipt_path"]), receipt
        )
    print(
        receipt["status"],
        receipt["world_count"],
        receipt["pair_count"],
        receipt["identity33_row_count"],
        disposition,
    )


if __name__ == "__main__":
    main()
