#!/usr/bin/env python3
"""Development-smoke-only split transaction protocol for Step28-v13 v1.13.

This module exercises the world-marker, recovery, smoke-finalization, seal, and
cleanup state machine in an ephemeral tree.  It deliberately cannot create a
formal split or turn the frozen non-committable development candidate into a
formal candidate.
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import importlib.util
import json
import os
import stat
import sys
import sysconfig
import tempfile
import threading
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CANDIDATE_POLICY_RAW_SHA256 = (
    "ddce84eeaf8c5e74067efacf497426c682acd14b62c13e87deb5ab8e53ebe6a0"
)
EXPECTED_CANDIDATE_POLICY_SELF_HASH = (
    "32f8a431311eba03e97068dc3b4597a24e4befe8867dd8311dde755c580a783d"
)
EXPECTED_CANDIDATE_SOURCE_SHA256 = (
    "73888224e44b606c6204d002397c2cec986c9d61d60222fdada56e5d1d048a55"
)


def _bootstrap_dependency_paths() -> set[Path]:
    candidates: list[Path] = []
    paths = sysconfig.get_paths()
    for key in ("purelib", "platlib"):
        value = paths.get(key)
        if value:
            candidates.append(Path(value))
    user_scheme = "nt_user" if os.name == "nt" else "posix_user"
    if user_scheme in sysconfig.get_scheme_names():
        for key in ("purelib", "platlib"):
            value = sysconfig.get_path(key, scheme=user_scheme)
            if value:
                candidates.append(Path(value))
    executable = Path(sys.executable).resolve()
    if os.name == "nt":
        candidates.append(executable.parent / "Lib" / "site-packages")
    else:
        version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        prefix = executable.parent.parent
        candidates.extend(
            (
                prefix / "lib" / version / "site-packages",
                prefix / "lib64" / version / "site-packages",
            )
        )
    return {candidate.resolve() for candidate in candidates if candidate.is_dir()}


def _bootstrap_verify_candidate_import_closure() -> None:
    """Verify the frozen Stage4A import closure before project code executes."""

    scripts_root = (ROOT / "scripts").resolve()
    resolved_paths = [Path(value or ".").resolve() for value in sys.path]
    if resolved_paths.count(scripts_root) != 1:
        raise RuntimeError("Canonical scripts directory is not unique on sys.path")
    scripts_index = resolved_paths.index(scripts_root)
    interpreter_roots = tuple(
        dict.fromkeys(
            Path(value).resolve()
            for value in (
                sys.base_prefix,
                sys.prefix,
            )
        )
    )
    dependency_paths = _bootstrap_dependency_paths()
    for entry, resolved in zip(sys.path[:scripts_index], resolved_paths[:scripts_index]):
        if not entry:
            raise RuntimeError("Current-directory import path precedes canonical scripts")
        if resolved not in dependency_paths and not any(
            root == resolved or root in resolved.parents for root in interpreter_roots
        ):
            raise RuntimeError("Non-interpreter import path precedes canonical scripts")
    preloaded = sorted(
        name
        for name in sys.modules
        if (
            name.startswith("step28_")
            or name == "step3_build_seller_profiles"
        )
        and name != __name__
    )
    if preloaded:
        raise RuntimeError("Project module was preloaded before frozen bootstrap")
    cached_bytecode = sorted(
        path
        for base in (ROOT / "scripts", ROOT / "tests")
        for path in base.rglob("*.pyc")
    )
    if cached_bytecode:
        raise RuntimeError("Project bytecode cache exists before frozen bootstrap")
    sys.dont_write_bytecode = True

    policy_path = ROOT / "schema" / "step28_v13_v1_13_candidate_selection_policy.json"
    source_path = ROOT / "scripts" / "step28_v13_v1_13_candidate_selection.py"
    try:
        policy_bytes = policy_path.read_bytes()
        source_bytes = source_path.read_bytes()
    except OSError as exc:
        raise RuntimeError("Frozen Stage4A import closure is unavailable") from exc
    if (
        hashlib.sha256(policy_bytes).hexdigest()
        != EXPECTED_CANDIDATE_POLICY_RAW_SHA256
        or hashlib.sha256(source_bytes).hexdigest()
        != EXPECTED_CANDIDATE_SOURCE_SHA256
    ):
        raise RuntimeError("Frozen Stage4A import root drifted before import")
    try:
        policy = json.loads(policy_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Frozen Stage4A policy cannot bootstrap imports") from exc
    if policy.get("canonical_self_hash") != EXPECTED_CANDIDATE_POLICY_SELF_HASH:
        raise RuntimeError("Frozen Stage4A policy self-hash pin drifted before import")
    frozen_inputs = policy.get("frozen_inputs")
    if not isinstance(frozen_inputs, dict) or not frozen_inputs:
        raise RuntimeError("Frozen Stage4A import closure is empty")
    for role, pin in frozen_inputs.items():
        if not isinstance(pin, dict) or set(pin) != {"path", "sha256", "size_bytes"}:
            raise RuntimeError(f"Frozen Stage4A import pin is malformed: {role}")
        relative = Path(pin["path"])
        if relative.is_absolute() or relative.as_posix() != pin["path"] or ".." in relative.parts:
            raise RuntimeError(f"Frozen Stage4A import path is unsafe: {role}")
        target = (ROOT / relative).resolve()
        try:
            target.relative_to(ROOT.resolve())
            payload = target.read_bytes()
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Frozen Stage4A import pin is unavailable: {role}") from exc
        if (
            type(pin["size_bytes"]) is not int
            or len(payload) != pin["size_bytes"]
            or hashlib.sha256(payload).hexdigest() != pin["sha256"]
        ):
            raise RuntimeError(f"Frozen Stage4A import bytes drifted: {role}")
        if relative.parts[0] == "scripts" and relative.suffix == ".py":
            module_name = relative.stem
            spec = importlib.util.find_spec(module_name)
            if spec is None or spec.origin is None:
                raise RuntimeError(f"Frozen Stage4A module cannot be resolved: {role}")
            if Path(spec.origin).resolve() != target:
                raise RuntimeError(f"Frozen Stage4A module resolved outside root: {role}")


_bootstrap_verify_candidate_import_closure()

import step28_v13_common as common
import step28_v13_identity_values as identity_values
import step28_v13_v1_13_candidate_selection as selection
import step28_v13_v1_13_natural_variation as natural


def _verify_loaded_project_module_origins() -> None:
    scripts_root = (ROOT / "scripts").resolve()
    for name, module in tuple(sys.modules.items()):
        if not (
            name.startswith("step28_")
            or name == "step3_build_seller_profiles"
        ):
            continue
        origin = getattr(module, "__file__", None)
        if not origin:
            raise RuntimeError(f"Loaded project module has no source origin: {name}")
        resolved = Path(origin).resolve()
        try:
            resolved.relative_to(scripts_root)
        except ValueError as exc:
            raise RuntimeError(
                f"Loaded project module escaped canonical scripts root: {name}"
            ) from exc
        if resolved.suffix != ".py":
            raise RuntimeError(f"Loaded project module is not source-backed: {name}")


_verify_loaded_project_module_origins()


DEFAULT_POLICY_PATH = (
    ROOT / "schema" / "step28_v13_v1_13_split_transaction_policy.json"
)
TRANSACTION_SOURCE_PATH = ROOT / "scripts" / "step28_v13_v1_13_split_transaction.py"
SOURCE_GUARD_PATH = ROOT / "scripts" / "step28_v13_v1_13_source_guard.py"
TRANSACTION_TEST_PATH = (
    ROOT / "tests" / "step28_v13_v1_13_split_transaction_contracts.py"
)
POLICY_VERSION = "2026-08-10-step28-v13-v1-13-split-transaction-policy-v1"
POLICY_STATUS = "DESIGN_SMOKE_ONLY_SPLIT_TRANSACTION_NO_FORMAL_AUTHORIZATION"
CLAIM_BOUNDARY = (
    "One pinned development-smoke world may exercise the transaction, recovery, "
    "smoke-finalization, seal, and cleanup state machine only inside an ephemeral "
    "workspace. This is not a formal split, formal canonical member set, formal "
    "durability certification, dataset, model, metric, or release."
)
ALLOWED_MODE = "development_smoke"
ALLOWED_SPLIT = "audit_a"
EXPECTED_WORLD_COUNT = 1
SYNTHETIC_PREDECESSOR_DOMAIN = (
    "step28-v13-v1.13-synthetic-predecessor-fixture-v1"
)
FORMAL_SPLIT_ORDER = ("train", "development", "audit_a", "audit_b")
WORKSPACE_PREFIX = "step28-v13-v1-13-transaction-smoke-"
WORLD_DIRECTORY_NAME = "world_000000"
WORLD_MARKER_NAME = "WORLD_ACCEPTED.json"
SPLIT_SEAL_NAME = "SPLIT_SEAL_COMPLETE.json"
CLEANUP_RECEIPT_NAME = "WORLD_CLEANUP_COMPLETE.json"
WORKSPACE_BOUNDARY_NAME = "DEVELOPMENT_SMOKE_ONLY.json"
TRANSACTIONS_DIRECTORY_NAME = "transactions"
FINAL_DIRECTORY_NAME = "smoke_final_projection"

FORMAL_AUTHORIZATION_KEYS = frozenset(
    {
        "audit_truth_access",
        "formal_capability_derivation",
        "formal_candidate_generation",
        "formal_dataset_generation",
        "formal_model_training",
        "formal_seed_ceremony",
    }
)
HEX = frozenset("0123456789abcdef")
WORLD_MEMBER_ROLES = (
    "allocation_delta",
    "exact_title_clone_qualification",
    "identity33",
    "item_document_hashes",
    "private_world",
    "profile_provenance",
    "redacted_items",
    "rejection_counts",
    "selection_context",
    "seller_document_hashes",
    "seller_profiles",
)
EXPECTED_WORLD_MEMBERS = {
    "allocation_delta": "allocation_delta.json",
    "exact_title_clone_qualification": "exact_title_clone_qualification.json",
    "identity33": "identity33.json",
    "item_document_hashes": "item_document_hashes.json",
    "private_world": "private_world.json",
    "profile_provenance": "profile_provenance.json",
    "redacted_items": "redacted_items.json",
    "rejection_counts": "rejection_counts.json",
    "selection_context": "selection_context.json",
    "seller_document_hashes": "seller_document_hashes.json",
    "seller_profiles": "seller_profiles.json",
}
FINAL_MEMBER_NAMES = (
    "worlds.jsonl",
    "redacted_items.jsonl",
    "seller_profiles.jsonl",
    "identity33.jsonl",
    "document_collision_attempts.jsonl",
    "document_collision_registry.json",
    "split_manifest.json",
)
PUBLIC_WORLD_FIELDS = ("world_uid",)


class SplitTransactionError(RuntimeError):
    """Base fail-closed transaction error."""


class SplitLockBusy(SplitTransactionError):
    """The stable development-smoke split lock is already held."""


class RecoveryCorruption(SplitTransactionError):
    """Persistent smoke state cannot be validated."""


class PublishConflict(SplitTransactionError):
    """An immutable destination already contains different bytes."""


def _canonical_bytes(value: Any) -> bytes:
    return common.canonical_json_bytes(value)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in HEX for character in value)
    ):
        raise SplitTransactionError(f"{label} is not a lowercase SHA-256")
    return value


def _require_plain_int(value: Any, *, label: str) -> int:
    if type(value) is not int:
        raise SplitTransactionError(f"{label} must be a plain integer")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: Sequence[str] | set[str], *, label: str
) -> None:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise SplitTransactionError(f"{label} keyset drift")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise SplitTransactionError(f"Duplicate JSON key in {key}")
        output[key] = value
    return output


def _decode_canonical(payload: bytes, *, label: str) -> Any:
    if not isinstance(payload, bytes):
        raise SplitTransactionError(f"{label} must be immutable bytes")
    try:
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=_reject_duplicate_pairs
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SplitTransactionError(f"{label} is not strict UTF-8 JSON") from exc
    if _canonical_bytes(value) != payload:
        raise SplitTransactionError(f"{label} is not canonical JSON")
    return value


def _with_self_hash(value: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(value)
    if "canonical_self_hash" in output:
        raise SplitTransactionError("Self-hash input already has canonical_self_hash")
    output["canonical_self_hash"] = common.canonical_sha256(output)
    return output


def _verify_self_hash(value: Mapping[str, Any], *, label: str) -> None:
    expected = _require_sha256(value.get("canonical_self_hash"), label=label)
    unsigned = dict(value)
    unsigned.pop("canonical_self_hash")
    if common.canonical_sha256(unsigned) != expected:
        raise SplitTransactionError(f"{label} canonical self-hash drift")


def _validate_policy(policy: Mapping[str, Any]) -> None:
    required = {
        "version",
        "status",
        "claim_boundary",
        "formal_authorizations",
        "allowed_mode",
        "allowed_split",
        "expected_world_count",
        "smoke_projection_only",
        "formal_canonical_member_set_claimed",
        "formal_split_semantics",
        "linux_formal_directory_durability_certified",
        "stable_lock",
        "frozen_candidate_selection_trust",
        "implementation_bundle_pins",
        "source_execution_boundary",
        "allocation_delta_semantics",
        "transaction_layout",
        "smoke_final_member_plan",
        "jsonl_serialization",
        "state_machine",
        "publish_protocol",
        "recovery_trust",
        "synthetic_architecture_tests",
        "canonical_self_hash",
    }
    _require_exact_keys(policy, required, label="transaction policy")
    world_count = _require_plain_int(
        policy["expected_world_count"], label="expected world count"
    )
    if (
        policy["version"] != POLICY_VERSION
        or policy["status"] != POLICY_STATUS
        or policy["claim_boundary"] != CLAIM_BOUNDARY
        or policy["allowed_mode"] != ALLOWED_MODE
        or policy["allowed_split"] != ALLOWED_SPLIT
        or world_count != EXPECTED_WORLD_COUNT
        or policy["smoke_projection_only"] is not True
        or policy["formal_canonical_member_set_claimed"] is not False
        or policy["formal_split_semantics"] is not False
        or policy["linux_formal_directory_durability_certified"] is not False
    ):
        raise SplitTransactionError("Transaction policy boundary drift")
    authorizations = policy["formal_authorizations"]
    _require_exact_keys(
        authorizations, FORMAL_AUTHORIZATION_KEYS, label="formal authorizations"
    )
    if any(value is not False for value in authorizations.values()):
        raise SplitTransactionError("Formal authorization became true")
    trust = policy["frozen_candidate_selection_trust"]
    expected_trust = {
        "policy_path": "schema/step28_v13_v1_13_candidate_selection_policy.json",
        "policy_raw_sha256": EXPECTED_CANDIDATE_POLICY_RAW_SHA256,
        "policy_canonical_self_hash": EXPECTED_CANDIDATE_POLICY_SELF_HASH,
        "source_path": "scripts/step28_v13_v1_13_candidate_selection.py",
        "source_sha256": EXPECTED_CANDIDATE_SOURCE_SHA256,
        "expected_accepted_state_sha256": (
            "6f14e23acbb0b7adbfda3e7044c9d5389338db30fd5692dee2a2aaff3802343f"
        ),
        "expected_world_uid": (
            "w_003497845547650a980473b05e249937bf825ad0eaefa424baec74f2bd2210f3"
        ),
        "expected_candidate_index": 0,
        "expected_candidates_examined": 1,
        "expected_rejected_candidate_count": 0,
        "expected_item_count": 105,
        "expected_seller_count": 28,
        "expected_allocation_delta_count": 84,
    }
    if _canonical_bytes(trust) != _canonical_bytes(expected_trust):
        raise SplitTransactionError("Frozen candidate trust pin drift")
    implementation = policy["implementation_bundle_pins"]
    _require_exact_keys(
        implementation,
        {
            "trust_status",
            "external_parent_receipt_required",
            "source_guard",
            "source",
            "contract_test",
        },
        label="implementation bundle pins",
    )
    if (
        implementation["trust_status"]
        != "INTERNAL_CLOSURE_ONLY_PENDING_EXTERNAL_GIT_PARENT"
        or implementation["external_parent_receipt_required"] is not True
    ):
        raise SplitTransactionError("Implementation bundle trust boundary drift")
    for role, expected_path in (
        ("source_guard", SOURCE_GUARD_PATH),
        ("source", TRANSACTION_SOURCE_PATH),
        ("contract_test", TRANSACTION_TEST_PATH),
    ):
        pin = implementation[role]
        _require_exact_keys(pin, {"path", "size_bytes", "sha256"}, label=role)
        relative_path = expected_path.relative_to(ROOT).as_posix()
        if pin["path"] != relative_path:
            raise SplitTransactionError(f"{role} path pin drift")
        size_bytes = _require_plain_int(pin["size_bytes"], label=f"{role} size")
        expected_sha256 = _require_sha256(pin["sha256"], label=f"{role} SHA-256")
        try:
            observed_size = expected_path.stat().st_size
            observed_sha256 = common.sha256_file(expected_path)
        except OSError as exc:
            raise SplitTransactionError(f"{role} implementation pin unavailable") from exc
        if size_bytes != observed_size or expected_sha256 != observed_sha256:
            raise SplitTransactionError(f"{role} implementation bytes drift")
    expected_sections: dict[str, Any] = {
        "source_execution_boundary": {
            "entrypoint": "scripts/step28_v13_v1_13_source_guard.py",
            "required_python_flags": ["-I", "-S", "-B"],
            "environment_isolation_required": True,
            "site_initialization_forbidden": True,
            "site_module_preloaded_forbidden": True,
            "third_party_paths_derived_from_interpreter_only": False,
            "user_site_reintroduced": True,
            "user_site_path_may_depend_on_os_environment": True,
            "pth_execution_forbidden": True,
            "interpreter_standard_library_before_scripts": True,
            "third_party_before_canonical_scripts": True,
            "third_party_dependency_bytes_pinned": False,
            "hostile_third_party_payload_protected": False,
            "hostile_or_environment_redirected_third_party_protected": False,
            "formal_environment_attestation_required": True,
            "main_script_source_only": True,
            "project_bytecode_cache_forbidden": True,
            "project_module_preload_forbidden": True,
            "canonical_scripts_path_unique": True,
            "ordinary_unittest_discovery_includes_focused_contracts": False,
            "ordinary_unittest_discovery_authoritative_for_stage4b": False,
            "focused_contracts_authoritative_entrypoint_only": True,
        },
        "stable_lock": {
            "domain": "step28-v13-v1.13-split-transaction-lock-v1",
            "identity_fields": [
                "canonical_repository_realpath",
                "transaction_policy_version",
                "allowed_mode",
                "allowed_split",
            ],
            "nonblocking": True,
            "lock_before_ephemeral_workspace_creation": True,
            "busy_is_research_failure": False,
            "windows_scope": "same_logon_session_only",
            "posix_lock_object": "canonical_repository_directory_inode",
            "repository_root_inode_must_remain_stable": True,
            "hostile_cross_session_or_repository_root_swap_protected": False,
            "formal_lock_upgrade_required": True,
        },
        "allocation_delta_semantics": {
            "element_type": "lowercase_sha256_of_normalized_identity_value",
            "derivation": "sorted unique trial_allocated minus committed_identity_hashes",
            "must_equal_all_private_identity_asset_value_hashes": True,
            "sufficient_to_reconstruct_live_identity_exclusion_set": True,
            "raw_identity_values_persisted_in_marker": False,
        },
        "transaction_layout": {
            "transactions_directory": TRANSACTIONS_DIRECTORY_NAME,
            "final_directory": FINAL_DIRECTORY_NAME,
            "world_directory_format": WORLD_DIRECTORY_NAME,
            "world_marker": WORLD_MARKER_NAME,
            "split_seal": SPLIT_SEAL_NAME,
            "cleanup_receipt": CLEANUP_RECEIPT_NAME,
            "workspace_boundary": WORKSPACE_BOUNDARY_NAME,
            "world_members": EXPECTED_WORLD_MEMBERS,
        },
        "smoke_final_member_plan": {
            "worlds.jsonl": {
                "source": "private_world.public.world",
                "order": ["world_uid"],
                "allowed_fields": list(PUBLIC_WORLD_FIELDS),
                "private_world_forbidden": True,
            },
            "redacted_items.jsonl": {
                "source": "redacted_items",
                "order": ["world_uid", "seller_uid", "item_uid"],
            },
            "seller_profiles.jsonl": {
                "source": "seller_profiles",
                "order": ["seller_uid"],
            },
            "identity33.jsonl": {
                "source": "identity33",
                "order": ["world_uid", "canonical_pair_uid"],
            },
            "document_collision_attempts.jsonl": {
                "source": "accepted_marker_private_audit_projection",
                "order": ["world_ordinal"],
            },
            "document_collision_registry.json": {
                "source": "accepted_marker_registry_deltas",
                "canonical_json_object": True,
            },
            "split_manifest.json": {
                "source": "all_other_smoke_final_member_commitments",
                "canonical_json_object": True,
            },
        },
        "jsonl_serialization": {
            "row_serialization": "canonical_json_bytes(single_row)",
            "row_separator_hex": "0a",
            "final_newline_required": True,
            "empty_file_bytes": "",
            "sorting": (
                "listed fields by UTF-8 bytes; numeric values remain exact JSON numbers"
            ),
            "bom_forbidden": True,
        },
        "state_machine": [
            "UNCOMMITTED_SAFE_DIRECTORY",
            "COMMITTED_PRE_SEAL_FULL_MEMBERS",
            "PARTIAL_FINAL_PRE_SEAL",
            "SEALED_CLEANUP_IN_PROGRESS",
            "SEALED_CLEANUP_COMPLETE",
        ],
        "publish_protocol": {
            "temporary_create": "same-directory O_CREAT|O_EXCL",
            "file_sync_before_publish": True,
            "publish": "atomic destination-exists-fails no-replace rename",
            "replace_calls_forbidden": True,
            "parent_directory_sync_after_publish": True,
            "cooperating_writers_must_honor_split_lock": True,
            "hostile_concurrent_parent_swap_protected": False,
            "formal_handle_relative_upgrade_required": True,
            "windows_boundary": (
                "development smoke proves file fsync and atomic visibility only; "
                "no formal directory-durability claim"
            ),
            "linux_boundary": (
                "formal use remains forbidden; a future formal policy must require "
                "supported renameat2 RENAME_NOREPLACE and directory fsync"
            ),
        },
        "recovery_trust": {
            "pre_seal_verifier": (
                "all marker members present; reconstruct accepted candidate; replay "
                "production documents; match frozen candidate golden"
            ),
            "post_seal_verifier": (
                "fixed render(0) replay from exact-pinned candidate sources derives "
                "expected marker, final bytes, seal, cleanup plan, and receipt without "
                "trusting disk outputs"
            ),
            "valid_marker_never_researches_candidates": True,
            "fixed_post_seal_render_is_not_candidate_search": True,
            "formal_500_world_authenticity_proven": False,
        },
        "synthetic_architecture_tests": {
            "minimum_marker_chain_length": 3,
            "formal_split_order": list(FORMAL_SPLIT_ORDER),
            "predecessor_fixture_domain": SYNTHETIC_PREDECESSOR_DOMAIN,
            "real_production_chain_world_count": 1,
            "synthetic_markers_are_scientific_results": False,
        },
    }
    for name, expected_section in expected_sections.items():
        if _canonical_bytes(policy[name]) != _canonical_bytes(expected_section):
            raise SplitTransactionError(f"Transaction policy section drift: {name}")
    _verify_self_hash(policy, label="transaction policy")


def load_policy(path: Path = DEFAULT_POLICY_PATH) -> dict[str, Any]:
    if path.resolve() != DEFAULT_POLICY_PATH.resolve():
        raise SplitTransactionError("Only the canonical transaction policy may pass")
    try:
        policy = common.load_json(path)
    except (OSError, common.ContractError) as exc:
        raise SplitTransactionError("Transaction policy is unreadable") from exc
    if not isinstance(policy, dict):
        raise SplitTransactionError("Transaction policy must be an object")
    _validate_policy(policy)
    return policy


def _verify_frozen_candidate_sources() -> dict[str, Any]:
    policy_path = ROOT / "schema" / "step28_v13_v1_13_candidate_selection_policy.json"
    source_path = ROOT / "scripts" / "step28_v13_v1_13_candidate_selection.py"
    if (
        common.sha256_file(policy_path) != EXPECTED_CANDIDATE_POLICY_RAW_SHA256
        or common.sha256_file(source_path) != EXPECTED_CANDIDATE_SOURCE_SHA256
    ):
        raise SplitTransactionError("Frozen candidate-selection raw bytes drift")
    candidate_policy = selection.load_policy()
    if (
        candidate_policy.get("canonical_self_hash")
        != EXPECTED_CANDIDATE_POLICY_SELF_HASH
    ):
        raise SplitTransactionError("Frozen candidate-selection self-hash drift")
    trust = load_policy()["frozen_candidate_selection_trust"]
    expected = candidate_policy["expected_smoke_selection"]
    comparisons = {
        "expected_accepted_state_sha256": expected["accepted_state_sha256"],
        "expected_world_uid": expected["world_uid"],
        "expected_candidate_index": expected["accepted_candidate_index"],
        "expected_candidates_examined": expected["candidates_examined"],
        "expected_rejected_candidate_count": expected["rejected_candidate_count"],
        "expected_item_count": expected["item_count"],
        "expected_seller_count": expected["seller_count"],
        "expected_allocation_delta_count": expected["allocation_delta_count"],
    }
    if any(trust[key] != value for key, value in comparisons.items()):
        raise SplitTransactionError("Candidate-selection golden pin drift")
    return candidate_policy


def _is_reparse(stat_result: os.stat_result) -> bool:
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(getattr(stat_result, "st_file_attributes", 0) & flag)


def _require_plain_single_link(path: Path, *, label: str) -> os.stat_result:
    try:
        result = os.lstat(common.filesystem_path(path))
    except OSError as exc:
        raise RecoveryCorruption(f"{label} is unavailable") from exc
    if (
        not stat.S_ISREG(result.st_mode)
        or result.st_nlink != 1
        or _is_reparse(result)
    ):
        raise RecoveryCorruption(f"{label} is not a plain single-link file")
    return result


def _require_plain_directory(path: Path, *, label: str) -> os.stat_result:
    try:
        result = os.lstat(common.filesystem_path(path))
    except OSError as exc:
        raise RecoveryCorruption(f"{label} is unavailable") from exc
    if not stat.S_ISDIR(result.st_mode) or _is_reparse(result):
        raise RecoveryCorruption(f"{label} is not a plain directory")
    return result


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev,
        left.st_ino,
        left.st_size,
        left.st_nlink,
    ) == (
        right.st_dev,
        right.st_ino,
        right.st_size,
        right.st_nlink,
    )


def _read_plain_file(path: Path, *, label: str) -> bytes:
    before = _require_plain_single_link(path, label=label)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(common.filesystem_path(path), flags)
    try:
        opened = os.fstat(descriptor)
        if not _same_file_identity(before, opened):
            raise RecoveryCorruption(f"{label} changed during no-follow open")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after_descriptor = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after_path = _require_plain_single_link(path, label=label)
    if (
        not _same_file_identity(opened, after_descriptor)
        or not _same_file_identity(after_descriptor, after_path)
    ):
        raise RecoveryCorruption(f"{label} changed while being read")
    payload = b"".join(chunks)
    if len(payload) != opened.st_size:
        raise RecoveryCorruption(f"{label} size changed while being read")
    return payload


def _fsync_directory(path: Path) -> None:
    """Sync directory entries on POSIX; Windows smoke makes no durability claim."""

    if os.name == "nt":
        return
    descriptor = os.open(common.filesystem_path(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _unlink_known_plain_file(
    token: _SplitLockToken, path: Path, *, label: str
) -> None:
    _assert_lock(token)
    _require_plain_single_link(path, label=label)
    os.unlink(common.filesystem_path(path))


def _publish_no_replace(
    token: _SplitLockToken, path: Path, payload: bytes
) -> None:
    _assert_lock(token)
    if not isinstance(payload, bytes):
        raise TypeError("Immutable publication requires bytes")
    parent = path.parent
    _require_plain_directory(parent, label="publication parent")
    stage = parent / f".{path.name}.pending"
    if os.path.lexists(common.filesystem_path(stage)):
        _unlink_known_plain_file(token, stage, label="stale known pending file")
    try:
        with open(common.filesystem_path(stage), "xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        common.atomic_rename_no_replace(stage, path)
        _fsync_directory(parent)
    finally:
        if os.path.lexists(common.filesystem_path(stage)):
            _unlink_known_plain_file(token, stage, label="failed pending file")


def _publish_or_verify(
    token: _SplitLockToken, path: Path, payload: bytes
) -> None:
    _assert_lock(token)
    if os.path.lexists(common.filesystem_path(path)):
        observed = _read_plain_file(path, label=f"existing {path.name}")
        if observed != payload:
            raise PublishConflict(f"Existing immutable file differs: {path.name}")
        return
    try:
        _publish_no_replace(token, path, payload)
    except OSError as exc:
        if exc.errno != errno.EEXIST and not isinstance(exc, FileExistsError):
            raise
        observed = _read_plain_file(path, label=f"raced {path.name}")
        if observed != payload:
            raise PublishConflict(f"Raced immutable file differs: {path.name}") from exc


def _stable_lock_identity() -> str:
    repository = os.path.normcase(os.fspath(ROOT.resolve()))
    return common.canonical_sha256(
        {
            "domain": "step28-v13-v1.13-split-transaction-lock-v1",
            "repository_realpath": repository,
            "transaction_policy_version": POLICY_VERSION,
            "mode": ALLOWED_MODE,
            "split": ALLOWED_SPLIT,
        }
    )


class _SplitLockToken:
    __slots__ = ("identity", "_nonce")

    def __init__(self, identity: str, nonce: object) -> None:
        self.identity = identity
        self._nonce = nonce


_PROCESS_LOCK_GUARD = threading.Lock()
_PROCESS_LOCK_IDENTITY: str | None = None
_ACTIVE_LOCK_TOKEN: _SplitLockToken | None = None


def _assert_lock(token: _SplitLockToken) -> None:
    if (
        not isinstance(token, _SplitLockToken)
        or _ACTIVE_LOCK_TOKEN is None
        or token is not _ACTIVE_LOCK_TOKEN
        or token._nonce is not _ACTIVE_LOCK_TOKEN._nonce
    ):
        raise SplitTransactionError("Split transaction operation lacks the active lock")


def _release_windows_mutex(kernel32: Any, handle: Any, *, acquired: bool) -> None:
    errors: list[str] = []
    if acquired and not bool(kernel32.ReleaseMutex(handle)):
        errors.append("ReleaseMutex")
    if not bool(kernel32.CloseHandle(handle)):
        errors.append("CloseHandle")
    if errors:
        raise SplitTransactionError(
            "Stable Windows split mutex cleanup failed: " + ", ".join(errors)
        )


@contextlib.contextmanager
def _exclusive_split_lock() -> Iterator[_SplitLockToken]:
    """Acquire the stable stage/split lock before any workspace exists."""

    global _PROCESS_LOCK_IDENTITY, _ACTIVE_LOCK_TOKEN
    identity = _stable_lock_identity()
    with _PROCESS_LOCK_GUARD:
        if _PROCESS_LOCK_IDENTITY is not None:
            raise SplitLockBusy("Development-smoke split transaction is already active")
        _PROCESS_LOCK_IDENTITY = identity
    acquired = False
    handle: Any = None
    descriptor: int | None = None
    token = _SplitLockToken(identity, object())
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.CreateMutexW.argtypes = (
                wintypes.LPVOID,
                wintypes.BOOL,
                wintypes.LPCWSTR,
            )
            kernel32.CreateMutexW.restype = wintypes.HANDLE
            kernel32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
            kernel32.WaitForSingleObject.restype = wintypes.DWORD
            kernel32.ReleaseMutex.argtypes = (wintypes.HANDLE,)
            kernel32.ReleaseMutex.restype = wintypes.BOOL
            kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
            kernel32.CloseHandle.restype = wintypes.BOOL
            handle = kernel32.CreateMutexW(
                None, False, f"Local\\Step28V13V113SplitTransaction-{identity}"
            )
            if not handle:
                raise SplitTransactionError("Unable to create stable split mutex")
            result = int(kernel32.WaitForSingleObject(handle, 0))
            if result == 258:
                raise SplitLockBusy("Development-smoke split transaction is busy")
            if result not in {0, 128}:
                raise SplitTransactionError("Unable to acquire stable split mutex")
            acquired = True
        else:
            import fcntl

            flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            descriptor = os.open(common.filesystem_path(ROOT), flags)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno in {errno.EACCES, errno.EAGAIN}:
                    raise SplitLockBusy(
                        "Development-smoke split transaction is busy"
                    ) from exc
                raise
            acquired = True
        _ACTIVE_LOCK_TOKEN = token
        yield token
    finally:
        _ACTIVE_LOCK_TOKEN = None
        try:
            if os.name == "nt" and handle:
                import ctypes

                kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
                kernel32.ReleaseMutex.argtypes = (ctypes.c_void_p,)
                kernel32.ReleaseMutex.restype = ctypes.c_int
                kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
                kernel32.CloseHandle.restype = ctypes.c_int
                _release_windows_mutex(kernel32, handle, acquired=acquired)
            elif descriptor is not None:
                if acquired:
                    import fcntl

                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)
        finally:
            with _PROCESS_LOCK_GUARD:
                _PROCESS_LOCK_IDENTITY = None


def _workspace_boundary_value(root: Path, *, policy: Mapping[str, Any]) -> dict[str, Any]:
    return _with_self_hash(
        {
            "artifact_type": "step28_v13_v1_13_development_smoke_workspace_boundary",
            "schema_version": "2026-08-10-v1",
            "status": POLICY_STATUS,
            "design_smoke_only": True,
            "formal_split_semantics": False,
            "formal_authorizations": dict(policy["formal_authorizations"]),
            "transaction_policy_version": policy["version"],
            "transaction_policy_canonical_self_hash": policy["canonical_self_hash"],
            "stable_lock_identity": _stable_lock_identity(),
            "workspace_realpath_sha256": _sha256_bytes(
                os.path.normcase(os.fspath(root.resolve())).encode("utf-8")
            ),
        }
    )


def _validate_workspace_root(root: Path) -> None:
    if not isinstance(root, Path) or not root.name.startswith(WORKSPACE_PREFIX):
        raise RecoveryCorruption("Workspace name is outside the smoke namespace")
    result = _require_plain_directory(root, label="smoke workspace root")
    if result.st_nlink < 1:
        raise RecoveryCorruption("Smoke workspace root link count drift")
    if Path(os.path.abspath(os.fspath(root))) != root.resolve():
        raise RecoveryCorruption("Smoke workspace root uses a path alias")


def _initialize_workspace(
    token: _SplitLockToken, root: Path, *, create: bool
) -> tuple[dict[str, Any], bytes]:
    _assert_lock(token)
    policy = load_policy()
    if create:
        if os.path.lexists(common.filesystem_path(root)):
            raise RecoveryCorruption("Fresh smoke workspace already exists")
        os.mkdir(common.filesystem_path(root))
        _fsync_directory(root.parent)
    _validate_workspace_root(root)
    boundary_path = root / WORKSPACE_BOUNDARY_NAME
    boundary_pending = root / f".{WORKSPACE_BOUNDARY_NAME}.pending"
    expected = _workspace_boundary_value(root, policy=policy)
    expected_bytes = _canonical_bytes(expected)
    if os.path.lexists(common.filesystem_path(boundary_path)):
        observed = _read_plain_file(boundary_path, label="workspace boundary")
        if observed != expected_bytes:
            raise RecoveryCorruption("Workspace boundary differs from this locked root")
    else:
        entry_names = {
            entry.name for entry in os.scandir(common.filesystem_path(root))
        }
        if entry_names == {boundary_pending.name}:
            _unlink_known_plain_file(
                token, boundary_pending, label="stale workspace-boundary pending file"
            )
            _fsync_directory(root)
        elif entry_names:
            raise RecoveryCorruption("Uninitialized smoke workspace is not empty")
        _publish_no_replace(token, boundary_path, expected_bytes)
    _verify_self_hash(expected, label="workspace boundary")
    return expected, expected_bytes


def _ensure_plain_directory(
    token: _SplitLockToken, path: Path, *, parent: Path, label: str
) -> None:
    _assert_lock(token)
    _require_plain_directory(parent, label=f"{label} parent")
    if os.path.lexists(common.filesystem_path(path)):
        _require_plain_directory(path, label=label)
        return
    os.mkdir(common.filesystem_path(path))
    _fsync_directory(parent)


def _entry_names(path: Path, *, label: str) -> set[str]:
    _require_plain_directory(path, label=label)
    return {entry.name for entry in os.scandir(common.filesystem_path(path))}


def _validate_workspace_entries(root: Path) -> None:
    allowed = {
        WORKSPACE_BOUNDARY_NAME,
        TRANSACTIONS_DIRECTORY_NAME,
        FINAL_DIRECTORY_NAME,
    }
    unknown = _entry_names(root, label="smoke workspace root") - allowed
    if unknown:
        raise RecoveryCorruption("Unknown workspace entry")


def _validate_world_directory_entries(
    world_directory: Path,
    *,
    policy: Mapping[str, Any],
    post_seal: bool,
    allow_cleanup_receipt_pending: bool = False,
) -> None:
    allowed = set(policy["transaction_layout"]["world_members"].values())
    allowed.add(WORLD_MARKER_NAME)
    if post_seal:
        allowed.add(CLEANUP_RECEIPT_NAME)
        if allow_cleanup_receipt_pending:
            allowed.add(f".{CLEANUP_RECEIPT_NAME}.pending")
    unknown = _entry_names(world_directory, label="world transaction directory") - allowed
    if unknown:
        raise RecoveryCorruption("Unknown world transaction entry")


def _validate_final_directory_entries(
    final_directory: Path, *, post_seal: bool
) -> None:
    allowed = set(FINAL_MEMBER_NAMES) | {SPLIT_SEAL_NAME}
    if not post_seal:
        allowed.update(f".{name}.pending" for name in tuple(allowed))
    unknown = _entry_names(final_directory, label="smoke final directory") - allowed
    if unknown:
        raise RecoveryCorruption("Unknown smoke final entry")


def _remove_safe_pre_seal_pending_files(
    token: _SplitLockToken, final_directory: Path
) -> None:
    _assert_lock(token)
    removed = False
    for name in (*FINAL_MEMBER_NAMES, SPLIT_SEAL_NAME):
        pending = final_directory / f".{name}.pending"
        if os.path.lexists(common.filesystem_path(pending)):
            _unlink_known_plain_file(
                token, pending, label="stale pre-seal pending file"
            )
            removed = True
    if removed:
        _fsync_directory(final_directory)


def _member_commitment(role: str, relative_path: str, payload: bytes) -> dict[str, Any]:
    if role not in WORLD_MEMBER_ROLES:
        raise SplitTransactionError("Unknown world-member role")
    if Path(relative_path).as_posix() != relative_path or len(Path(relative_path).parts) != 1:
        raise SplitTransactionError("World-member path is not a direct canonical child")
    return {
        "role": role,
        "path": relative_path,
        "size_bytes": len(payload),
        "sha256": _sha256_bytes(payload),
    }


def _world_member_payloads(
    accepted: selection.AcceptedDevelopmentCandidate,
    *,
    policy: Mapping[str, Any],
) -> dict[str, bytes]:
    member_names = policy["transaction_layout"]["world_members"]
    payloads = {
        "allocation_delta": _canonical_bytes(list(accepted.allocation_delta)),
        "exact_title_clone_qualification": accepted.exact_title_clone_qualification_bytes,
        "identity33": accepted.identity33_bytes,
        "item_document_hashes": accepted.item_hash_rows_bytes,
        "private_world": accepted.world_bytes,
        "profile_provenance": accepted.profile_provenance_bytes,
        "redacted_items": accepted.redacted_items_bytes,
        "rejection_counts": accepted.rejection_counts_bytes,
        "selection_context": accepted.selection_context_bytes,
        "seller_document_hashes": accepted.seller_hash_rows_bytes,
        "seller_profiles": accepted.profiles_bytes,
    }
    if set(payloads) != set(WORLD_MEMBER_ROLES):
        raise AssertionError("World-member payload plan drift")
    output: dict[str, bytes] = {}
    for role in WORLD_MEMBER_ROLES:
        filename = str(member_names[role])
        if filename in output:
            raise SplitTransactionError("Duplicate world-member filename")
        output[filename] = payloads[role]
    return output


def _accepted_scalar_projection(
    accepted: selection.AcceptedDevelopmentCandidate,
) -> dict[str, Any]:
    byte_fields = {
        "rejection_counts_bytes",
        "identity33_bytes",
        "profile_provenance_bytes",
        "world_bytes",
        "redacted_items_bytes",
        "profiles_bytes",
        "item_hash_rows_bytes",
        "seller_hash_rows_bytes",
        "selection_context_bytes",
        "exact_title_clone_qualification_bytes",
    }
    output: dict[str, Any] = {}
    for field in fields(selection.AcceptedDevelopmentCandidate):
        if field.name in byte_fields:
            continue
        value = getattr(accepted, field.name)
        output[field.name] = list(value) if isinstance(value, tuple) else value
    return output


def _transaction_genesis(
    *, policy: Mapping[str, Any], workspace_boundary_sha256: str
) -> str:
    return common.canonical_sha256(
        {
            "domain": "step28-v13-v1.13-development-smoke-transaction-genesis-v1",
            "transaction_policy_version": policy["version"],
            "transaction_policy_canonical_self_hash": policy["canonical_self_hash"],
            "candidate_selection_policy_raw_sha256": (
                EXPECTED_CANDIDATE_POLICY_RAW_SHA256
            ),
            "candidate_selection_policy_canonical_self_hash": (
                EXPECTED_CANDIDATE_POLICY_SELF_HASH
            ),
            "mode": ALLOWED_MODE,
            "split": ALLOWED_SPLIT,
            "expected_world_count": EXPECTED_WORLD_COUNT,
            "predecessor_seal_pins": [],
            "workspace_boundary_sha256": workspace_boundary_sha256,
        }
    )


def _marker_value(
    accepted: selection.AcceptedDevelopmentCandidate,
    *,
    policy: Mapping[str, Any],
    boundary_bytes: bytes,
) -> dict[str, Any]:
    payloads = _world_member_payloads(accepted, policy=policy)
    names_by_role = policy["transaction_layout"]["world_members"]
    manifest = [
        _member_commitment(role, str(names_by_role[role]), payloads[str(names_by_role[role])])
        for role in WORLD_MEMBER_ROLES
    ]
    return _with_self_hash(
        {
            "artifact_type": "step28_v13_v1_13_development_smoke_world_accepted",
            "schema_version": "2026-08-10-v1",
            "status": "DEVELOPMENT_SMOKE_WORLD_ACCEPTED_NOT_FORMAL",
            "design_smoke_only": True,
            "formal_commit": False,
            "formal_split_semantics": False,
            "source_candidate_committable": False,
            "formal_authorizations": dict(policy["formal_authorizations"]),
            "transaction_policy_version": policy["version"],
            "transaction_policy_canonical_self_hash": policy["canonical_self_hash"],
            "candidate_selection_policy_raw_sha256": (
                EXPECTED_CANDIDATE_POLICY_RAW_SHA256
            ),
            "candidate_selection_policy_canonical_self_hash": (
                EXPECTED_CANDIDATE_POLICY_SELF_HASH
            ),
            "candidate_selection_source_sha256": EXPECTED_CANDIDATE_SOURCE_SHA256,
            "workspace_boundary_sha256": _sha256_bytes(boundary_bytes),
            "mode": ALLOWED_MODE,
            "split": ALLOWED_SPLIT,
            "expected_world_count": EXPECTED_WORLD_COUNT,
            "world_ordinal": 0,
            "world_uid": accepted.world_uid,
            "previous_world_marker_raw_sha256": _transaction_genesis(
                policy=policy,
                workspace_boundary_sha256=_sha256_bytes(boundary_bytes),
            ),
            "predecessor_seal_pins": [],
            "accepted_candidate": _accepted_scalar_projection(accepted),
            "member_manifest": manifest,
        }
    )


def _validate_marker_envelope(
    marker: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
    boundary_bytes: bytes,
) -> None:
    required = {
        "artifact_type",
        "schema_version",
        "status",
        "design_smoke_only",
        "formal_commit",
        "formal_split_semantics",
        "source_candidate_committable",
        "formal_authorizations",
        "transaction_policy_version",
        "transaction_policy_canonical_self_hash",
        "candidate_selection_policy_raw_sha256",
        "candidate_selection_policy_canonical_self_hash",
        "candidate_selection_source_sha256",
        "workspace_boundary_sha256",
        "mode",
        "split",
        "expected_world_count",
        "world_ordinal",
        "world_uid",
        "previous_world_marker_raw_sha256",
        "predecessor_seal_pins",
        "accepted_candidate",
        "member_manifest",
        "canonical_self_hash",
    }
    _require_exact_keys(marker, required, label="world marker")
    expected_world_count = _require_plain_int(
        marker["expected_world_count"], label="marker expected world count"
    )
    world_ordinal = _require_plain_int(
        marker["world_ordinal"], label="marker world ordinal"
    )
    if (
        marker["artifact_type"]
        != "step28_v13_v1_13_development_smoke_world_accepted"
        or marker["schema_version"] != "2026-08-10-v1"
        or marker["status"] != "DEVELOPMENT_SMOKE_WORLD_ACCEPTED_NOT_FORMAL"
        or marker["design_smoke_only"] is not True
        or marker["formal_commit"] is not False
        or marker["formal_split_semantics"] is not False
        or marker["source_candidate_committable"] is not False
        or marker["formal_authorizations"] != policy["formal_authorizations"]
        or marker["transaction_policy_version"] != policy["version"]
        or marker["transaction_policy_canonical_self_hash"]
        != policy["canonical_self_hash"]
        or marker["candidate_selection_policy_raw_sha256"]
        != EXPECTED_CANDIDATE_POLICY_RAW_SHA256
        or marker["candidate_selection_policy_canonical_self_hash"]
        != EXPECTED_CANDIDATE_POLICY_SELF_HASH
        or marker["candidate_selection_source_sha256"]
        != EXPECTED_CANDIDATE_SOURCE_SHA256
        or marker["workspace_boundary_sha256"] != _sha256_bytes(boundary_bytes)
        or marker["mode"] != ALLOWED_MODE
        or marker["split"] != ALLOWED_SPLIT
        or expected_world_count != EXPECTED_WORLD_COUNT
        or world_ordinal != 0
        or marker["predecessor_seal_pins"] != []
        or marker["previous_world_marker_raw_sha256"]
        != _transaction_genesis(
            policy=policy,
            workspace_boundary_sha256=_sha256_bytes(boundary_bytes),
        )
    ):
        raise RecoveryCorruption("World marker smoke boundary drift")
    _verify_self_hash(marker, label="world marker")


def _load_member_payloads(
    world_directory: Path,
    marker: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
    allow_cleanup_missing: bool,
) -> dict[str, bytes | None]:
    expected_names = policy["transaction_layout"]["world_members"]
    manifest = marker["member_manifest"]
    if not isinstance(manifest, list) or len(manifest) != len(WORLD_MEMBER_ROLES):
        raise RecoveryCorruption("World marker member manifest length drift")
    output: dict[str, bytes | None] = {}
    for position, role in enumerate(WORLD_MEMBER_ROLES):
        entry = manifest[position]
        _require_exact_keys(entry, {"role", "path", "size_bytes", "sha256"}, label="member")
        filename = str(expected_names[role])
        if entry["role"] != role or entry["path"] != filename:
            raise RecoveryCorruption("World marker member order/path drift")
        size = _require_plain_int(entry["size_bytes"], label="member size")
        digest = _require_sha256(entry["sha256"], label="member hash")
        path = world_directory / filename
        if not os.path.lexists(common.filesystem_path(path)):
            if allow_cleanup_missing:
                output[role] = None
                continue
            raise RecoveryCorruption(f"Committed member is missing: {filename}")
        payload = _read_plain_file(path, label=f"world member {filename}")
        if len(payload) != size or _sha256_bytes(payload) != digest:
            raise RecoveryCorruption(f"Committed member drift: {filename}")
        output[role] = payload
    return output


def _accepted_from_marker_and_members(
    marker: Mapping[str, Any], member_payloads: Mapping[str, bytes | None]
) -> selection.AcceptedDevelopmentCandidate:
    scalar = marker["accepted_candidate"]
    if not isinstance(scalar, Mapping):
        raise RecoveryCorruption("Accepted candidate marker projection is not an object")
    byte_mapping = {
        "rejection_counts_bytes": "rejection_counts",
        "identity33_bytes": "identity33",
        "profile_provenance_bytes": "profile_provenance",
        "world_bytes": "private_world",
        "redacted_items_bytes": "redacted_items",
        "profiles_bytes": "seller_profiles",
        "item_hash_rows_bytes": "item_document_hashes",
        "seller_hash_rows_bytes": "seller_document_hashes",
        "selection_context_bytes": "selection_context",
        "exact_title_clone_qualification_bytes": "exact_title_clone_qualification",
    }
    expected_scalar_fields = {
        field.name for field in fields(selection.AcceptedDevelopmentCandidate)
    } - set(byte_mapping)
    _require_exact_keys(scalar, expected_scalar_fields, label="accepted scalar projection")
    kwargs = dict(scalar)
    for name in ("item_registry_delta", "seller_registry_delta", "allocation_delta"):
        value = kwargs[name]
        if not isinstance(value, list):
            raise RecoveryCorruption(f"Accepted {name} is not a list")
        kwargs[name] = tuple(value)
    for field_name, role in byte_mapping.items():
        payload = member_payloads.get(role)
        if payload is None:
            raise RecoveryCorruption("Cannot reconstruct accepted candidate after cleanup")
        kwargs[field_name] = payload
    try:
        return selection.AcceptedDevelopmentCandidate(**kwargs)
    except TypeError as exc:
        raise RecoveryCorruption("Accepted candidate reconstruction failed") from exc


def _expected_smoke_projection(
    accepted: selection.AcceptedDevelopmentCandidate,
    *,
    candidate_policy: Mapping[str, Any],
) -> dict[str, Any]:
    expected = candidate_policy["expected_smoke_selection"]
    return {
        "world_uid": accepted.world_uid,
        "accepted_candidate_index": accepted.candidate_index,
        "candidates_examined": accepted.candidates_examined,
        "rejected_candidate_count": accepted.rejected_candidate_count,
        "item_count": accepted.item_registry_delta_count,
        "seller_count": accepted.seller_registry_delta_count,
        "selection_context_sha256": accepted.selection_context_sha256,
        "exact_title_clone_qualification_sha256": (
            accepted.exact_title_clone_qualification_sha256
        ),
        "redacted_items_sha256": accepted.redacted_items_sha256,
        "profiles_sha256": accepted.profiles_sha256,
        "item_registry_delta_sha256": accepted.item_registry_delta_sha256,
        "seller_registry_delta_sha256": accepted.seller_registry_delta_sha256,
        "allocation_delta_count": accepted.allocation_delta_count,
        "allocation_delta_sha256": accepted.allocation_delta_sha256,
        "accepted_state_sha256": accepted.accepted_state_sha256,
    }


def _validate_persisted_smoke_candidate(
    accepted: selection.AcceptedDevelopmentCandidate,
) -> None:
    """Validate persisted candidate bytes without constructing a selector."""

    candidate_policy = _verify_frozen_candidate_sources()
    if (
        not isinstance(accepted, selection.AcceptedDevelopmentCandidate)
        or accepted.version
        != "2026-08-10-step28-v13-v1-13-accepted-development-candidate-v1"
        or accepted.mode != ALLOWED_MODE
        or accepted.split != ALLOWED_SPLIT
        or accepted.design_smoke_only is not True
        or accepted.committable is not False
        or accepted.candidate_index != 0
        or accepted.candidates_examined != 1
        or accepted.rejected_candidate_count != 0
    ):
        raise RecoveryCorruption("Persisted candidate envelope drift")
    pairs = (
        (accepted.rejection_counts_bytes, accepted.rejection_counts_sha256),
        (accepted.identity33_bytes, accepted.identity33_sha256),
        (accepted.profile_provenance_bytes, accepted.profile_provenance_sha256),
        (accepted.world_bytes, accepted.world_sha256),
        (accepted.redacted_items_bytes, accepted.redacted_items_sha256),
        (accepted.profiles_bytes, accepted.profiles_sha256),
        (accepted.item_hash_rows_bytes, accepted.item_hash_rows_sha256),
        (accepted.seller_hash_rows_bytes, accepted.seller_hash_rows_sha256),
        (accepted.selection_context_bytes, accepted.selection_context_sha256),
        (
            accepted.exact_title_clone_qualification_bytes,
            accepted.exact_title_clone_qualification_sha256,
        ),
    )
    for payload, digest in pairs:
        _decode_canonical(payload, label="persisted accepted member")
        if _sha256_bytes(payload) != _require_sha256(digest, label="member hash"):
            raise RecoveryCorruption("Persisted accepted member hash drift")
    replay_candidate = natural.AssembledDevelopmentCandidate(
        candidate_index=accepted.candidate_index,
        world_bytes=accepted.world_bytes,
        world_sha256=accepted.world_sha256,
        profiles_bytes=accepted.profiles_bytes,
        profiles_sha256=accepted.profiles_sha256,
        profile_provenance_bytes=accepted.profile_provenance_bytes,
        profile_provenance_sha256=accepted.profile_provenance_sha256,
        identity33_bytes=accepted.identity33_bytes,
        identity33_sha256=accepted.identity33_sha256,
        natural_output_sha256=accepted.natural_output_sha256,
        candidate_invariant_sha256=accepted.candidate_invariant_sha256,
        identity_parent_sha256=accepted.identity_parent_sha256,
    )
    replayed = selection._replay_final_document_observation(replay_candidate)
    for name in (
        "redacted_items_bytes",
        "redacted_items_sha256",
        "profiles_bytes",
        "profiles_sha256",
        "item_hash_rows_bytes",
        "item_hash_rows_sha256",
        "seller_hash_rows_bytes",
        "seller_hash_rows_sha256",
    ):
        if getattr(replayed, name) != getattr(accepted, name):
            raise RecoveryCorruption("Persisted candidate disagrees with production replay")
    item_rows = selection._hash_rows(
        accepted.item_hash_rows_bytes,
        fields=selection.HASH_ROW_FIELDS,
        label="persisted item hash rows",
    )
    seller_rows = selection._hash_rows(
        accepted.seller_hash_rows_bytes,
        fields=selection.SELLER_HASH_ROW_FIELDS,
        label="persisted seller hash rows",
    )
    item_delta = tuple(sorted(row["document_sha256"] for row in item_rows))
    seller_delta = tuple(sorted(row["document_sha256"] for row in seller_rows))
    if (
        item_delta != accepted.item_registry_delta
        or seller_delta != accepted.seller_registry_delta
        or accepted.item_registry_delta_count != len(item_delta)
        or accepted.seller_registry_delta_count != len(seller_delta)
        or accepted.item_registry_delta_sha256
        != common.canonical_sha256(list(item_delta))
        or accepted.seller_registry_delta_sha256
        != common.canonical_sha256(list(seller_delta))
        or accepted.allocation_delta_count != len(accepted.allocation_delta)
        or accepted.allocation_delta != tuple(sorted(set(accepted.allocation_delta)))
        or accepted.allocation_delta_sha256
        != common.canonical_sha256(list(accepted.allocation_delta))
    ):
        raise RecoveryCorruption("Persisted candidate registry delta drift")
    world = replay_candidate.thaw_world()
    observed_identity_hashes = tuple(
        sorted(
            identity_values.value_hash(str(row["identity_value"]))
            for row in world["private"]["identity_assets"]
        )
    )
    if observed_identity_hashes != accepted.allocation_delta:
        raise RecoveryCorruption("Allocation delta cannot reconstruct identity exclusions")
    if (
        common.canonical_sha256(selection._accepted_state_projection(accepted))
        != accepted.accepted_state_sha256
    ):
        raise RecoveryCorruption("Persisted accepted-state projection drift")
    actual = _expected_smoke_projection(
        accepted, candidate_policy=candidate_policy
    )
    if actual != candidate_policy["expected_smoke_selection"]:
        raise RecoveryCorruption("Persisted candidate differs from frozen smoke golden")


def _fixed_replay_candidate_zero(
    token: _SplitLockToken,
) -> selection.AcceptedDevelopmentCandidate:
    """Independently render frozen candidate zero; never search another candidate."""

    _assert_lock(token)
    candidate_policy = _verify_frozen_candidate_sources()
    expected = candidate_policy["expected_smoke_selection"]
    if (
        expected["accepted_candidate_index"] != 0
        or expected["candidates_examined"] != 1
        or expected["rejected_candidate_count"] != 0
    ):
        raise RecoveryCorruption("Frozen replay is not pinned to candidate zero")
    session = natural.DevelopmentSmokeVariationSession()
    material = session.trusted_selection_material()
    rendered = session.render(0)
    observation = selection._replay_final_document_observation(rendered)
    context = selection._build_smoke_collision_context()
    classification = selection._validate_collision_classification(
        selection._classify_document_collisions(
            item_hash_rows_bytes=observation.item_hash_rows_bytes,
            seller_hash_rows_bytes=observation.seller_hash_rows_bytes,
            context=context,
        )
    )
    if classification.has_collision:
        raise RecoveryCorruption("Frozen candidate zero no longer has zero collisions")
    accepted = selection._build_accepted_candidate(
        candidate=rendered,
        observation=observation,
        material=material,
        context=context,
        rejection_counts={name: 0 for name in selection.COLLISION_CATEGORIES},
    )
    _validate_persisted_smoke_candidate(accepted)
    return accepted


def _commit_world(
    token: _SplitLockToken,
    root: Path,
    accepted: selection.AcceptedDevelopmentCandidate,
    *,
    policy: Mapping[str, Any],
    boundary_bytes: bytes,
) -> tuple[dict[str, Any], bytes]:
    _assert_lock(token)
    _validate_persisted_smoke_candidate(accepted)
    transactions = root / TRANSACTIONS_DIRECTORY_NAME
    _ensure_plain_directory(
        token, transactions, parent=root, label="smoke transactions directory"
    )
    world_directory = transactions / WORLD_DIRECTORY_NAME
    if os.path.lexists(common.filesystem_path(world_directory)):
        raise PublishConflict("World transaction directory already exists")
    os.mkdir(common.filesystem_path(world_directory))
    _fsync_directory(transactions)
    payloads = _world_member_payloads(accepted, policy=policy)
    marker = _marker_value(accepted, policy=policy, boundary_bytes=boundary_bytes)
    marker_bytes = _canonical_bytes(marker)
    for role in WORLD_MEMBER_ROLES:
        filename = str(policy["transaction_layout"]["world_members"][role])
        _publish_no_replace(token, world_directory / filename, payloads[filename])
    _fsync_directory(world_directory)
    _publish_no_replace(token, world_directory / WORLD_MARKER_NAME, marker_bytes)
    return marker, marker_bytes


def _allowed_uncommitted_names(policy: Mapping[str, Any]) -> set[str]:
    names = set(policy["transaction_layout"]["world_members"].values())
    names.add(WORLD_MARKER_NAME)
    names.update(f".{name}.pending" for name in tuple(names))
    return names


def _delete_safe_uncommitted_world(
    token: _SplitLockToken,
    world_directory: Path,
    *,
    transactions: Path,
    policy: Mapping[str, Any],
) -> None:
    _assert_lock(token)
    if world_directory.parent.resolve() != transactions.resolve():
        raise RecoveryCorruption("Uncommitted directory is outside transaction root")
    if world_directory.name != WORLD_DIRECTORY_NAME:
        raise RecoveryCorruption("Uncommitted directory name drift")
    _require_plain_directory(world_directory, label="uncommitted world directory")
    allowed = _allowed_uncommitted_names(policy)
    entries = sorted(
        os.scandir(common.filesystem_path(world_directory)),
        key=lambda entry: entry.name.encode("utf-8"),
    )
    for entry in entries:
        if entry.name not in allowed:
            raise RecoveryCorruption("Unknown entry blocks safe transaction cleanup")
        path = world_directory / entry.name
        _unlink_known_plain_file(
            token, path, label="uncommitted transaction member"
        )
    os.rmdir(common.filesystem_path(world_directory))
    _fsync_directory(transactions)


def verify_committed_world_pre_seal(
    token: _SplitLockToken,
    root: Path,
    *,
    policy: Mapping[str, Any],
    boundary_bytes: bytes,
) -> tuple[selection.AcceptedDevelopmentCandidate, dict[str, Any], bytes]:
    """Require every committed member and replay it before a valid seal exists."""

    _assert_lock(token)
    seal_path = root / FINAL_DIRECTORY_NAME / SPLIT_SEAL_NAME
    if os.path.lexists(common.filesystem_path(seal_path)):
        raise RecoveryCorruption("Pre-seal verifier cannot run after seal publication")
    world_directory = root / TRANSACTIONS_DIRECTORY_NAME / WORLD_DIRECTORY_NAME
    _require_plain_directory(world_directory, label="committed world directory")
    _validate_world_directory_entries(
        world_directory, policy=policy, post_seal=False
    )
    marker_bytes = _read_plain_file(
        world_directory / WORLD_MARKER_NAME, label="world accepted marker"
    )
    marker = _decode_canonical(marker_bytes, label="world accepted marker")
    if not isinstance(marker, dict):
        raise RecoveryCorruption("World marker is not an object")
    _validate_marker_envelope(marker, policy=policy, boundary_bytes=boundary_bytes)
    members = _load_member_payloads(
        world_directory, marker, policy=policy, allow_cleanup_missing=False
    )
    accepted = _accepted_from_marker_and_members(marker, members)
    _validate_persisted_smoke_candidate(accepted)
    expected_marker = _marker_value(
        accepted, policy=policy, boundary_bytes=boundary_bytes
    )
    if marker_bytes != _canonical_bytes(expected_marker):
        raise RecoveryCorruption("World marker differs from reconstructed candidate")
    return accepted, marker, marker_bytes


def _scan_pre_seal_state(
    token: _SplitLockToken,
    root: Path,
    *,
    policy: Mapping[str, Any],
    boundary_bytes: bytes,
) -> tuple[selection.AcceptedDevelopmentCandidate, dict[str, Any], bytes] | None:
    _assert_lock(token)
    transactions = root / TRANSACTIONS_DIRECTORY_NAME
    _ensure_plain_directory(
        token, transactions, parent=root, label="smoke transactions directory"
    )
    entries = sorted(
        os.scandir(common.filesystem_path(transactions)),
        key=lambda entry: entry.name.encode("utf-8"),
    )
    if [entry.name for entry in entries] not in ([], [WORLD_DIRECTORY_NAME]):
        raise RecoveryCorruption("Unexpected or discontinuous world transaction")
    for entry in entries:
        world_directory = transactions / entry.name
        _require_plain_directory(world_directory, label="world transaction")
        marker_path = world_directory / WORLD_MARKER_NAME
        if not os.path.lexists(common.filesystem_path(marker_path)):
            _delete_safe_uncommitted_world(
                token,
                world_directory,
                transactions=transactions,
                policy=policy,
            )
            return None
        return verify_committed_world_pre_seal(
            token,
            root,
            policy=policy,
            boundary_bytes=boundary_bytes,
        )
    return None


def _utf8_sort_rows(
    rows: Sequence[Mapping[str, Any]], *, fields: Sequence[str], label: str
) -> list[dict[str, Any]]:
    if not fields or len(fields) != len(set(fields)):
        raise SplitTransactionError(f"{label} sort fields drift")
    output = [dict(row) for row in rows]
    if any(any(field not in row for field in fields) for row in output):
        raise SplitTransactionError(f"{label} row lacks a sort field")
    output.sort(
        key=lambda row: tuple(str(row[field]).encode("utf-8") for field in fields)
    )
    return output


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_bytes(dict(row)) + b"\n" for row in rows)


def _commitment(path: str, payload: bytes) -> dict[str, Any]:
    if Path(path).as_posix() != path or len(Path(path).parts) != 1:
        raise SplitTransactionError("Final member path is not canonical")
    return {"path": path, "size_bytes": len(payload), "sha256": _sha256_bytes(payload)}


def _final_member_payloads(
    accepted: selection.AcceptedDevelopmentCandidate,
    marker: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
) -> dict[str, bytes]:
    world = _decode_canonical(accepted.world_bytes, label="private world")
    if not isinstance(world, dict) or set(world) != {"private", "public"}:
        raise RecoveryCorruption("Private world public/private envelope drift")
    public = world["public"]
    if not isinstance(public, dict) or "world" not in public:
        raise RecoveryCorruption("Private world lacks public world projection")
    source_public_world = public["world"]
    if not isinstance(source_public_world, dict) or set(source_public_world) != set(
        PUBLIC_WORLD_FIELDS
    ):
        raise RecoveryCorruption("Public world projection field set drift")
    public_world = {
        field: source_public_world[field] for field in PUBLIC_WORLD_FIELDS
    }
    if public_world["world_uid"] != accepted.world_uid:
        raise RecoveryCorruption("Public world UID differs from accepted candidate")
    redacted = _decode_canonical(accepted.redacted_items_bytes, label="redacted items")
    profiles = _decode_canonical(accepted.profiles_bytes, label="seller profiles")
    identity33 = _decode_canonical(accepted.identity33_bytes, label="identity33")
    rejection_counts = _decode_canonical(
        accepted.rejection_counts_bytes, label="rejection counts"
    )
    if (
        not isinstance(redacted, list)
        or not isinstance(profiles, list)
        or not isinstance(identity33, list)
        or not isinstance(rejection_counts, dict)
    ):
        raise RecoveryCorruption("Smoke final projection sources have wrong types")
    attempts = [
        {
            "world_ordinal": 0,
            "world_uid": accepted.world_uid,
            "accepted_candidate_index": accepted.candidate_index,
            "candidates_examined": accepted.candidates_examined,
            "rejected_candidate_count": accepted.rejected_candidate_count,
            "rejection_counts": rejection_counts,
            "candidate_parent_full_state_sha256": (
                accepted.candidate_parent_full_state_sha256
            ),
            "frozen_trial_identity_full_state_sha256": (
                accepted.frozen_trial_identity_full_state_sha256
            ),
            "candidate_invariant_sha256": accepted.candidate_invariant_sha256,
            "accepted_state_sha256": accepted.accepted_state_sha256,
        }
    ]
    output: dict[str, bytes] = {
        "worlds.jsonl": _jsonl_bytes([public_world]),
        "redacted_items.jsonl": _jsonl_bytes(
            _utf8_sort_rows(
                redacted,
                fields=("world_uid", "seller_uid", "item_uid"),
                label="redacted items",
            )
        ),
        "seller_profiles.jsonl": _jsonl_bytes(
            _utf8_sort_rows(profiles, fields=("seller_uid",), label="seller profiles")
        ),
        "identity33.jsonl": _jsonl_bytes(
            _utf8_sort_rows(
                identity33,
                fields=("world_uid", "canonical_pair_uid"),
                label="identity33",
            )
        ),
        "document_collision_attempts.jsonl": _jsonl_bytes(attempts),
    }
    registry = _with_self_hash(
        {
            "artifact_type": "step28_v13_v1_13_development_smoke_document_collision_registry",
            "schema_version": "2026-08-10-v1",
            "status": "SMOKE_PROJECTION_ONLY_NOT_FORMAL",
            "design_smoke_only": True,
            "formal_split_semantics": False,
            "formal_canonical_member_set_claimed": False,
            "transaction_policy_canonical_self_hash": policy["canonical_self_hash"],
            "mode": ALLOWED_MODE,
            "split": ALLOWED_SPLIT,
            "expected_world_count": EXPECTED_WORLD_COUNT,
            "item_document_hashes": list(accepted.item_registry_delta),
            "item_document_count": accepted.item_registry_delta_count,
            "item_document_hashes_sha256": accepted.item_registry_delta_sha256,
            "seller_document_hashes": list(accepted.seller_registry_delta),
            "seller_document_count": accepted.seller_registry_delta_count,
            "seller_document_hashes_sha256": accepted.seller_registry_delta_sha256,
            "identity_value_hashes": list(accepted.allocation_delta),
            "identity_value_hash_count": accepted.allocation_delta_count,
            "identity_value_hashes_sha256": accepted.allocation_delta_sha256,
            "predecessor_seal_pins": [],
            "source_world_marker_canonical_self_hash": marker["canonical_self_hash"],
        }
    )
    output["document_collision_registry.json"] = _canonical_bytes(registry)
    manifest_members = [
        _commitment(name, output[name])
        for name in FINAL_MEMBER_NAMES
        if name != "split_manifest.json"
    ]
    manifest = _with_self_hash(
        {
            "artifact_type": "step28_v13_v1_13_development_smoke_split_manifest",
            "schema_version": "2026-08-10-v1",
            "status": "SMOKE_FINAL_PROJECTION_ONLY_NOT_FORMAL",
            "design_smoke_only": True,
            "formal_split_semantics": False,
            "formal_canonical_member_set_claimed": False,
            "transaction_policy_version": policy["version"],
            "transaction_policy_canonical_self_hash": policy["canonical_self_hash"],
            "formal_authorizations": dict(policy["formal_authorizations"]),
            "mode": ALLOWED_MODE,
            "split": ALLOWED_SPLIT,
            "expected_world_count": EXPECTED_WORLD_COUNT,
            "source_world_marker_canonical_self_hash": marker["canonical_self_hash"],
            "members": manifest_members,
        }
    )
    output["split_manifest.json"] = _canonical_bytes(manifest)
    if tuple(output) != FINAL_MEMBER_NAMES:
        raise SplitTransactionError("Final member construction order drift")
    return output


def _seal_value(
    *,
    marker: Mapping[str, Any],
    marker_bytes: bytes,
    final_payloads: Mapping[str, bytes],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    final_manifest = [_commitment(name, final_payloads[name]) for name in FINAL_MEMBER_NAMES]
    cleanup_plan = [dict(entry) for entry in marker["member_manifest"]]
    return _with_self_hash(
        {
            "artifact_type": "step28_v13_v1_13_development_smoke_split_seal_complete",
            "schema_version": "2026-08-10-v1",
            "status": "DEVELOPMENT_SMOKE_SEAL_COMPLETE_NOT_FORMAL",
            "design_smoke_only": True,
            "formal_split_semantics": False,
            "formal_canonical_member_set_claimed": False,
            "source_candidate_committable": False,
            "formal_authorizations": dict(policy["formal_authorizations"]),
            "transaction_policy_version": policy["version"],
            "transaction_policy_canonical_self_hash": policy["canonical_self_hash"],
            "candidate_selection_policy_raw_sha256": (
                EXPECTED_CANDIDATE_POLICY_RAW_SHA256
            ),
            "candidate_selection_policy_canonical_self_hash": (
                EXPECTED_CANDIDATE_POLICY_SELF_HASH
            ),
            "candidate_selection_source_sha256": EXPECTED_CANDIDATE_SOURCE_SHA256,
            "mode": ALLOWED_MODE,
            "split": ALLOWED_SPLIT,
            "expected_world_count": EXPECTED_WORLD_COUNT,
            "world_marker_count": 1,
            "ordered_world_marker_raw_sha256": [_sha256_bytes(marker_bytes)],
            "ordered_world_marker_raw_sha256_root": common.canonical_sha256(
                [_sha256_bytes(marker_bytes)]
            ),
            "last_world_marker_raw_sha256": _sha256_bytes(marker_bytes),
            "last_world_marker_canonical_self_hash": marker["canonical_self_hash"],
            "predecessor_seal_pins": [],
            "final_members": final_manifest,
            "document_collision_registry_sha256": _sha256_bytes(
                final_payloads["document_collision_registry.json"]
            ),
            "split_manifest_sha256": _sha256_bytes(
                final_payloads["split_manifest.json"]
            ),
            "cleanup_eligible_world_members": cleanup_plan,
            "cleanup_receipt_name": CLEANUP_RECEIPT_NAME,
        }
    )


def _cleanup_receipt_value(
    *,
    marker: Mapping[str, Any],
    marker_bytes: bytes,
    seal_bytes: bytes,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    return _with_self_hash(
        {
            "artifact_type": "step28_v13_v1_13_development_smoke_world_cleanup_complete",
            "schema_version": "2026-08-10-v1",
            "status": "DEVELOPMENT_SMOKE_WORLD_CLEANUP_COMPLETE_NOT_FORMAL",
            "design_smoke_only": True,
            "formal_split_semantics": False,
            "formal_authorizations": dict(policy["formal_authorizations"]),
            "transaction_policy_version": policy["version"],
            "transaction_policy_canonical_self_hash": policy["canonical_self_hash"],
            "world_ordinal": 0,
            "world_uid": marker["world_uid"],
            "world_marker_name": WORLD_MARKER_NAME,
            "world_marker_raw_sha256": _sha256_bytes(marker_bytes),
            "world_marker_canonical_self_hash": marker["canonical_self_hash"],
            "split_seal_name": SPLIT_SEAL_NAME,
            "split_seal_raw_sha256": _sha256_bytes(seal_bytes),
            "deleted_members": [dict(entry) for entry in marker["member_manifest"]],
            "retained_members": [WORLD_MARKER_NAME, CLEANUP_RECEIPT_NAME],
        }
    )


@dataclass(frozen=True)
class _FrozenExpectedState:
    accepted: selection.AcceptedDevelopmentCandidate
    marker: dict[str, Any]
    marker_bytes: bytes
    final_payloads: dict[str, bytes]
    seal: dict[str, Any]
    seal_bytes: bytes
    cleanup_receipt: dict[str, Any]
    cleanup_receipt_bytes: bytes


def _expected_state_from_accepted(
    accepted: selection.AcceptedDevelopmentCandidate,
    *,
    policy: Mapping[str, Any],
    boundary_bytes: bytes,
) -> _FrozenExpectedState:
    _validate_persisted_smoke_candidate(accepted)
    marker = _marker_value(accepted, policy=policy, boundary_bytes=boundary_bytes)
    marker_bytes = _canonical_bytes(marker)
    final_payloads = _final_member_payloads(
        accepted, marker, policy=policy
    )
    seal = _seal_value(
        marker=marker,
        marker_bytes=marker_bytes,
        final_payloads=final_payloads,
        policy=policy,
    )
    seal_bytes = _canonical_bytes(seal)
    cleanup = _cleanup_receipt_value(
        marker=marker,
        marker_bytes=marker_bytes,
        seal_bytes=seal_bytes,
        policy=policy,
    )
    return _FrozenExpectedState(
        accepted=accepted,
        marker=marker,
        marker_bytes=marker_bytes,
        final_payloads=final_payloads,
        seal=seal,
        seal_bytes=seal_bytes,
        cleanup_receipt=cleanup,
        cleanup_receipt_bytes=_canonical_bytes(cleanup),
    )


def _publish_final_and_seal(
    token: _SplitLockToken,
    root: Path,
    state: _FrozenExpectedState,
) -> None:
    _assert_lock(token)
    final_directory = root / FINAL_DIRECTORY_NAME
    _ensure_plain_directory(
        token, final_directory, parent=root, label="smoke final projection directory"
    )
    _validate_final_directory_entries(final_directory, post_seal=False)
    _remove_safe_pre_seal_pending_files(token, final_directory)
    _validate_final_directory_entries(final_directory, post_seal=False)
    seal_path = final_directory / SPLIT_SEAL_NAME
    if os.path.lexists(common.filesystem_path(seal_path)):
        raise RecoveryCorruption("Pre-seal publisher received an existing seal")
    for name in FINAL_MEMBER_NAMES:
        _publish_or_verify(token, final_directory / name, state.final_payloads[name])
    _fsync_directory(final_directory)
    _publish_no_replace(token, seal_path, state.seal_bytes)


def _verify_final_and_seal(
    root: Path, state: _FrozenExpectedState
) -> None:
    final_directory = root / FINAL_DIRECTORY_NAME
    _require_plain_directory(final_directory, label="smoke final projection directory")
    _validate_final_directory_entries(final_directory, post_seal=True)
    for name in FINAL_MEMBER_NAMES:
        observed = _read_plain_file(
            final_directory / name, label=f"sealed final member {name}"
        )
        if observed != state.final_payloads[name]:
            raise RecoveryCorruption(
                f"Frozen replay expected final bytes differ: {name}"
            )
    observed_seal = _read_plain_file(
        final_directory / SPLIT_SEAL_NAME, label="smoke split seal"
    )
    if observed_seal != state.seal_bytes:
        raise RecoveryCorruption("Frozen replay expected seal bytes differ")


def verify_frozen_smoke_post_seal_projection(
    token: _SplitLockToken,
    root: Path,
    *,
    policy: Mapping[str, Any],
    boundary_bytes: bytes,
) -> _FrozenExpectedState:
    """Derive post-seal truth only from pinned sources and fixed render(0)."""

    _assert_lock(token)
    accepted = _fixed_replay_candidate_zero(token)
    state = _expected_state_from_accepted(
        accepted, policy=policy, boundary_bytes=boundary_bytes
    )
    world_directory = root / TRANSACTIONS_DIRECTORY_NAME / WORLD_DIRECTORY_NAME
    marker_path = world_directory / WORLD_MARKER_NAME
    observed_marker = _read_plain_file(marker_path, label="sealed world marker")
    if observed_marker != state.marker_bytes:
        raise RecoveryCorruption("Frozen replay expected marker bytes differ")
    _verify_final_and_seal(root, state)
    return state


def _finish_cleanup(
    token: _SplitLockToken,
    root: Path,
    state: _FrozenExpectedState,
) -> None:
    _assert_lock(token)
    world_directory = root / TRANSACTIONS_DIRECTORY_NAME / WORLD_DIRECTORY_NAME
    _require_plain_directory(world_directory, label="sealed world directory")
    receipt_path = world_directory / CLEANUP_RECEIPT_NAME
    receipt_pending = world_directory / f".{CLEANUP_RECEIPT_NAME}.pending"
    receipt_exists = os.path.lexists(common.filesystem_path(receipt_path))
    _validate_world_directory_entries(
        world_directory,
        policy=load_policy(),
        post_seal=True,
        allow_cleanup_receipt_pending=not receipt_exists,
    )
    if not receipt_exists and os.path.lexists(common.filesystem_path(receipt_pending)):
        _unlink_known_plain_file(
            token, receipt_pending, label="stale cleanup-receipt pending file"
        )
        _fsync_directory(world_directory)
    if receipt_exists:
        observed = _read_plain_file(receipt_path, label="world cleanup receipt")
        if observed != state.cleanup_receipt_bytes:
            raise RecoveryCorruption("Frozen replay expected cleanup receipt differs")
        for entry in state.marker["member_manifest"]:
            if os.path.lexists(
                common.filesystem_path(world_directory / entry["path"])
            ):
                raise RecoveryCorruption(
                    "Cleanup-complete world member reappeared after receipt"
                )
    else:
        for entry in state.marker["member_manifest"]:
            path = world_directory / entry["path"]
            if os.path.lexists(common.filesystem_path(path)):
                observed = _read_plain_file(
                    path, label="cleanup-eligible world member"
                )
                if (
                    len(observed) != entry["size_bytes"]
                    or _sha256_bytes(observed) != entry["sha256"]
                ):
                    raise RecoveryCorruption("Cleanup-eligible member drift")
                _unlink_known_plain_file(
                    token, path, label="cleanup-eligible world member"
                )
        _fsync_directory(world_directory)
        _publish_no_replace(token, receipt_path, state.cleanup_receipt_bytes)
    for entry in state.marker["member_manifest"]:
        if os.path.lexists(common.filesystem_path(world_directory / entry["path"])):
            raise RecoveryCorruption("Cleanup receipt exists before member deletion")
    retained = {
        WORLD_MARKER_NAME,
        CLEANUP_RECEIPT_NAME,
    }
    observed_names = {
        entry.name for entry in os.scandir(common.filesystem_path(world_directory))
    }
    if observed_names != retained:
        raise RecoveryCorruption("Sealed world retained-member set drift")


def verify_sealed_world_post_seal(
    token: _SplitLockToken,
    root: Path,
    *,
    policy: Mapping[str, Any],
    boundary_bytes: bytes,
) -> _FrozenExpectedState:
    """Verify sealed state without requiring cleanup-eligible members."""

    _assert_lock(token)
    _validate_workspace_entries(root)
    state = verify_frozen_smoke_post_seal_projection(
        token,
        root,
        policy=policy,
        boundary_bytes=boundary_bytes,
    )
    world_directory = root / TRANSACTIONS_DIRECTORY_NAME / WORLD_DIRECTORY_NAME
    members = _load_member_payloads(
        world_directory,
        state.marker,
        policy=policy,
        allow_cleanup_missing=True,
    )
    for entry in state.marker["member_manifest"]:
        role = entry["role"]
        payload = members[role]
        if payload is not None and (
            len(payload) != entry["size_bytes"]
            or _sha256_bytes(payload) != entry["sha256"]
        ):
            raise RecoveryCorruption("Surviving cleanup member drift")
    _finish_cleanup(token, root, state)
    return state


def _run_locked_workspace(
    token: _SplitLockToken,
    root: Path,
    *,
    create_workspace: bool,
) -> dict[str, Any]:
    _assert_lock(token)
    policy = load_policy()
    _verify_frozen_candidate_sources()
    _boundary, boundary_bytes = _initialize_workspace(
        token, root, create=create_workspace
    )
    _validate_workspace_entries(root)
    final_directory = root / FINAL_DIRECTORY_NAME
    seal_path = final_directory / SPLIT_SEAL_NAME
    if os.path.lexists(common.filesystem_path(seal_path)):
        state = verify_sealed_world_post_seal(
            token,
            root,
            policy=policy,
            boundary_bytes=boundary_bytes,
        )
        recovery_state = "SEALED_CLEANUP_COMPLETE"
    else:
        recovered = _scan_pre_seal_state(
            token,
            root,
            policy=policy,
            boundary_bytes=boundary_bytes,
        )
        if recovered is None:
            selector = selection.DevelopmentSmokeCandidateSelector()
            accepted = selector.select()
            selector.validate_completed_candidate(accepted)
            marker, marker_bytes = _commit_world(
                token,
                root,
                accepted,
                policy=policy,
                boundary_bytes=boundary_bytes,
            )
        else:
            accepted, marker, marker_bytes = recovered
        state = _expected_state_from_accepted(
            accepted,
            policy=policy,
            boundary_bytes=boundary_bytes,
        )
        if marker != state.marker or marker_bytes != state.marker_bytes:
            raise RecoveryCorruption("Pre-seal marker differs from expected state")
        _publish_final_and_seal(token, root, state)
        # The post-seal trust transition must use a fresh fixed render(0), not
        # the accepted object that just produced the seal.
        state = verify_sealed_world_post_seal(
            token,
            root,
            policy=policy,
            boundary_bytes=boundary_bytes,
        )
        recovery_state = "SEALED_CLEANUP_COMPLETE"
    return {
        "status": "PASS_DEVELOPMENT_SMOKE_TRANSACTION_STATE_MACHINE_ONLY",
        "design_smoke_only": True,
        "formal_split_semantics": False,
        "formal_canonical_member_set_claimed": False,
        "formal_authorizations": dict(policy["formal_authorizations"]),
        "formal_seeds_generated": 0,
        "formal_candidates_generated": 0,
        "formal_rows_generated": 0,
        "formal_models_trained": 0,
        "formal_metrics_generated": 0,
        "mode": ALLOWED_MODE,
        "split": ALLOWED_SPLIT,
        "expected_world_count": EXPECTED_WORLD_COUNT,
        "source_candidate_committable": state.accepted.committable,
        "world_uid": state.accepted.world_uid,
        "accepted_candidate_index": state.accepted.candidate_index,
        "accepted_state_sha256": state.accepted.accepted_state_sha256,
        "world_marker_raw_sha256": _sha256_bytes(state.marker_bytes),
        "smoke_split_seal_raw_sha256": _sha256_bytes(state.seal_bytes),
        "cleanup_receipt_raw_sha256": _sha256_bytes(
            state.cleanup_receipt_bytes
        ),
        "smoke_final_member_count": len(state.final_payloads),
        "recovery_state": recovery_state,
        "ephemeral_workspace_retained": False,
        "formal_500_world_authenticity_proven": False,
        "linux_formal_directory_durability_certified": False,
    }


def run_development_smoke() -> dict[str, Any]:
    """Run the sole public, zero-argument, ephemeral smoke transaction."""

    with _exclusive_split_lock() as token:
        # The workspace is intentionally created only after the stable split
        # lock has been acquired.
        with tempfile.TemporaryDirectory(prefix=WORKSPACE_PREFIX) as temporary:
            root = Path(temporary)
            return _run_locked_workspace(token, root, create_workspace=False)


def _synthetic_marker_bytes(
    *,
    world_ordinal: int,
    previous_raw_sha256: str,
    item_hashes: Sequence[str],
    seller_hashes: Sequence[str],
    identity_hashes: Sequence[str],
) -> bytes:
    """Build a hash-only fixture marker; never a scientific artifact."""

    ordinal = _require_plain_int(world_ordinal, label="synthetic world ordinal")
    if ordinal < 0:
        raise SplitTransactionError("Synthetic world ordinal cannot be negative")
    item_values = tuple(sorted(item_hashes))
    seller_values = tuple(sorted(seller_hashes))
    identity_values_hashes = tuple(sorted(identity_hashes))
    for label, values in (
        ("synthetic item", item_values),
        ("synthetic seller", seller_values),
        ("synthetic identity", identity_values_hashes),
    ):
        if len(values) != len(set(values)):
            raise SplitTransactionError(f"{label} hashes are duplicated")
        for value in values:
            _require_sha256(value, label=label)
    value = _with_self_hash(
        {
            "artifact_type": "step28_v13_v1_13_synthetic_marker_fixture",
            "schema_version": "2026-08-10-v1",
            "scientific_result": False,
            "world_ordinal": ordinal,
            "previous_world_marker_raw_sha256": _require_sha256(
                previous_raw_sha256, label="synthetic predecessor"
            ),
            "item_document_hashes": list(item_values),
            "seller_document_hashes": list(seller_values),
            "identity_value_hashes": list(identity_values_hashes),
        }
    )
    return _canonical_bytes(value)


def validate_synthetic_marker_chain(
    marker_bytes: Sequence[bytes], *, genesis_sha256: str
) -> dict[str, tuple[str, ...]]:
    """Exercise multi-world continuity and cumulative-set semantics only."""

    previous = _require_sha256(genesis_sha256, label="synthetic genesis")
    cumulative = {"item": set(), "seller": set(), "identity": set()}
    for expected_ordinal, payload in enumerate(marker_bytes):
        marker = _decode_canonical(payload, label="synthetic marker")
        required = {
            "artifact_type",
            "schema_version",
            "scientific_result",
            "world_ordinal",
            "previous_world_marker_raw_sha256",
            "item_document_hashes",
            "seller_document_hashes",
            "identity_value_hashes",
            "canonical_self_hash",
        }
        _require_exact_keys(marker, required, label="synthetic marker")
        _verify_self_hash(marker, label="synthetic marker")
        observed_ordinal = _require_plain_int(
            marker["world_ordinal"], label="synthetic marker world ordinal"
        )
        if (
            marker["artifact_type"]
            != "step28_v13_v1_13_synthetic_marker_fixture"
            or marker["schema_version"] != "2026-08-10-v1"
            or marker["scientific_result"] is not False
            or observed_ordinal != expected_ordinal
            or marker["previous_world_marker_raw_sha256"] != previous
        ):
            raise RecoveryCorruption("Synthetic marker chain is discontinuous")
        for group, field in (
            ("item", "item_document_hashes"),
            ("seller", "seller_document_hashes"),
            ("identity", "identity_value_hashes"),
        ):
            values = marker[field]
            if (
                not isinstance(values, list)
                or values != sorted(values)
                or len(values) != len(set(values))
            ):
                raise RecoveryCorruption("Synthetic registry delta drift")
            for value in values:
                _require_sha256(value, label=f"synthetic {group}")
            if cumulative[group].intersection(values):
                raise RecoveryCorruption("Synthetic current-split registry collision")
            cumulative[group].update(values)
        previous = _sha256_bytes(payload)
    return {key: tuple(sorted(values)) for key, values in cumulative.items()}


def synthetic_predecessor_fixture_pins(split: str) -> tuple[dict[str, str], ...]:
    """Derive independent hash-only predecessor truth for architecture tests."""

    if split not in FORMAL_SPLIT_ORDER:
        raise SplitTransactionError("Synthetic predecessor split is unknown")
    expected_splits = FORMAL_SPLIT_ORDER[: FORMAL_SPLIT_ORDER.index(split)]
    output = []
    for predecessor in expected_splits:
        def fixture_hash(role: str) -> str:
            return common.canonical_sha256(
                {
                    "domain": SYNTHETIC_PREDECESSOR_DOMAIN,
                    "scientific_result": False,
                    "predecessor_split": predecessor,
                    "role": role,
                }
            )

        output.append(
            {
                "split": predecessor,
                "split_seal_raw_sha256": fixture_hash("split_seal"),
                "document_registry_raw_sha256": fixture_hash(
                    "document_collision_registry"
                ),
            }
        )
    return tuple(output)


def validate_synthetic_predecessor_pins(
    *, split: str, pins: Sequence[Mapping[str, Any]]
) -> None:
    """Validate predecessor order and bytes against independent fixture truth."""

    expected = synthetic_predecessor_fixture_pins(split)
    if not isinstance(pins, Sequence) or isinstance(pins, (str, bytes, Mapping)):
        raise SplitTransactionError("Synthetic predecessor pins must be a sequence")
    for pin in pins:
        _require_exact_keys(
            pin,
            {"split", "split_seal_raw_sha256", "document_registry_raw_sha256"},
            label="synthetic predecessor pin",
        )
        _require_sha256(pin["split_seal_raw_sha256"], label="predecessor seal")
        _require_sha256(
            pin["document_registry_raw_sha256"], label="predecessor registry"
        )
    if _canonical_bytes(list(pins)) != _canonical_bytes(list(expected)):
        raise RecoveryCorruption("Synthetic predecessor fixture truth drift")


def main() -> None:
    if len(sys.argv) != 1:
        raise SplitTransactionError("The development-smoke runner accepts no arguments")
    summary = run_development_smoke()
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
