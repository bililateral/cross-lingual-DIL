#!/usr/bin/env python3
"""Build fixed, nonjoinable Step28-v13 development integrity receipts.

These helpers deliberately implement only the in-process development smoke
path.  They validate the same data relationships that future isolated
sealers must validate, but they do not claim OS custody, access-hook, or
external-parent evidence.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import step28_v13_candidate_sampling as candidate_sampling
import step28_v13_common as common
import step28_v13_structure as structure


DEPLOYMENT_VERSION = (
    "2026-07-28-step28-v13-dataset-custody-deployment-v7-draft"
)
SPLITS = ("train", "development", "audit_a", "audit_b")
HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
MULTISET_PREFIX = b"step28-v13-canonical-multiset-v1"
COMPONENT_NAMES = (
    "controller_membership",
    "controller_style_groups",
    "identity_asset_decisions",
    "mechanism_assignments",
    "negative_flags",
    "positive_targets",
    "registered_override_decisions",
    "repeat_decisions",
    "seller_markets",
    "solver_trace",
)
COMPARISON_RECEIPT_KEYS = {
    "version",
    "world_uid",
    "mode",
    "split",
    "scope",
    "evidence_level",
    "independent_typed_dgp_replay_pass",
    "independent_decision_implementation",
    "formal_custody_seal",
    "producer_private_input_used_by_replayer",
    "structure_key_serialized",
    "full_typed_projection_exact",
    "producer_typed_projection_sha256",
    "replayer_typed_projection_sha256",
    "observed_uid_pool_audit",
    "component_receipts",
}
OBSERVED_UID_AUDIT_KEYS = {
    "seller_uid_pool_sha256",
    "seller_count",
    "all_item_count",
    "nonempty_title_item_count",
    "nonempty_description_item_count",
    "all_item_uid_pool_sha256",
    "nonempty_title_item_uid_pool_sha256",
    "nonempty_description_item_uid_pool_sha256",
}
COMPONENT_RECEIPT_KEYS = {
    "producer_row_count",
    "replayer_row_count",
    "producer_sha256",
    "replayer_sha256",
    "exact_equal",
}
CANDIDATE_CONTEXT_VERSION = (
    "2026-07-28-step28-v13-candidate-sealer-context-v1-draft"
)
CANDIDATE_CONTEXT_KEYS = {
    "version",
    "parent_policy_version",
    "mode",
    "split",
    "registered_world_uids_utf8_sorted",
    "registered_split_scope_sha256",
    "policy_parent_sha256",
    "candidate_policy_projection_sha256",
    "receipt_top_level_exact_keys",
    "receipt_spec",
    "canonical_self_hash",
}


def canonical_multiset_sha256(
    rows: Sequence[Mapping[str, Any]],
) -> str:
    """Hash a duplicate-preserving, input-order-independent row multiset."""

    encoded_rows: list[bytes] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise common.ContractError(
                "Canonical multiset rows must be mappings"
            )
        encoded_rows.append(common.canonical_json_bytes(dict(row)))
    encoded_rows.sort()
    framed = bytearray(MULTISET_PREFIX)
    framed.append(0)
    framed.extend(len(encoded_rows).to_bytes(8, "big", signed=False))
    for encoded in encoded_rows:
        framed.extend(len(encoded).to_bytes(8, "big", signed=False))
        framed.extend(encoded)
    return hashlib.sha256(framed).hexdigest()


def pretty_json_sha256(value: Any) -> str:
    """Hash the exact bytes emitted by common.write_json."""

    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _deployment_path(policy: Mapping[str, Any]) -> Path:
    return common.repo_path(
        str(policy["security"]["dataset_custody_deployment"]["path"])
    )


def _load_deployment(policy: Mapping[str, Any]) -> dict[str, Any]:
    path = _deployment_path(policy)
    deployment = common.load_json(path)
    if (
        deployment.get("version") != DEPLOYMENT_VERSION
        or deployment.get("formal_generation_enabled") is not False
        or deployment.get("parent_policy", {}).get("path")
        != "schema/step28_v13_synthetic_chinese_dataset_policy.json"
    ):
        raise common.ContractError(
            "Step28-v13 integrity deployment contract drift"
        )
    return deployment


def validate_deployment_contract(
    policy: Mapping[str, Any],
) -> None:
    """Fail closed on the development receipt/capability contract."""

    deployment = _load_deployment(policy)
    schemas = deployment.get(
        "fixed_nonjoinable_aggregate_receipt_schemas"
    )
    if not isinstance(schemas, Mapping):
        raise common.ContractError(
            "Step28-v13 aggregate receipt schema registry is absent"
        )
    expected_roles = {
        "render_integrity",
        "candidate_integrity",
        "m1_derangement_integrity",
        "independent_dgp_comparison",
        "structural_audit",
    }
    reserved = {
        "common_top_level_exact_keys",
        "common_constraints",
        "canonical_multiset_hash_encoding",
    }
    if set(schemas).difference(reserved) != expected_roles:
        raise common.ContractError(
            "Step28-v13 aggregate receipt role registry drift"
        )
    if (
        schemas["common_top_level_exact_keys"]
        != [
            "version",
            "evidence_level",
            "fixed_boolean_gates",
            "fixed_counts",
            "input_parent_hashes",
            "aggregate_content_hashes",
            "canonical_self_hash",
        ]
        or schemas["canonical_multiset_hash_encoding"].get(
            "version_prefix_utf8_then_nul"
        )
        != MULTISET_PREFIX.decode("ascii")
    ):
        raise common.ContractError(
            "Step28-v13 common receipt/framing schema drift"
        )
    source_closure_contract = deployment.get(
        "development_source_closure_contract"
    )
    if (
        not isinstance(source_closure_contract, Mapping)
        or set(source_closure_contract)
        != {
            "algorithm",
            "all_discovered_repo_local_dependencies_bound",
            "every_registered_member_hash_perturbation_changes_closure",
            "role_local_imports_are_lazy",
            "nonliteral_dynamic_imports_fail_closed",
            "registered_alias_and_indirect_patterns_fail_closed",
            "direct_reflection_primitives_fail_closed",
            "relative_imports_fail_closed",
            "relative_dynamic_targets_fail_closed",
            "unregistered_reflection_is_not_claimed_as_development_sandbox",
            "conservative_repo_local_superset_allowed",
            "third_party_environment_lock_required_for_formal",
            "formal_fresh_process_loaded_module_inventory_implemented",
        }
        or source_closure_contract.get(
            "all_discovered_repo_local_dependencies_bound"
        )
        is not True
        or source_closure_contract.get(
            "every_registered_member_hash_perturbation_changes_closure"
        )
        is not True
        or source_closure_contract.get("role_local_imports_are_lazy")
        is not True
        or source_closure_contract.get(
            "nonliteral_dynamic_imports_fail_closed"
        )
        is not True
        or source_closure_contract.get(
            "registered_alias_and_indirect_patterns_fail_closed"
        )
        != [
            "import importlib as alias",
            "import builtins",
            "from importlib import import_module",
            "non-direct importlib.import_module attribute use",
            "getattr or vars applied directly to importlib or builtins",
            "attribute named import_module or __import__ on another object",
        ]
        or source_closure_contract.get(
            "direct_reflection_primitives_fail_closed"
        )
        != [
            "__builtins__ name",
            "compile direct call",
            "eval direct call",
            "exec direct call",
            "globals direct call",
            "locals direct call",
        ]
        or source_closure_contract.get(
            "relative_imports_fail_closed"
        )
        is not True
        or source_closure_contract.get(
            "relative_dynamic_targets_fail_closed"
        )
        is not True
        or source_closure_contract.get(
            "unregistered_reflection_is_not_claimed_as_development_sandbox"
        )
        is not True
        or source_closure_contract.get(
            "conservative_repo_local_superset_allowed"
        )
        is not True
        or source_closure_contract.get(
            "third_party_environment_lock_required_for_formal"
        )
        is not True
        or source_closure_contract.get(
            "formal_fresh_process_loaded_module_inventory_implemented"
        )
        is not False
    ):
        raise common.ContractError(
            "Step28-v13 development source closure contract drift"
        )
    for role in expected_roles:
        spec = schemas[role]
        if not isinstance(spec, Mapping):
            raise common.ContractError(
                f"Step28-v13 receipt spec is not a mapping: {role}"
            )
        for key_name in (
            "fixed_boolean_gates_exact_keys",
            "fixed_counts_exact_keys",
            "input_parent_hashes_exact_keys",
            "aggregate_content_hashes_exact_keys",
        ):
            values = spec.get(key_name)
            if (
                not isinstance(values, list)
                or not values
                or any(not isinstance(value, str) or not value for value in values)
                or len(values) != len(set(values))
            ):
                raise common.ContractError(
                    f"Step28-v13 receipt exact-key registry drift: "
                    f"{role}.{key_name}"
                )
        if (
            "registered_split_scope_sha256"
            not in spec["input_parent_hashes_exact_keys"]
        ):
            raise common.ContractError(
                f"Step28-v13 receipt omits split scope: {role}"
            )
    capabilities = deployment.get("capabilities")
    candidate_worker = (
        capabilities.get("candidate_worker")
        if isinstance(capabilities, Mapping)
        else None
    )
    candidate_capability = (
        capabilities.get("candidate_integrity_projection_sealer")
        if isinstance(capabilities, Mapping)
        else None
    )
    required_candidate_worker_reads = {
        "raw_observed_sellers_and_items_for_one_split",
        "complete_model_pair_universe",
        (
            "public_candidate_policy_projection_without_"
            "structure_or_oracle_secrets"
        ),
        "frozen_step3_and_step4_code_and_schemas",
        "one_split_candidate_key",
    }
    required_candidate_reads = {
        "raw_observed_sellers_and_items_for_one_split",
        "complete_model_pair_universe",
        "candidate_pair_projection",
        "candidate_sampling_audit",
        "frozen_step3_and_step4_code_and_schemas",
        "one_split_candidate_key",
        (
            "public_candidate_policy_projection_without_"
            "structure_or_oracle_secrets"
        ),
    }
    if (
        not isinstance(candidate_worker, Mapping)
        or set(candidate_worker.get("read_roles", []))
        != required_candidate_worker_reads
        or "full_parent_policy"
        not in set(candidate_worker.get("forbidden_roles", []))
        or candidate_worker.get(
            "public_candidate_projection_values_exact_not_key_only"
        )
        is not True
        or candidate_worker.get(
            "public_candidate_static_contract_sha256"
        )
        != candidate_sampling.CANDIDATE_STATIC_CONTRACT_SHA256
        or not isinstance(candidate_capability, Mapping)
        or set(candidate_capability.get("read_roles", []))
        != required_candidate_reads
        or "full_parent_policy"
        not in set(candidate_capability.get("forbidden_roles", []))
        or candidate_capability.get(
            "public_candidate_projection_values_exact_not_key_only"
        )
        is not True
        or candidate_capability.get(
            "public_candidate_static_contract_sha256"
        )
        != candidate_sampling.CANDIDATE_STATIC_CONTRACT_SHA256
    ):
        raise common.ContractError(
            "Candidate integrity replay capability boundary drift"
        )
    for split in SPLITS:
        expected_receipt_roles(policy, split=split)


def _receipt_spec(
    policy: Mapping[str, Any], role: str
) -> Mapping[str, Any]:
    schemas = _load_deployment(policy)[
        "fixed_nonjoinable_aggregate_receipt_schemas"
    ]
    spec = schemas.get(role)
    if not isinstance(spec, Mapping):
        raise common.ContractError(
            f"Unknown Step28-v13 aggregate receipt role: {role}"
        )
    return spec


def _validate_hash(value: Any, *, label: str) -> None:
    if not isinstance(value, str) or HASH_PATTERN.fullmatch(value) is None:
        raise common.ContractError(f"{label} is not a lowercase SHA-256")


def validate_aggregate_receipt(
    policy: Mapping[str, Any],
    *,
    role: str,
    receipt: Mapping[str, Any],
) -> None:
    """Recursively enforce the exact scalar-only receipt schema."""

    deployment = _load_deployment(policy)
    schemas = deployment["fixed_nonjoinable_aggregate_receipt_schemas"]
    spec = _receipt_spec(policy, role)
    _validate_aggregate_receipt_against_spec(
        role=role,
        receipt=receipt,
        top_keys=schemas["common_top_level_exact_keys"],
        spec=spec,
    )


def _validate_aggregate_receipt_against_spec(
    *,
    role: str,
    receipt: Mapping[str, Any],
    top_keys: Sequence[str],
    spec: Mapping[str, Any],
) -> None:
    """Validate a receipt using only its public exact-schema contract."""

    top_key_set = set(top_keys)
    if set(receipt) != top_key_set:
        raise common.ContractError(
            f"{role} aggregate receipt top-level schema drift"
        )
    if (
        receipt["version"] != spec["release_exact_version"]
        or receipt["evidence_level"]
        != spec["release_exact_evidence_level"]
    ):
        raise common.ContractError(
            f"{role} aggregate receipt version/evidence drift"
        )
    object_specs = (
        ("fixed_boolean_gates", "fixed_boolean_gates_exact_keys"),
        ("fixed_counts", "fixed_counts_exact_keys"),
        ("input_parent_hashes", "input_parent_hashes_exact_keys"),
        (
            "aggregate_content_hashes",
            "aggregate_content_hashes_exact_keys",
        ),
    )
    for object_name, keys_name in object_specs:
        value = receipt[object_name]
        if not isinstance(value, Mapping) or set(value) != set(
            spec[keys_name]
        ):
            raise common.ContractError(
                f"{role} aggregate receipt {object_name} schema drift"
            )
        if any(isinstance(item, (Mapping, list, tuple)) for item in value.values()):
            raise common.ContractError(
                f"{role} aggregate receipt contains a nested object"
            )
    if any(
        type(value) is not bool
        for value in receipt["fixed_boolean_gates"].values()
    ):
        raise common.ContractError(
            f"{role} aggregate receipt gate type drift"
        )
    if any(
        type(value) is not int or value < 0
        for value in receipt["fixed_counts"].values()
    ):
        raise common.ContractError(
            f"{role} aggregate receipt count type drift"
        )
    for object_name in ("input_parent_hashes", "aggregate_content_hashes"):
        for key, value in receipt[object_name].items():
            _validate_hash(value, label=f"{role}.{object_name}.{key}")
    _validate_hash(receipt["canonical_self_hash"], label=f"{role}.self")
    expected_self_hash = common.canonical_sha256(
        {
            key: value
            for key, value in receipt.items()
            if key != "canonical_self_hash"
        }
    )
    if receipt["canonical_self_hash"] != expected_self_hash:
        raise common.ContractError(
            f"{role} aggregate receipt self-hash mismatch"
        )


def _build_receipt(
    policy: Mapping[str, Any],
    *,
    role: str,
    fixed_boolean_gates: Mapping[str, bool],
    fixed_counts: Mapping[str, int],
    input_parent_hashes: Mapping[str, str],
    aggregate_content_hashes: Mapping[str, str],
) -> dict[str, Any]:
    spec = _receipt_spec(policy, role)
    schemas = _load_deployment(policy)[
        "fixed_nonjoinable_aggregate_receipt_schemas"
    ]
    return _build_receipt_against_spec(
        role=role,
        spec=spec,
        top_keys=schemas["common_top_level_exact_keys"],
        fixed_boolean_gates=fixed_boolean_gates,
        fixed_counts=fixed_counts,
        input_parent_hashes=input_parent_hashes,
        aggregate_content_hashes=aggregate_content_hashes,
    )


def _build_receipt_against_spec(
    *,
    role: str,
    spec: Mapping[str, Any],
    top_keys: Sequence[str],
    fixed_boolean_gates: Mapping[str, bool],
    fixed_counts: Mapping[str, int],
    input_parent_hashes: Mapping[str, str],
    aggregate_content_hashes: Mapping[str, str],
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "version": spec["release_exact_version"],
        "evidence_level": spec["release_exact_evidence_level"],
        "fixed_boolean_gates": dict(fixed_boolean_gates),
        "fixed_counts": dict(fixed_counts),
        "input_parent_hashes": dict(input_parent_hashes),
        "aggregate_content_hashes": dict(aggregate_content_hashes),
    }
    receipt["canonical_self_hash"] = common.canonical_sha256(receipt)
    _validate_aggregate_receipt_against_spec(
        role=role,
        receipt=receipt,
        top_keys=top_keys,
        spec=spec,
    )
    return receipt


def _expected_world_uids(
    policy: Mapping[str, Any],
    *,
    mode: str,
    split: str,
    worlds: Sequence[Mapping[str, Any]],
) -> list[str]:
    if mode != "development_smoke" or split not in SPLITS:
        raise common.ContractError(
            "Integrity receipts are development-smoke only"
        )
    world_schema = policy["relational_integrity"][
        "observed_core_schemas"
    ]["worlds.csv"]
    if any(list(row) != world_schema for row in worlds):
        raise common.ContractError("Integrity receipt world schema drift")
    world_uids = common.utf8_sort(
        str(row["world_uid"]) for row in worlds
    )
    registered_world_uids = common.utf8_sort(
        str(row["world_uid"])
        for row in structure.build_mode_world_pool(policy, mode=mode)
        if str(row["split"]) == split
    )
    expected_count = int(policy["modes"][mode]["world_counts"][split])
    if (
        len(world_uids) != expected_count
        or len(set(world_uids)) != expected_count
        or len(registered_world_uids) != expected_count
        or len(set(registered_world_uids)) != expected_count
        or world_uids != registered_world_uids
    ):
        raise common.ContractError(
            "Integrity receipt registered split world set drift"
        )
    return world_uids


def _registered_split_scope_sha256(
    policy: Mapping[str, Any],
    *,
    mode: str,
    split: str,
) -> str:
    world_uids = common.utf8_sort(
        str(row["world_uid"])
        for row in structure.build_mode_world_pool(policy, mode=mode)
        if str(row["split"]) == split
    )
    expected_count = int(policy["modes"][mode]["world_counts"][split])
    if (
        split not in SPLITS
        or len(world_uids) != expected_count
        or len(set(world_uids)) != expected_count
    ):
        raise common.ContractError(
            "Registered split scope cannot be constructed"
        )
    return common.canonical_sha256(
        {
            "policy_version": str(policy["version"]),
            "mode": mode,
            "split": split,
            "registered_world_uids_utf8_sorted": world_uids,
        }
    )


def _rows_for_world(
    rows: Sequence[Mapping[str, Any]],
    *,
    world_uid: str,
    strip_world_uid: bool = False,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source in rows:
        if str(source.get("world_uid", "")) != world_uid:
            continue
        row = dict(source)
        if strip_world_uid:
            if next(iter(row), None) != "world_uid":
                raise common.ContractError(
                    "Prepended private world_uid order drift"
                )
            row.pop("world_uid")
        output.append(row)
    return output


def _tagged_rows(
    *tables: tuple[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    return [
        {"table": table, "row": dict(row)}
        for table, rows in tables
        for row in rows
    ]


def _policy_parent_sha256(policy: Mapping[str, Any]) -> str:
    file_policy = common.load_json(common.DEFAULT_POLICY_PATH)
    if common.canonical_json_bytes(file_policy) != common.canonical_json_bytes(
        dict(policy)
    ):
        raise common.ContractError(
            "Integrity receipt policy differs from the exact policy file"
        )
    return common.sha256_file(common.DEFAULT_POLICY_PATH)


def build_candidate_integrity_context(
    policy: Mapping[str, Any],
    *,
    candidate_policy: Mapping[str, Any],
    mode: str,
    split: str,
) -> dict[str, Any]:
    """Project receipt metadata before entering candidate sealer custody."""

    common.validate_policy(policy, mode=mode)
    candidate_sampling.validate_public_candidate_policy(
        candidate_policy, mode=mode, split=split
    )
    expected_candidate_policy = (
        candidate_sampling.build_public_candidate_policy(
            policy, mode=mode, split=split
        )
    )
    if common.canonical_json_bytes(
        dict(candidate_policy)
    ) != common.canonical_json_bytes(expected_candidate_policy):
        raise common.ContractError(
            "Candidate policy projection differs from the exact "
            "registered policy projection"
        )
    deployment = _load_deployment(policy)
    schemas = deployment[
        "fixed_nonjoinable_aggregate_receipt_schemas"
    ]
    registered_world_uids = common.utf8_sort(
        str(row["world_uid"])
        for row in structure.build_mode_world_pool(policy, mode=mode)
        if str(row["split"]) == split
    )
    context = {
        "version": CANDIDATE_CONTEXT_VERSION,
        "parent_policy_version": str(policy["version"]),
        "mode": mode,
        "split": split,
        "registered_world_uids_utf8_sorted": registered_world_uids,
        "registered_split_scope_sha256": (
            _registered_split_scope_sha256(
                policy, mode=mode, split=split
            )
        ),
        "policy_parent_sha256": _policy_parent_sha256(policy),
        "candidate_policy_projection_sha256": common.canonical_sha256(
            dict(candidate_policy)
        ),
        "receipt_top_level_exact_keys": copy.deepcopy(
            schemas["common_top_level_exact_keys"]
        ),
        "receipt_spec": copy.deepcopy(schemas["candidate_integrity"]),
    }
    context["canonical_self_hash"] = common.canonical_sha256(context)
    _validate_candidate_integrity_context(
        context,
        candidate_policy=candidate_policy,
        mode=mode,
        split=split,
    )
    return context


def _validate_candidate_integrity_context(
    context: Mapping[str, Any],
    *,
    candidate_policy: Mapping[str, Any],
    mode: str,
    split: str,
) -> None:
    candidate_sampling.validate_public_candidate_policy(
        candidate_policy, mode=mode, split=split
    )
    if (
        not isinstance(context, Mapping)
        or set(context) != CANDIDATE_CONTEXT_KEYS
        or context.get("version") != CANDIDATE_CONTEXT_VERSION
        or context.get("parent_policy_version") != common.POLICY_VERSION
        or context.get("mode") != mode
        or context.get("split") != split
        or context.get("candidate_policy_projection_sha256")
        != common.canonical_sha256(dict(candidate_policy))
        or context.get("canonical_self_hash")
        != common.canonical_sha256(
            {
                key: value
                for key, value in context.items()
                if key != "canonical_self_hash"
            }
        )
    ):
        raise common.ContractError(
            "Candidate integrity public sealer context drift"
        )
    world_uids = context["registered_world_uids_utf8_sorted"]
    if (
        not isinstance(world_uids, list)
        or not world_uids
        or any(not isinstance(value, str) or not value for value in world_uids)
        or world_uids
        != sorted(world_uids, key=lambda value: value.encode("utf-8"))
        or len(world_uids) != len(set(world_uids))
        or context["registered_split_scope_sha256"]
        != common.canonical_sha256(
            {
                "policy_version": str(context["parent_policy_version"]),
                "mode": mode,
                "split": split,
                "registered_world_uids_utf8_sorted": world_uids,
            }
        )
    ):
        raise common.ContractError(
            "Candidate integrity registered world context drift"
        )
    _validate_hash(
        context["policy_parent_sha256"],
        label="candidate context policy parent",
    )
    _validate_hash(
        context["candidate_policy_projection_sha256"],
        label="candidate context policy projection",
    )
    _validate_hash(
        context["registered_split_scope_sha256"],
        label="candidate context split scope",
    )
    top_keys = context["receipt_top_level_exact_keys"]
    spec = context["receipt_spec"]
    if (
        not isinstance(top_keys, list)
        or top_keys
        != [
            "version",
            "evidence_level",
            "fixed_boolean_gates",
            "fixed_counts",
            "input_parent_hashes",
            "aggregate_content_hashes",
            "canonical_self_hash",
        ]
        or not isinstance(spec, Mapping)
    ):
        raise common.ContractError(
            "Candidate integrity receipt contract context drift"
        )


def build_render_integrity_receipt(
    policy: Mapping[str, Any],
    *,
    mode: str,
    split: str,
    template: Mapping[str, Any],
    worlds: Sequence[Mapping[str, Any]],
    sellers: Sequence[Mapping[str, Any]],
    items: Sequence[Mapping[str, Any]],
    redacted_items: Sequence[Mapping[str, Any]],
    parsed_identity_occurrences: Sequence[Mapping[str, Any]],
    identity_slots_audit: Sequence[Mapping[str, Any]],
    noise_slots_audit: Sequence[Mapping[str, Any]],
    render_asts: Sequence[Mapping[str, Any]],
    override_audit: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Recompute parser/render/redaction integrity into one scalar receipt."""

    import step28_v13_production_chain as production

    common.validate_policy(policy, mode=mode)
    tables = {
        "worlds": worlds,
        "sellers": sellers,
        "items": items,
        "redacted_items": redacted_items,
        "parsed_identity_occurrences": parsed_identity_occurrences,
        "identity_slots_audit": identity_slots_audit,
        "noise_slots_audit": noise_slots_audit,
        "render_asts": render_asts,
        "override_audit": override_audit,
    }
    observed_schemas = policy["relational_integrity"][
        "observed_core_schemas"
    ]
    exact_schemas = {
        "worlds": list(observed_schemas["worlds.csv"]),
        "sellers": list(observed_schemas["sellers.csv"]),
        "items": list(observed_schemas["items.jsonl"]),
        "redacted_items": list(
            observed_schemas["redacted_items.jsonl"]
        ),
        "parsed_identity_occurrences": list(
            observed_schemas[
                "parsed_identity_occurrences.structural_audit_private.csv"
            ]
        ),
        "identity_slots_audit": list(
            policy["relational_integrity"][
                "renderer_identity_slots_audit_schema"
            ]
        ),
        "noise_slots_audit": list(
            policy["relational_integrity"][
                "renderer_noise_slots_audit_schema"
            ]
        ),
        "render_asts": list(
            observed_schemas["render_asts.jsonl_private"]
        ),
        "override_audit": [
            "world_uid",
            "override_kind",
            "asset_index",
            "canonical_pair_uid",
            "seller_uid_left",
            "seller_uid_right",
            "item_uid_left",
            "item_uid_right",
        ],
    }
    for name, rows in tables.items():
        schema = exact_schemas[name]
        if any(
            not isinstance(row, Mapping) or list(row) != schema
            for row in rows
        ):
            raise common.ContractError(
                f"Render integrity {name} schema/order drift"
            )
    world_uids = _expected_world_uids(
        policy,
        mode=mode,
        split=split,
        worlds=tables["worlds"],
    )
    consumed: dict[str, list[dict[str, Any]]] = {
        name: [] for name in tables
    }
    consumed["worlds"].extend(dict(row) for row in tables["worlds"])
    per_world: list[dict[str, Any]] = []
    for world_uid in world_uids:
        world_sellers = _rows_for_world(
            tables["sellers"], world_uid=world_uid
        )
        world_items = _rows_for_world(
            tables["items"], world_uid=world_uid
        )
        item_uids = {str(row["item_uid"]) for row in world_items}
        world_parsed = _rows_for_world(
            tables["parsed_identity_occurrences"],
            world_uid=world_uid,
        )
        world_redacted = _rows_for_world(
            tables["redacted_items"], world_uid=world_uid
        )
        world_identity_slots = [
            dict(row)
            for row in tables["identity_slots_audit"]
            if str(row.get("item_uid", "")) in item_uids
        ]
        world_noise_slots = [
            dict(row)
            for row in tables["noise_slots_audit"]
            if str(row.get("item_uid", "")) in item_uids
        ]
        world_render_asts = _rows_for_world(
            tables["render_asts"], world_uid=world_uid
        )
        world_overrides = _rows_for_world(
            tables["override_audit"],
            world_uid=world_uid,
            strip_world_uid=True,
        )
        consumed["sellers"].extend(world_sellers)
        consumed["items"].extend(world_items)
        consumed["redacted_items"].extend(world_redacted)
        consumed["parsed_identity_occurrences"].extend(world_parsed)
        consumed["identity_slots_audit"].extend(world_identity_slots)
        consumed["noise_slots_audit"].extend(world_noise_slots)
        consumed["render_asts"].extend(world_render_asts)
        consumed["override_audit"].extend(
            _rows_for_world(
                tables["override_audit"],
                world_uid=world_uid,
            )
        )
        parser_audit = production.validate_parser_against_private_plan(
            policy,
            mode=mode,
            split=split,
            sellers=world_sellers,
            items=world_items,
            parsed_rows=world_parsed,
            identity_slots_audit=world_identity_slots,
            noise_slots_audit=world_noise_slots,
            render_asts=world_render_asts,
        )
        redaction_audit = production.validate_redaction_against_private_plan(
            policy,
            mode=mode,
            split=split,
            template=template,
            sellers=world_sellers,
            items=world_items,
            redacted_items=world_redacted,
            parsed_rows=world_parsed,
            identity_slots_audit=world_identity_slots,
            noise_slots_audit=world_noise_slots,
            render_asts=world_render_asts,
            override_audit=world_overrides,
        )
        if (
            parser_audit["exact_rows_and_flags"] is not True
            or redaction_audit[
                "post_redaction_seller_minimums_pass"
            ]
            is not True
        ):
            raise common.ContractError(
                "Render integrity recomputation returned a failed gate"
            )
        per_world.append(
            {
                "world_uid": world_uid,
                "item_count": len(world_items),
                "planned_identity_slot_count": int(
                    parser_audit["planned_must_extract_count"]
                ),
                "parsed_identity_occurrence_count": int(
                    parser_audit["actual_parser_row_count"]
                ),
                "noise_slot_count": int(
                    parser_audit["ast_noise_slot_count"]
                ),
                "parser_rows_and_flags_exact": True,
                "planned_identity_surface_residue_count": int(
                    redaction_audit[
                        "planned_identity_surface_residue_count"
                    ]
                ),
                "parsed_identity_surface_residue_count": int(
                    redaction_audit[
                        "parsed_identity_surface_residue_count"
                    ]
                ),
                "context_guard_residue_count": int(
                    redaction_audit["context_guard_residue_count"]
                ),
                "must_ignore_changed_count": int(
                    redaction_audit["must_ignore_changed_count"]
                ),
                "base_description_changed_count": int(
                    redaction_audit["base_description_changed_count"]
                ),
                "title_changed_count": int(
                    redaction_audit["title_changed_count"]
                ),
                "post_redaction_seller_minimums_pass": True,
            }
        )

    for name, rows in tables.items():
        if (
            len(consumed[name]) != len(rows)
            or canonical_multiset_sha256(consumed[name])
            != canonical_multiset_sha256(rows)
        ):
            raise common.ContractError(
                f"Render integrity {name} contains an unconsumed row"
            )

    identity_residue_count = sum(
        int(row["planned_identity_surface_residue_count"])
        + int(row["parsed_identity_surface_residue_count"])
        + int(row["context_guard_residue_count"])
        for row in per_world
    )
    must_ignore_mismatch_count = sum(
        int(row["must_ignore_changed_count"]) for row in per_world
    )
    base_text_mismatch_count = sum(
        int(row["base_description_changed_count"]) for row in per_world
    )
    title_mismatch_count = sum(
        int(row["title_changed_count"]) for row in per_world
    )
    fixed_counts = {
        "split_world_count": len(world_uids),
        "split_item_count": len(tables["items"]),
        "planned_identity_slot_count": len(
            tables["identity_slots_audit"]
        ),
        "parsed_identity_occurrence_count": len(
            tables["parsed_identity_occurrences"]
        ),
        "noise_slot_count": len(tables["noise_slots_audit"]),
        "identity_residue_count": identity_residue_count,
        "must_ignore_mismatch_count": must_ignore_mismatch_count,
        "base_text_mismatch_count": base_text_mismatch_count,
        "title_mismatch_count": title_mismatch_count,
    }
    fixed_gates = {
        "parser_rows_and_flags_exact": all(
            row["parser_rows_and_flags_exact"] is True
            for row in per_world
        ),
        "parser_item_universe_exact": (
            sum(int(row["item_count"]) for row in per_world)
            == len(tables["items"])
            == len(tables["redacted_items"])
            == len(tables["render_asts"])
        ),
        "identity_redaction_complete": identity_residue_count == 0,
        "base_text_exact": base_text_mismatch_count == 0,
        "must_ignore_text_preserved": must_ignore_mismatch_count == 0,
        "titles_unchanged": title_mismatch_count == 0,
        "post_redaction_seller_minimums_pass": all(
            row["post_redaction_seller_minimums_pass"] is True
            for row in per_world
        ),
    }
    if not all(fixed_gates.values()):
        raise common.ContractError("Render integrity aggregate gate failed")
    renderer_audit_rows = _tagged_rows(
        ("identity_slots", tables["identity_slots_audit"]),
        ("noise_slots", tables["noise_slots_audit"]),
        ("override_audit", tables["override_audit"]),
    )
    return _build_receipt(
        policy,
        role="render_integrity",
        fixed_boolean_gates=fixed_gates,
        fixed_counts=fixed_counts,
        input_parent_hashes={
            "observed_parent_sha256": canonical_multiset_sha256(
                tables["items"]
            ),
            "redacted_parent_sha256": canonical_multiset_sha256(
                tables["redacted_items"]
            ),
            "parser_parent_sha256": canonical_multiset_sha256(
                tables["parsed_identity_occurrences"]
            ),
            "renderer_audit_parent_sha256": (
                canonical_multiset_sha256(renderer_audit_rows)
            ),
            "render_ast_parent_sha256": canonical_multiset_sha256(
                tables["render_asts"]
            ),
            "policy_parent_sha256": _policy_parent_sha256(policy),
            "registered_split_scope_sha256": (
                _registered_split_scope_sha256(
                    policy, mode=mode, split=split
                )
            ),
        },
        aggregate_content_hashes={
            "parser_rows_multiset_sha256": canonical_multiset_sha256(
                tables["parsed_identity_occurrences"]
            ),
            "redacted_items_multiset_sha256": canonical_multiset_sha256(
                tables["redacted_items"]
            ),
            "render_integrity_multiset_sha256": (
                canonical_multiset_sha256(per_world)
            ),
        },
    )


def _hamilton_quotas(
    layer_sizes: Mapping[str, int],
    *,
    total_slots: int,
    trigger_priority: Sequence[str],
) -> dict[str, int]:
    total_size = sum(int(layer_sizes.get(name, 0)) for name in trigger_priority)
    if total_size < total_slots:
        raise common.ContractError(
            "Candidate receipt Hamilton universe is too small"
        )
    quotas: dict[str, int] = {}
    remainders: list[tuple[int, int, str]] = []
    allocated = 0
    for priority_index, name in enumerate(trigger_priority):
        size = int(layer_sizes.get(name, 0))
        numerator = total_slots * size
        quota, remainder = divmod(numerator, total_size)
        quotas[name] = quota
        allocated += quota
        if size:
            remainders.append((-remainder, priority_index, name))
    for _remainder, _priority, name in sorted(remainders)[
        : total_slots - allocated
    ]:
        quotas[name] += 1
    if sum(quotas.values()) != total_slots:
        raise common.ContractError(
            "Candidate receipt Hamilton allocation drift"
        )
    return quotas


def _independently_validate_candidate_trigger_projection(
    candidate_policy: Mapping[str, Any],
    *,
    mode: str,
    split: str,
    sellers: Sequence[Mapping[str, Any]],
    raw_observed_items: Sequence[Mapping[str, Any]],
    complete_pair_endpoints: Sequence[Mapping[str, Any]],
    candidate_sampling_audit: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    """Rebuild Step3/Step4 evidence without calling the C40 producer.

    The candidate producer and this sealer intentionally share the frozen
    Step3/Step4 definitions, which are the registered source of candidate
    evidence.  Everything after those frozen rows--trigger projection,
    primary-layer choice, lexical serialization, and structural projection--
    is recomputed here instead of trusting ``build_world_c40``.
    """

    frozen_inputs = candidate_policy["frozen_inputs"]
    step3_path = common.verify_file_pin(
        frozen_inputs["step3_parser_profile_code"],
        label="Candidate sealer Step3 producer",
    )
    common.verify_file_pin(
        frozen_inputs["step3_profile_schema"],
        label="Candidate sealer Step3 schema",
    )
    step4_path = common.verify_file_pin(
        frozen_inputs["step4_candidate_code"],
        label="Candidate sealer Step4 producer",
    )
    step4_schema_path = common.verify_file_pin(
        frozen_inputs["step4_candidate_schema"],
        label="Candidate sealer Step4 schema",
    )
    profiles_mod = importlib.import_module("step28_v13_profiles")
    step3_mod = importlib.import_module("step3_build_seller_profiles")
    step4_mod = importlib.import_module("step4_build_silver_candidates")
    if (
        common.sha256_file(step3_path)
        != common.sha256_file(Path(step3_mod.__file__).resolve())
        or common.sha256_file(step4_path)
        != common.sha256_file(Path(step4_mod.__file__).resolve())
    ):
        raise common.ContractError(
            "Candidate sealer frozen dependency import drift"
        )

    profile_policy = {
        "modes": {mode: {}},
        "frozen_inputs": {
            "step3_profile_schema": frozen_inputs[
                "step3_profile_schema"
            ]
        },
        "relational_integrity": {
            "observed_core_schemas": candidate_policy[
                "observed_core_schemas"
            ]
        },
    }
    raw_profiles, raw_profile_audit = profiles_mod.build_world_profiles(
        profile_policy,
        mode=mode,
        split=split,
        sellers=sellers,
        items=raw_observed_items,
    )
    step4_schema = common.load_json(step4_schema_path)
    filtering = step4_schema["filtering_policy"]
    stopwords = {
        str(value).lower()
        for value in filtering["contact_noise_stopwords"]
    }
    step4_profiles = step4_mod.build_seller_profiles(
        rows=raw_profiles,
        data_bucket=f"step28_v13_{mode}_{split}",
        language="zh",
        stopwords=stopwords,
        min_config=filtering["content_minimums"],
        pgp_alias_map={},
    )
    design = candidate_policy["candidate_design"]
    derived_config = {
        key: value
        for key, value in design["step4_derived_config"].items()
        if key != "pgp_alias_map"
    }
    step4_mod.compute_retrieval_weights(
        step4_profiles, derived_config
    )
    step4_rows = step4_mod.build_candidates_for_pool(
        step4_profiles,
        derived_config,
        "zh",
        filtering["duplicate_cluster_limits"],
    )

    pair_uids = {
        str(row["canonical_pair_uid"])
        for row in complete_pair_endpoints
    }
    if (
        len(pair_uids) != len(complete_pair_endpoints)
        or len(pair_uids) != 378
    ):
        raise common.ContractError(
            "Candidate sealer complete-pair universe drift"
        )
    step4_index: dict[str, dict[str, Any]] = {}
    for source in step4_rows:
        row = dict(source)
        pair_uid = str(row.get("pair_uid", ""))
        if pair_uid not in pair_uids or pair_uid in step4_index:
            raise common.ContractError(
                "Candidate sealer frozen Step4 pair drift"
            )
        step4_index[pair_uid] = row

    priority = [
        str(value)
        for value in design["primary_trigger_priority"]
    ]
    reachable = priority[:-1]
    fallback = priority[-1]
    supported_rules = {*reachable, "structural_support"}
    audit_index = {
        str(row["canonical_pair_uid"]): row
        for row in candidate_sampling_audit
    }
    if (
        len(audit_index) != len(candidate_sampling_audit)
        or set(audit_index) != pair_uids
    ):
        raise common.ContractError(
            "Candidate sealer sampling-audit universe drift"
        )
    for pair_uid in common.utf8_sort(pair_uids):
        source = step4_index.get(pair_uid)
        source_flags: set[str] = set()
        lexical_similarity = 0.0
        if source is not None:
            source_flags = set(
                filter(
                    None,
                    str(source["candidate_rule_hits"]).split("|"),
                )
            )
            if not source_flags.issubset(supported_rules):
                raise common.ContractError(
                    "Candidate sealer Step4 rule universe drift"
                )
            lexical_similarity = float(source["lexical_similarity"])
        expected_flags = [
            name for name in reachable if name in source_flags
        ]
        expected_primary = (
            expected_flags[0] if expected_flags else fallback
        )
        observed = audit_index[pair_uid]
        if (
            str(observed["trigger_flags"])
            != "|".join(expected_flags)
            or str(observed["primary_trigger"]) != expected_primary
            or str(observed["lexical_similarity"])
            != f"{lexical_similarity:.6f}"
            or str(observed["structural_support_flag"])
            != (
                "true"
                if "structural_support" in source_flags
                else "false"
            )
        ):
            raise common.ContractError(
                "Candidate sealer independent trigger replay drift"
            )
    return {
        "ephemeral_raw_profile_sha256": common.canonical_sha256(
            raw_profiles
        ),
        "step3_profile_audit_sha256": common.canonical_sha256(
            raw_profile_audit
        ),
        "frozen_step4_candidate_rows_sha256": (
            common.canonical_sha256(step4_rows)
        ),
    }


def build_candidate_integrity_receipt(
    candidate_context: Mapping[str, Any],
    *,
    candidate_policy: Mapping[str, Any],
    candidate_key_hex: str,
    mode: str,
    split: str,
    worlds: Sequence[Mapping[str, Any]],
    sellers: Sequence[Mapping[str, Any]],
    raw_observed_items: Sequence[Mapping[str, Any]],
    complete_pair_endpoints: Sequence[Mapping[str, Any]],
    candidate_pairs: Sequence[Mapping[str, Any]],
    candidate_sampling_audit: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Recompute complete-pair/C40/audit closure without labels or oracle."""

    _validate_candidate_integrity_context(
        candidate_context,
        candidate_policy=candidate_policy,
        mode=mode,
        split=split,
    )
    tables = {
        "worlds": worlds,
        "sellers": sellers,
        "raw_observed_items": raw_observed_items,
        "complete_model_pair_endpoints": complete_pair_endpoints,
        "candidate_pairs": candidate_pairs,
        "candidate_sampling_audit": candidate_sampling_audit,
    }
    if any(list(row) != ["world_uid"] for row in tables["worlds"]):
        raise common.ContractError(
            "Candidate integrity world schema/order drift"
        )
    world_uids = common.utf8_sort(
        str(row["world_uid"]) for row in tables["worlds"]
    )
    if world_uids != candidate_context[
        "registered_world_uids_utf8_sorted"
    ]:
        raise common.ContractError(
            "Candidate integrity registered world set drift"
        )
    pairs = tables["complete_model_pair_endpoints"]
    candidates = tables["candidate_pairs"]
    audits = tables["candidate_sampling_audit"]
    pair_schema = candidate_policy[
        "complete_model_pair_endpoints_schema"
    ]
    candidate_schema = candidate_policy["candidate_design"][
        "public_safe_projection_columns"
    ]
    audit_schema = candidate_policy["candidate_design"][
        "sampling_audit_projection_columns"
    ]
    seller_schema = candidate_policy["observed_core_schemas"][
        "sellers.csv"
    ]
    item_schema = candidate_policy["observed_core_schemas"][
        "items.jsonl"
    ]
    if (
        any(list(row) != seller_schema for row in sellers)
        or any(list(row) != item_schema for row in raw_observed_items)
        or any(list(row) != pair_schema for row in pairs)
        or any(list(row) != candidate_schema for row in candidates)
        or any(list(row) != audit_schema for row in audits)
    ):
        raise common.ContractError(
            "Candidate integrity input schema/order drift"
        )
    pair_index: dict[tuple[str, str], dict[str, Any]] = {}
    for source in pairs:
        row = dict(source)
        world_uid = str(row["world_uid"])
        pair_uid = str(row["canonical_pair_uid"])
        left = str(row["seller_uid_left"])
        right = str(row["seller_uid_right"])
        key = (world_uid, pair_uid)
        if (
            world_uid not in world_uids
            or key in pair_index
            or left == right
            or common.utf8_sort((left, right)) != [left, right]
            or pair_uid != common.canonical_pair_uid(left, right)
        ):
            raise common.ContractError(
                "Candidate integrity complete-pair lineage drift"
            )
        pair_index[key] = row
    candidate_index: dict[tuple[str, str], dict[str, Any]] = {}
    for source in candidates:
        row = dict(source)
        key = (str(row["world_uid"]), str(row["canonical_pair_uid"]))
        parent = pair_index.get(key)
        if (
            key in candidate_index
            or parent is None
            or any(str(row[name]) != str(parent[name]) for name in pair_schema)
        ):
            raise common.ContractError(
                "Candidate integrity C40 endpoint drift"
            )
        candidate_index[key] = row
    audit_index: dict[tuple[str, str], dict[str, Any]] = {}
    priority = list(
        candidate_policy["candidate_design"]["primary_trigger_priority"]
    )
    reachable = priority[:-1]
    fallback = priority[-1]
    candidate_key_hex = str(candidate_key_hex)
    if (
        re.fullmatch(r"[0-9a-f]{64}", candidate_key_hex) is None
        or len(bytes.fromhex(candidate_key_hex)) != 32
    ):
        raise common.ContractError(
            "Candidate integrity key encoding drift"
        )
    for source in audits:
        row = dict(source)
        key = (str(row["world_uid"]), str(row["canonical_pair_uid"]))
        if (
            key in audit_index
            or key not in pair_index
            or row["primary_trigger"] not in priority
            or HASH_PATTERN.fullmatch(str(row["hmac_digest_hex"])) is None
            or str(row["hmac_digest_hex"])
            != common.hmac_digest(
                candidate_key_hex,
                str(row["world_uid"]),
                str(row["canonical_pair_uid"]),
            ).hex()
            or str(row["selected_bool"]) not in {"true", "false"}
            or str(row["structural_support_flag"]) not in {"true", "false"}
            or re.fullmatch(r"-?\d+\.\d{6}", str(row["lexical_similarity"]))
            is None
        ):
            raise common.ContractError(
                "Candidate integrity sampling-audit value drift"
            )
        flags = str(row["trigger_flags"]).split("|") if row["trigger_flags"] else []
        expected_flag_order = [name for name in reachable if name in flags]
        if (
            flags != expected_flag_order
            or len(flags) != len(set(flags))
            or (
                row["primary_trigger"]
                != (flags[0] if flags else fallback)
            )
            or type(row["layer_size"]) is not int
            or type(row["layer_quota"]) is not int
            or int(row["layer_size"]) < 0
            or not 0 <= int(row["layer_quota"]) <= int(row["layer_size"])
        ):
            raise common.ContractError(
                "Candidate integrity trigger lineage drift"
            )
        probability = str(row["design_inclusion_probability"])
        expected_probability = (
            f"{int(row['layer_quota']) / int(row['layer_size']):.12f}"
            if int(row["layer_size"])
            else ""
        )
        if probability != expected_probability:
            raise common.ContractError(
                "Candidate integrity inclusion-probability drift"
            )
        selected = str(row["selected_bool"]) == "true"
        rank = str(row["selected_rank"])
        if (selected and re.fullmatch(r"[1-9]|[1-3][0-9]|40", rank) is None) or (
            not selected and rank
        ):
            raise common.ContractError(
                "Candidate integrity selected-rank drift"
            )
        audit_index[key] = row

    per_world_lineage: list[dict[str, Any]] = []
    for world_uid in world_uids:
        world_pair_keys = {
            key for key in pair_index if key[0] == world_uid
        }
        world_candidate_keys = {
            key for key in candidate_index if key[0] == world_uid
        }
        world_audit = [
            row
            for key, row in audit_index.items()
            if key[0] == world_uid
        ]
        if (
            len(world_pair_keys) != 378
            or len(world_candidate_keys) != 40
            or len(world_audit) != 378
            or {key for key in audit_index if key[0] == world_uid}
            != world_pair_keys
        ):
            raise common.ContractError(
                "Candidate integrity per-world cardinality drift"
            )
        layer_sizes = Counter(
            str(row["primary_trigger"]) for row in world_audit
        )
        production_count = sum(layer_sizes[name] for name in reachable)
        if production_count >= 40:
            expected_quotas = _hamilton_quotas(
                layer_sizes,
                total_slots=40,
                trigger_priority=reachable,
            )
            expected_quotas[fallback] = 0
        else:
            expected_quotas = {
                name: layer_sizes[name] for name in reachable
            }
            expected_quotas[fallback] = 40 - production_count
        selected_keys: set[tuple[str, str]] = set()
        for name in priority:
            layer_rows = [
                row
                for row in world_audit
                if str(row["primary_trigger"]) == name
            ]
            if any(
                int(row["layer_size"]) != layer_sizes[name]
                or int(row["layer_quota"]) != expected_quotas[name]
                for row in layer_rows
            ):
                raise common.ContractError(
                    "Candidate integrity layer size/quota drift"
                )
            expected_selected = {
                (
                    str(row["world_uid"]),
                    str(row["canonical_pair_uid"]),
                )
                for row in sorted(
                    layer_rows,
                    key=lambda row: (
                        bytes.fromhex(str(row["hmac_digest_hex"])),
                        str(row["canonical_pair_uid"]).encode("utf-8"),
                    ),
                )[: expected_quotas[name]]
            }
            observed_selected = {
                (
                    str(row["world_uid"]),
                    str(row["canonical_pair_uid"]),
                )
                for row in layer_rows
                if str(row["selected_bool"]) == "true"
            }
            if expected_selected != observed_selected:
                raise common.ContractError(
                    "Candidate integrity within-layer HMAC selection drift"
                )
            selected_keys.update(observed_selected)
        ranks = {
            (
                str(row["world_uid"]),
                str(row["canonical_pair_uid"]),
            ): int(row["selected_rank"])
            for row in world_audit
            if str(row["selected_bool"]) == "true"
        }
        ordered_candidate_keys = [
            (str(row["world_uid"]), str(row["canonical_pair_uid"]))
            for row in candidates
            if str(row["world_uid"]) == world_uid
        ]
        expected_global_order = sorted(
            world_candidate_keys,
            key=lambda key: (
                common.hmac_digest(
                    candidate_key_hex,
                    world_uid,
                    "selected_global_rank",
                    key[1],
                ),
                key[1].encode("utf-8"),
            ),
        )
        expected_ranks = {
            key: rank
            for rank, key in enumerate(expected_global_order, start=1)
        }
        if (
            selected_keys != world_candidate_keys
            or set(ranks) != world_candidate_keys
            or sorted(ranks.values()) != list(range(1, 41))
            or ranks != expected_ranks
            or ordered_candidate_keys != expected_global_order
        ):
            raise common.ContractError(
                "Candidate integrity selected projection/rank drift"
            )
        per_world_lineage.append(
            {
                "world_uid": world_uid,
                "complete_pair_count": len(world_pair_keys),
                "candidate_pair_count": len(world_candidate_keys),
                "sampling_audit_count": len(world_audit),
                "selected_rank_set_exact": True,
                "layer_quota_sum": sum(expected_quotas.values()),
                "selected_keyset_exact": True,
            }
        )

    replay_lineage_rows: list[dict[str, Any]] = []
    consumed_sellers: list[dict[str, Any]] = []
    consumed_items: list[dict[str, Any]] = []
    for world_uid in world_uids:
        world_sellers = _rows_for_world(
            sellers, world_uid=world_uid
        )
        world_items = _rows_for_world(
            raw_observed_items, world_uid=world_uid
        )
        world_pairs = _rows_for_world(
            pairs, world_uid=world_uid
        )
        world_candidates = _rows_for_world(
            candidates, world_uid=world_uid
        )
        world_audits = _rows_for_world(
            audits, world_uid=world_uid
        )
        replayed_candidates, replayed_audits, replay_audit = (
            candidate_sampling.build_world_c40(
                candidate_policy,
                candidate_key_hex=candidate_key_hex,
                mode=mode,
                split=split,
                sellers=world_sellers,
                raw_observed_items=world_items,
                complete_pair_endpoints=world_pairs,
            )
        )
        independent_trigger_audit = (
            _independently_validate_candidate_trigger_projection(
                candidate_policy,
                mode=mode,
                split=split,
                sellers=world_sellers,
                raw_observed_items=world_items,
                complete_pair_endpoints=world_pairs,
                candidate_sampling_audit=world_audits,
            )
        )
        if (
            replayed_candidates != world_candidates
            or replayed_audits != world_audits
            or replay_audit[
                "labels_or_oracle_or_model_scores_read"
            ]
            is not False
            or replay_audit[
                "ephemeral_step4_raw_evidence_persisted"
            ]
            is not False
            or independent_trigger_audit[
                "ephemeral_raw_profile_sha256"
            ]
            != replay_audit["ephemeral_raw_profile_sha256"]
            or independent_trigger_audit[
                "step3_profile_audit_sha256"
            ]
            != replay_audit["step3_profile_audit_sha256"]
        ):
            raise common.ContractError(
                "Candidate integrity frozen Step3/Step4 replay drift"
            )
        consumed_sellers.extend(world_sellers)
        consumed_items.extend(world_items)
        replay_lineage_rows.append(
            {
                "world_uid": world_uid,
                "raw_observed_item_input_sha256": str(
                    replay_audit["raw_observed_item_input_sha256"]
                ),
                "ephemeral_raw_profile_sha256": str(
                    replay_audit["ephemeral_raw_profile_sha256"]
                ),
                "step3_profile_audit_sha256": str(
                    replay_audit["step3_profile_audit_sha256"]
                ),
                "candidate_safe_projection_sha256": str(
                    replay_audit["candidate_safe_projection_sha256"]
                ),
                "candidate_sampling_audit_sha256": str(
                    replay_audit["candidate_sampling_audit_sha256"]
                ),
                "frozen_step4_candidate_rows_sha256": str(
                    independent_trigger_audit[
                        "frozen_step4_candidate_rows_sha256"
                    ]
                ),
            }
        )
    if (
        len(consumed_sellers) != len(sellers)
        or canonical_multiset_sha256(consumed_sellers)
        != canonical_multiset_sha256(sellers)
        or len(consumed_items) != len(raw_observed_items)
        or canonical_multiset_sha256(consumed_items)
        != canonical_multiset_sha256(raw_observed_items)
    ):
        raise common.ContractError(
            "Candidate integrity raw input contains an unconsumed row"
        )

    fixed_counts = {
        "split_world_count": len(world_uids),
        "split_complete_pair_count": len(pair_index),
        "split_candidate_pair_count": len(candidate_index),
        "duplicate_candidate_count": len(candidates) - len(candidate_index),
        "foreign_key_mismatch_count": 0,
    }
    fixed_gates = {
        "complete_pair_universe_exact": len(pair_index)
        == 378 * len(world_uids),
        "candidate_schema_exact": True,
        "candidate_keys_unique": len(candidate_index) == len(candidates),
        "sampling_lineage_exact": True,
        "label_and_oracle_fields_absent_from_sealer_arguments": True,
    }
    if not all(fixed_gates.values()):
        raise common.ContractError("Candidate integrity aggregate gate failed")
    return _build_receipt_against_spec(
        role="candidate_integrity",
        spec=candidate_context["receipt_spec"],
        top_keys=candidate_context["receipt_top_level_exact_keys"],
        fixed_boolean_gates=fixed_gates,
        fixed_counts=fixed_counts,
        input_parent_hashes={
            "candidate_policy_projection_sha256": str(
                candidate_context[
                    "candidate_policy_projection_sha256"
                ]
            ),
            "candidate_key_commitment_sha256": common.sha256_bytes(
                bytes.fromhex(candidate_key_hex)
            ),
            "observed_seller_parent_sha256": canonical_multiset_sha256(
                sellers
            ),
            "raw_observed_item_parent_sha256": (
                canonical_multiset_sha256(raw_observed_items)
            ),
            "complete_pair_parent_sha256": canonical_multiset_sha256(pairs),
            "candidate_parent_sha256": canonical_multiset_sha256(candidates),
            "sampling_audit_parent_sha256": canonical_multiset_sha256(audits),
            "policy_parent_sha256": str(
                candidate_context["policy_parent_sha256"]
            ),
            "registered_split_scope_sha256": str(
                candidate_context["registered_split_scope_sha256"]
            ),
        },
        aggregate_content_hashes={
            "complete_pair_multiset_sha256": canonical_multiset_sha256(
                pairs
            ),
            "candidate_pair_multiset_sha256": canonical_multiset_sha256(
                candidates
            ),
            "sampling_lineage_multiset_sha256": (
                canonical_multiset_sha256(
                    [
                        {
                            **row,
                            **replay_lineage_rows[index],
                        }
                        for index, row in enumerate(per_world_lineage)
                    ]
                )
            ),
        },
    )


def _registered_seed_ids(
    policy: Mapping[str, Any], *, mode: str
) -> list[str]:
    return [
        "rws_" + hashlib.sha256(bytes.fromhex(seed_hex)).hexdigest()
        for seed_hex in policy["randomness"][mode]["rewire_key_hexes"]
    ]


def build_m1_derangement_integrity_receipt(
    policy: Mapping[str, Any],
    *,
    mode: str,
    split: str,
    m2_identity33_all_pairs: Sequence[Mapping[str, Any]],
    candidate_pairs: Sequence[Mapping[str, Any]],
    complete_pair_endpoints: Sequence[Mapping[str, Any]],
    placebos: Sequence[Mapping[str, Any]],
    support_preflight: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay all five mappings and the support comparator before sealing M1."""

    import step28_v13_feature_derangement as feature_derangement
    import step28_v13_placebo_support as placebo_support

    common.validate_policy(policy, mode=mode)
    if mode != "development_smoke" or split != "train":
        raise common.ContractError(
            "M1 derangement receipt is development-smoke train only"
        )
    feature_names = list(policy["history_features"]["feature_names"])
    matrix_schema = ["canonical_pair_uid", "world_uid", *feature_names]
    mapping_schema = policy["placebo"][
        "feature_derangement_mapping_schema"
    ]
    pair_schema = policy["relational_integrity"][
        "pair_projection_contract"
    ]["complete_model_pair_endpoints_schema"]
    candidate_schema = policy["candidate_design"][
        "public_safe_projection_columns"
    ]
    if (
        any(list(row) != matrix_schema for row in m2_identity33_all_pairs)
        or any(list(row) != pair_schema for row in complete_pair_endpoints)
        or any(list(row) != candidate_schema for row in candidate_pairs)
    ):
        raise common.ContractError(
            "M1 integrity base input schema/order drift"
        )
    m2_index = {
        (str(row["world_uid"]), str(row["canonical_pair_uid"])): dict(row)
        for row in m2_identity33_all_pairs
    }
    endpoint_index = {
        (str(row["world_uid"]), str(row["canonical_pair_uid"])): (
            str(row["seller_uid_left"]),
            str(row["seller_uid_right"]),
        )
        for row in complete_pair_endpoints
    }
    candidate_keys = {
        (str(row["world_uid"]), str(row["canonical_pair_uid"]))
        for row in candidate_pairs
    }
    if (
        not m2_index
        or len(m2_index) != len(m2_identity33_all_pairs)
        or len(endpoint_index) != len(complete_pair_endpoints)
        or set(m2_index) != set(endpoint_index)
        or len(candidate_keys) != len(candidate_pairs)
        or not candidate_keys.issubset(endpoint_index)
    ):
        raise common.ContractError("M1 integrity base keyset drift")
    expected_seed_ids = _registered_seed_ids(policy, mode=mode)
    by_seed = {str(row.get("rewire_seed_id", "")): row for row in placebos}
    if (
        len(placebos) != 5
        or len(by_seed) != 5
        or set(by_seed) != set(expected_seed_ids)
    ):
        raise common.ContractError(
            "M1 integrity registered replicate set drift"
        )

    all_m1_rows: list[dict[str, Any]] = []
    all_mapping_rows: list[dict[str, Any]] = []
    fixed_point_count = 0
    endpoint_overlap_count = 0
    multiset_mismatch_count = 0
    for seed_hex, seed_id in zip(
        policy["randomness"][mode]["rewire_key_hexes"],
        expected_seed_ids,
        strict=True,
    ):
        output = by_seed[seed_id]
        required_output_keys = {
            "rewire_seed_id",
            "identity33_all_pairs",
            "feature_derangement_mapping",
            "joint_vector_multiset_exact_by_world_and_universe",
            "endpoint_disjoint_bijection_exact",
            "labels_or_controller_inputs_read",
            "candidate_trigger_or_audit_inputs_read",
            "canonical_self_hash",
        }
        if (
            set(output) != required_output_keys
            or output["joint_vector_multiset_exact_by_world_and_universe"]
            is not True
            or output["endpoint_disjoint_bijection_exact"] is not True
            or output["labels_or_controller_inputs_read"] is not False
            or output["candidate_trigger_or_audit_inputs_read"] is not False
            or output["canonical_self_hash"]
            != common.canonical_sha256(
                {
                    key: value
                    for key, value in output.items()
                    if key != "canonical_self_hash"
                }
            )
        ):
            raise common.ContractError(
                "M1 integrity source result envelope drift"
            )
        m1_rows = output["identity33_all_pairs"]
        mapping_rows = output["feature_derangement_mapping"]
        if (
            len(m1_rows) != 3780
            or len(mapping_rows) != 3780
            or any(list(row) != matrix_schema for row in m1_rows)
            or any(list(row) != mapping_schema for row in mapping_rows)
        ):
            raise common.ContractError(
                "M1 integrity matrix/mapping schema or count drift"
            )
        m1_index = {
            (str(row["world_uid"]), str(row["canonical_pair_uid"])): dict(row)
            for row in m1_rows
        }
        if len(m1_index) != len(m1_rows) or set(m1_index) != set(m2_index):
            raise common.ContractError("M1 integrity output keyset drift")
        mapping_index: dict[tuple[str, str], dict[str, Any]] = {}
        source_keys_by_stratum: dict[
            tuple[str, str], set[tuple[str, str]]
        ] = defaultdict(set)
        destination_keys_by_stratum: dict[
            tuple[str, str], set[tuple[str, str]]
        ] = defaultdict(set)
        for source in mapping_rows:
            row = dict(source)
            world_uid = str(row["world_uid"])
            destination_uid = str(row["destination_pair_uid"])
            source_uid = str(row["source_pair_uid"])
            universe = str(row["universe"])
            destination_key = (world_uid, destination_uid)
            source_key = (world_uid, source_uid)
            stratum = (world_uid, universe)
            expected_universe = (
                "primary_c40"
                if destination_key in candidate_keys
                else "secondary_complement"
            )
            if (
                str(row["rewire_seed_id"]) != seed_id
                or universe != expected_universe
                or destination_key in mapping_index
                or destination_key not in m2_index
                or source_key not in m2_index
                or (
                    (source_key in candidate_keys)
                    != (destination_key in candidate_keys)
                )
                or type(row["endpoint_disjoint_bool"]) is not bool
                or row["endpoint_disjoint_bool"] is not True
            ):
                raise common.ContractError(
                    "M1 integrity mapping lineage drift"
                )
            overlap = bool(
                set(endpoint_index[destination_key]).intersection(
                    endpoint_index[source_key]
                )
            )
            endpoint_overlap_count += int(overlap)
            fixed_point_count += int(destination_key == source_key)
            source_vector = [
                str(m2_index[source_key][name]) for name in feature_names
            ]
            destination_vector = [
                str(m1_index[destination_key][name])
                for name in feature_names
            ]
            if (
                overlap
                or destination_key == source_key
                or source_vector != destination_vector
                or str(row["feature_vector_sha256"])
                != common.canonical_sha256(source_vector)
            ):
                raise common.ContractError(
                    "M1 integrity endpoint/vector replay mismatch"
                )
            mapping_index[destination_key] = row
            source_keys_by_stratum[stratum].add(source_key)
            destination_keys_by_stratum[stratum].add(destination_key)
        if len(mapping_index) != 3780:
            raise common.ContractError(
                "M1 integrity destination mapping is not bijective"
            )
        for stratum, destinations in destination_keys_by_stratum.items():
            sources = source_keys_by_stratum[stratum]
            expected_count = (
                40 if stratum[1] == "primary_c40" else 338
            )
            if (
                len(destinations) != expected_count
                or sources != destinations
            ):
                raise common.ContractError(
                    "M1 integrity stratum source/destination bijection drift"
                )
            original_vectors = Counter(
                tuple(
                    str(m2_index[key][name]) for name in feature_names
                )
                for key in destinations
            )
            observed_vectors = Counter(
                tuple(
                    str(m1_index[key][name]) for name in feature_names
                )
                for key in destinations
            )
            if original_vectors != observed_vectors:
                multiset_mismatch_count += 1
        replayed = feature_derangement.build_one_feature_derangement(
            policy,
            mode=mode,
            split=split,
            seed_hex=str(seed_hex),
            m2_identity33_all_pairs=m2_identity33_all_pairs,
            candidate_pairs=candidate_pairs,
            complete_pair_endpoints=complete_pair_endpoints,
        )
        if replayed != output:
            raise common.ContractError(
                "M1 integrity deterministic replay mismatch"
            )
        all_m1_rows.extend(
            {"rewire_seed_id": seed_id, **dict(row)} for row in m1_rows
        )
        all_mapping_rows.extend(dict(row) for row in mapping_rows)

    recomputed_support = placebo_support.run_support_comparability_preflight(
        policy,
        mode=mode,
        split=split,
        m2_identity33_all_pairs=m2_identity33_all_pairs,
        candidate_pairs=candidate_pairs,
        complete_pair_endpoints=complete_pair_endpoints,
        placebos=placebos,
    )
    if dict(support_preflight) != recomputed_support:
        raise common.ContractError(
            "M1 integrity support-preflight deterministic replay mismatch"
        )
    support_failure_count = sum(
        int(row["primary_validity_pass"] is not True)
        for row in recomputed_support["seed_results"]
    )
    fixed_gates = {
        "input_universe_exact": len(m2_index) == 3780,
        "destination_source_bijection_exact": True,
        "endpoint_disjoint_exact": endpoint_overlap_count == 0,
        "joint_vector_multiset_exact": multiset_mismatch_count == 0,
        (
            "labels_controller_and_candidate_evidence_fields_"
            "absent_from_sealer_arguments"
        ): True,
        "deterministic_replay_exact": True,
        "support_comparability_pass": support_failure_count == 0,
    }
    if not all(fixed_gates.values()) or fixed_point_count:
        raise common.ContractError(
            "M1 derangement integrity aggregate gate failed"
        )
    return _build_receipt(
        policy,
        role="m1_derangement_integrity",
        fixed_boolean_gates=fixed_gates,
        fixed_counts={
            "split_world_count": 10,
            "m1_replicate_count": len(placebos),
            "m2_identity33_row_count": len(m2_identity33_all_pairs),
            "m1_identity33_row_count": len(all_m1_rows),
            "mapping_row_count": len(all_mapping_rows),
            "fixed_point_count": fixed_point_count,
            "endpoint_overlap_count": endpoint_overlap_count,
            "multiset_mismatch_count": multiset_mismatch_count,
            "support_failure_count": support_failure_count,
        },
        input_parent_hashes={
            "m2_identity33_parent_sha256": canonical_multiset_sha256(
                m2_identity33_all_pairs
            ),
            "complete_pair_parent_sha256": canonical_multiset_sha256(
                complete_pair_endpoints
            ),
            "candidate_endpoint_parent_sha256": (
                canonical_multiset_sha256(candidate_pairs)
            ),
            "m1_output_parent_sha256": canonical_multiset_sha256(
                all_m1_rows
            ),
            "derangement_mapping_parent_sha256": (
                canonical_multiset_sha256(all_mapping_rows)
            ),
            "policy_parent_sha256": _policy_parent_sha256(policy),
            "registered_split_scope_sha256": (
                _registered_split_scope_sha256(
                    policy, mode=mode, split=split
                )
            ),
        },
        aggregate_content_hashes={
            "m2_identity33_multiset_sha256": canonical_multiset_sha256(
                m2_identity33_all_pairs
            ),
            "m1_identity33_multiset_sha256": canonical_multiset_sha256(
                all_m1_rows
            ),
            "derangement_mapping_multiset_sha256": (
                canonical_multiset_sha256(all_mapping_rows)
            ),
            "support_preflight_sha256": common.canonical_sha256(
                recomputed_support
            ),
        },
    )


def _validate_comparison_receipt(
    receipt: Mapping[str, Any],
    *,
    mode: str,
    split: str,
) -> None:
    import step28_v13_independent_dgp_comparator as dgp_comparator

    if set(receipt) != COMPARISON_RECEIPT_KEYS:
        raise common.ContractError(
            "Independent DGP comparison receipt schema drift"
        )
    if (
        receipt["version"]
        != "2026-07-28-step28-v13-independent-comparison-v2-draft"
        or receipt["mode"] != mode
        or receipt["split"] != split
        or receipt["scope"] != dgp_comparator.REPLAY_SCOPE
        or receipt["evidence_level"] != dgp_comparator.EVIDENCE_LEVEL
        or receipt["independent_typed_dgp_replay_pass"] is not True
        or receipt["independent_decision_implementation"] is not True
        or receipt["formal_custody_seal"] is not False
        or receipt["producer_private_input_used_by_replayer"] is not False
        or receipt["structure_key_serialized"] is not False
        or receipt["full_typed_projection_exact"] is not True
    ):
        raise common.ContractError(
            "Independent DGP comparison receipt envelope drift"
        )
    for key in (
        "producer_typed_projection_sha256",
        "replayer_typed_projection_sha256",
    ):
        _validate_hash(receipt[key], label=f"DGP comparison {key}")
    if (
        receipt["producer_typed_projection_sha256"]
        != receipt["replayer_typed_projection_sha256"]
    ):
        raise common.ContractError(
            "Independent DGP typed projection hash mismatch"
        )
    uid_audit = receipt["observed_uid_pool_audit"]
    if not isinstance(uid_audit, Mapping) or set(uid_audit) != OBSERVED_UID_AUDIT_KEYS:
        raise common.ContractError(
            "Independent DGP observed UID audit schema drift"
        )
    for key, value in uid_audit.items():
        if key.endswith("_sha256"):
            _validate_hash(value, label=f"DGP UID audit {key}")
        elif type(value) is not int or value < 0:
            raise common.ContractError(
                "Independent DGP observed UID audit count drift"
            )
    components = receipt["component_receipts"]
    if not isinstance(components, Mapping) or set(components) != set(
        COMPONENT_NAMES
    ):
        raise common.ContractError(
            "Independent DGP component set drift"
        )
    for name, component in components.items():
        if (
            not isinstance(component, Mapping)
            or set(component) != COMPONENT_RECEIPT_KEYS
            or type(component["producer_row_count"]) is not int
            or type(component["replayer_row_count"]) is not int
            or int(component["producer_row_count"]) < 0
            or int(component["replayer_row_count"]) < 0
            or component["producer_row_count"]
            != component["replayer_row_count"]
            or component["exact_equal"] is not True
        ):
            raise common.ContractError(
                f"Independent DGP component receipt drift: {name}"
            )
        _validate_hash(
            component["producer_sha256"],
            label=f"DGP component {name} producer",
        )
        _validate_hash(
            component["replayer_sha256"],
            label=f"DGP component {name} replayer",
        )
        if component["producer_sha256"] != component["replayer_sha256"]:
            raise common.ContractError(
                f"Independent DGP component hash mismatch: {name}"
            )


def build_independent_dgp_comparison_receipt(
    policy: Mapping[str, Any],
    *,
    mode: str,
    split: str,
    worlds: Sequence[Mapping[str, Any]],
    per_world_comparison_receipts: Sequence[Mapping[str, Any]],
    producer_typed_dgp_projections: Sequence[Mapping[str, Any]],
    independent_replay_ledgers: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate replay ledgers and comparisons into a fixed scalar receipt."""

    import step28_v13_independent_dgp_comparator as dgp_comparator

    common.validate_policy(policy, mode=mode)
    world_uids = _expected_world_uids(
        policy,
        mode=mode,
        split=split,
        worlds=worlds,
    )
    audit_by_world: dict[str, Mapping[str, Any]] = {}
    for comparison in per_world_comparison_receipts:
        if not isinstance(comparison, Mapping):
            raise common.ContractError(
                "Independent DGP comparison receipt is not a mapping"
            )
        world_uid = str(comparison.get("world_uid", ""))
        if (
            world_uid in audit_by_world
            or world_uid not in world_uids
        ):
            raise common.ContractError(
                "Independent DGP world comparison audit set drift"
            )
        _validate_comparison_receipt(
            comparison, mode=mode, split=split
        )
        if str(comparison["world_uid"]) != world_uid:
            raise common.ContractError(
                "Independent DGP comparison world lineage drift"
            )
        audit_by_world[world_uid] = comparison
    if set(audit_by_world) != set(world_uids):
        raise common.ContractError(
            "Independent DGP comparison complete world set drift"
        )
    expected_graph = str(
        policy["identity_design"]["mechanism_by_split"][split]
    )
    replay_by_world: dict[str, Mapping[str, Any]] = {}
    for ledger in independent_replay_ledgers:
        if not isinstance(ledger, Mapping):
            raise common.ContractError(
                "Independent DGP replay ledger is not a mapping"
            )
        world_uid = str(ledger.get("world_uid", ""))
        if (
            world_uid in replay_by_world
            or world_uid not in world_uids
        ):
            raise common.ContractError(
                "Independent DGP replay ledger world set drift"
            )
        try:
            replay_tables = dgp_comparator._validate_replay_envelope(
                ledger
            )
        except dgp_comparator.IndependentDgpComparisonError as exc:
            raise common.ContractError(
                "Independent DGP replay ledger envelope drift"
            ) from exc
        comparison = audit_by_world[world_uid]
        if (
            ledger["mode"] != mode
            or ledger["split"] != split
            or ledger["world_uid"] != world_uid
            or ledger["graph_name"] != expected_graph
            or ledger["typed_replay_sha256"]
            != comparison["replayer_typed_projection_sha256"]
            or dict(ledger["observed_uid_pool_audit"])
            != dict(comparison["observed_uid_pool_audit"])
        ):
            raise common.ContractError(
                "Independent DGP replay/comparison lineage drift"
            )
        for component_name in COMPONENT_NAMES:
            value = replay_tables[component_name]
            row_count = 1 if component_name == "solver_trace" else len(value)
            component = comparison["component_receipts"][
                component_name
            ]
            if (
                int(component["replayer_row_count"]) != row_count
                or component["replayer_sha256"]
                != common.canonical_sha256(value)
            ):
                raise common.ContractError(
                    "Independent DGP replay component receipt drift: "
                    f"{component_name}"
                )
        replay_by_world[world_uid] = ledger
    if set(replay_by_world) != set(world_uids):
        raise common.ContractError(
            "Independent DGP replay ledger set is incomplete"
        )
    producer_projection_keys = {
        "version",
        "scope",
        "mode",
        "split",
        "world_uid",
        "graph_name",
        "tables",
        "typed_projection_sha256",
        "canonical_self_hash",
    }
    projection_by_world: dict[str, Mapping[str, Any]] = {}
    for projection in producer_typed_dgp_projections:
        if not isinstance(projection, Mapping):
            raise common.ContractError(
                "Independent DGP producer projection is not a mapping"
            )
        world_uid = str(projection.get("world_uid", ""))
        if (
            set(projection) != producer_projection_keys
            or world_uid in projection_by_world
            or world_uid not in world_uids
            or projection["version"]
            != dgp_comparator.PRODUCER_PROJECTION_VERSION
            or projection["scope"] != dgp_comparator.REPLAY_SCOPE
            or projection["mode"] != mode
            or projection["split"] != split
            or projection["graph_name"] != expected_graph
        ):
            raise common.ContractError(
                "Independent DGP producer projection envelope drift"
            )
        expected_self_hash = common.canonical_sha256(
            {
                key: value
                for key, value in projection.items()
                if key != "canonical_self_hash"
            }
        )
        if projection["canonical_self_hash"] != expected_self_hash:
            raise common.ContractError(
                "Independent DGP producer projection self-hash drift"
            )
        try:
            producer_tables = dgp_comparator._validate_tables(
                projection["tables"],
                label="AGGREGATE_PRODUCER",
            )
        except dgp_comparator.IndependentDgpComparisonError as exc:
            raise common.ContractError(
                "Independent DGP producer table schema drift"
            ) from exc
        typed_hash = common.canonical_sha256(producer_tables)
        comparison = audit_by_world[world_uid]
        if (
            projection["typed_projection_sha256"] != typed_hash
            or comparison["producer_typed_projection_sha256"]
            != typed_hash
        ):
            raise common.ContractError(
                "Independent DGP producer/comparison typed hash drift"
            )
        for component_name in COMPONENT_NAMES:
            value = producer_tables[component_name]
            row_count = 1 if component_name == "solver_trace" else len(value)
            component = comparison["component_receipts"][
                component_name
            ]
            if (
                int(component["producer_row_count"]) != row_count
                or component["producer_sha256"]
                != common.canonical_sha256(value)
            ):
                raise common.ContractError(
                    "Independent DGP producer component receipt drift: "
                    f"{component_name}"
                )
        projection_by_world[world_uid] = projection
    if set(projection_by_world) != set(world_uids):
        raise common.ContractError(
            "Independent DGP producer projection world set drift"
        )
    per_world_receipts = [audit_by_world[uid] for uid in world_uids]
    component_rows: list[dict[str, Any]] = []
    observed_uid_rows: list[dict[str, Any]] = []
    producer_total = 0
    replayer_total = 0
    component_mismatch_count = 0
    for receipt in per_world_receipts:
        world_uid = str(receipt["world_uid"])
        observed_uid_rows.append(
            {
                "world_uid": world_uid,
                **dict(receipt["observed_uid_pool_audit"]),
            }
        )
        for component_name in COMPONENT_NAMES:
            component = receipt["component_receipts"][component_name]
            producer_total += int(component["producer_row_count"])
            replayer_total += int(component["replayer_row_count"])
            mismatch = (
                component["exact_equal"] is not True
                or component["producer_row_count"]
                != component["replayer_row_count"]
                or component["producer_sha256"]
                != component["replayer_sha256"]
            )
            component_mismatch_count += int(mismatch)
            component_rows.append(
                {
                    "world_uid": world_uid,
                    "component_name": component_name,
                    **dict(component),
                }
            )
    per_world_replay_ledgers = [
        replay_by_world[uid] for uid in world_uids
    ]
    fixed_gates = {
        "complete_registered_world_set_exact": True,
        "all_worlds_exact": all(
            row["full_typed_projection_exact"] is True
            for row in per_world_receipts
        ),
        "independent_decision_implementation": all(
            row["independent_decision_implementation"] is True
            for row in per_world_receipts
        ),
        "producer_private_input_argument_used_by_replayer_absent": all(
            row["producer_private_input_used_by_replayer"] is False
            for row in per_world_receipts
        ),
        "structure_key_serialized_in_comparison_receipt_absent": all(
            row["structure_key_serialized"] is False
            for row in per_world_receipts
        ),
        "component_counts_and_hashes_exact": (
            component_mismatch_count == 0
        ),
    }
    if not all(fixed_gates.values()):
        raise common.ContractError(
            "Independent DGP comparison aggregate gate failed"
        )
    return _build_receipt(
        policy,
        role="independent_dgp_comparison",
        fixed_boolean_gates=fixed_gates,
        fixed_counts={
            "split_world_count": len(world_uids),
            "component_count": len(COMPONENT_NAMES),
            "producer_component_row_count": producer_total,
            "replayer_component_row_count": replayer_total,
            "component_mismatch_count": component_mismatch_count,
        },
        input_parent_hashes={
            "per_world_comparison_parent_sha256": (
                canonical_multiset_sha256(per_world_receipts)
            ),
            "producer_projection_parent_sha256": (
                canonical_multiset_sha256(
                    producer_typed_dgp_projections
                )
            ),
            "independent_replay_parent_sha256": (
                canonical_multiset_sha256(per_world_replay_ledgers)
            ),
            "policy_parent_sha256": _policy_parent_sha256(policy),
            "registered_split_scope_sha256": (
                _registered_split_scope_sha256(
                    policy, mode=mode, split=split
                )
            ),
        },
        aggregate_content_hashes={
            "per_world_comparison_multiset_sha256": (
                canonical_multiset_sha256(per_world_receipts)
            ),
            "component_summary_multiset_sha256": (
                canonical_multiset_sha256(component_rows)
            ),
            "observed_uid_pool_audit_multiset_sha256": (
                canonical_multiset_sha256(observed_uid_rows)
            ),
        },
    )


def expected_receipt_roles(
    policy: Mapping[str, Any], *, split: str
) -> list[str]:
    if split not in SPLITS:
        raise common.ContractError(
            "Structural audit split is outside the registered set"
        )
    schema = _load_deployment(policy)[
        "structural_auditor_parent_seal_projection_schema"
    ]
    roles = list(schema["role_exact_values_by_split"][split])
    expected_count = int(schema["exact_record_count_by_split"][split])
    if (
        len(roles) != expected_count
        or len(set(roles)) != expected_count
        or roles != sorted(roles, key=lambda value: value.encode("utf-8"))
    ):
        raise common.ContractError(
            "Structural audit registered role set/order drift"
        )
    return roles


SOURCE_CLOSURE_ENTRY_PATHS = {
    "training_ready_dataset_builder": (
        "scripts/step28_v13_build_training_ready_dataset.py",
    ),
    "render_integrity": (
        "scripts/step28_v13_integrity_receipts.py",
        "scripts/step28_v13_production_chain.py",
    ),
    "candidate_integrity": (
        "scripts/step28_v13_integrity_receipts.py",
        "scripts/step28_v13_candidate_sampling.py",
        "scripts/step28_v13_profiles.py",
        "scripts/step3_build_seller_profiles.py",
        "scripts/step4_build_silver_candidates.py",
    ),
    "m1_derangement_integrity": (
        "scripts/step28_v13_integrity_receipts.py",
        "scripts/step28_v13_feature_derangement.py",
        "scripts/step28_v13_placebo_support.py",
    ),
    "independent_dgp_comparison": (
        "scripts/step28_v13_integrity_receipts.py",
        "scripts/step28_v13_independent_dgp_comparator.py",
        "scripts/step28_v13_independent_private_dgp_replay.py",
        "scripts/step28_v13_producer_dgp_projection.py",
    ),
}
SOURCE_CLOSURE_DATA_PATHS = {
    "training_ready_dataset_builder": (),
    "render_integrity": (
        "schema/step28_v13_dataset_custody_deployment.json",
    ),
    "candidate_integrity": (
        "schema/step28_v13_dataset_custody_deployment.json",
        "schema/step3_seller_profile_schema.json",
        "schema/step4_silver_candidate_schema.json",
    ),
    "m1_derangement_integrity": (
        "schema/step28_v13_dataset_custody_deployment.json",
    ),
    "independent_dgp_comparison": (
        "schema/step28_v13_dataset_custody_deployment.json",
    ),
}


def _repo_local_module_path(module_name: str) -> str | None:
    """Resolve one absolute Python import to a repository-local source."""

    if (
        not module_name
        or module_name.startswith(".")
        or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_.]*", module_name
        )
    ):
        return None
    candidate = (
        common.ROOT
        / "scripts"
        / Path(*module_name.split("."))
    ).with_suffix(".py")
    if not candidate.is_file():
        return None
    resolved = candidate.resolve()
    scripts_root = (common.ROOT / "scripts").resolve()
    if not resolved.is_relative_to(scripts_root):
        raise common.ContractError(
            "Source closure import escaped the scripts root"
        )
    return resolved.relative_to(common.ROOT.resolve()).as_posix()


def _repo_local_import_paths(relative_path: str) -> set[str]:
    """Parse all static and literal dynamic repo-local imports."""

    path = common.ROOT / Path(relative_path)
    try:
        tree = ast.parse(
            path.read_text(encoding="utf-8-sig"),
            filename=relative_path,
        )
    except (OSError, SyntaxError, UnicodeError) as exc:
        raise common.ContractError(
            f"Source closure cannot parse {relative_path}"
        ) from exc
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    module_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if (
                    alias.name == "importlib"
                    and alias.asname not in {None, "importlib"}
                ):
                    raise common.ContractError(
                        "Source closure forbids aliased dynamic imports"
                    )
                if alias.name == "builtins":
                    raise common.ContractError(
                        "Source closure forbids indirect dynamic imports"
                    )
            module_names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level != 0:
                raise common.ContractError(
                    "Source closure forbids relative imports"
                )
            if (
                node.module == "importlib"
                and any(
                    alias.name == "import_module"
                    for alias in node.names
                )
            ) or (
                node.module == "builtins"
                and any(
                    alias.name == "__import__"
                    for alias in node.names
                )
            ):
                raise common.ContractError(
                    "Source closure forbids aliased dynamic imports"
                )
            if node.module:
                module_names.add(node.module)
        elif isinstance(node, ast.Attribute):
            parent = parents.get(node)
            direct_call = (
                isinstance(parent, ast.Call)
                and parent.func is node
            )
            if (
                isinstance(node.value, ast.Name)
                and node.value.id == "importlib"
            ):
                if node.attr != "import_module" or not direct_call:
                    raise common.ContractError(
                        "Source closure forbids indirect dynamic imports"
                    )
            elif node.attr in {"import_module", "__import__"}:
                raise common.ContractError(
                    "Source closure forbids indirect dynamic imports"
                )
        elif (
            isinstance(node, ast.Name)
            and node.id in {
                "__builtins__",
                "__import__",
                "importlib",
            }
        ):
            parent = parents.get(node)
            if node.id == "__builtins__":
                raise common.ContractError(
                    "Source closure forbids direct reflection primitives"
                )
            if node.id == "__import__" and not (
                isinstance(parent, ast.Call)
                and parent.func is node
            ):
                raise common.ContractError(
                    "Source closure forbids indirect dynamic imports"
                )
            if node.id == "importlib" and not (
                isinstance(parent, ast.Attribute)
                and parent.value is node
            ):
                raise common.ContractError(
                    "Source closure forbids indirect dynamic imports"
                )
        elif isinstance(node, ast.Call):
            dynamic_name: str | None = None
            dynamic_call = False
            if (
                isinstance(node.func, ast.Name)
                and node.func.id
                in {"compile", "eval", "exec", "globals", "locals"}
            ):
                raise common.ContractError(
                    "Source closure forbids direct reflection primitives"
                )
            if (
                isinstance(node.func, ast.Name)
                and node.func.id in {"getattr", "vars"}
                and node.args
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id in {"importlib", "builtins"}
            ) or (
                isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value
                in {"import_module", "__import__"}
            ):
                raise common.ContractError(
                    "Source closure forbids indirect dynamic imports"
                )
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "importlib"
                and node.func.attr == "import_module"
            ):
                dynamic_call = True
                dynamic_name = (
                    str(node.args[0].value)
                    if node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                    else None
                )
            elif (
                isinstance(node.func, ast.Name)
                and node.func.id in {"import_module", "__import__"}
            ):
                dynamic_call = True
                dynamic_name = (
                    str(node.args[0].value)
                    if node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                    else None
                )
            if dynamic_call and dynamic_name is None:
                raise common.ContractError(
                    "Source closure forbids nonliteral dynamic imports"
                )
            if dynamic_name is not None:
                if dynamic_name.startswith("."):
                    raise common.ContractError(
                        "Source closure forbids relative dynamic imports"
                    )
                if re.fullmatch(
                    r"[A-Za-z_][A-Za-z0-9_.]*", dynamic_name
                ) is None:
                    raise common.ContractError(
                        "Source closure dynamic import target is invalid"
                    )
                module_names.add(dynamic_name)
    return {
        path
        for module_name in module_names
        if (path := _repo_local_module_path(module_name)) is not None
    }


def _source_closure_members(role: str) -> tuple[str, ...]:
    """Return the recursively discovered exact repo-local role closure."""

    try:
        entries = SOURCE_CLOSURE_ENTRY_PATHS[role]
        data_paths = SOURCE_CLOSURE_DATA_PATHS[role]
    except KeyError as exc:
        raise common.ContractError(
            f"Unknown structural parent role: {role}"
        ) from exc
    members = set(data_paths)
    pending = list(entries)
    parsed: set[str] = set()
    while pending:
        relative_path = pending.pop()
        if relative_path in parsed:
            continue
        path = common.ROOT / Path(relative_path)
        if (
            path.suffix != ".py"
            or not path.is_file()
            or not path.resolve().is_relative_to(common.ROOT.resolve())
        ):
            raise common.ContractError(
                "Source closure entry is missing or outside the repository"
            )
        parsed.add(relative_path)
        members.add(relative_path)
        pending.extend(
            _repo_local_import_paths(relative_path).difference(parsed)
        )
    for relative_path in members:
        if not (common.ROOT / Path(relative_path)).is_file():
            raise common.ContractError(
                f"Source closure member is missing: {relative_path}"
            )
    return tuple(
        sorted(members, key=lambda value: value.encode("utf-8"))
    )


def _source_closure_sha256(role: str) -> str:
    return common.canonical_sha256(
        [
            {
                "repo_relative_path": name,
                "sha256": common.sha256_file(
                    common.ROOT / Path(name)
                ),
            }
            for name in _source_closure_members(role)
        ]
    )


def _development_access_summary_sha256(role: str) -> str:
    return common.canonical_sha256(
        {
            "role": role,
            "mode": "development_in_process",
            "audit_hook_installed": False,
            "external_custody_parent_verified": False,
            "formal_forbidden_open_count_attested": False,
        }
    )


def build_development_parent_projections(
    policy: Mapping[str, Any],
    *,
    split: str,
    receipts: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Create explicit self-hash parents for development wiring tests only."""

    roles = expected_receipt_roles(policy, split=split)
    if set(receipts) != set(roles):
        raise common.ContractError(
            "Development parent receipt role set drift"
        )
    projections: list[dict[str, str]] = []
    for role in roles:
        receipt = receipts[role]
        validate_aggregate_receipt(
            policy, role=role, receipt=receipt
        )
        projections.append(
            {
                "role": role,
                "file_sha256": pretty_json_sha256(receipt),
                "content_sha256": str(receipt["canonical_self_hash"]),
                "source_closure_sha256": _source_closure_sha256(role),
                "access_summary_sha256": (
                    _development_access_summary_sha256(role)
                ),
            }
        )
    return projections


def _validate_parent_projections(
    policy: Mapping[str, Any],
    *,
    split: str,
    receipts: Mapping[str, Mapping[str, Any]],
    parent_projections: Sequence[Mapping[str, Any]],
) -> None:
    schema = _load_deployment(policy)[
        "structural_auditor_parent_seal_projection_schema"
    ]
    roles = expected_receipt_roles(policy, split=split)
    if (
        len(parent_projections) != len(roles)
        or [
            str(row.get("role", "")) for row in parent_projections
        ]
        != roles
    ):
        raise common.ContractError(
            "Structural parent projection role set/order drift"
        )
    exact_keys = set(schema["record_exact_keys"])
    for row in parent_projections:
        role = str(row.get("role", ""))
        if set(row) != exact_keys or role not in receipts:
            raise common.ContractError(
                "Structural parent projection schema drift"
            )
        for key, value in row.items():
            if key != "role":
                _validate_hash(
                    value, label=f"structural parent {role}.{key}"
                )
        if (
            row["content_sha256"]
            != receipts[role]["canonical_self_hash"]
            or row["file_sha256"]
            != pretty_json_sha256(receipts[role])
            or row["source_closure_sha256"]
            != _source_closure_sha256(role)
            or row["access_summary_sha256"]
            != _development_access_summary_sha256(role)
        ):
            raise common.ContractError(
                "Structural parent projection receipt hash mismatch"
            )


def build_structural_audit_receipt(
    policy: Mapping[str, Any],
    *,
    mode: str,
    split: str,
    receipts: Mapping[str, Mapping[str, Any]],
    parent_projections: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Audit only fixed receipts and parent projections, never raw tables."""

    common.validate_policy(policy, mode=mode)
    roles = expected_receipt_roles(policy, split=split)
    if set(receipts) != set(roles):
        raise common.ContractError(
            "Structural auditor aggregate receipt role set drift"
        )
    for role in roles:
        validate_aggregate_receipt(
            policy, role=role, receipt=receipts[role]
        )
    expected_scope_hash = _registered_split_scope_sha256(
        policy, mode=mode, split=split
    )
    expected_policy_hash = _policy_parent_sha256(policy)
    if any(
        receipt["input_parent_hashes"].get(
            "registered_split_scope_sha256"
        )
        != expected_scope_hash
        for receipt in receipts.values()
    ) or any(
        receipt["input_parent_hashes"].get("policy_parent_sha256")
        != expected_policy_hash
        for receipt in receipts.values()
    ):
        raise common.ContractError(
            "Structural auditor policy or registered split scope mismatch"
        )
    expected_candidate_policy = (
        candidate_sampling.build_public_candidate_policy(
            policy, mode=mode, split=split
        )
    )
    candidate_parents = receipts["candidate_integrity"][
        "input_parent_hashes"
    ]
    if (
        candidate_parents["candidate_policy_projection_sha256"]
        != common.canonical_sha256(expected_candidate_policy)
        or candidate_parents["candidate_key_commitment_sha256"]
        != common.sha256_bytes(
            bytes.fromhex(
                str(policy["randomness"][mode]["candidate_key_hex"])
            )
        )
    ):
        raise common.ContractError(
            "Structural auditor candidate policy/key parent mismatch"
        )
    _validate_parent_projections(
        policy,
        split=split,
        receipts=receipts,
        parent_projections=parent_projections,
    )
    world_counts = {
        int(receipts[role]["fixed_counts"]["split_world_count"])
        for role in roles
    }
    failed_gates = [
        (role, gate)
        for role in roles
        for gate, passed in receipts[role]["fixed_boolean_gates"].items()
        if passed is not True
    ]
    if len(world_counts) != 1 or failed_gates:
        raise common.ContractError(
            "Structural auditor received a failed aggregate receipt"
        )
    boolean_rows = [
        {"role": role, "gate": gate, "value": passed}
        for role in roles
        for gate, passed in receipts[role]["fixed_boolean_gates"].items()
    ]
    count_rows = [
        {"role": role, "count": name, "value": value}
        for role in roles
        for name, value in receipts[role]["fixed_counts"].items()
    ]
    content_hash_rows = [
        {"role": role, "hash_name": name, "sha256": value}
        for role in roles
        for name, value in receipts[role][
            "aggregate_content_hashes"
        ].items()
    ]
    return _build_receipt(
        policy,
        role="structural_audit",
        fixed_boolean_gates={
            "receipt_role_set_exact": True,
            "receipt_recursive_schemas_exact": True,
            "receipt_self_hashes_exact": True,
            "all_integrity_boolean_gates_pass": True,
            "parent_projection_role_set_exact": True,
            "parent_content_and_file_hashes_match": True,
            "cross_receipt_world_counts_exact": True,
            "development_non_custody_boundary_explicit": True,
        },
        fixed_counts={
            "split_world_count": next(iter(world_counts)),
            "aggregate_receipt_count": len(receipts),
            "parent_projection_count": len(parent_projections),
            "failed_integrity_gate_count": len(failed_gates),
            "formal_custody_parent_seal_count": 0,
        },
        input_parent_hashes={
            "aggregate_receipt_set_sha256": canonical_multiset_sha256(
                [
                    {"role": role, "receipt": dict(receipts[role])}
                    for role in roles
                ]
            ),
            "parent_projection_set_sha256": canonical_multiset_sha256(
                parent_projections
            ),
            "deployment_schema_parent_sha256": common.sha256_file(
                _deployment_path(policy)
            ),
            "registered_split_scope_sha256": expected_scope_hash,
        },
        aggregate_content_hashes={
            "receipt_boolean_vector_sha256": canonical_multiset_sha256(
                boolean_rows
            ),
            "receipt_count_vector_sha256": canonical_multiset_sha256(
                count_rows
            ),
            "receipt_content_hash_vector_sha256": (
                canonical_multiset_sha256(content_hash_rows)
            ),
        },
    )
