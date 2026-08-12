#!/usr/bin/env python3
"""Shared scientific-only contracts for the Step28-v13 v1.13 dataset builder."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import step28_v13_common as common
import step28_v13_structure as structure


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_PATH = (
    ROOT / "schema" / "step28_v13_v1_13_scientific_dataset_builder_policy.json"
)
POLICY_VERSION = "2026-08-11-step28-v13-v1-13-scientific-dataset-builder-v2"
POLICY_STATUS = "DESIGN_PREFLIGHT_ONLY"
SPLITS = ("train", "development", "audit_a", "audit_b")
DESIGN_MODES = ("small_smoke", "design_preflight")
EXECUTION_MODES = (*DESIGN_MODES, "formal")
HEX_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ScientificBuilderError(common.ContractError):
    """Fail-closed error for the scientific dataset construction layer."""


@dataclass(frozen=True)
class ExecutionContext:
    execution_mode: str
    base_mode: str
    effective_policy: dict[str, Any]
    world_records: tuple[dict[str, Any], ...]
    document_variation_key: bytes
    anonymous_handle_key: bytes
    output_root: Path
    scientific_use_forbidden: bool


def _canonical_clone(value: Any) -> Any:
    return json.loads(common.canonical_json_bytes(value).decode("utf-8"))


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or HEX_SHA256_RE.fullmatch(value) is None:
        raise ScientificBuilderError(f"{label} must be lowercase SHA-256")
    return value


def _verify_self_hash(policy: Mapping[str, Any]) -> None:
    claimed = _require_sha256(
        policy.get("canonical_self_hash"), label="scientific policy self-hash"
    )
    payload = dict(policy)
    payload.pop("canonical_self_hash", None)
    observed = common.canonical_sha256(payload)
    if observed != claimed:
        raise ScientificBuilderError(
            f"Scientific policy self-hash drift: claimed={claimed} observed={observed}"
        )


def _verify_pin(spec: Mapping[str, Any], *, label: str) -> Path:
    expected_keys = {"path", "size_bytes", "sha256"}
    if set(spec) != expected_keys:
        raise ScientificBuilderError(f"{label} pin keyset drift")
    path = common.repo_path(str(spec["path"]))
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ScientificBuilderError(f"Pinned {label} is unavailable") from exc
    if (
        isinstance(spec["size_bytes"], bool)
        or not isinstance(spec["size_bytes"], int)
        or size != spec["size_bytes"]
        or common.sha256_file(path) != _require_sha256(spec["sha256"], label=label)
    ):
        raise ScientificBuilderError(f"Pinned {label} bytes drift")
    return path


def _validate_key_block(block: Mapping[str, Any], *, label: str) -> set[str]:
    expected = {
        "id_namespace_key_hex",
        "structure_key_hex",
        "id_key_hex",
        "identity_value_key_hex",
        "text_key_hex",
        "candidate_key_hex",
        "query_key_hex",
        "document_variation_key_hex",
        "anonymous_handle_key_hex",
        "rewire_key_hexes",
    }
    if set(block) != expected:
        raise ScientificBuilderError(f"{label} key block schema drift")
    scalars = [
        _require_sha256(block[name], label=f"{label}.{name}")
        for name in expected - {"rewire_key_hexes"}
    ]
    rewires = block["rewire_key_hexes"]
    if not isinstance(rewires, list) or len(rewires) != 5:
        raise ScientificBuilderError(f"{label} requires five rewire keys")
    values = [
        *scalars,
        *(
            _require_sha256(value, label=f"{label}.rewire_key_hexes")
            for value in rewires
        ),
    ]
    if len(values) != len(set(values)):
        raise ScientificBuilderError(f"{label} reuses a random authority")
    return set(values)


def _collect_random_authorities(value: Any) -> set[str]:
    """Collect every 32-byte hex authority from a base randomness tree."""

    if isinstance(value, Mapping):
        output: set[str] = set()
        for child in value.values():
            output.update(_collect_random_authorities(child))
        return output
    if isinstance(value, (list, tuple)):
        output = set()
        for child in value:
            output.update(_collect_random_authorities(child))
        return output
    if isinstance(value, str) and HEX_SHA256_RE.fullmatch(value):
        return {value}
    return set()


def validate_policy(policy: Mapping[str, Any]) -> None:
    expected = {
        "version",
        "status",
        "claim_boundary",
        "scientific_contract",
        "base_dataset_policy",
        "historical_collision_policy",
        "implementation",
        "split_order",
        "world_contract",
        "exact_title_clone_endpoint_qualification",
        "model_mount_contract",
        "execution_modes",
        "candidate_selection",
        "public_preflight_keys",
        "formal_authorization",
        "canonical_self_hash",
    }
    if set(policy) != expected:
        raise ScientificBuilderError("Scientific builder policy keyset drift")
    _verify_self_hash(policy)
    if policy.get("version") != POLICY_VERSION or policy.get("status") != POLICY_STATUS:
        raise ScientificBuilderError("Scientific builder policy version/status drift")
    if tuple(policy.get("split_order", ())) != SPLITS:
        raise ScientificBuilderError("Scientific split order drift")
    _verify_pin(policy["scientific_contract"], label="scientific contract")
    base_policy_path = _verify_pin(
        policy["base_dataset_policy"], label="base dataset policy"
    )
    _verify_pin(
        policy["historical_collision_policy"], label="historical collision policy"
    )
    implementation = policy["implementation"]
    if not isinstance(implementation, Mapping) or tuple(implementation) != (
        "scientific_common",
        "scientific_world",
        "dataset_builder",
    ):
        raise ScientificBuilderError("Scientific implementation universe drift")
    for role, spec in implementation.items():
        _verify_pin(spec, label=f"scientific implementation {role}")

    world = policy["world_contract"]
    if world != {
        "sellers_per_world": 28,
        "controllers_per_world": 12,
        "dyad_controllers": 8,
        "triad_controllers": 4,
        "all_unordered_pairs_per_world": 378,
        "positive_pairs_per_world": 20,
        "negative_pairs_per_world": 358,
        "identity_feature_count": 33,
    }:
        raise ScientificBuilderError("Scientific world contract drift")

    qualification = policy["exact_title_clone_endpoint_qualification"]
    if qualification != {
        "version": "2026-08-11-step28-v13-v1-13-exact-clone-endpoints-v1",
        "selection_stage": "after_base_world_before_identity_remap_and_candidate_rendering",
        "selection_authority": "existing_split_structure_key",
        "source_requirements": ["title_nonempty"],
        "target_requirements": ["title_nonempty", "description_nonempty"],
        "seller_pair_and_direction_frozen": True,
        "item_endpoint_reselection_only": True,
        "unregistered_clone_residue_forbidden": True,
        "labels_or_model_scores_read": False,
        "shortcut_probe_results_read": False,
        "expected_exact_title_clone_count_per_world": 2,
    }:
        raise ScientificBuilderError(
            "Exact-title clone endpoint-qualification contract drift"
        )

    mount = policy["model_mount_contract"]
    if mount != {
        "model_seller_profile_path": "observed/model_seller_profiles.jsonl",
        "seller_profile_join_only_fields": ["seller_uid"],
        "seller_profile_text_feature_source_fields": [
            "category_concat_top",
            "signature_title_concat",
            "title_concat_top",
            "signature_description_concat",
            "description_concat_top",
        ],
        "seller_profile_numeric_feature_source_fields": [
            "item_count",
            "title_length_stats",
            "description_length_stats",
            "style_stats",
        ],
        "seller_profile_length_stat_fields": ["median"],
        "seller_profile_style_stat_fields": [
            "digit_ratio_mean",
            "punct_ratio_mean",
            "repeated_title_share",
            "repeated_description_share",
            "max_category_share",
        ],
        "redacted_item_path": "observed/redacted_items.jsonl",
        "redacted_item_join_only_fields": [
            "item_uid",
            "seller_uid",
            "world_uid",
        ],
        "redacted_item_text_feature_source_fields": ["title", "description"],
        "automatic_feature_discovery_forbidden": True,
        "full_seller_profile_mount_forbidden": True,
    }:
        raise ScientificBuilderError("Scientific model-mount contract drift")

    modes = policy["execution_modes"]
    if not isinstance(modes, dict) or tuple(modes) != EXECUTION_MODES:
        raise ScientificBuilderError("Scientific execution-mode order drift")
    expected_counts = {
        "small_smoke": {split: 1 for split in SPLITS},
        "design_preflight": {
            "train": 50,
            "development": 50,
            "audit_a": 2,
            "audit_b": 2,
        },
        "formal": {split: 500 for split in SPLITS},
    }
    output_roots: set[str] = set()
    for name in EXECUTION_MODES:
        spec = modes[name]
        required_flag = (
            "scientific_use_forbidden"
            if name in DESIGN_MODES
            else "scientific_use_forbidden_until_root_quality_pass"
        )
        if set(spec) != {required_flag, "world_counts", "output_root"}:
            raise ScientificBuilderError(f"Execution-mode schema drift: {name}")
        if spec[required_flag] is not True or spec["world_counts"] != expected_counts[name]:
            raise ScientificBuilderError(f"Execution-mode boundary drift: {name}")
        relative = str(spec["output_root"])
        path = common.repo_path(relative)
        reports_root = (ROOT / "reports").resolve()
        if reports_root not in path.parents or relative in output_roots:
            raise ScientificBuilderError(f"Unsafe or reused output root: {name}")
        output_roots.add(relative)

    selection = policy["candidate_selection"]
    if (
        selection.get("candidate_limit") != 32
        or selection.get("identity_value_maximum_counter") != 128
        or selection.get("labels_or_model_scores_read") is not False
        or selection.get("shortcut_probe_results_read") is not False
        or selection.get("advance_reason_allowlist")
        != [
            "same_world_item_document",
            "same_world_seller_document",
            "historical_item_document",
            "historical_seller_document",
            "current_dataset_item_document",
            "current_dataset_seller_document",
        ]
    ):
        raise ScientificBuilderError("Candidate-selection contract drift")

    preflight = policy["public_preflight_keys"]
    if not isinstance(preflight, dict) or tuple(preflight) != DESIGN_MODES:
        raise ScientificBuilderError("Public preflight-key mode drift")
    all_preflight_keys: set[str] = set()
    for name in DESIGN_MODES:
        values = _validate_key_block(preflight[name], label=name)
        if values & all_preflight_keys:
            raise ScientificBuilderError("Preflight modes reuse random authorities")
        all_preflight_keys.update(values)
    base_policy = common.load_json(base_policy_path)
    if not isinstance(base_policy, dict) or "randomness" not in base_policy:
        raise ScientificBuilderError("Pinned base randomness tree is unavailable")
    reused = all_preflight_keys & _collect_random_authorities(
        base_policy["randomness"]
    )
    if reused:
        raise ScientificBuilderError(
            "Preflight random authority reuses a pinned base authority"
        )

    authorization = policy["formal_authorization"]
    if set(authorization) != {
        "enabled",
        "formal_seed_receipt_path",
        "formal_seed_receipt_sha256",
        "formal_structure_key_commitments",
    }:
        raise ScientificBuilderError("Formal authorization schema drift")
    commitments = authorization["formal_structure_key_commitments"]
    if not isinstance(commitments, dict) or tuple(commitments) != SPLITS:
        raise ScientificBuilderError("Formal structure commitment order drift")
    if authorization["enabled"] is not False or any(
        authorization[name] is not None
        for name in ("formal_seed_receipt_path", "formal_seed_receipt_sha256")
    ) or any(value is not None for value in commitments.values()):
        raise ScientificBuilderError(
            "Design-only policy must contain zero formal seed material"
        )


def load_policy(path: Path = DEFAULT_POLICY_PATH) -> dict[str, Any]:
    if path.resolve() != DEFAULT_POLICY_PATH.resolve():
        raise ScientificBuilderError("Only the canonical scientific policy may pass")
    value = common.load_json(path)
    if not isinstance(value, dict):
        raise ScientificBuilderError("Scientific policy must be a JSON object")
    validate_policy(value)
    return value


def _replace_development_stream(
    effective: dict[str, Any], keys: Mapping[str, Any]
) -> None:
    effective["randomness"]["development_smoke"] = {
        "id_namespace_key_hex": keys["id_namespace_key_hex"],
        "structure_key_hex": keys["structure_key_hex"],
        "id_key_hex": keys["id_key_hex"],
        "identity_value_key_hex": keys["identity_value_key_hex"],
        "text_key_hex": keys["text_key_hex"],
        "candidate_key_hex": keys["candidate_key_hex"],
        "query_key_hex": keys["query_key_hex"],
        "rewire_key_hexes": list(keys["rewire_key_hexes"]),
    }


def build_execution_context(
    policy: Mapping[str, Any], *, execution_mode: str
) -> ExecutionContext:
    validate_policy(policy)
    if execution_mode not in EXECUTION_MODES:
        raise ScientificBuilderError(f"Unknown execution mode: {execution_mode}")
    base_path = _verify_pin(policy["base_dataset_policy"], label="base dataset policy")
    base = common.load_json(base_path)
    if not isinstance(base, dict):
        raise ScientificBuilderError("Base dataset policy must be an object")
    effective = _canonical_clone(base)
    mode_spec = policy["execution_modes"][execution_mode]
    effective_mode: str
    document_variation_key: bytes
    anonymous_handle_key: bytes
    if execution_mode in DESIGN_MODES:
        effective_mode = "development_smoke"
        keys = policy["public_preflight_keys"][execution_mode]
        _replace_development_stream(effective, keys)
        effective["modes"][effective_mode]["world_counts"] = _canonical_clone(
            mode_spec["world_counts"]
        )
        document_variation_key = bytes.fromhex(keys["document_variation_key_hex"])
        anonymous_handle_key = bytes.fromhex(keys["anonymous_handle_key_hex"])
        scientific_use_forbidden = True
    else:
        authorization = policy["formal_authorization"]
        if authorization["enabled"] is not True:
            raise ScientificBuilderError(
                "Formal generation remains disabled before the one-shot seed ceremony"
            )
        raise ScientificBuilderError(
            "Formal seed-receipt loading is intentionally unavailable in this design-only release"
        )
    common.validate_policy(effective, mode=effective_mode)
    records = tuple(
        _canonical_clone(row)
        for row in structure.build_mode_world_pool(effective, mode=effective_mode)
    )
    expected_total = sum(int(value) for value in mode_spec["world_counts"].values())
    if len(records) != expected_total:
        raise ScientificBuilderError("Effective world-pool cardinality drift")
    return ExecutionContext(
        execution_mode=execution_mode,
        base_mode=effective_mode,
        effective_policy=effective,
        world_records=records,
        document_variation_key=document_variation_key,
        anonymous_handle_key=anonymous_handle_key,
        output_root=common.repo_path(str(mode_spec["output_root"])),
        scientific_use_forbidden=scientific_use_forbidden,
    )


def load_release_inputs(
    context: ExecutionContext,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    template, fixture = common.validate_policy_release_documents(
        context.effective_policy, mode=context.base_mode
    )
    style_spec = context.effective_policy["style_reference_boundary"][
        "generator_release_inputs"
    ]["profile"]
    style_profile = common.load_json(
        common.verify_file_pin(style_spec, label="scientific style profile")
    )
    if not isinstance(style_profile, dict):
        raise ScientificBuilderError("Scientific style profile must be an object")
    common.validate_independent_replay_public_domains(
        context.effective_policy,
        template=template,
        style_profile=style_profile,
    )
    return template, fixture, style_profile


def key_commitment(key: bytes) -> str:
    if not isinstance(key, bytes) or len(key) != 32:
        raise ScientificBuilderError("Random authority must contain 32 bytes")
    return hashlib.sha256(key).hexdigest()
