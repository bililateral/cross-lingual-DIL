#!/usr/bin/env python3
"""Shared contracts for the Step28-v13 v1.13 v9 implementation-only builder."""

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
    ROOT / "schema" / "step28_v13_v1_13_scientific_dataset_builder_policy_v9.json"
)
POLICY_VERSION = "2026-08-14-step28-v13-v1-13-scientific-dataset-builder-v9"
POLICY_STATUS = "DESIGN_IMPLEMENTATION_ONLY_NO_REBUILD_OR_TRAINING"
EXPECTED_CLAIM_BOUNDARY = (
    "This policy authorizes implementation tests and in-memory causal replay through "
    "the already exposed train ordinal 283 only. It creates no dataset publication, "
    "formal seed, training-qualified row, model, or scientific metric."
)
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
        "quality_audit_v8_design_scale_amendment",
        "v8_build_execution_failure_record",
        "v8_second_build_execution_failure_record",
        "v8_third_build_execution_failure_record",
        "v8_fourth_build_execution_failure_record",
        "v8_fifth_build_execution_failure_record",
        "v9_document_capacity_repair_contract",
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
        "single_attempt_random_authority",
        "retired_public_preflight_authorities",
        "formal_authorization",
        "canonical_self_hash",
    }
    if set(policy) != expected:
        raise ScientificBuilderError("Scientific builder policy keyset drift")
    _verify_self_hash(policy)
    if policy.get("version") != POLICY_VERSION or policy.get("status") != POLICY_STATUS:
        raise ScientificBuilderError("Scientific builder policy version/status drift")
    if policy.get("claim_boundary") != EXPECTED_CLAIM_BOUNDARY:
        raise ScientificBuilderError("Scientific claim boundary drift")
    if tuple(policy.get("split_order", ())) != SPLITS:
        raise ScientificBuilderError("Scientific split order drift")
    _verify_pin(policy["scientific_contract"], label="scientific contract")
    _verify_pin(
        policy["quality_audit_v8_design_scale_amendment"],
        label="quality audit v8 design-scale amendment",
    )
    _verify_pin(
        policy["v8_build_execution_failure_record"],
        label="v8 build execution failure record",
    )
    _verify_pin(
        policy["v8_second_build_execution_failure_record"],
        label="v8 second build execution failure record",
    )
    _verify_pin(
        policy["v8_third_build_execution_failure_record"],
        label="v8 third build execution failure record",
    )
    _verify_pin(
        policy["v8_fourth_build_execution_failure_record"],
        label="v8 fourth build execution failure record",
    )
    _verify_pin(
        policy["v8_fifth_build_execution_failure_record"],
        label="v8 fifth build execution failure record",
    )
    _verify_pin(
        policy["v9_document_capacity_repair_contract"],
        label="v9 document-capacity repair contract",
    )
    base_policy_path = _verify_pin(
        policy["base_dataset_policy"], label="base dataset policy"
    )
    _verify_pin(
        policy["historical_collision_policy"], label="historical collision policy"
    )
    implementation = policy["implementation"]
    if not isinstance(implementation, Mapping) or tuple(implementation) != (
        "scientific_common",
        "candidate_text_templates",
        "document_capacity",
        "pure_natural_renderer",
        "scientific_world",
        "dataset_builder",
        "causal_replay_0_283",
        "scientific_contract_tests",
        "production_chain",
        "quality_channel_contract",
        "quality_channel_policy",
        "quality_channel_views",
        "quality_channel_policy_validator",
        "quality_channel_materializer",
        "quality_channel_tests",
        "quality_channel_materializer_tests",
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
        "seller_profile_surface_paths": {
            "surface_full": "observed/model_seller_profiles.jsonl",
            "surface_code_masked": "observed/model_seller_profiles.code_masked.jsonl",
            "surface_code_neutralized": "observed/model_seller_profiles.code_neutralized.jsonl",
        },
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
        "redacted_item_surface_paths": {
            "surface_full": "observed/redacted_items.jsonl",
            "surface_code_masked": "observed/redacted_items.code_masked.jsonl",
            "surface_code_neutralized": "observed/redacted_items.code_neutralized.jsonl",
        },
        "public_code_probe_input_path": "private/public_code_probe_input.jsonl",
        "text_probe_eligibility_input_path": "private/text_probe_eligibility_input.jsonl",
        "channel_structure_audit_path": "private/channel_structure_audit.jsonl",
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
            "train": 500,
            "development": 500,
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
    attribute_variation = selection.get("attribute_variation_repair")
    capacity_repair = selection.get("document_capacity_repair")
    if (
        selection.get("candidate_limit") != 32
        or selection.get("identity_value_maximum_counter") != 128
        or selection.get("labels_or_model_scores_read") is not False
        or selection.get("shortcut_probe_results_read") is not False
        or capacity_repair
        != {
            "version": "2026-08-14-step28-v13-v1-13-document-capacity-v9",
            "execution_stage": "after_baseline_identity33_before_candidate_view",
            "changed_render_fields": [
                "code",
                "title_nonempty",
                "description_nonempty",
            ],
            "seller_slot_source": "base_uid_creation_ordinal",
            "item_slot_source": "base_uid_creation_ordinal",
            "world_stride": 256,
            "seller_stride": 8,
            "feistel_rounds": 6,
            "feistel_half_bits": 20,
            "code_format": "Q[A-P]{10}",
            "candidate_zero_lineage_reference_before_collision": True,
            "labels_controllers_candidates_or_registries_read": False,
        }
        or attribute_variation
        != {
            "authority": "existing_derived_candidate_key_only",
            "domain": (
                "step28-v13-v1.13-v8.attribute.semantic-orbit."
                "keyed-rotation-v2"
            ),
            "mapping": "frozen_content_slot_keyed_cyclic_rotation",
            "semantic_slot_id": "product_version_or_specification_attribute",
            "strict_synonymy_required": False,
            "candidate_semantics": (
                "admissible_alternative_realizations_within_one_content_slot"
            ),
            "semantic_orbits": [
                ["标准版", "组合版", "多规格"],
                ["轻量版", "更新版", "通用版"],
                ["可选配色"],
                ["分批交付"],
                ["附使用说明"],
                ["支持自选参数"],
                ["含基础售后"],
            ],
            "candidate_only_extension": {
                "values": ["通用版"],
                "base_attribute_sampling_forbidden": True,
                "restricted_candidate_view_only": True,
                "existing_candidate_authority_only": True,
                "baseline_projection_bytes_must_match_previous_v8": True,
            },
            "full_cross_style_visible_equality_required": True,
            "shared_sequential_rng_reads": 0,
            "historical_or_current_registry_reads": False,
            "labels_or_model_scores_read": False,
        }
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

    attempt = policy["single_attempt_random_authority"]
    if attempt != {
        "attempt_index": 1,
        "total_world_count": 1004,
        "alternate_authority_forbidden": True,
        "alternate_output_root_forbidden": True,
        "build_execution_failure_reuses_same_authority": True,
        "audit_execution_failure_reuses_same_dataset_root": True,
        "data_quality_failure_closes_v8_permanently": True,
    }:
        raise ScientificBuilderError("Single-attempt authority contract drift")
    retired = policy["retired_public_preflight_authorities"]
    if (
        not isinstance(retired, Mapping)
        or set(retired) != {"count", "sorted_values_sha256"}
        or retired.get("count") != 28
        or retired.get("sorted_values_sha256")
        != "c2774d5ee66f05fca9fa7dcc4624b994b57be2d7ea414964bce32cb4e1dc6e81"
        or all_preflight_keys
        & {
            "0165fe7ba9d44d0c1aa895a65ed5ce02c8c3e85e52ae0b60f2eeaeaaa6ba5774",
            "524fc8ab2d2ad1e350412e94d2d63df4ad98b9aa63bf9343e260f5cfd1551af9",
            "bf747e10aa5842b1d49a854e09e0fd3fd57fddff5c38fe0915ab6afc3a768251",
            "feb0421051541d003f9dd9945ec5a6734bcf309ea1ba89c1fa29a6228d5f176c",
            "d956c1a032e0e96afd4b931ce4dd0e438fd929040b33e965a6e760462292aa8b",
            "f7cd9f35604ca4707b227f5c8966e1e80b2f096b05f3086e02d836619882898d",
            "cb464cba67aadaf7caad25d670f9fe359ff6e118ef9df5e41e44d322e391cbba",
            "1320bfa7e397b663616a456231e12f962130e2917496b1392b35cac70112f01e",
            "a5250ce575b1d4202b3ea133bc4bc0c5ed192315e5584b1ed67e7d475c3daeaa",
            "717dbd277bd11a370c89adb3622c8038f2dfab38ad9c152b23d37d179bc7d52d",
            "ff487621729d2eba779252c1c4c5dce99e93d23b86e116d3abd6b92359cd8875",
            "dd012846485c21c9f70242bb2b95b3f401f54a9e9e5ede876a4c67349238705b",
            "51d16a44a6008ad045453e16e0ca5a1712f7d91efc55e29f6142ab0b1e9179cb",
            "ff3d46965c70a6d252015583399ee9e5e72e82e18c4b9e3a54b3de589c92d1b9",
            "2000a3785337edb97c5d220bdec059fb388f45d87c7f0e5e1c39096c8c45df76",
            "fdec79069589dd8d9c4fa47082614cbb2edd220b7649d7711b505681bdf02dc1",
            "5658f15380dab0ccf69586f8b482ee03607d607a8e687292ba8354be12e745ba",
            "6066945fe233bff11a68b87201610945d548752a1ed83d082cb858c8164a429f",
            "4c456f322a9d7d5675ca1b68a08038cbd1e781c9516a451bbeecda7abcf8586e",
            "07252342ee2d4784a6baddb21b8880a2e22f31d2e407c92da457412148a86160",
            "2c7d10aa00478473833ae7d5702ecd98ae34e7b8da68e178d4bcb8251eea89c6",
            "bd728f88090dd0ae10a2015cc94dfb948411cf7a11a81cbc63ce181ecedbd295",
            "f96b536cacc9f4dbdd02cb15173cf0a7de852afd1ca85ed0263fe151da19da09",
            "9fa78ce0f049a4d86446ec8ccf41d4acfcbe67b8514fd2ddc4096fe45fdb0c6d",
            "88415a4cea65bacdf44664a9e70136380d6d942dbc6994089126edd228a7799c",
            "5acd599370e22fd0d12eb3def9fb1fad5f233c44f545ac7132074563950e91b9",
            "9678b2d1b800455827765db04da0350954277ab30a6d103308a4c9c4b1cdd704",
            "82137ff42853e28aeebeb4e844641a34e612bb4432a0afbfc5f5e6ee43260291",
        }
    ):
        raise ScientificBuilderError("Retired preflight authority contract drift")

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
