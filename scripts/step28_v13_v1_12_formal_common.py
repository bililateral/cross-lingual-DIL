#!/usr/bin/env python3
"""Shared clean-room primitives for the Step28-v13 v1.12 formal build.

The module cannot create a master seed.  It accepts one split-scoped
capability bundle, derives no additional authority, and materializes complete
K28 worlds through the pinned successful generation components.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import math
import os
import platform
import subprocess
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import step28_v13_v1_12_preceremony as preceremony


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DRAFT_PATH = (
    ROOT / "schema" / "step28_v13_v1_12_formal_build_draft.json"
)
DEFAULT_PRELOCK_PATH = (
    ROOT / "schema" / "step28_v13_v1_12_formal_prelock.json"
)
DEFAULT_EXECUTION_LOCK_PATH = (
    ROOT
    / "reports"
    / "step28_synthetic_chinese_dataset"
    / "release_inputs"
    / "v1_12_cleanroom_20260803"
    / "train_development_execution_lock.json"
)
DEFAULT_AUDIT_A_LOCK_PATH = (
    ROOT
    / "reports"
    / "step28_synthetic_chinese_dataset"
    / "release_inputs"
    / "v1_12_cleanroom_20260803"
    / "audit_a_generation_lock.json"
)
DEFAULT_AUDIT_B_LOCK_PATH = (
    ROOT
    / "reports"
    / "step28_synthetic_chinese_dataset"
    / "release_inputs"
    / "v1_12_cleanroom_20260803"
    / "audit_b_generation_lock.json"
)
SPLITS = ("train", "development", "audit_a", "audit_b")
IDENTITY_ASSETS_PER_WORLD = {
    "train": 84,
    "development": 84,
    "audit_a": 84,
    "audit_b": 89,
}
MASTER_DOMAIN = b"step28-v13-v1.12"
GENERATOR_ROLES = (
    "structure",
    "id_namespace",
    "id",
    "identity_bootstrap",
    "identity_remap",
    "text",
    "query",
)
M1_ROLES = tuple(f"m1_r{index:02d}" for index in range(1, 6))
STRUCTURE_ENVIRONMENTS = {
    split: f"STEP28_V13_{split.upper()}_STRUCTURE_KEY_HEX"
    for split in SPLITS
}


class FormalBuildError(ValueError):
    """Raised when the v1.12 formal-build boundary fails closed."""


def expected_identity_asset_count(
    draft: Mapping[str, Any], *, split: str, world_count: int
) -> int:
    """Return the frozen split-specific identity-asset count."""

    if split not in SPLITS or not 1 <= int(world_count) <= 500:
        raise FormalBuildError("Identity-asset count scope is malformed")
    configured = draft.get("dataset_shape", {}).get(
        "identity_assets_per_world_by_split", {}
    )
    if configured != IDENTITY_ASSETS_PER_WORLD:
        raise FormalBuildError("Split-specific identity-asset shape drift")
    return int(world_count) * IDENTITY_ASSETS_PER_WORLD[split]


def require_canonical_path(
    path: Path, expected: Path, *, label: str
) -> Path:
    resolved = path.resolve()
    if resolved != expected.resolve():
        raise FormalBuildError(f"{label} is not at its canonical path")
    return resolved


def _verify_pin(spec: Mapping[str, Any], *, label: str) -> Path:
    return preceremony.verify_file_pin(spec, label=label)


def load_and_validate_draft(
    path: Path = DEFAULT_DRAFT_PATH,
) -> dict[str, Any]:
    """Validate the mutable prelock draft without granting formal authority."""

    path = require_canonical_path(
        path, DEFAULT_DRAFT_PATH, label="v1.12 formal-build draft"
    )
    draft = preceremony.load_json_strict(path)
    preceremony.validate_canonical_self_hash(
        draft, label="v1.12 formal-build draft"
    )
    if (
        draft.get("status")
        != "DRAFT_IMPLEMENTATION_NO_SEED_OR_DATA_AUTHORIZATION"
        or draft.get("run_id")
        != "v13_training_ready_v1_12_cleanroom_20260803"
        or set(draft.get("authorizations", {}).values()) != {False}
    ):
        raise FormalBuildError("Formal-build draft authorization drift")

    _verify_pin(draft["contract"], label="v1.12 formal build contract")
    baseline = draft["preceremony_baseline"]
    baseline_policy_path = preceremony._repo_path(
        str(baseline["policy_path"])
    )
    if (
        preceremony.sha256_file(baseline_policy_path)
        != baseline["policy_sha256"]
    ):
        raise FormalBuildError("Preceremony policy pin drift")
    baseline_policy = preceremony.load_json_strict(baseline_policy_path)
    preceremony.validate_canonical_self_hash(
        baseline_policy, label="pinned preceremony policy"
    )
    if (
        baseline_policy["canonical_self_hash"]
        != baseline["policy_canonical_self_hash"]
    ):
        raise FormalBuildError("Preceremony policy self-hash drift")
    validated_baseline = preceremony.validate_policy(baseline_policy_path)

    baseline_receipt_path = preceremony._repo_path(
        str(baseline["receipt_path"])
    )
    if (
        preceremony.sha256_file(baseline_receipt_path)
        != baseline["receipt_sha256"]
    ):
        raise FormalBuildError("Preceremony receipt pin drift")
    baseline_receipt = preceremony.load_json_strict(baseline_receipt_path)
    preceremony.validate_canonical_self_hash(
        baseline_receipt, label="pinned preceremony receipt"
    )
    if (
        baseline_receipt["canonical_self_hash"]
        != baseline["receipt_canonical_self_hash"]
        or baseline_receipt.get("status")
        != "PASS_DESIGN_ONLY_NO_FORMAL_AUTHORIZATION"
        or baseline_receipt.get("formal_seed_or_key_access") is not False
        or int(baseline_receipt.get("formal_dataset_rows_produced", -1)) != 0
    ):
        raise FormalBuildError("Preceremony receipt semantic drift")

    _verify_pin(
        draft["base_generation_policy"],
        label="pinned v1.2 pure DGP policy",
    )
    support = draft.get("base_generation_support_inputs", {})
    if set(support) != {"dataset_custody_deployment"}:
        raise FormalBuildError("Base DGP support-input set drift")
    _verify_pin(
        support["dataset_custody_deployment"],
        label="pinned dataset custody deployment",
    )
    historical_path, historical_release = _load_pinned_self_hashed_json(
        draft["historical_success_release"],
        label="pinned historical successful v1.2 release",
    )
    if (
        historical_release.get("status")
        != "PASS_DATASET_ONLY_READY_FOR_M0_M1_M2"
        or historical_release.get("run_id")
        != "v13_training_ready_v1_2_order_repair_20260731"
        or draft["historical_success_release"].get("allowed_use")
        != (
            "observed_uid_and_exact_visible_document_intersection_plus_"
            "privileged_identity_value_hash_derivation_only_no_c40_or_labels"
        )
        or historical_path.name != "release_manifest.json"
    ):
        raise FormalBuildError("Historical successful release boundary drift")

    coverage_path, coverage = _load_pinned_self_hashed_json(
        draft["historical_identity_exclusion_coverage"],
        label="historical identity exclusion coverage receipt",
    )
    coverage_spec = draft["historical_identity_exclusion_coverage"]
    expected_split_counts = {
        "train": 42_000,
        "development": 42_000,
        "audit_a": 42_000,
        "audit_b": 44_500,
    }
    for split, expected_count in expected_split_counts.items():
        record = coverage.get("historical_v1_2_splits", {}).get(split, {})
        _verify_pin(
            record.get("split_manifest", {}),
            label=f"historical {split} split manifest coverage source",
        )
        _verify_pin(
            record.get("identity_assets", {}),
            label=f"historical {split} identity coverage source",
        )
        if (
            int(record.get("row_count", -1)) != expected_count
            or int(record.get("unique_value_hash_count", -1))
            != expected_count
            or preceremony.HEX_SHA256_RE.fullmatch(
                str(record.get("ordered_value_hash_digest", ""))
            )
            is None
        ):
            raise FormalBuildError(
                f"Historical {split} identity coverage record drift"
            )
    _verify_pin(
        coverage.get("producer", {}),
        label="historical identity coverage producer",
    )
    _verify_pin(
        coverage.get("original_identity_deny_registry", {}),
        label="historical original identity deny registry",
    )
    coverage_historical = coverage.get("historical_v1_2_identity_union", {})
    coverage_original = coverage.get("original_identity_deny_registry", {})
    coverage_old_boundary = coverage.get(
        "original_deny_plus_historical_v1_2", {}
    )
    coverage_archive = coverage.get("combined_exclusion_archive", {})
    coverage_counts = coverage.get("coverage", {})
    baseline_archive_spec = validated_baseline["policy"][
        "failed_identity_exclusion_archive"
    ]
    if (
        coverage_path.name
        != "historical_identity_exclusion_coverage_receipt.json"
        or coverage.get("version")
        != (
            "2026-08-03-step28-v13-v1-12-historical-identity-"
            "coverage-receipt-v1"
        )
        or coverage.get("status")
        != (
            "PASS_HISTORICAL_IDENTITY_EXCLUSION_COVERAGE_"
            "NO_FORMAL_AUTHORIZATION"
        )
        or coverage.get("run_id") != draft["run_id"]
        or coverage.get("normalization")
        != "SHA256(UTF-8(NFC(casefold(strip(value)))))"
        or coverage.get("formal_seed_or_key_access") is not False
        or int(coverage.get("formal_dataset_rows_produced", -1)) != 0
        or coverage.get("scientific_metrics_produced") is not False
        or coverage.get("raw_identity_values_persisted_in_receipt") is not False
        or coverage.get("formal_authorizations_after_audit")
        != validated_baseline["policy"]["authorizations"]
        or {
            key: coverage["historical_release_manifest"].get(key)
            for key in ("path", "sha256", "size_bytes", "canonical_self_hash")
        }
        != {
            key: draft["historical_success_release"].get(key)
            for key in ("path", "sha256", "size_bytes", "canonical_self_hash")
        }
        or int(coverage_historical.get("unique_value_hash_count", -1))
        != int(coverage_spec["historical_v1_2_unique_value_hash_count"])
        or coverage_historical.get("ordered_value_hash_digest")
        != coverage_spec["historical_v1_2_ordered_value_hash_digest"]
        or int(coverage_historical.get("intersection_with_original_deny_count", -1))
        != 0
        or int(coverage_original.get("unique_value_hash_count", -1))
        != int(coverage_spec["original_deny_unique_value_hash_count"])
        or coverage_original.get("ordered_value_hash_digest")
        != coverage_spec["original_deny_ordered_value_hash_digest"]
        or int(coverage_old_boundary.get("unique_value_hash_count", -1))
        != int(coverage_spec["old_boundary_unique_value_hash_count"])
        or coverage_old_boundary.get("ordered_value_hash_digest")
        != coverage_spec["old_boundary_ordered_value_hash_digest"]
        or coverage_archive.get("path") != baseline_archive_spec["path"]
        or coverage_archive.get("sha256") != baseline_archive_spec["sha256"]
        or int(coverage_archive.get("size_bytes", -1))
        != int(baseline_archive_spec["size_bytes"])
        or coverage_archive.get("canonical_self_hash")
        != baseline_archive_spec["canonical_self_hash"]
        or int(coverage_archive.get("base_unique_value_hash_count", -1))
        != 745_496
        or int(coverage_archive.get("combined_unique_value_hash_count", -1))
        != len(validated_baseline["failed_identity_hashes"])
        or int(coverage_counts.get("original_deny_present_count", -1))
        != 112_996
        or int(coverage_counts.get("original_deny_missing_count", -1)) != 0
        or int(coverage_counts.get("historical_v1_2_present_count", -1))
        != 170_500
        or int(coverage_counts.get("historical_v1_2_missing_count", -1))
        != 0
        or int(coverage_counts.get("old_boundary_present_count", -1))
        != 283_496
        or int(coverage_counts.get("old_boundary_missing_count", -1)) != 0
        or coverage_counts.get("all_historical_identity_values_forbidden")
        is not True
    ):
        raise FormalBuildError(
            "Historical identity exclusion coverage receipt drift"
        )

    collision = draft["identity_collision_resolution"]
    if (
        collision.get("strategy") != "first_admissible_per_asset_counter"
        or int(collision.get("maximum_counter", -1)) != 1024
        or int(collision.get("historical_forbidden_identity_hash_count", -1))
        != len(validated_baseline["failed_identity_hashes"])
        or int(collision.get("forbidden_master_commitment_count", -1))
        != len(validated_baseline["forbidden_master_commitments"])
        or collision.get("same_run_hash_set_monotonic") is not True
        or collision.get("visible_identity_free_text_collision_forbidden")
        is not True
        or collision.get("master_retry_forbidden") is not True
    ):
        raise FormalBuildError("Formal identity collision boundary drift")
    shape = draft["dataset_shape"]
    if (
        tuple(shape["split_order"]) != SPLITS
        or int(shape["worlds_per_split"]) != 500
        or int(shape["sellers_per_world"]) != 28
        or int(shape["controllers_per_world"]) != 12
        or int(shape["complete_pairs_per_world"]) != 378
        or int(shape["positive_pairs_per_world"]) != 20
        or int(shape["negative_pairs_per_world"]) != 358
        or int(shape["pairs_per_split"]) != 189000
        or int(shape["positive_pairs_per_split"]) != 10000
        or int(shape["negative_pairs_per_split"]) != 179000
        or shape.get("identity_assets_per_world_by_split")
        != IDENTITY_ASSETS_PER_WORLD
        or shape.get("identity_assets_per_split")
        != {
            split: 500 * count
            for split, count in IDENTITY_ASSETS_PER_WORLD.items()
        }
        or int(shape["retrieval_queries_per_world"]) != 28
        or int(shape["retrieval_gallery_per_query"]) != 27
        or int(shape["retrieval_relevant_relations_per_world"]) != 40
        or shape["c40_in_any_premodel_member"] is not False
    ):
        raise FormalBuildError("Formal full-378 dataset shape drift")

    randomness = draft["randomness"]
    if (
        int(randomness["master_seed_bytes"]) != 32
        or tuple(randomness["ceremony_split_order"]) != SPLITS
        or int(randomness["master_draws_per_split"]) != 1
        or randomness["derivation"] != "HMAC-SHA256"
        or randomness["derivation_prefix"] != MASTER_DOMAIN.decode("ascii")
        or randomness["separator_hex"] != "1f"
        or tuple(randomness["generator_capability_roles"])
        != GENERATOR_ROLES
        or tuple(randomness["train_m1_roles"]) != M1_ROLES
        or randomness["master_mounted_to_generator"] is not False
        or randomness["master_mounted_to_model"] is not False
        or randomness["one_m1_key_per_fit_process"] is not True
        or randomness["seed_replacement_or_screening"] is not False
        or preceremony.HEX_SHA256_RE.fullmatch(
            str(randomness["design_only_master_hex"])
        )
        is None
    ):
        raise FormalBuildError("Formal capability derivation contract drift")

    release = draft["release"]
    public_members = list(release["public_common_members"])
    private_members = list(release["private_common_members"])
    audit_ladder = release.get("audit_generation_ladder", {})
    if (
        len(public_members) != len(set(public_members))
        or len(private_members) != len(set(private_members))
        or any("c40" in value.casefold() for value in public_members)
        or any("c40" in value.casefold() for value in private_members)
        or release["staging_under_private_root"] is not True
        or release["no_replace_publish"] is not True
        or release.get("split_manifest_version")
        != "2026-08-03-step28-v13-v1-12-full378-split-manifest-v1"
        or release.get("release_manifest_version")
        != "2026-08-03-step28-v13-v1-12-full378-release-manifest-v1"
        or release["raw_master_or_capability_in_public_release"] is not False
        or release["audit_truth_in_public_release"] is not False
        or audit_ladder
        != {
            "audit_a_lock": DEFAULT_AUDIT_A_LOCK_PATH.relative_to(
                ROOT
            ).as_posix(),
            "audit_a_authorized_before_audit_b": True,
            "audit_b_lock": DEFAULT_AUDIT_B_LOCK_PATH.relative_to(
                ROOT
            ).as_posix(),
            "audit_b_requires_published_audit_a": True,
            "both_audits_authorized_by_one_lock": False,
        }
    ):
        raise FormalBuildError("Formal member/custody contract drift")
    for split in SPLITS:
        supervision = list(release["public_supervision_members"][split])
        if supervision != (
            ["supervision/classification_labels.csv"]
            if split in {"train", "development"}
            else []
        ):
            raise FormalBuildError("Public supervision boundary drift")
    private_root = preceremony._repo_path(str(release["private_root"]))
    ignored = subprocess.run(
        [
            "git",
            "check-ignore",
            "-q",
            "--",
            private_root.relative_to(ROOT).as_posix(),
        ],
        cwd=ROOT,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if ignored.returncode != 0:
        raise FormalBuildError("Formal private custody root is not Git ignored")

    shortcut = draft["shortcut_preflight"]
    if (
        int(shortcut["design_only_train_worlds"]) != 500
        or int(shortcut["design_only_development_worlds"]) != 500
        or int(shortcut["rows_per_world"]) != 378
        or int(shortcut["fold_count"]) != 5
        or int(shortcut["bootstrap_replicates"]) != 9999
        or int(shortcut["logistic_maximum_iterations"]) != 100
        or float(shortcut["logistic_gradient_tolerance"]) != 1e-9
        or int(shortcut["null_nuisance_feature_count"]) != 14
        or len(shortcut["join_nuisance_features"]) != 10
        or shortcut["intended_market_style_or_identity_features_in_null_gate"]
        is not False
        or shortcut["exact_numeric_preflight_required_before_seed"] is not True
    ):
        raise FormalBuildError("Exact shortcut preflight contract drift")
    mounts = draft["model_mounts"]
    if (
        mounts["m0_text_fields_in_order"]
        != [
            "category_concat_top",
            "signature_title_concat",
            "title_concat_top",
            "signature_description_concat",
            "description_concat_top",
        ]
        or mounts["profile_text_and_all_other_profile_fields_model_forbidden"]
        is not True
        or mounts["fit_columns"] != ["p0", "identity33_numeric_only"]
        or mounts["join_columns_removed_before_fit"] is not True
    ):
        raise FormalBuildError("Formal model mount field boundary drift")
    text_diagnostic = draft["visible_text_diagnostic"]
    if (
        text_diagnostic["probe"] != "fixed_hashing_char_trigram_cosine"
        or text_diagnostic["ngram_range"] != [3, 3]
        or int(text_diagnostic["feature_dimension"]) != 65536
        or text_diagnostic["alternate_sign"] is not False
        or text_diagnostic["normalization"] != "l2"
        or text_diagnostic["formal_gate"] is not False
    ):
        raise FormalBuildError("Visible-text diagnostic contract drift")
    if set(draft["implementation_freeze"].values()) != {False}:
        raise FormalBuildError("Draft falsely claims a frozen implementation")

    return {
        "draft": draft,
        "baseline": validated_baseline,
        "baseline_receipt": baseline_receipt,
    }


def _derive(master: bytes, *, split: str, role: str) -> bytes:
    if len(master) != 32 or split not in SPLITS or not role:
        raise FormalBuildError("Capability derivation input is malformed")
    message = b"\x1f".join(
        (MASTER_DOMAIN, split.encode("ascii"), role.encode("ascii"))
    )
    return hmac.new(master, message, hashlib.sha256).digest()


def derive_capabilities(master: bytes, *, split: str) -> dict[str, Any]:
    """Derive the complete ceremony-side capability set for one split."""

    generator = {
        role: _derive(master, split=split, role=role).hex()
        for role in GENERATOR_ROLES
    }
    m1 = (
        {
            role: _derive(master, split=split, role=role).hex()
            for role in M1_ROLES
        }
        if split == "train"
        else {}
    )
    raw_values = [*generator.values(), *m1.values()]
    if len(raw_values) != len(set(raw_values)):
        raise FormalBuildError("Derived capability values collide")
    return {"split": split, "generator": generator, "m1": m1}


def capability_commitments(capabilities: Mapping[str, Any]) -> dict[str, Any]:
    split = str(capabilities.get("split", ""))
    if split not in SPLITS:
        raise FormalBuildError("Capability bundle split is malformed")

    def commitments(values: Mapping[str, str]) -> dict[str, str]:
        output: dict[str, str] = {}
        for role, value in values.items():
            if preceremony.HEX_SHA256_RE.fullmatch(str(value)) is None:
                raise FormalBuildError("Capability value is not 32-byte hex")
            output[str(role)] = hashlib.sha256(bytes.fromhex(str(value))).hexdigest()
        return output

    return {
        "split": split,
        "generator": commitments(capabilities["generator"]),
        "m1": commitments(capabilities["m1"]),
    }


def build_execution_policy(
    *,
    draft: Mapping[str, Any],
    split: str,
    generator_capabilities: Mapping[str, str],
    structure_commitments: Mapping[str, str],
) -> dict[str, Any]:
    """Create an in-memory DGP policy; raw capabilities are never persisted."""

    import step28_v13_common as common

    if (
        split not in SPLITS
        or set(generator_capabilities) != set(GENERATOR_ROLES)
        or set(structure_commitments) != set(SPLITS)
    ):
        raise FormalBuildError("Execution policy capability/keyset drift")
    for value in [*generator_capabilities.values(), *structure_commitments.values()]:
        if preceremony.HEX_SHA256_RE.fullmatch(str(value)) is None:
            raise FormalBuildError("Execution policy key/commitment is malformed")
    base_path = preceremony._repo_path(
        str(draft["base_generation_policy"]["path"])
    )
    policy = common.load_json(base_path)
    policy["status"] = "FROZEN"
    policy["formal_generation_enabled"] = True
    for name in SPLITS:
        policy["modes"]["formal"]["world_counts"][name] = int(
            draft["dataset_shape"]["worlds_per_split"]
        )
    policy["modes"]["formal"]["run_id"] = str(draft["run_id"])
    policy["modes"]["formal"]["output_root"] = str(
        draft["release"]["public_root"]
    )
    policy["modes"]["formal"]["power_design_path"] = str(
        draft["contract"]["path"]
    )
    policy["modes"]["formal"]["power_design_sha256"] = str(
        draft["contract"]["sha256"]
    )
    policy["security"]["dataset_custody_deployment"]["sha256"] = str(
        draft["base_generation_support_inputs"][
            "dataset_custody_deployment"
        ]["sha256"]
    )
    identity_types = list(policy["identity_design"]["identity_types"])
    bootstrap_salt = int(
        draft["identity_collision_resolution"][
            "temporary_bootstrap_salt_counter_per_type"
        ]
    )
    if (
        bootstrap_salt != 0
        or draft["identity_collision_resolution"][
            "temporary_bootstrap_values_persisted"
        ]
        is not False
        or draft["identity_collision_resolution"][
            "final_per_asset_remap_required"
        ]
        is not True
    ):
        raise FormalBuildError("Temporary identity bootstrap contract drift")
    policy["identity_design"]["identity_value_generation"][
        "salt_selection"
    ]["formal_per_type_salt_counters"] = {
        identity_type: bootstrap_salt for identity_type in identity_types
    }
    stream = policy["randomness"]["formal"]
    stream["id_namespace_key_hex"] = str(
        generator_capabilities["id_namespace"]
    )
    stream["id_key_hex"] = str(generator_capabilities["id"])
    stream["identity_value_key_hex"] = str(
        generator_capabilities["identity_bootstrap"]
    )
    stream["text_key_hex"] = str(generator_capabilities["text"])
    custody = stream["label_bearing_structure_keys"]
    for name in SPLITS:
        custody[name]["sha256_commitment"] = str(
            structure_commitments[name]
        )
    common.validate_policy(policy, mode="formal")
    if (
        hashlib.sha256(
            bytes.fromhex(str(generator_capabilities["structure"]))
        ).hexdigest()
        != structure_commitments[split]
    ):
        raise FormalBuildError("Current structure capability commitment mismatch")
    return policy


@contextmanager
def mounted_structure_capability(
    *, split: str, structure_key_hex: str
) -> Iterator[None]:
    """Mount exactly one structure capability for a bounded in-process call."""

    if split not in SPLITS or preceremony.HEX_SHA256_RE.fullmatch(
        structure_key_hex
    ) is None:
        raise FormalBuildError("Structure capability mount input is malformed")
    present = [
        variable
        for variable in STRUCTURE_ENVIRONMENTS.values()
        if os.environ.get(variable) is not None
    ]
    if present:
        raise FormalBuildError(
            f"Structure capability environment is not empty: {present}"
        )
    variable = STRUCTURE_ENVIRONMENTS[split]
    os.environ[variable] = structure_key_hex
    try:
        yield
    finally:
        observed = os.environ.pop(variable, None)
        if observed != structure_key_hex:
            raise FormalBuildError("Structure capability mount changed during use")


def load_release_inputs(
    execution_policy: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    import step28_v13_common as common

    common.validate_policy(execution_policy, mode="formal")
    template_path = common.verify_file_pin(
        execution_policy["template_library"],
        label="v1.12 formal template library",
    )
    fixture_path = common.verify_file_pin(
        execution_policy["identity_design"][
            "role_template_parser_flag_fixture"
        ],
        label="v1.12 externally frozen parser fixture",
    )
    template = common.load_json(template_path)
    fixture = common.load_json(fixture_path)
    if (
        template.get("version")
        != execution_policy["template_library"]["required_version"]
        or fixture.get("version")
        != execution_policy["identity_design"][
            "role_template_parser_flag_fixture"
        ]["required_version"]
        or fixture.get("status") != "DRAFT_PASS_NOT_FROZEN"
    ):
        raise FormalBuildError("Pinned parser/template input semantic drift")
    style_profile = common.load_json(
        common.verify_file_pin(
            execution_policy["style_reference_boundary"][
                "generator_release_inputs"
            ]["profile"],
            label="v1.12 formal style reference",
        )
    )
    common.validate_independent_replay_public_domains(
        execution_policy,
        template=template,
        style_profile=style_profile,
    )
    return template, fixture, style_profile


def split_world_records(
    execution_policy: Mapping[str, Any], *, split: str
) -> list[dict[str, Any]]:
    import step28_v13_structure as structure

    records = [
        dict(row)
        for row in structure.build_mode_world_pool(
            execution_policy, mode="formal"
        )
        if row["split"] == split
    ]
    records.sort(key=lambda row: int(row["split_ordinal"]))
    if (
        len(records) != 500
        or [int(row["split_ordinal"]) for row in records]
        != list(range(500))
        or len({str(row["world_uid"]) for row in records}) != 500
    ):
        raise FormalBuildError("Formal split world-record pool drift")
    return records


def _history_item_index(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        {
            "world_uid": str(row["world_uid"]),
            "seller_uid": str(row["seller_uid"]),
            "item_uid": str(row["item_uid"]),
            "time_bucket": int(row["time_bucket"]),
        }
        for row in items
    ]
    rows.sort(
        key=lambda row: (
            row["world_uid"].encode("utf-8"),
            row["seller_uid"].encode("utf-8"),
            row["item_uid"].encode("utf-8"),
        )
    )
    return rows


def _retrieval_rows(
    *,
    split: str,
    world_uid: str,
    seller_uids: Sequence[str],
    controller_membership: Sequence[Mapping[str, Any]],
    query_key_hex: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    seller_to_controller = {
        str(row["seller_uid"]): str(row["controller_uid"])
        for row in controller_membership
    }
    ordered = sorted((str(value) for value in seller_uids), key=str.encode)
    if len(ordered) != 28 or set(ordered) != set(seller_to_controller):
        raise FormalBuildError("Retrieval seller/controller keyset drift")
    key = bytes.fromhex(query_key_hex)
    queries: list[dict[str, str]] = []
    qrels: list[dict[str, str]] = []
    for seller_uid in ordered:
        query_uid = "qry_" + hmac.new(
            key,
            b"\x1f".join(
                (
                    MASTER_DOMAIN,
                    b"query",
                    split.encode("ascii"),
                    world_uid.encode("utf-8"),
                    seller_uid.encode("utf-8"),
                )
            ),
            hashlib.sha256,
        ).hexdigest()
        queries.append(
            {
                "query_uid": query_uid,
                "world_uid": world_uid,
                "seller_uid": seller_uid,
            }
        )
        for gallery_uid in ordered:
            if (
                gallery_uid != seller_uid
                and seller_to_controller[gallery_uid]
                == seller_to_controller[seller_uid]
            ):
                qrels.append(
                    {
                        "query_uid": query_uid,
                        "world_uid": world_uid,
                        "query_seller_uid": seller_uid,
                        "gallery_seller_uid": gallery_uid,
                        "relevance": "1",
                    }
                )
    if len(queries) != 28 or len(qrels) != 40:
        raise FormalBuildError("Retrieval query/qrel cardinality drift")
    return queries, qrels


def _build_formal_history_attestation(
    *,
    policy: Mapping[str, Any],
    split: str,
    world_uid: str,
    sellers: Sequence[Mapping[str, Any]],
    raw_items: Sequence[Mapping[str, Any]],
    history_safe_occurrences: Sequence[Mapping[str, Any]],
    history_item_index: Sequence[Mapping[str, Any]],
    parsed_rows: Sequence[Mapping[str, Any]],
    identity_slots_audit: Sequence[Mapping[str, Any]],
    noise_slots_audit: Sequence[Mapping[str, Any]],
    render_asts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Create the formal parent-sealed attestation required by identity33."""

    import step28_v13_common as common
    import step28_v13_production_chain as production

    parser_audit = production.validate_parser_against_private_plan(
        policy,
        mode="formal",
        split=split,
        sellers=sellers,
        items=raw_items,
        parsed_rows=parsed_rows,
        identity_slots_audit=identity_slots_audit,
        noise_slots_audit=noise_slots_audit,
        render_asts=render_asts,
    )
    expected_safe = production.project_history_safe_occurrences(
        policy,
        mode="formal",
        split=split,
        sellers=sellers,
        items=raw_items,
        parsed_rows=parsed_rows,
    )
    expected_index = _history_item_index(raw_items)
    if (
        common.canonical_json_bytes(list(history_safe_occurrences))
        != common.canonical_json_bytes(expected_safe)
        or common.canonical_json_bytes(list(history_item_index))
        != common.canonical_json_bytes(expected_index)
        or parser_audit.get("exact_rows_and_flags") is not True
        or int(parser_audit.get("actual_parser_row_count", -1))
        != len(parsed_rows)
        or len(parsed_rows) != len(history_safe_occurrences)
    ):
        raise FormalBuildError("Formal history parent did not replay exactly")
    parent_seal = common.canonical_sha256(
        {
            "version": "2026-08-03-step28-v13-v1-12-history-parent-v1",
            "split": split,
            "world_uid": world_uid,
            "sellers_sha256": common.canonical_sha256(list(sellers)),
            "raw_items_sha256": common.canonical_sha256(list(raw_items)),
            "identity_slots_sha256": common.canonical_sha256(
                list(identity_slots_audit)
            ),
            "noise_slots_sha256": common.canonical_sha256(
                list(noise_slots_audit)
            ),
            "render_asts_sha256": common.canonical_sha256(list(render_asts)),
            "parser_audit_sha256": common.canonical_sha256(parser_audit),
        }
    )
    producer_path = ROOT / "scripts" / "step28_v13_production_chain.py"
    payload: dict[str, Any] = {
        "version": "2026-07-27-step28-v13-history-projection-attestation-v1",
        "mode": "formal",
        "split": split,
        "world_uid": world_uid,
        "history_safe_occurrence_count": len(history_safe_occurrences),
        "history_safe_occurrences_sha256": common.canonical_rows_sha256(
            history_safe_occurrences,
            order_fields=(
                "world_uid",
                "seller_uid",
                "item_uid",
                "source_field",
                "contact_type",
                "normalized_value",
            ),
        ),
        "history_item_index_count": len(history_item_index),
        "history_item_index_sha256": common.canonical_rows_sha256(
            history_item_index,
            order_fields=("world_uid", "seller_uid", "item_uid"),
        ),
        "parser_artifact_row_count": len(parsed_rows),
        "parser_artifact_sha256": common.canonical_rows_sha256(
            parsed_rows,
            order_fields=(
                "world_uid",
                "seller_uid",
                "item_uid",
                "source_field",
                "contact_type",
                "normalized_value",
            ),
        ),
        "parser_exact_replay": True,
        "private_plan_exact": True,
        "projection_producer_path": "scripts/step28_v13_production_chain.py",
        "projection_producer_sha256": common.sha256_file(producer_path),
        "step3_parser_code_sha256": str(
            policy["frozen_inputs"]["step3_parser_profile_code"]["sha256"]
        ),
        "custody_status": "FORMAL_PARENT_SEALED",
        "custody_parent_seal_sha256": parent_seal,
    }
    payload["canonical_self_hash"] = common.canonical_sha256(payload)
    return payload


def _build_formal_identity33(
    *,
    policy: Mapping[str, Any],
    split: str,
    history_safe_occurrences: Sequence[Mapping[str, Any]],
    history_item_index: Sequence[Mapping[str, Any]],
    projection_attestations: Sequence[Mapping[str, Any]],
    complete_model_pair_endpoints: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Release the previously unreachable formal parent-seal validation path."""

    import step28_v13_common as common
    import step28_v13_history_features as history_features

    common.validate_policy(policy, mode="formal")
    common.verify_file_pin(
        policy["frozen_inputs"]["identity_history_code"],
        label="v1.12 formal identity-history feature code",
    )
    feature_names, excluded_names = history_features._feature_contract(policy)
    pair_rows_by_world, sellers_by_world = history_features._validate_pair_rows(
        policy, complete_model_pair_endpoints
    )
    item_index = history_features._validate_history_item_index(
        policy,
        rows=history_item_index,
        sellers_by_world=sellers_by_world,
    )
    history_rows_by_world = history_features._validate_history_rows(
        policy,
        mode="formal",
        split=split,
        rows=history_safe_occurrences,
        sellers_by_world=sellers_by_world,
        item_index=item_index,
    )
    expected_worlds = set(pair_rows_by_world)
    item_rows_by_world: dict[str, list[Mapping[str, Any]]] = {
        world_uid: [] for world_uid in expected_worlds
    }
    for row in history_item_index:
        item_rows_by_world.setdefault(str(row["world_uid"]), []).append(row)
    expected_fields = [
        "version",
        "mode",
        "split",
        "world_uid",
        "history_safe_occurrence_count",
        "history_safe_occurrences_sha256",
        "history_item_index_count",
        "history_item_index_sha256",
        "parser_artifact_row_count",
        "parser_artifact_sha256",
        "parser_exact_replay",
        "private_plan_exact",
        "projection_producer_path",
        "projection_producer_sha256",
        "step3_parser_code_sha256",
        "custody_status",
        "custody_parent_seal_sha256",
        "canonical_self_hash",
    ]
    producer_sha = common.sha256_file(
        ROOT / "scripts" / "step28_v13_production_chain.py"
    )
    attestation_worlds: set[str] = set()
    for attestation in projection_attestations:
        if list(attestation) != expected_fields:
            raise FormalBuildError("Formal projection attestation schema/order drift")
        world_uid = str(attestation["world_uid"])
        body = {
            name: attestation[name]
            for name in expected_fields
            if name != "canonical_self_hash"
        }
        safe_rows = list(history_rows_by_world.get(world_uid, []))
        item_rows = item_rows_by_world.get(world_uid, [])
        if (
            world_uid in attestation_worlds
            or world_uid not in expected_worlds
            or attestation["version"]
            != "2026-07-27-step28-v13-history-projection-attestation-v1"
            or attestation["mode"] != "formal"
            or attestation["split"] != split
            or attestation["canonical_self_hash"]
            != common.canonical_sha256(body)
            or attestation["parser_exact_replay"] is not True
            or attestation["private_plan_exact"] is not True
            or attestation["projection_producer_path"]
            != "scripts/step28_v13_production_chain.py"
            or attestation["projection_producer_sha256"] != producer_sha
            or attestation["step3_parser_code_sha256"]
            != policy["frozen_inputs"]["step3_parser_profile_code"]["sha256"]
            or attestation["custody_status"] != "FORMAL_PARENT_SEALED"
            or preceremony.HEX_SHA256_RE.fullmatch(
                str(attestation["custody_parent_seal_sha256"])
            )
            is None
            or int(attestation["history_safe_occurrence_count"])
            != len(safe_rows)
            or attestation["history_safe_occurrences_sha256"]
            != common.canonical_rows_sha256(
                safe_rows,
                order_fields=(
                    "world_uid",
                    "seller_uid",
                    "item_uid",
                    "source_field",
                    "contact_type",
                    "normalized_value",
                ),
            )
            or int(attestation["history_item_index_count"])
            != len(item_rows)
            or attestation["history_item_index_sha256"]
            != common.canonical_rows_sha256(
                item_rows,
                order_fields=("world_uid", "seller_uid", "item_uid"),
            )
            or int(attestation["parser_artifact_row_count"]) != len(safe_rows)
            or preceremony.HEX_SHA256_RE.fullmatch(
                str(attestation["parser_artifact_sha256"])
            )
            is None
        ):
            raise FormalBuildError("Formal projection attestation validation failed")
        attestation_worlds.add(world_uid)
    if attestation_worlds != expected_worlds:
        raise FormalBuildError("Formal projection attestation world closure failed")
    return history_features._compute_identity33(
        policy,
        feature_names=feature_names,
        excluded_names=excluded_names,
        pair_rows_by_world=pair_rows_by_world,
        history_rows_by_world=history_rows_by_world,
        history_safe_occurrence_count=len(history_safe_occurrences),
        history_item_index_count=len(history_item_index),
    )


def materialize_world_bundle(
    *,
    execution_policy: dict[str, Any],
    template: dict[str, Any],
    fixture: dict[str, Any],
    style_profile: dict[str, Any],
    split: str,
    world_record: Mapping[str, Any],
    generator_capabilities: Mapping[str, str],
    historical_forbidden_hashes: frozenset[str],
    allocated_identity_hashes: set[str],
    maximum_identity_counter: int,
    force_first_candidate_collision: bool = False,
) -> dict[str, Any]:
    """Build and independently validate one complete full-378 world."""

    import step28_v13_common as common
    import step28_v13_identity_values as identity_values
    import step28_v13_production_chain as production
    import step28_v13_profiles as profiles
    import step28_v13_world_builder as world_builder

    if split not in SPLITS or set(generator_capabilities) != set(GENERATOR_ROLES):
        raise FormalBuildError("World materialization capability boundary drift")
    world_uid = str(world_record["world_uid"])
    with mounted_structure_capability(
        split=split,
        structure_key_hex=str(generator_capabilities["structure"]),
    ):
        world = world_builder.build_world(
            policy=execution_policy,
            template=template,
            fixture=fixture,
            style_profile=style_profile,
            mode="formal",
            world_record=world_record,
            structure_key_hex=str(generator_capabilities["structure"]),
        )
    world, identity_allocation = preceremony.remap_world_identity_values(
        world,
        template=template,
        key_hex=str(generator_capabilities["identity_remap"]),
        historical_forbidden=historical_forbidden_hashes,
        allocated_in_current_run=allocated_identity_hashes,
        maximum_counter=maximum_identity_counter,
        force_first_candidate_collision=force_first_candidate_collision,
    )
    public = world["public"]
    private = world["private"]
    parsed = production.parse_observed_world(
        execution_policy,
        mode="formal",
        split=split,
        sellers=public["sellers"],
        items=public["items"],
    )
    parser_audit = production.validate_parser_against_private_plan(
        execution_policy,
        mode="formal",
        split=split,
        sellers=public["sellers"],
        items=public["items"],
        parsed_rows=parsed,
        identity_slots_audit=private["identity_slots_audit"],
        noise_slots_audit=private["noise_slots_audit"],
        render_asts=private["render_asts"],
    )
    projection = preceremony.project_registered_visible_text(
        policy=execution_policy,
        template=template,
        sellers=public["sellers"],
        items=public["items"],
        parsed_rows=parsed,
    )
    history_rows = production.project_history_safe_occurrences(
        execution_policy,
        mode="formal",
        split=split,
        sellers=public["sellers"],
        items=public["items"],
        parsed_rows=parsed,
    )
    item_index = _history_item_index(public["items"])
    attestation = _build_formal_history_attestation(
        policy=execution_policy,
        split=split,
        world_uid=world_uid,
        sellers=public["sellers"],
        raw_items=public["items"],
        history_safe_occurrences=history_rows,
        history_item_index=item_index,
        parsed_rows=parsed,
        identity_slots_audit=private["identity_slots_audit"],
        noise_slots_audit=private["noise_slots_audit"],
        render_asts=private["render_asts"],
    )
    identity33, identity33_audit = _build_formal_identity33(
        policy=execution_policy,
        split=split,
        history_safe_occurrences=history_rows,
        history_item_index=item_index,
        projection_attestations=[attestation],
        complete_model_pair_endpoints=public["complete_model_pair_endpoints"],
    )
    seller_profiles, profile_audit = profiles.build_world_profiles(
        execution_policy,
        mode="formal",
        split=split,
        sellers=public["sellers"],
        items=projection["profile_safe_items"],
    )
    labels = preceremony.validate_full_pair_labels(
        pair_rows=public["complete_model_pair_endpoints"],
        controller_membership=private["controller_membership"],
        expected_world_uid=world_uid,
    )
    queries, qrels = _retrieval_rows(
        split=split,
        world_uid=world_uid,
        seller_uids=[str(row["seller_uid"]) for row in public["sellers"]],
        controller_membership=private["controller_membership"],
        query_key_hex=str(generator_capabilities["query"]),
    )
    mechanism_audit = preceremony.validate_world_scoped_mechanism_slots(
        private["mechanism_assignments"],
        expected_world_count=1,
    )
    identity_hashes = sorted(
        identity_values.value_hash(str(row["identity_value"]))
        for row in private["identity_assets"]
    )
    if (
        len(identity_hashes) != len(set(identity_hashes))
        or set(identity_hashes) & historical_forbidden_hashes
        or int(identity33_audit["feature_count"]) != 33
        or len(identity33) != 378
        or len(seller_profiles) != 28
        or len(labels) != 378
        or sum(int(row["label"]) for row in labels) != 20
        or parser_audit.get("exact_rows_and_flags") is not True
        or int(profile_audit["seller_count"]) != 28
        or mechanism_audit["world_scoped_unique_key_count"] != 12
    ):
        raise FormalBuildError("Materialized world aggregate contract drift")
    return {
        "split": split,
        "world_uid": world_uid,
        "public": {
            "worlds": [{"world_uid": world_uid}],
            "sellers": [dict(row) for row in public["sellers"]],
            "redacted_items": projection["redacted_items"],
            "seller_profiles": seller_profiles,
            "complete_model_pair_endpoints": [
                dict(row) for row in public["complete_model_pair_endpoints"]
            ],
            "identity33_all_pairs": identity33,
            "retrieval_queries": queries,
        },
        "private": {
            "controller_membership": [
                dict(row) for row in private["controller_membership"]
            ],
            "controller_style_groups": [
                dict(row) for row in private["controller_style_groups"]
            ],
            "mechanism_assignments": [
                dict(row) for row in private["mechanism_assignments"]
            ],
            "identity_assets": [dict(row) for row in private["identity_assets"]],
            "positive_targets": [
                dict(row) for row in private["positive_targets"]
            ],
            "negative_flags": [dict(row) for row in private["negative_flags"]],
            "classification_labels": labels,
            "retrieval_qrels": qrels,
            "raw_identity_bearing_items": [
                dict(row) for row in public["items"]
            ],
            "history_safe_occurrences": history_rows,
            "history_item_index": item_index,
            "parsed_identity_occurrences": parsed,
            "renderer_identity_slots": [
                dict(row) for row in private["identity_slots_audit"]
            ],
            "renderer_identity_slot_edits": [
                dict(row) for row in private["identity_slots_edit"]
            ],
            "renderer_noise_slots": [
                dict(row) for row in private["noise_slots_audit"]
            ],
            "registered_override_audit": [
                dict(row) for row in private["override_audit"]
            ],
            "render_asts": [dict(row) for row in private["render_asts"]],
            "solver_audit": [dict(private["solver_audit"])],
            "projection_attestations": [dict(attestation)],
        },
        "audit": {
            "identity_allocation": identity_allocation,
            "identity_value_hashes": identity_hashes,
            "identity33_audit": identity33_audit,
            "parser_audit_sha256": common.canonical_sha256(parser_audit),
            "profile_audit_sha256": common.canonical_sha256(profile_audit),
            "mechanism_scope": mechanism_audit,
        },
    }


def aggregate_world_counts(worlds: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    world_uids = [str(row["world_uid"]) for row in worlds]
    public_pair_count = sum(
        len(row["public"]["complete_model_pair_endpoints"]) for row in worlds
    )
    labels = [
        label
        for row in worlds
        for label in row["private"]["classification_labels"]
    ]
    return {
        "world_count": len(set(world_uids)),
        "pair_count": public_pair_count,
        "positive_count": sum(int(row["label"]) for row in labels),
        "negative_count": sum(1 - int(row["label"]) for row in labels),
        "identity_asset_count": sum(
            len(row["private"]["identity_assets"]) for row in worlds
        ),
        "retrieval_query_count": sum(
            len(row["public"]["retrieval_queries"]) for row in worlds
        ),
        "retrieval_qrel_count": sum(
            len(row["private"]["retrieval_qrels"]) for row in worlds
        ),
    }


def runtime_versions() -> dict[str, str]:
    import numpy
    import scipy
    import sklearn

    return {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
    }


def _load_pinned_self_hashed_json(
    spec: Mapping[str, Any], *, label: str
) -> tuple[Path, dict[str, Any]]:
    path = _verify_pin(spec, label=label)
    document = preceremony.load_json_strict(path)
    preceremony.validate_canonical_self_hash(document, label=label)
    if (
        "canonical_self_hash" in spec
        and document["canonical_self_hash"] != spec["canonical_self_hash"]
    ):
        raise FormalBuildError(f"Pinned canonical self-hash drift: {label}")
    return path, document


def load_and_validate_prelock(
    path: Path = DEFAULT_PRELOCK_PATH,
) -> dict[str, Any]:
    """Validate the frozen seed-only prelock and every pinned byte."""

    path = require_canonical_path(
        path, DEFAULT_PRELOCK_PATH, label="v1.12 formal prelock"
    )
    prelock = preceremony.load_json_strict(path)
    preceremony.validate_canonical_self_hash(
        prelock, label="v1.12 formal prelock"
    )
    expected_authorizations = {
        "formal_seed_ceremony": True,
        "formal_train_generation": False,
        "formal_development_generation": False,
        "formal_audit_a_generation": False,
        "formal_audit_b_generation": False,
        "model_training": False,
        "audit_truth_unsealing": False,
    }
    if (
        prelock.get("version")
        != "2026-08-03-step28-v13-v1-12-formal-prelock-v1"
        or prelock.get("status") != "READY_FOR_SEED_CEREMONY_ONLY"
        or prelock.get("run_id")
        != "v13_training_ready_v1_12_cleanroom_20260803"
        or prelock.get("authorizations") != expected_authorizations
        or prelock.get("formal_master_or_capability_created") is not False
    ):
        raise FormalBuildError("Formal prelock authorization/status drift")

    draft_path, draft = _load_pinned_self_hashed_json(
        prelock["formal_build_draft"], label="v1.12 pinned formal draft"
    )
    validated_draft = load_and_validate_draft(draft_path)
    if draft != validated_draft["draft"]:
        raise FormalBuildError("Formal prelock draft replay drift")

    evidence = prelock.get("design_evidence", {})
    if set(evidence) != {
        "two_world_persisted_stage",
        "exact_shortcut_preflight",
        "full_repository_tests",
    }:
        raise FormalBuildError("Formal prelock evidence set drift")
    _two_path, two_world = _load_pinned_self_hashed_json(
        evidence["two_world_persisted_stage"],
        label="v1.12 two-world persisted-stage receipt",
    )
    if (
        two_world.get("status")
        != "PASS_DESIGN_ONLY_TWO_WORLD_PERSISTED_REPLAY"
        or int(two_world.get("design_world_count", -1)) != 2
        or int(two_world.get("design_pair_count", -1)) != 756
        or int(two_world.get("m1_structural_receipt_count", -1)) != 5
        or two_world.get("formal_authorization_used") is not False
        or int(two_world.get("formal_rows_produced", -1)) != 0
        or two_world.get("formal_seed_or_key_access") is not False
        or two_world.get("producer_path")
        != "scripts/step28_v13_v1_12_generate_split.py"
        or two_world.get("producer_sha256")
        != preceremony.sha256_file(
            ROOT / "scripts" / "step28_v13_v1_12_generate_split.py"
        )
        or two_world.get("formal_common_sha256")
        != preceremony.sha256_file(Path(__file__))
        or two_world.get("formal_build_draft_sha256")
        != preceremony.sha256_file(draft_path)
        or two_world.get("runtime_versions") != runtime_versions()
        or two_world.get("temporary_stage_deleted_on_exit") is not True
        or len(two_world.get("stages", [])) != 2
    ):
        raise FormalBuildError("Two-world persisted-stage evidence drift")
    stages = two_world["stages"]
    if (
        [stage.get("split") for stage in stages] != ["train", "development"]
        or any(
            stage.get("status")
            != "PASS_DESIGN_ONLY_PERSISTED_STAGE_REPLAY"
            or int(stage.get("world_count", -1)) != 1
            or int(stage.get("pair_count", -1)) != 378
            for stage in stages
        )
    ):
        raise FormalBuildError("Two-world persisted-stage replay detail drift")
    _shortcut_path, shortcut = _load_pinned_self_hashed_json(
        evidence["exact_shortcut_preflight"],
        label="v1.12 exact shortcut receipt",
    )
    if (
        shortcut.get("status")
        != "PASS_DESIGN_ONLY_EXACT_SHORTCUT_PREFLIGHT"
        or int(shortcut.get("train_world_count", -1)) != 500
        or int(shortcut.get("development_world_count", -1)) != 500
        or int(shortcut.get("train_row_count", -1)) != 189000
        or int(shortcut.get("development_row_count", -1)) != 189000
        or shortcut.get("gates")
        != {
            "all_logistic_optimizer_audits": True,
            "combined_ap_uplift": True,
            "combined_auc": True,
            "combined_bootstrap_ap_uplift": True,
            "combined_bootstrap_auc": True,
            "single_feature_auc": True,
        }
        or shortcut.get("formal_seed_or_key_access") is not False
        or int(shortcut.get("formal_rows_produced", -1)) != 0
        or shortcut.get("formal_authorization_used") is not False
        or shortcut.get("scientific_metrics_produced") is not False
        or shortcut.get("producer_path")
        != "scripts/step28_v13_v1_12_exact_shortcut_preflight.py"
        or shortcut.get("producer_sha256")
        != preceremony.sha256_file(
            ROOT
            / "scripts"
            / "step28_v13_v1_12_exact_shortcut_preflight.py"
        )
        or shortcut.get("formal_common_sha256")
        != preceremony.sha256_file(Path(__file__))
        or shortcut.get("formal_build_draft_sha256")
        != preceremony.sha256_file(draft_path)
        or shortcut.get("runtime_versions") != runtime_versions()
        or int(shortcut.get("feature_count", -1)) != 24
        or int(shortcut.get("bootstrap_replicates", -1))
        != int(draft["shortcut_preflight"]["bootstrap_replicates"])
        or shortcut.get("fast_full_parity", {}).get("status")
        != "PASS_FAST_FULL_NULL_NUISANCE_PARITY"
        or int(shortcut.get("fast_full_parity", {}).get("pair_count", -1))
        != 756
    ):
        raise FormalBuildError("Exact shortcut evidence drift")
    shortcut_config = draft["shortcut_preflight"]
    if (
        float(shortcut.get("point_max_single_symmetric_auc", float("inf")))
        > float(shortcut_config["single_feature_maximum_symmetric_auc"])
        or float(shortcut.get("point_max_combined_symmetric_auc", float("inf")))
        > float(shortcut_config["combined_maximum_symmetric_auc"])
        or float(
            shortcut.get(
                "bootstrap_95_upper_max_combined_symmetric_auc",
                float("inf"),
            )
        )
        > float(
            shortcut_config["combined_bootstrap_95_upper_symmetric_auc"]
        )
        or float(shortcut.get("point_max_combined_ap_uplift", float("inf")))
        > float(shortcut_config["combined_maximum_ap_uplift"])
        or float(
            shortcut.get(
                "bootstrap_95_upper_max_combined_ap_uplift", float("inf")
            )
        )
        > float(shortcut_config["combined_bootstrap_95_upper_ap_uplift"])
    ):
        raise FormalBuildError("Exact shortcut numeric gate replay failed")
    optimizer = shortcut.get("optimizer_audit", {})
    optimizer_audits = [
        *list(optimizer.get("fold_logistic_optimizer_audits", [])),
        optimizer.get("full_train_logistic_optimizer_audit"),
    ]
    if len(optimizer_audits) != int(shortcut_config["fold_count"]) + 1:
        raise FormalBuildError("Exact shortcut optimizer audit count drift")
    optimizer_keys = {
        "solver_success",
        "convergence_warning_count",
        "iteration_count",
        "maximum_iterations",
        "normalized_gradient",
        "gradient_tolerance",
        "objective_finite",
        "preceremony_exact_configuration",
    }
    for audit in optimizer_audits:
        if not isinstance(audit, Mapping):
            raise FormalBuildError("Exact shortcut optimizer audit is malformed")
        preceremony.validate_optimizer_audit(
            {name: audit.get(name) for name in optimizer_keys}
        )
        if (
            float(audit.get("l2", float("nan")))
            != float(shortcut_config["logistic_l2"])
            or not isinstance(audit.get("objective"), (int, float))
            or not math.isfinite(float(audit["objective"]))
        ):
            raise FormalBuildError("Exact shortcut optimizer parameters drift")
    _tests_path, tests = _load_pinned_self_hashed_json(
        evidence["full_repository_tests"],
        label="v1.12 full repository test receipt",
    )
    if (
        tests.get("status") != "PASS_FULL_REPOSITORY_TESTS"
        or int(tests.get("failed_count", -1)) != 0
        or int(tests.get("error_count", -1)) != 0
        or int(tests.get("test_count", -1)) < 1
        or tests.get("formal_seed_or_key_access") is not False
        or int(tests.get("formal_rows_produced", -1)) != 0
        or int(tests.get("passed_count", -1))
        + int(tests.get("skipped_count", -1))
        != int(tests.get("test_count", -1))
        or tests.get("runtime_versions") != runtime_versions()
    ):
        raise FormalBuildError("Full repository test evidence drift")

    closure = prelock.get("source_closure", {})
    members = list(closure.get("members", []))
    paths = [str(record.get("path", "")) for record in members]
    if (
        len(members) != int(closure.get("member_count", -1))
        or paths != sorted(paths, key=lambda value: value.encode("utf-8"))
        or len(set(paths)) != len(paths)
        or preceremony.canonical_sha256(members)
        != closure.get("canonical_sha256")
        or int(closure.get("failed_version_member_count", -1)) != 0
        or int(closure.get("c40_member_count", -1)) != 0
        or any("c40" in value.casefold() for value in paths)
    ):
        raise FormalBuildError("Formal source closure registry drift")
    for index, member in enumerate(members):
        _verify_pin(member, label=f"v1.12 formal source closure {index}")

    if (
        tests.get("source_closure_canonical_sha256")
        != closure.get("canonical_sha256")
        or int(tests.get("source_closure_member_count", -1))
        != int(closure.get("member_count", -1))
    ):
        raise FormalBuildError("Full-test/source-closure evidence drift")

    if prelock.get("dependency_versions") != runtime_versions():
        raise FormalBuildError("Formal dependency version drift")
    custody = prelock.get("custody", {})
    release_inputs_root = DEFAULT_EXECUTION_LOCK_PATH.parent.relative_to(
        ROOT
    ).as_posix()
    expected_private_root = str(draft["release"]["private_root"])
    if (
        custody.get("private_root") != expected_private_root
        or custody.get("public_root") != draft["release"]["public_root"]
        or custody.get("private_seed_bundle_root")
        != f"{expected_private_root}/seed_custody"
        or custody.get("private_seed_stage_root")
        != f"{expected_private_root}/_seed_ceremony_stage"
        or custody.get("seed_ceremony_start_receipt_path")
        != f"{release_inputs_root}/seed_ceremony_start_receipt.json"
        or custody.get("public_ceremony_receipt_path")
        != f"{release_inputs_root}/seed_ceremony_receipt.json"
        or custody.get("train_development_execution_lock_path")
        != DEFAULT_EXECUTION_LOCK_PATH.relative_to(ROOT).as_posix()
        or custody.get("train_development_quality_receipt_path")
        != f"{release_inputs_root}/train_development_quality_gate.json"
        or custody.get("audit_a_generation_lock_path")
        != DEFAULT_AUDIT_A_LOCK_PATH.relative_to(ROOT).as_posix()
        or custody.get("audit_b_generation_lock_path")
        != DEFAULT_AUDIT_B_LOCK_PATH.relative_to(ROOT).as_posix()
        or custody.get("permanent_failure_receipt_path")
        != f"{release_inputs_root}/permanent_failure_receipt.json"
        or custody.get("master_mounted_to_generator") is not False
        or custody.get("master_mounted_to_model") is not False
        or custody.get("one_draw_per_split_no_retry") is not True
        or custody.get("private_root_git_ignored") is not True
    ):
        raise FormalBuildError("Formal prelock custody boundary drift")
    return {
        "prelock": prelock,
        "draft": draft,
        "baseline": validated_draft["baseline"],
        "source_members": members,
        "two_world_receipt": two_world,
        "shortcut_receipt": shortcut,
        "test_receipt": tests,
    }


def load_and_validate_execution_lock(
    path: Path = DEFAULT_EXECUTION_LOCK_PATH,
) -> dict[str, Any]:
    """Validate the post-ceremony train/development execution lock."""

    path = require_canonical_path(
        path,
        DEFAULT_EXECUTION_LOCK_PATH,
        label="v1.12 train/development execution lock",
    )
    lock = preceremony.load_json_strict(path)
    preceremony.validate_canonical_self_hash(
        lock, label="v1.12 train/development execution lock"
    )
    validated_prelock = load_and_validate_prelock(
        preceremony._repo_path(str(lock["prelock"]["path"]))
    )
    prelock = validated_prelock["prelock"]
    if (
        preceremony.sha256_file(
            preceremony._repo_path(str(lock["prelock"]["path"]))
        )
        != lock["prelock"]["sha256"]
        or prelock["canonical_self_hash"]
        != lock["prelock"]["canonical_self_hash"]
    ):
        raise FormalBuildError("Execution lock prelock pin drift")
    expected_authorizations = {
        "formal_seed_ceremony": False,
        "formal_train_generation": True,
        "formal_development_generation": True,
        "formal_audit_a_generation": False,
        "formal_audit_b_generation": False,
        "model_training": False,
        "audit_truth_unsealing": False,
    }
    if (
        lock.get("version")
        != "2026-08-03-step28-v13-v1-12-train-development-lock-v1"
        or lock.get("status")
        != "READY_FOR_TRAIN_DEVELOPMENT_GENERATION"
        or lock.get("run_id") != prelock["run_id"]
        or lock.get("authorizations") != expected_authorizations
        or lock.get("source_closure_canonical_sha256")
        != prelock["source_closure"]["canonical_sha256"]
    ):
        raise FormalBuildError("Execution lock authorization/status drift")
    _receipt_path, receipt = _load_pinned_self_hashed_json(
        lock["ceremony_receipt"], label="v1.12 seed ceremony receipt"
    )
    start_path, start_receipt = _load_pinned_self_hashed_json(
        receipt["ceremony_start_receipt"],
        label="v1.12 seed ceremony start receipt",
    )
    if (
        receipt.get("status") != "PASS_ONE_SHOT_SEED_CEREMONY"
        or receipt.get("run_id") != prelock["run_id"]
        or receipt.get("raw_master_or_capability_serialized_publicly")
        is not False
        or receipt.get("master_draw_count") != 4
        or receipt.get("one_draw_per_split_no_retry") is not True
        or start_receipt.get("status")
        != "SEED_CEREMONY_STARTED_NO_REDRAW"
        or start_receipt.get("run_id") != prelock["run_id"]
        or start_receipt.get("master_draw_count_at_start_receipt") != 0
        or start_receipt.get("raw_master_or_capability_present") is not False
        or start_path
        != preceremony._repo_path(
            str(prelock["custody"]["seed_ceremony_start_receipt_path"])
        )
    ):
        raise FormalBuildError("Seed ceremony receipt semantic drift")
    if lock.get("master_commitments") != receipt.get("master_commitments"):
        raise FormalBuildError("Execution lock master commitment drift")
    if (
        lock.get("generator_capability_commitments")
        != receipt.get("generator_capability_commitments")
        or lock.get("m1_capability_commitments")
        != receipt.get("m1_capability_commitments")
    ):
        raise FormalBuildError("Execution lock capability commitment drift")
    private_files = lock.get("private_generator_capability_files", {})
    if set(private_files) != set(SPLITS):
        raise FormalBuildError("Execution lock private generator file set drift")
    for split, spec in private_files.items():
        _verify_pin(
            spec, label=f"v1.12 private generator capability {split}"
        )
        if preceremony.HEX_SHA256_RE.fullmatch(
            str(spec.get("canonical_self_hash", ""))
        ) is None:
            raise FormalBuildError(
                "Execution lock private generator self-hash pin is malformed"
            )
    private_m1_files = lock.get("private_m1_capability_files", {})
    if set(private_m1_files) != set(M1_ROLES):
        raise FormalBuildError("Execution lock private M1 file set drift")
    for role, spec in private_m1_files.items():
        _verify_pin(spec, label=f"v1.12 private M1 capability {role}")
        if preceremony.HEX_SHA256_RE.fullmatch(
            str(spec.get("canonical_self_hash", ""))
        ) is None:
            raise FormalBuildError(
                "Execution lock private M1 self-hash pin is malformed"
            )
    return {
        **validated_prelock,
        "execution_lock": lock,
        "ceremony_receipt": receipt,
        "ceremony_start_receipt": start_receipt,
    }


def load_split_generator_capabilities(
    *,
    split: str,
    execution_lock_path: Path = DEFAULT_EXECUTION_LOCK_PATH,
) -> tuple[dict[str, str], dict[str, str], dict[str, Any]]:
    """Load one split capability without ever opening its master document."""

    if split not in SPLITS:
        raise FormalBuildError("Unknown formal split capability request")
    validated = load_and_validate_execution_lock(execution_lock_path)
    lock = validated["execution_lock"]
    if lock["authorizations"][f"formal_{split}_generation"] is not True:
        raise FormalBuildError(f"Formal split is not authorized: {split}")
    _path, document = _load_pinned_self_hashed_json(
        lock["private_generator_capability_files"][split],
        label=f"v1.12 private generator capability {split}",
    )
    capabilities = document.get("generator_capabilities", {})
    if (
        document.get("version")
        != "2026-08-03-step28-v13-v1-12-generator-capabilities-v1"
        or document.get("run_id") != lock["run_id"]
        or document.get("split") != split
        or set(capabilities) != set(GENERATOR_ROLES)
        or document.get("master_present") is not False
        or capability_commitments(
            {"split": split, "generator": capabilities, "m1": {}}
        )["generator"]
        != lock["generator_capability_commitments"][split]
    ):
        raise FormalBuildError("Private generator capability document drift")
    return (
        {str(key): str(value) for key, value in capabilities.items()},
        {
            name: str(lock["generator_capability_commitments"][name]["structure"])
            for name in SPLITS
        },
        validated,
    )


def load_train_m1_capability(
    *,
    role: str,
    execution_lock_path: Path = DEFAULT_EXECUTION_LOCK_PATH,
) -> tuple[str, dict[str, Any]]:
    """Open exactly one train M1 key document for one bounded process."""

    if role not in M1_ROLES:
        raise FormalBuildError("Unknown train M1 capability role")
    validated = load_and_validate_execution_lock(execution_lock_path)
    lock = validated["execution_lock"]
    if lock["authorizations"]["formal_train_generation"] is not True:
        raise FormalBuildError("Train M1 capability is not authorized")
    files = lock.get("private_m1_capability_files", {})
    if set(files) != set(M1_ROLES):
        raise FormalBuildError("Execution lock private M1 file set drift")
    _path, document = _load_pinned_self_hashed_json(
        files[role], label=f"v1.12 private M1 capability {role}"
    )
    value = str(document.get("rewire_key_hex", ""))
    if (
        document.get("version")
        != "2026-08-03-step28-v13-v1-12-m1-capability-v1"
        or document.get("run_id") != lock["run_id"]
        or document.get("split") != "train"
        or document.get("role") != role
        or preceremony.HEX_SHA256_RE.fullmatch(value) is None
        or hashlib.sha256(bytes.fromhex(value)).hexdigest()
        != lock["m1_capability_commitments"][role]
        or document.get("master_present") is not False
        or document.get("other_m1_capability_present") is not False
    ):
        raise FormalBuildError("Private M1 capability document drift")
    return value, validated


def _load_and_validate_audit_a_lock(path: Path) -> dict[str, Any]:
    path = require_canonical_path(
        path, DEFAULT_AUDIT_A_LOCK_PATH, label="v1.12 Audit A lock"
    )
    lock = preceremony.load_json_strict(path)
    preceremony.validate_canonical_self_hash(
        lock, label="v1.12 Audit A generation lock"
    )
    parent_path = preceremony._repo_path(str(lock["parent_execution_lock"]["path"]))
    validated = load_and_validate_execution_lock(parent_path)
    parent = validated["execution_lock"]
    expected_authorizations = {
        "formal_seed_ceremony": False,
        "formal_train_generation": False,
        "formal_development_generation": False,
        "formal_audit_a_generation": True,
        "formal_audit_b_generation": False,
        "model_training": False,
        "audit_truth_unsealing": False,
    }
    if (
        lock.get("version")
        != "2026-08-03-step28-v13-v1-12-audit-a-generation-lock-v1"
        or lock.get("status") != "READY_FOR_AUDIT_A_GENERATION_ONLY"
        or lock.get("run_id") != parent["run_id"]
        or lock.get("authorizations") != expected_authorizations
        or lock.get("formal_split_order") != ["audit_a"]
        or lock.get("audit_b_requires_published_audit_a") is not True
        or preceremony.sha256_file(parent_path)
        != lock["parent_execution_lock"]["sha256"]
        or parent["canonical_self_hash"]
        != lock["parent_execution_lock"]["canonical_self_hash"]
        or lock.get("source_closure_canonical_sha256")
        != parent["source_closure_canonical_sha256"]
        or lock.get("master_commitments") != parent["master_commitments"]
        or lock.get("generator_capability_commitments")
        != parent["generator_capability_commitments"]
        or lock.get("private_generator_capability_files")
        != parent["private_generator_capability_files"]
    ):
        raise FormalBuildError("Audit generation lock status/lineage drift")
    quality_path, quality = _load_pinned_self_hashed_json(
        lock["train_development_quality_receipt"],
        label="v1.12 train/development quality receipt",
    )
    if (
        quality.get("status")
        != "PASS_FORMAL_TRAIN_DEVELOPMENT_QUALITY_GATE"
        or quality.get("run_id") != parent["run_id"]
        or quality.get("c40_generated_or_read") is not False
        or quality.get("audit_a_or_b_truth_read") is not False
    ):
        raise FormalBuildError("Audit lock quality evidence drift")
    publications = lock.get("train_development_publication_receipts", {})
    if set(publications) != {"train", "development"}:
        raise FormalBuildError("Audit lock publication evidence set drift")
    publication_documents: dict[str, dict[str, Any]] = {}
    for split, spec in publications.items():
        _publication_path, document = _load_pinned_self_hashed_json(
            spec, label=f"v1.12 formal {split} publication receipt"
        )
        if (
            document.get("status")
            != "PASS_FORMAL_SPLIT_PUBLISHED_NO_REPLACE"
            or document.get("split") != split
            or document.get("quality_receipt")["sha256"]
            != preceremony.sha256_file(quality_path)
        ):
            raise FormalBuildError("Audit lock publication receipt drift")
        publication_documents[split] = document
    return {
        **validated,
        "audit_lock": lock,
        "audit_a_lock": lock,
        "quality_receipt": quality,
        "publication_receipts": publication_documents,
    }


def _load_and_validate_audit_b_lock(path: Path) -> dict[str, Any]:
    path = require_canonical_path(
        path, DEFAULT_AUDIT_B_LOCK_PATH, label="v1.12 Audit B lock"
    )
    lock = preceremony.load_json_strict(path)
    preceremony.validate_canonical_self_hash(
        lock, label="v1.12 Audit B generation lock"
    )
    parent_path = preceremony._repo_path(
        str(lock["parent_audit_a_lock"]["path"])
    )
    validated = _load_and_validate_audit_a_lock(parent_path)
    parent = validated["audit_a_lock"]
    expected_authorizations = {
        "formal_seed_ceremony": False,
        "formal_train_generation": False,
        "formal_development_generation": False,
        "formal_audit_a_generation": False,
        "formal_audit_b_generation": True,
        "model_training": False,
        "audit_truth_unsealing": False,
    }
    if (
        lock.get("version")
        != "2026-08-03-step28-v13-v1-12-audit-b-generation-lock-v1"
        or lock.get("status") != "READY_FOR_AUDIT_B_GENERATION_ONLY"
        or lock.get("run_id") != parent["run_id"]
        or lock.get("authorizations") != expected_authorizations
        or lock.get("formal_split_order") != ["audit_b"]
        or lock.get("audit_a_published_before_authorization") is not True
        or preceremony.sha256_file(parent_path)
        != lock["parent_audit_a_lock"]["sha256"]
        or parent["canonical_self_hash"]
        != lock["parent_audit_a_lock"]["canonical_self_hash"]
        or lock.get("source_closure_canonical_sha256")
        != parent["source_closure_canonical_sha256"]
        or lock.get("master_commitments") != parent["master_commitments"]
        or lock.get("generator_capability_commitments")
        != parent["generator_capability_commitments"]
        or lock.get("private_generator_capability_files")
        != parent["private_generator_capability_files"]
    ):
        raise FormalBuildError("Audit B generation lock status/lineage drift")
    publication_path, publication = _load_pinned_self_hashed_json(
        lock["audit_a_publication_receipt"],
        label="v1.12 Audit A publication receipt",
    )
    quality_path, quality = _load_pinned_self_hashed_json(
        lock["audit_a_quality_receipt"],
        label="v1.12 Audit A sealed quality receipt",
    )
    if (
        publication_path
        != DEFAULT_AUDIT_A_LOCK_PATH.parent / "audit_a_publication_receipt.json"
        or publication.get("status")
        != "PASS_FORMAL_SPLIT_PUBLISHED_NO_REPLACE"
        or publication.get("run_id") != parent["run_id"]
        or publication.get("split") != "audit_a"
        or publication.get("quality_receipt", {}).get("sha256")
        != preceremony.sha256_file(quality_path)
        or publication.get("quality_receipt", {}).get("canonical_self_hash")
        != quality.get("canonical_self_hash")
        or quality_path
        != DEFAULT_AUDIT_A_LOCK_PATH.parent / "audit_a_quality_gate.json"
        or quality.get("status")
        != "PASS_FORMAL_SEALED_AUDIT_SPLIT_QUALITY"
        or quality.get("run_id") != parent["run_id"]
        or quality.get("split") != "audit_a"
        or quality.get("classification_labels_parsed") is not False
        or quality.get("retrieval_qrels_parsed") is not False
        or quality.get("controller_membership_parsed") is not False
        or quality.get("model_training_or_prediction_started") is not False
    ):
        raise FormalBuildError("Audit B lock Audit A evidence drift")
    return {
        **validated,
        "audit_lock": lock,
        "audit_b_lock": lock,
        "audit_a_publication_receipt": publication,
        "audit_a_quality_receipt": quality,
    }


def load_and_validate_audit_lock(
    path: Path = DEFAULT_AUDIT_A_LOCK_PATH,
) -> dict[str, Any]:
    """Validate exactly one rung of the Audit A then Audit B ladder."""

    resolved = path.resolve()
    if resolved == DEFAULT_AUDIT_A_LOCK_PATH.resolve():
        return _load_and_validate_audit_a_lock(resolved)
    if resolved == DEFAULT_AUDIT_B_LOCK_PATH.resolve():
        return _load_and_validate_audit_b_lock(resolved)
    raise FormalBuildError("Audit generation lock is not at a canonical path")


def load_audit_generator_capabilities(
    *, split: str, audit_lock_path: Path = DEFAULT_AUDIT_A_LOCK_PATH
) -> tuple[dict[str, str], dict[str, str], dict[str, Any]]:
    """Load one Audit A/B generator bundle after target-quality authorization."""

    if split not in {"audit_a", "audit_b"}:
        raise FormalBuildError("Audit capability request is not Audit A/B")
    validated = load_and_validate_audit_lock(audit_lock_path)
    lock = validated["audit_lock"]
    if lock["authorizations"][f"formal_{split}_generation"] is not True:
        raise FormalBuildError(f"Formal audit split is not authorized: {split}")
    _path, document = _load_pinned_self_hashed_json(
        lock["private_generator_capability_files"][split],
        label=f"v1.12 private generator capability {split}",
    )
    capabilities = document.get("generator_capabilities", {})
    if (
        document.get("version")
        != "2026-08-03-step28-v13-v1-12-generator-capabilities-v1"
        or document.get("run_id") != lock["run_id"]
        or document.get("split") != split
        or set(capabilities) != set(GENERATOR_ROLES)
        or document.get("master_present") is not False
        or capability_commitments(
            {"split": split, "generator": capabilities, "m1": {}}
        )["generator"]
        != lock["generator_capability_commitments"][split]
    ):
        raise FormalBuildError("Private audit generator capability drift")
    return (
        {str(key): str(value) for key, value in capabilities.items()},
        {
            name: str(lock["generator_capability_commitments"][name]["structure"])
            for name in SPLITS
        },
        validated,
    )
