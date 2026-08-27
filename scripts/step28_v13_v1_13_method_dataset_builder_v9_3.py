#!/usr/bin/env python3
"""Build the V9.3 1,004-world method-qualification root without pair truth."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any

import step28_v13_common as common
import step28_v13_structure as structure
import step28_v13_v1_13_balanced_schedule_v9_3 as balanced
import step28_v13_v1_13_method_policy_v9_3 as method_policy_module
import step28_v13_v1_13_method_world_v9_3 as method_world
import step28_v13_v1_13_registered_negative_plan_v9_3 as negative_plan
import step28_v13_v1_13_scientific_common_v9 as scientific
import step28_v13_v1_13_structure_matrix_v9_3 as structure_matrix


VERSION = (
    "2026-08-27-step28-v13-v1-13-method-dataset-builder-"
    "v9-3-r2-user-accepted-residual-22"
)
MODE = "development_smoke"
WORLD_COUNTS = {"train": 500, "development": 500, "audit_a": 2, "audit_b": 2}
SPLITS = tuple(WORLD_COUNTS)
DEFAULT_OUTPUT = common.repo_path(
    "reports/step28_v13_v1_13_scientific_builder/"
    "design_preflight_v9_3_r2_20260827/method_qualification_1004"
)
DEFAULT_AUTHORITY = common.repo_path(
    "private_custody/step28_v13_v1_13_v9_3_r2_method_random_authority.json"
)
AUTHORITY_VERSION = (
    "2026-08-27-step28-v13-v1-13-v9-3-r2-method-random-authority"
)
DEFAULT_PREBUILD_STRUCTURE_GATE_RESULT = common.repo_path(
    "reports/step28_v13_v1_13_balanced_schedule_v9_3/"
    "registered_negative_structure_gate_r2_20260827/"
    "structure_gate_result.json"
)
PAIR_FIELDS = tuple(method_world.PAIR_KEY_FIELDS)
IDENTITY_FIELDS: tuple[str, ...] | None = None


class MethodDatasetBuilderV93Error(common.ContractError):
    """Raised when the V9.3 method root cannot be published exactly."""


def _self_hash(payload: Mapping[str, Any]) -> str:
    value = deepcopy(dict(payload))
    value["canonical_self_sha256"] = None
    return common.canonical_sha256(value)


def _prebuild_gate_self_hash(payload: Mapping[str, Any]) -> str:
    """Reproduce the structure gate's remove-field self-hash convention."""

    value = deepcopy(dict(payload))
    value.pop("canonical_self_sha256", None)
    return common.canonical_sha256(value)


def _validate_authority(payload: Mapping[str, Any]) -> dict[str, Any]:
    expected_keys = {
        "version",
        "status",
        "canonical_self_sha256",
        "single_use",
        "world_counts",
        "method_policy_canonical_self_sha256",
        "keys",
    }
    if set(payload) != expected_keys:
        raise MethodDatasetBuilderV93Error("Random-authority schema drift")
    keys = payload.get("keys")
    key_fields = {
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
    if not isinstance(keys, Mapping) or set(keys) != key_fields:
        raise MethodDatasetBuilderV93Error("Random-authority key block drift")
    scalar_names = sorted(key_fields - {"rewire_key_hexes"})
    scalars = [str(keys[name]) for name in scalar_names]
    rewires = keys["rewire_key_hexes"]
    if not isinstance(rewires, list) or len(rewires) != 5:
        raise MethodDatasetBuilderV93Error("Random authority needs five rewire keys")
    values = [*scalars, *(str(value) for value in rewires)]
    if (
        payload.get("version") != AUTHORITY_VERSION
        or payload.get("status") != "FROZEN_FRESH_SINGLE_USE"
        or payload.get("single_use") is not True
        or payload.get("world_counts") != WORLD_COUNTS
        or payload.get("canonical_self_sha256") != _self_hash(payload)
        or any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in values
        )
        or len(values) != len(set(values))
    ):
        raise MethodDatasetBuilderV93Error("Random-authority value drift")
    return deepcopy(dict(payload))


def _validate_input_file_record(record: Mapping[str, Any]) -> None:
    if set(record) != {"path", "size_bytes", "sha256"}:
        raise MethodDatasetBuilderV93Error("Prebuild input file-record drift")
    path = common.repo_path(str(record["path"]))
    if (
        not path.is_file()
        or type(record["size_bytes"]) is not int
        or path.stat().st_size != record["size_bytes"]
        or common.sha256_file(path) != record["sha256"]
    ):
        raise MethodDatasetBuilderV93Error(
            f"Prebuild input payload drift: {record['path']}"
        )


def _validate_prebuild_structure_gate_result(
    payload: Mapping[str, Any],
    *,
    result_path: Path,
    method_policy: Mapping[str, Any],
    authority: Mapping[str, Any],
    authority_path: Path,
) -> dict[str, Any]:
    contract = method_policy["prebuild_structure_gate_contract"]
    expected_path = common.repo_path(str(contract["result_path"]))
    expected_fields = {
        "version",
        "status",
        "scientific_pass",
        "claim_boundary",
        "count_family_coverage",
        "matrix_commitments",
        "label_access_counts",
        "label_commitments",
        "probe_result",
        "probe_contract_audit",
        "hard_gates",
        "natural_text_generated",
        "identity_assets_generated",
        "method_root_generated",
        "audit_a_b_truth_read_count",
        "m0_m1_m2_m3_training_authorized",
        "plan_root_validation",
        "inputs",
        "canonical_self_sha256",
    }
    if (
        result_path.resolve() != expected_path.resolve()
        or result_path.resolve()
        != DEFAULT_PREBUILD_STRUCTURE_GATE_RESULT.resolve()
        or not result_path.is_file()
        or set(payload) != expected_fields
        or payload["version"] != contract["version"]
        or payload["status"] != contract["required_pass_status"]
        or payload["scientific_pass"] is not True
        or payload["claim_boundary"]
        != "frozen_registered_structure_probes_did_not_detect_a_shortcut_above_threshold"
        or payload["canonical_self_sha256"] != _prebuild_gate_self_hash(payload)
        or payload["natural_text_generated"] is not False
        or payload["identity_assets_generated"] is not False
        or payload["method_root_generated"] is not False
        or payload["audit_a_b_truth_read_count"] != 0
        or payload["m0_m1_m2_m3_training_authorized"] is not False
        or payload["label_access_counts"]
        != {"train": 1, "development": 1, "audit_a": 0, "audit_b": 0}
    ):
        raise MethodDatasetBuilderV93Error(
            "Successful prebuild structure-gate result drift"
        )
    coverage = payload["count_family_coverage"]
    if (
        not isinstance(coverage, Mapping)
        or common.canonical_sha256(coverage.get("coverage"))
        != contract["finite_preregistered_projection_map_sha256"]
        or coverage.get("coverage_semantics")
        != "frozen_view_projection_diagnostic_not_cellwise_or_cross_view_interaction_complete"
        or coverage.get("seller_slot_and_noise_visible_models_remain_separate")
        is not True
        or coverage.get("cross_view_interactions_tested") is not False
        or coverage.get("theoretical_5_324_cell_balance_certified") is not False
    ):
        raise MethodDatasetBuilderV93Error(
            "Prebuild finite projection-map boundary drift"
        )
    probe_audit = payload["probe_contract_audit"]
    if (
        not isinstance(probe_audit, Mapping)
        or probe_audit.get("model_count") != 4
        or probe_audit.get("matrix_concatenation_used") is not False
        or probe_audit.get("average_precision_baseline") != 20 / 378
        or probe_audit.get("bootstrap_replicates") != 9999
        or probe_audit.get("bootstrap_world_count") != 500
        or probe_audit.get("bootstrap_score_family_size") != 4
        or probe_audit.get("bootstrap_draws_raw_i8_c_sha256")
        != method_policy["bootstrap"]["draws_raw_i8_c_sha256"]
    ):
        raise MethodDatasetBuilderV93Error("Prebuild probe-contract audit drift")
    expected_gate_thresholds = {
        "single_feature_maximum_symmetric_roc_auc": method_policy[
            "quality_gates"
        ]["maximum_single_feature_symmetric_roc_auc"],
        "family_maximum_symmetric_roc_auc": method_policy["quality_gates"][
            "maximum_family_symmetric_roc_auc"
        ],
        "family_maximum_average_precision_uplift": method_policy[
            "quality_gates"
        ]["maximum_family_average_precision_uplift"],
        "bootstrap_95_upper_symmetric_roc_auc": method_policy["quality_gates"][
            "bootstrap_95_upper_symmetric_roc_auc"
        ],
        "bootstrap_95_upper_average_precision_uplift": method_policy[
            "quality_gates"
        ]["bootstrap_95_upper_average_precision_uplift"],
    }
    hard_gates = payload["hard_gates"]
    if (
        not isinstance(hard_gates, list)
        or [row.get("gate") for row in hard_gates]
        != list(expected_gate_thresholds)
    ):
        raise MethodDatasetBuilderV93Error("Prebuild hard-gate registry drift")
    for row in hard_gates:
        name = row["gate"]
        if (
            set(row) != {"gate", "observed", "maximum", "passed"}
            or row["passed"] is not True
            or not math.isfinite(float(row["observed"]))
            or not math.isfinite(float(row["maximum"]))
            or float(row["maximum"]) != float(expected_gate_thresholds[name])
            or float(row["observed"]) > float(row["maximum"])
        ):
            raise MethodDatasetBuilderV93Error(
                f"Prebuild hard gate did not pass exactly: {name}"
            )
    inputs = payload["inputs"]
    expected_input_fields = {
        "method_policy",
        "method_policy_canonical_self_sha256",
        "parent_policy",
        "text_template",
        "style_profile",
        "runtime_source_files",
        "authority_commitment",
        "balanced_schedule_files",
        "published_plan_root_files",
        "joint_signatures",
    }
    if not isinstance(inputs, Mapping) or set(inputs) != expected_input_fields:
        raise MethodDatasetBuilderV93Error(
            "Prebuild structure-gate input ledger drift"
        )
    authority_commitment = inputs.get("authority_commitment", {})
    if (
        inputs.get("method_policy_canonical_self_sha256")
        != method_policy["canonical_self_sha256"]
        or authority_commitment
        != {
            "path": authority_path.resolve()
            .relative_to(common.ROOT.resolve())
            .as_posix(),
            "file_sha256": common.sha256_file(authority_path),
            "canonical_self_sha256": authority["canonical_self_sha256"],
            "key_values_recorded": False,
        }
    ):
        raise MethodDatasetBuilderV93Error(
            "Prebuild structure-gate authority or policy binding drift"
        )
    if (
        set(inputs["balanced_schedule_files"]) != {"train", "development"}
        or set(inputs["published_plan_root_files"])
        != {
            "construction_receipt.json",
            "development_registered_negative_plan.json",
            "development_residual_disclosure.json",
            "train_registered_negative_plan.json",
            "train_residual_disclosure.json",
        }
        or set(inputs["runtime_source_files"])
        != {
            "prebuild_structure_gate",
            "plan_builder",
            "plan_validator",
            "dataset_builder",
            "world_builder",
            "method_world",
            "structure_matrix",
            "probe_core",
            "balanced_schedule_validator",
            "joint_signature_validator",
        }
    ):
        raise MethodDatasetBuilderV93Error(
            "Prebuild structure-gate nested input ledger drift"
        )
    expected_direct_paths = {
        "method_policy": method_policy_module.POLICY_PATH,
        "parent_policy": str(method_policy["frozen_inputs"]["parent_policy"]),
        "text_template": str(method_policy["frozen_inputs"]["text_template"]),
        "joint_signatures": str(
            method_policy["frozen_inputs"]["joint_noise_signature"]
        ),
    }
    if any(
        inputs[name].get("path") != expected
        for name, expected in expected_direct_paths.items()
    ):
        raise MethodDatasetBuilderV93Error(
            "Prebuild structure-gate direct input path drift"
        )
    schedule_root = str(method_policy["frozen_inputs"]["balanced_schedule_root"])
    plan_root = str(method_policy["frozen_inputs"]["registered_negative_plan_root"])
    if any(
        inputs["balanced_schedule_files"][split].get("path")
        != f"{schedule_root}/{split}_balanced_schedule.json"
        for split in ("train", "development")
    ) or any(
        record.get("path") != f"{plan_root}/{name}"
        for name, record in inputs["published_plan_root_files"].items()
    ):
        raise MethodDatasetBuilderV93Error(
            "Prebuild schedule or plan-root input path drift"
        )
    frozen_pin_paths = {
        str(row["path"]) for row in method_policy.get("frozen_file_pins", [])
    }
    if not {
        str(record.get("path"))
        for record in inputs["runtime_source_files"].values()
    }.issubset(frozen_pin_paths):
        raise MethodDatasetBuilderV93Error(
            "Prebuild runtime source is outside frozen method-policy pins"
        )
    _validate_input_file_record(inputs["method_policy"])
    for record in inputs["balanced_schedule_files"].values():
        _validate_input_file_record(record)
    for record in inputs["published_plan_root_files"].values():
        _validate_input_file_record(record)
    _validate_input_file_record(inputs["joint_signatures"])
    _validate_input_file_record(inputs["parent_policy"])
    _validate_input_file_record(inputs["text_template"])
    _validate_input_file_record(inputs["style_profile"])
    for record in inputs["runtime_source_files"].values():
        _validate_input_file_record(record)
    return {
        "version": payload["version"],
        "status": payload["status"],
        "scientific_pass": True,
        "file_sha256": common.sha256_file(result_path),
        "canonical_self_sha256": payload["canonical_self_sha256"],
        "hard_gates": deepcopy(hard_gates),
        "probe_contract_audit": deepcopy(dict(probe_audit)),
        "finite_preregistered_projection_map_sha256": contract[
            "finite_preregistered_projection_map_sha256"
        ],
    }


def _effective_policy(base: Mapping[str, Any], keys: Mapping[str, Any]) -> dict[str, Any]:
    effective = json.loads(common.canonical_json_bytes(base).decode("utf-8"))
    scientific._replace_development_stream(effective, keys)
    effective["modes"][MODE]["world_counts"] = dict(WORLD_COUNTS)
    common.validate_policy(effective, mode=MODE)
    return effective


class _LineWriter:
    def __init__(self, path: Path, *, fields: Sequence[str] | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.handle = path.open("wb")
        self.fields = tuple(fields) if fields is not None else None
        self.digest = hashlib.sha256()
        self.row_count = 0
        if self.fields is not None:
            self._write_csv_values(self.fields)

    def _write(self, value: bytes) -> None:
        self.handle.write(value)
        self.digest.update(value)

    def _write_csv_values(self, values: Sequence[Any]) -> None:
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(values)
        self._write(buffer.getvalue().encode("utf-8"))

    def write(self, row: Mapping[str, Any]) -> None:
        if self.fields is None:
            self._write(common.canonical_json_bytes(dict(row)) + b"\n")
        else:
            if tuple(row) != self.fields:
                raise MethodDatasetBuilderV93Error(
                    f"CSV schema/order drift: {self.path.name}"
                )
            self._write_csv_values([row[name] for name in self.fields])
        self.row_count += 1

    def close(self, root: Path) -> dict[str, Any]:
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        return {
            "path": self.path.relative_to(root).as_posix(),
            "size_bytes": self.path.stat().st_size,
            "sha256": self.digest.hexdigest(),
            "row_count": self.row_count,
            "format": "csv" if self.fields is not None else "jsonl",
            "fields": list(self.fields) if self.fields is not None else None,
        }

    def abort(self) -> None:
        if not self.handle.closed:
            self.handle.close()


class _SplitWriters:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.writers = {
            "worlds": _LineWriter(root / "observed/worlds.jsonl"),
            "sellers": _LineWriter(root / "observed/sellers.jsonl"),
            "endpoints": _LineWriter(
                root / "observed/complete_pair_endpoints.csv", fields=PAIR_FIELDS
            ),
            "original_items": _LineWriter(root / "observed/original_redacted_items.jsonl"),
            "original_profiles": _LineWriter(root / "observed/original_model_seller_profiles.jsonl"),
            "deranged_items": _LineWriter(root / "observed/deranged_redacted_items.jsonl"),
            "deranged_profiles": _LineWriter(root / "observed/deranged_model_seller_profiles.jsonl"),
            "identity33": None,
            "controller_membership": _LineWriter(root / "private/controller_membership.jsonl"),
            "override_audit": _LineWriter(root / "private/override_audit.jsonl"),
            "render_asts": _LineWriter(root / "private/render_asts.jsonl"),
            "identity_slots": _LineWriter(root / "private/identity_slots_audit.jsonl"),
            "noise_slots": _LineWriter(root / "private/noise_slots_audit.jsonl"),
            "identity_assets": _LineWriter(root / "private/identity_assets.jsonl"),
            "mechanism_assignments": _LineWriter(
                root / "private/mechanism_assignments.jsonl"
            ),
            "seller_structure": _LineWriter(
                root / "private/seller_slot_structure.csv",
                fields=(*PAIR_FIELDS, *structure_matrix.SELLER_SLOT_RAW_FIELDS),
            ),
            "noise_structure": _LineWriter(
                root / "private/noise_visible_structure.csv",
                fields=(*PAIR_FIELDS, *structure_matrix.NOISE_VISIBLE_RAW_FIELDS),
            ),
            "world_audit": _LineWriter(root / "private/world_generation_audit.jsonl"),
        }

    def write_identity33(self, rows: Sequence[Mapping[str, Any]]) -> None:
        global IDENTITY_FIELDS
        if not rows:
            raise MethodDatasetBuilderV93Error("Identity33 world rows are empty")
        fields = tuple(rows[0])
        if IDENTITY_FIELDS is None:
            IDENTITY_FIELDS = fields
        if fields != IDENTITY_FIELDS or any(tuple(row) != fields for row in rows):
            raise MethodDatasetBuilderV93Error("Identity33 schema/order drift")
        if self.writers["identity33"] is None:
            self.writers["identity33"] = _LineWriter(
                self.root / "observed/identity33_all_pairs.csv", fields=fields
            )
        for row in rows:
            self.writers["identity33"].write(row)

    def close(self, root: Path) -> list[dict[str, Any]]:
        if self.writers["identity33"] is None:
            raise MethodDatasetBuilderV93Error("Identity33 writer was never initialized")
        return [writer.close(root) for writer in self.writers.values()]

    def abort(self) -> None:
        for writer in self.writers.values():
            if writer is not None:
                writer.abort()


def _write_world(writers: _SplitWriters, built: Mapping[str, Any]) -> dict[str, int]:
    public = built["public"]
    private = built["private_without_truth"]
    writers.writers["worlds"].write(
        {
            "world_uid": built["world_uid"],
            "split": built["split"],
            "split_ordinal": built["split_ordinal"],
            "candidate_index": built["candidate_index"],
        }
    )
    mapping = (
        ("sellers", public["sellers"]),
        ("endpoints", public["complete_pair_endpoints"]),
        ("original_items", public["original_redacted_items"]),
        ("original_profiles", public["original_model_seller_profiles"]),
        ("deranged_items", public["deranged_redacted_items"]),
        ("deranged_profiles", public["deranged_model_seller_profiles"]),
        ("controller_membership", private["controller_membership"]),
        ("override_audit", private["override_audit"]),
        ("render_asts", private["render_asts"]),
        ("identity_slots", private["identity_slots_audit"]),
        ("noise_slots", private["noise_slots_audit"]),
        ("identity_assets", private["identity_assets"]),
        ("mechanism_assignments", private["mechanism_assignments"]),
        ("seller_structure", public["seller_slot_structure_rows"]),
        ("noise_structure", public["noise_visible_structure_rows"]),
    )
    for name, rows in mapping:
        for row in rows:
            writers.writers[name].write(row)
    writers.write_identity33(public["identity33"])
    writers.writers["world_audit"].write(
        {
            "world_uid": built["world_uid"],
            **built["audit"],
            "controller_membership_sha256": common.canonical_sha256(
                private["controller_membership"]
            ),
            "override_audit_sha256": common.canonical_sha256(
                private["override_audit"]
            ),
        }
    )
    return {
        "seller_count": len(public["sellers"]),
        "pair_count": len(public["complete_pair_endpoints"]),
        "item_count": len(public["original_redacted_items"]),
        "profile_count": len(public["original_model_seller_profiles"]),
        "identity33_count": len(public["identity33"]),
        "identity_asset_count": len(private["identity_assets"]),
        "mechanism_assignment_count": len(private["mechanism_assignments"]),
    }


def _verify_files(root: Path, records: Sequence[Mapping[str, Any]]) -> None:
    expected = {str(row["path"]): row for row in records}
    observed = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file() and path.name != "root_manifest.json"
    }
    if set(expected) != set(observed):
        raise MethodDatasetBuilderV93Error("Persisted file universe drift")
    for relative, record in expected.items():
        path = observed[relative]
        if (
            path.stat().st_size != record["size_bytes"]
            or common.sha256_file(path) != record["sha256"]
        ):
            raise MethodDatasetBuilderV93Error(f"Persisted file hash drift: {relative}")


def _json_file_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": common.sha256_file(path),
        "row_count": 1,
        "format": "json",
        "fields": None,
    }


def _build_once(
    *, output_root: Path, authority_path: Path, consume_authority: bool = True
) -> dict[str, Any]:
    global IDENTITY_FIELDS
    IDENTITY_FIELDS = None
    output_root = output_root.resolve()
    authority_path = authority_path.resolve()
    if consume_authority and (
        output_root != DEFAULT_OUTPUT.resolve()
        or authority_path != DEFAULT_AUTHORITY.resolve()
    ):
        raise MethodDatasetBuilderV93Error(
            "Formal build must use the frozen output and private-authority paths"
        )
    method_policy = method_policy_module.load_policy()
    if (
        method_policy.get("status")
        != "FROZEN_METHOD_QUALIFICATION_INPUTS_NOT_TRAINING_QUALIFIED"
    ):
        raise MethodDatasetBuilderV93Error(
            "Method build is forbidden before all input and implementation pins freeze"
        )
    authority = _validate_authority(common.load_json(authority_path))
    if (
        authority["method_policy_canonical_self_sha256"]
        != method_policy["canonical_self_sha256"]
    ):
        raise MethodDatasetBuilderV93Error("Authority/policy binding drift")
    prebuild_gate_path = common.repo_path(
        str(
            method_policy["prebuild_structure_gate_contract"][
                "result_path"
            ]
        )
    )
    prebuild_gate_audit = _validate_prebuild_structure_gate_result(
        common.load_json(prebuild_gate_path),
        result_path=prebuild_gate_path,
        method_policy=method_policy,
        authority=authority,
        authority_path=authority_path,
    )
    consumed_path = authority_path.with_name(authority_path.stem + ".consumed.json")
    if consumed_path.exists():
        raise MethodDatasetBuilderV93Error("Random authority was already consumed")

    frozen = method_policy["frozen_inputs"]
    base = common.load_json(common.repo_path(str(frozen["parent_policy"])))
    effective = _effective_policy(base, authority["keys"])
    template = common.load_json(common.repo_path(str(frozen["text_template"])))
    fixture = common.load_json(
        common.repo_path(
            str(
                effective["identity_design"][
                    "role_template_parser_flag_fixture"
                ]["path"]
            )
        )
    )
    style_profile = common.load_json(
        common.verify_file_pin(
            effective["style_reference_boundary"]["generator_release_inputs"]["profile"],
            label="V9.3 style profile",
        )
    )
    signatures = common.load_json(common.repo_path(str(frozen["joint_noise_signature"])))
    blind = common.load_json(
        common.repo_path(str(frozen["blind_audit_design"]["path"]))
    )
    schedule_root = common.repo_path(str(frozen["balanced_schedule_root"]))
    plan_root = common.repo_path(str(frozen["registered_negative_plan_root"]))
    schedules = {
        split: common.load_json(schedule_root / f"{split}_balanced_schedule.json")
        for split in ("train", "development")
    }
    plans = {
        split: common.load_json(plan_root / f"{split}_registered_negative_plan.json")
        for split in ("train", "development")
    }
    for split in ("train", "development"):
        balanced.validate_schedule(schedules[split])
        negative_plan.validate_plan(
            plans[split],
            schedules[split],
            signatures,
            expected_version=negative_plan.BOUNDED_RESIDUAL_VERSION,
            require_exact_balance=False,
            success_status=negative_plan.BOUNDED_RESIDUAL_STATUS,
        )
    negative_plan.validate_train_development_plan_pair(
        plans["train"],
        plans["development"],
        schedules["train"],
        schedules["development"],
        signatures,
        expected_version=negative_plan.BOUNDED_RESIDUAL_VERSION,
        require_exact_balance=False,
        plan_success_status=negative_plan.BOUNDED_RESIDUAL_STATUS,
        pair_success_status=(
            "PASS_INDEPENDENT_BOUNDED_RESIDUAL_SPLIT_PLAN_PAIR_"
            "PENDING_STRUCTURE_GATE"
        ),
    )

    records = tuple(structure.build_mode_world_pool(effective, mode=MODE))
    if len(records) != 1004 or Counter(row["split"] for row in records) != Counter(WORLD_COUNTS):
        raise MethodDatasetBuilderV93Error("Effective 1,004-world pool drift")
    building = output_root.with_name(output_root.name + ".building")
    if output_root.exists() or building.exists():
        raise MethodDatasetBuilderV93Error("Method output path already exists")
    if consume_authority:
        authority_path.replace(consumed_path)
        authority_path = consumed_path
    building.mkdir(parents=True)
    writers = {split: _SplitWriters(building / split) for split in SPLITS}
    split_totals = {split: Counter() for split in SPLITS}
    world_uids: set[str] = set()
    seller_uids: set[str] = set()
    item_uids: set[str] = set()
    original_documents: set[str] = set()
    deranged_documents: set[str] = set()
    try:
        for index, world_record in enumerate(records):
            split = str(world_record["split"])
            built = method_world.build_method_world(
                policy=effective,
                template=template,
                fixture=fixture,
                style_profile=style_profile,
                mode=MODE,
                world_record=world_record,
                structure_key_hex=common.structure_key_for_split(
                    effective, mode=MODE, split=split
                ),
                balanced_schedule=schedules.get(split),
                registered_negative_plan=plans.get(split),
                joint_signatures=signatures,
                blind_audit_design=blind if split.startswith("audit_") else None,
                candidate_index=0,
            )
            if built["world_uid"] in world_uids:
                raise MethodDatasetBuilderV93Error("World UID collision")
            world_uids.add(built["world_uid"])
            for seller in built["public"]["sellers"]:
                uid = str(seller["seller_uid"])
                if uid in seller_uids:
                    raise MethodDatasetBuilderV93Error("Cross-world seller UID collision")
                seller_uids.add(uid)
            original_by_uid = {
                str(row["item_uid"]): row
                for row in built["public"]["original_redacted_items"]
            }
            deranged_by_uid = {
                str(row["item_uid"]): row
                for row in built["public"]["deranged_redacted_items"]
            }
            if set(original_by_uid) != set(deranged_by_uid):
                raise MethodDatasetBuilderV93Error("Counterfactual item universe drift")
            for uid in common.utf8_sort(original_by_uid):
                if uid in item_uids:
                    raise MethodDatasetBuilderV93Error("Cross-world item UID collision")
                item_uids.add(uid)
                for row, seen, label in (
                    (original_by_uid[uid], original_documents, "original"),
                    (deranged_by_uid[uid], deranged_documents, "deranged"),
                ):
                    digest = common.canonical_sha256(
                        [str(row["title"]), str(row["description"])]
                    )
                    if digest in seen:
                        raise MethodDatasetBuilderV93Error(
                            f"Cross-item {label} document collision"
                        )
                    seen.add(digest)
            split_totals[split].update(_write_world(writers[split], built))
            split_totals[split]["world_count"] += 1
            if (index + 1) % 25 == 0 or index + 1 == len(records):
                print(
                    json.dumps(
                        {
                            "event": "method_worlds_built",
                            "completed": index + 1,
                            "total": len(records),
                            "split": split,
                            "split_ordinal": world_record["split_ordinal"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        file_records: list[dict[str, Any]] = []
        split_manifests: dict[str, dict[str, Any]] = {}
        for split in SPLITS:
            records_for_split = writers[split].close(building)
            file_records.extend(records_for_split)
            manifest = {
                "version": VERSION,
                "split": split,
                "totals": dict(split_totals[split]),
                "files": records_for_split,
                "pair_truth_materialized": False,
                "canonical_self_sha256": None,
            }
            manifest["canonical_self_sha256"] = _self_hash(manifest)
            manifest_path = building / split / "split_manifest.json"
            common.write_json(manifest_path, manifest)
            file_records.append(_json_file_record(manifest_path, building))
            split_manifests[split] = manifest
    except BaseException:
        for split_writer in writers.values():
            split_writer.abort()
        raise
    _verify_files(building, file_records)
    root_manifest = {
        "version": VERSION,
        "status": "BUILT_NOT_QUALITY_AUDITED_NOT_TRAINING_QUALIFIED",
        "method_policy_canonical_self_sha256": method_policy["canonical_self_sha256"],
        "random_authority_canonical_self_sha256": authority["canonical_self_sha256"],
        "prebuild_structure_gate": prebuild_gate_audit,
        "world_counts": dict(WORLD_COUNTS),
        "world_uid_count": len(world_uids),
        "seller_uid_count": len(seller_uids),
        "item_uid_count": len(item_uids),
        "original_document_count": len(original_documents),
        "deranged_document_count": len(deranged_documents),
        "pair_truth_materialized": False,
        "audit_truth_read_count": 0,
        "split_manifest_self_hashes": {
            split: split_manifests[split]["canonical_self_sha256"] for split in SPLITS
        },
        "files": file_records,
        "canonical_self_sha256": None,
    }
    root_manifest["canonical_self_sha256"] = _self_hash(root_manifest)
    common.write_json(building / "root_manifest.json", root_manifest)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    building.replace(output_root)
    _verify_files(output_root, file_records)
    persisted_manifest = common.load_json(output_root / "root_manifest.json")
    if (
        persisted_manifest != root_manifest
        or persisted_manifest.get("canonical_self_sha256") != _self_hash(persisted_manifest)
    ):
        raise MethodDatasetBuilderV93Error("Published root-manifest closure drift")
    return root_manifest


def build(
    *, output_root: Path, authority_path: Path, consume_authority: bool = True
) -> dict[str, Any]:
    """Run one build and delete only newly created partial dataset payloads."""

    resolved_output = output_root.resolve()
    building = resolved_output.with_name(resolved_output.name + ".building")
    output_existed_before = resolved_output.exists()
    building_existed_before = building.exists()
    try:
        return _build_once(
            output_root=resolved_output,
            authority_path=authority_path.resolve(),
            consume_authority=consume_authority,
        )
    except BaseException:
        if not building_existed_before and building.exists():
            shutil.rmtree(building)
        if not output_existed_before and resolved_output.exists():
            shutil.rmtree(resolved_output)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--authority", type=Path, default=DEFAULT_AUTHORITY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build(
        output_root=args.output_root.resolve(),
        authority_path=args.authority.resolve(),
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
