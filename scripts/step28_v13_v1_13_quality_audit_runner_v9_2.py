#!/usr/bin/env python3
"""Run the complete V9.2 method-root audit under a consumed one-shot receipt."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any

import step28_v13_common as common
import step28_v13_structure as structure
import step28_v13_v1_13_document_capacity_v9 as document_capacity
import step28_v13_v1_13_quality_audit_runner_v9 as v9_runner
import step28_v13_v1_13_quality_complete_evidence_v9_2 as complete_evidence
import step28_v13_v1_13_quality_policy_v9_2 as policy_module
import step28_v13_v1_13_quality_probe_preparer_v9 as preparer_v9
import step28_v13_v1_13_quality_probe_preparer_v9_2 as preparer_v9_2
import step28_v13_v1_13_quality_probe_validator_v9_2 as validator_v9_2
import step28_v13_v1_13_quality_result_assembler_v9_2 as result_assembler
import step28_v13_v1_13_quality_structure_aggregator_v9_2 as structure_v9_2
import step28_v13_v1_13_quality_truth_capability_v9_2 as truth_capability
import step28_v13_v1_13_scientific_dataset_builder_v9 as builder_v9
import step28_v13_v1_13_scientific_dataset_builder_v9_2 as builder_v9_2


ROOT = Path(__file__).resolve().parents[1]
VERSION = "2026-08-23-step28-v13-v1-13-quality-audit-runner-v9-2"
AUTHORIZATION_VERSION = (
    "2026-08-23-step28-v13-v1-13-quality-run-authorization-v9-2"
)
AUTHORIZATION_PATH = (
    ROOT
    / "private_custody"
    / "step28_v13_v1_13_v9_2_quality_run_authorization.json"
)
REQUIRED_REVIEW_FINAL_LINE = "允许运行一次V9.2方法资格根质量审计"
SPLITS = ("train", "development", "audit_a", "audit_b")
SURFACE_FILES = {
    "surface_full": (
        "observed/redacted_items.jsonl",
        "observed/model_seller_profiles.jsonl",
    ),
    "surface_code_masked": (
        "observed/redacted_items.code_masked.jsonl",
        "observed/model_seller_profiles.code_masked.jsonl",
    ),
    "surface_code_neutralized": (
        "observed/redacted_items.code_neutralized.jsonl",
        "observed/model_seller_profiles.code_neutralized.jsonl",
    ),
    "surface_style_deranged_full": (
        builder_v9_2.COUNTERFACTUAL_ITEM_PATH,
        builder_v9_2.COUNTERFACTUAL_PROFILE_PATH,
    ),
}
ENDPOINT_PATH = "observed/complete_model_pair_endpoints.csv"
WORLDS_PATH = "observed/worlds.jsonl"
PUBLIC_CODE_PATH = "private/public_code_probe_input.jsonl"
ELIGIBILITY_PATH = "private/text_probe_eligibility_input.jsonl"
STRUCTURE_AUDIT_PATH = "private/channel_structure_audit.jsonl"
EXPECTED_SPLIT_DATA_PATHS = tuple(
    sorted(
        (*v9_runner.EXPECTED_SPLIT_DATA_PATHS, *builder_v9_2.MODEL_INPUT_PATHS[6:]),
        key=lambda value: value.encode("utf-8"),
    )
)
HEX_64 = re.compile(r"^[0-9a-f]{64}$")
GIT_OBJECT = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
CAPABILITIES = {
    "quality_audit_run": True,
    "metric_generation": True,
    "audit_a_b_truth_open": False,
    "formal_500_by_4": False,
    "model_training": False,
    "model_metric_generation": False,
}
DATASET_GATE_STAGES = {
    "root_manifest_and_physical_universe",
    "eight_input_label_free_loading",
    "public_uid_and_structure_closure",
    "freeze_28_text_2_code_slot_and_2_masks",
    "single_train_and_development_truth_open",
}
AUTHORIZATION_FIELDS = {
    "version",
    "status",
    "canonical_self_hash",
    "single_use",
    "receipt_generation_by_repository_code_forbidden",
    "quality_policy",
    "design_root_manifest",
    "complete_evidence_output_path",
    "capabilities",
    "private_key_material",
    "git_commit",
    "git_tree",
    "review_response_sha256",
    "review_final_line",
}


class QualityAuditRunnerV92Error(ValueError):
    """Raised when a V9.2 audit boundary or dataset gate drifts."""


class AuditorExecutionV92Error(RuntimeError):
    """Raised for mechanical failures that imply no dataset conclusion."""


def _canonical_self_hash(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("canonical_self_hash", None)
    return common.canonical_sha256(payload)


def _sha256_pin(path: Path, *, include_self_hash: bool = False) -> dict[str, Any]:
    output: dict[str, Any] = {
        "path": path.resolve().relative_to(ROOT.resolve()).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": common.sha256_file(path),
    }
    if include_self_hash:
        value = v9_runner._load_json(path)
        output["canonical_self_hash"] = value.get("canonical_self_hash")
    return output


def _current_git_identity() -> tuple[str, str]:
    def run(*args: str) -> str:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise AuditorExecutionV92Error("Git identity could not be verified") from exc
        return result.stdout.strip()

    return run("rev-parse", "HEAD"), run("rev-parse", "HEAD^{tree}")


def _safe_repo_file(relative: object, *, expected_name: str | None = None) -> Path:
    if not isinstance(relative, str) or not relative:
        raise QualityAuditRunnerV92Error("Authorization path is absent")
    path = (ROOT / relative).resolve()
    if ROOT.resolve() not in path.parents or (
        expected_name is not None and path.name != expected_name
    ):
        raise QualityAuditRunnerV92Error("Authorization path is unsafe")
    return path


def validate_run_authorization(
    value: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
    verify_bound_files: bool,
) -> dict[str, Any]:
    """Validate the future external receipt without creating or widening it."""

    normalized = json.loads(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    )
    if set(normalized) != AUTHORIZATION_FIELDS:
        raise QualityAuditRunnerV92Error("Quality-run authorization schema drift")
    if (
        normalized.get("version") != AUTHORIZATION_VERSION
        or normalized.get("status")
        != "ONE_SHOT_V9_2_METHOD_ROOT_QUALITY_AUDIT_AUTHORIZED"
        or normalized.get("canonical_self_hash") != _canonical_self_hash(normalized)
        or normalized.get("single_use") is not True
        or normalized.get("receipt_generation_by_repository_code_forbidden")
        is not True
        or normalized.get("capabilities") != CAPABILITIES
        or normalized.get("review_final_line") != REQUIRED_REVIEW_FINAL_LINE
    ):
        raise QualityAuditRunnerV92Error("Quality-run authorization identity drift")
    policy_pin = normalized.get("quality_policy")
    expected_policy_pin = {
        **_sha256_pin(policy_module.DEFAULT_POLICY_PATH),
        "canonical_self_hash": policy["canonical_self_hash"],
    }
    if policy_pin != expected_policy_pin:
        raise QualityAuditRunnerV92Error("Quality policy/run receipt binding drift")
    root_pin = normalized.get("design_root_manifest")
    if not isinstance(root_pin, Mapping) or set(root_pin) != {
        "path",
        "size_bytes",
        "sha256",
        "canonical_self_hash",
    }:
        raise QualityAuditRunnerV92Error("Design-root pin schema drift")
    root_path = _safe_repo_file(root_pin["path"], expected_name="root_manifest.json")
    reports_root = (ROOT / "reports").resolve()
    if reports_root not in root_path.parents:
        raise QualityAuditRunnerV92Error("Design-root pin is outside reports")
    output_path = _safe_repo_file(normalized["complete_evidence_output_path"])
    if (
        reports_root not in output_path.parents
        or output_path.name != "complete_quality_evidence.json"
        or root_path.parent in output_path.parents
        or output_path == root_path
    ):
        raise QualityAuditRunnerV92Error("Complete-evidence output path drift")
    key_material = normalized.get("private_key_material")
    if not isinstance(key_material, Mapping) or set(key_material) != {
        "id_key_hex",
        "document_variation_key_hex",
    } or any(
        not isinstance(key_material[name], str)
        or HEX_64.fullmatch(key_material[name]) is None
        for name in key_material
    ):
        raise QualityAuditRunnerV92Error("Private quality key material drift")
    if key_material["id_key_hex"] == key_material["document_variation_key_hex"]:
        raise QualityAuditRunnerV92Error("Private quality authorities were reused")
    if (
        GIT_OBJECT.fullmatch(str(normalized.get("git_commit", ""))) is None
        or GIT_OBJECT.fullmatch(str(normalized.get("git_tree", ""))) is None
        or HEX_64.fullmatch(str(normalized.get("review_response_sha256", "")))
        is None
    ):
        raise QualityAuditRunnerV92Error("Review/git commitment drift")
    if verify_bound_files:
        if (
            not root_path.is_file()
            or root_path.stat().st_size != root_pin["size_bytes"]
            or common.sha256_file(root_path) != root_pin["sha256"]
        ):
            raise QualityAuditRunnerV92Error("Design-root manifest byte pin drift")
        root_value = v9_runner._load_json(root_path)
        if (
            root_value.get("canonical_self_hash")
            != root_pin["canonical_self_hash"]
            or v9_runner._canonical_self_hash(root_value)
            != root_pin["canonical_self_hash"]
        ):
            raise QualityAuditRunnerV92Error("Design-root self-hash pin drift")
        commit, tree = _current_git_identity()
        if (commit, tree) != (normalized["git_commit"], normalized["git_tree"]):
            raise QualityAuditRunnerV92Error("Reviewed git identity drift")
    return normalized


def load_run_authorization(
    path: Path = AUTHORIZATION_PATH,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if path.resolve() != AUTHORIZATION_PATH.resolve():
        raise QualityAuditRunnerV92Error("Alternate quality authorization is forbidden")
    policy = policy_module.load_policy()
    if not path.is_file():
        raise QualityAuditRunnerV92Error(
            "V9.2 quality audit remains unauthorized; external receipt is absent"
        )
    value = v9_runner._load_json(path)
    return policy, validate_run_authorization(
        value, policy=policy, verify_bound_files=False
    )


def _consume_authorization(path: Path, authorization: Mapping[str, Any]) -> Path:
    consumed = path.with_name(path.stem + ".consumed.json")
    if consumed.exists():
        raise QualityAuditRunnerV92Error("Quality authorization was already consumed")
    expected = common.canonical_json_bytes(authorization) + b"\n"
    if path.read_bytes() != expected:
        raise QualityAuditRunnerV92Error("Quality authorization bytes are noncanonical")
    try:
        path.replace(consumed)
    except OSError as exc:
        raise AuditorExecutionV92Error("Quality authorization could not be consumed") from exc
    if consumed.read_bytes() != expected:
        raise AuditorExecutionV92Error("Consumed quality authorization bytes drift")
    return consumed


def _manifest_records(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    try:
        return v9_runner._manifest_records(manifest)
    except v9_runner.QualityAuditRunnerError as exc:
        raise QualityAuditRunnerV92Error(str(exc)) from exc


def _load_root_manifests(
    *, dataset_root: Path, root_pin: truth_capability.RootManifestPin
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest_path = dataset_root / "root_manifest.json"
    if (
        not manifest_path.is_file()
        or manifest_path.stat().st_size != root_pin.size_bytes
        or common.sha256_file(manifest_path) != root_pin.sha256
    ):
        raise QualityAuditRunnerV92Error("Root manifest pin drift")
    root_manifest = v9_runner._load_json(manifest_path)
    if (
        root_manifest.get("canonical_self_hash") != root_pin.canonical_self_hash
        or v9_runner._canonical_self_hash(root_manifest) != root_pin.canonical_self_hash
    ):
        raise QualityAuditRunnerV92Error("Root manifest self-hash drift")
    split_hashes = root_manifest.get("split_manifest_self_hashes")
    if not isinstance(split_hashes, Mapping) or set(split_hashes) != set(SPLITS):
        raise QualityAuditRunnerV92Error("Root/split manifest binding drift")
    manifests: dict[str, dict[str, Any]] = {}
    expected_files = {"root_manifest.json"}
    for split in SPLITS:
        split_path = dataset_root / split / "split_manifest.json"
        manifest = v9_runner._load_json(split_path)
        if (
            v9_runner._canonical_self_hash(manifest)
            != manifest.get("canonical_self_hash")
            or split_hashes[split] != manifest["canonical_self_hash"]
            or set(_manifest_records(manifest)) != set(EXPECTED_SPLIT_DATA_PATHS)
        ):
            raise QualityAuditRunnerV92Error("V9.2 split manifest closure drift")
        manifests[split] = manifest
        expected_files.add(f"{split}/split_manifest.json")
        expected_files.update(
            f"{split}/{relative}" for relative in EXPECTED_SPLIT_DATA_PATHS
        )
    actual_files = {
        path.relative_to(dataset_root).as_posix()
        for path in dataset_root.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        raise QualityAuditRunnerV92Error("V9.2 design-root physical universe drift")
    return root_manifest, manifests


def _load_split_label_free(
    *, dataset_root: Path, split: str, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    records = _manifest_records(manifest)
    required = {
        WORLDS_PATH,
        ENDPOINT_PATH,
        PUBLIC_CODE_PATH,
        ELIGIBILITY_PATH,
        STRUCTURE_AUDIT_PATH,
        *(path for pair in SURFACE_FILES.values() for path in pair),
    }
    if not required <= set(records):
        raise QualityAuditRunnerV92Error("Required label-free V9.2 input is absent")
    paths: dict[str, Path] = {}
    sources: dict[str, preparer_v9.SourceCommitment] = {}
    for relative in sorted(required, key=lambda value: value.encode("utf-8")):
        try:
            paths[relative], sources[relative] = v9_runner._verified_source(
                dataset_root=dataset_root,
                split=split,
                relative=relative,
                record=records[relative],
            )
        except v9_runner.QualityAuditRunnerError as exc:
            raise QualityAuditRunnerV92Error(str(exc)) from exc
    worlds = v9_runner._load_jsonl(
        paths[WORLDS_PATH], expected_rows=int(records[WORLDS_PATH]["row_count"])
    )
    endpoints = v9_runner._load_csv(
        paths[ENDPOINT_PATH],
        expected_rows=int(records[ENDPOINT_PATH]["row_count"]),
        expected_fields=preparer_v9.ENDPOINT_FIELDS,
    )
    return {
        "worlds": worlds,
        "endpoints": endpoints,
        "public_code": v9_runner._load_jsonl(
            paths[PUBLIC_CODE_PATH],
            expected_rows=int(records[PUBLIC_CODE_PATH]["row_count"]),
        ),
        "eligibility": v9_runner._load_jsonl(
            paths[ELIGIBILITY_PATH],
            expected_rows=int(records[ELIGIBILITY_PATH]["row_count"]),
        ),
        "structure_audit": v9_runner._load_jsonl(
            paths[STRUCTURE_AUDIT_PATH],
            expected_rows=int(records[STRUCTURE_AUDIT_PATH]["row_count"]),
        ),
        "surface_rows": {
            surface: (
                v9_runner._load_jsonl(
                    paths[item_path],
                    expected_rows=int(records[item_path]["row_count"]),
                ),
                v9_runner._load_jsonl(
                    paths[profile_path],
                    expected_rows=int(records[profile_path]["row_count"]),
                ),
            )
            for surface, (item_path, profile_path) in SURFACE_FILES.items()
        },
        "sources": sources,
        "paths": paths,
    }


def _registry_sha256(values: set[str]) -> str:
    return common.canonical_sha256(
        sorted(values, key=lambda value: value.encode("utf-8"))
    )


def _validate_public_closure(
    *,
    root_manifest: Mapping[str, Any],
    manifests: Mapping[str, Mapping[str, Any]],
    loaded: Mapping[str, Mapping[str, Any]],
) -> None:
    registries = {
        split: {kind: set() for kind in ("world", "seller", "item", "pair")}
        for split in SPLITS
    }
    for split in SPLITS:
        data = loaded[split]
        worlds = tuple(data["worlds"])
        if any(
            not isinstance(row, Mapping)
            or set(row) != {"world_uid", "split_ordinal"}
            or type(row["world_uid"]) is not str
            or type(row["split_ordinal"]) is not int
            for row in worlds
        ) or [row["split_ordinal"] for row in worlds] != list(range(len(worlds))):
            raise QualityAuditRunnerV92Error("World registry schema/order drift")
        ordered_worlds = tuple(str(row["world_uid"]) for row in worlds)
        row_keys, _, sellers_by_world = preparer_v9._validate_endpoints(
            data["endpoints"], ordered_world_uids=ordered_worlds
        )
        if any(len(values) != 28 for values in sellers_by_world.values()):
            raise QualityAuditRunnerV92Error("Per-world seller count drift")
        full_items = data["surface_rows"]["surface_full"][0]
        if any(set(row) != set(builder_v9.MODEL_REDACTED_ITEM_FIELDS) for row in full_items):
            raise QualityAuditRunnerV92Error("Model item exact schema drift")
        item_keys = {
            (str(row["world_uid"]), str(row["seller_uid"]), str(row["item_uid"]))
            for row in full_items
        }
        seller_uids = {str(row["seller_uid"]) for row in full_items}
        if len(item_keys) != len(full_items):
            raise QualityAuditRunnerV92Error("Duplicate model item key")
        for surface in structure_v9_2.MODEL_SURFACES:
            items, profiles = data["surface_rows"][surface]
            item_count_by_seller = {seller_uid: 0 for seller_uid in seller_uids}
            for item in items:
                seller_uid = str(item.get("seller_uid", ""))
                if seller_uid not in item_count_by_seller:
                    raise QualityAuditRunnerV92Error(
                        "Surface item seller is outside the full universe"
                    )
                item_count_by_seller[seller_uid] += 1
            if (
                any(set(row) != set(builder_v9.MODEL_REDACTED_ITEM_FIELDS) for row in items)
                or {
                    (str(row["world_uid"]), str(row["seller_uid"]), str(row["item_uid"]))
                    for row in items
                }
                != item_keys
                or any(set(row) != set(builder_v9.MODEL_PROFILE_FIELDS) for row in profiles)
                or {str(row["seller_uid"]) for row in profiles} != seller_uids
                or len(profiles) != len(seller_uids)
                or any(
                    type(row["item_count"]) is not int
                    or row["item_count"]
                    != item_count_by_seller[str(row["seller_uid"])]
                    for row in profiles
                )
            ):
                raise QualityAuditRunnerV92Error("Four-surface public key closure drift")
        public_sellers = {str(row["seller_uid"]) for row in data["public_code"]}
        preparer_v9._parse_public_rows(
            data["public_code"],
            ordered_worlds=ordered_worlds,
            sellers_by_world=sellers_by_world,
            expected_sellers_per_world=28,
        )
        eligibility = tuple(data["eligibility"])
        if (
            any(
                set(row) != set(preparer_v9.ELIGIBILITY_FIELDS)
                or type(row["text_probe_eligible"]) is not bool
                for row in eligibility
            )
            or {
                (str(row["world_uid"]), str(row["canonical_pair_uid"]))
                for row in eligibility
            }
            != set(row_keys)
        ):
            raise QualityAuditRunnerV92Error("Text eligibility closure drift")
        eligibility_by_world = {world_uid: [] for world_uid in ordered_worlds}
        for row in eligibility:
            eligibility_by_world[str(row["world_uid"])].append(row)
        if any(
            len(rows) != 378
            or sum(row["text_probe_eligible"] is False for row in rows) != 6
            for rows in eligibility_by_world.values()
        ):
            raise QualityAuditRunnerV92Error(
                "Per-world text eligibility cardinality drift"
            )
        structure_rows = tuple(data["structure_audit"])
        if (
            any(set(row) != set(structure_v9_2.EXTENSION_FIELDS) for row in structure_rows)
            or {str(row["world_uid"]) for row in structure_rows} != set(ordered_worlds)
            or len(structure_rows) != len(ordered_worlds)
        ):
            raise QualityAuditRunnerV92Error("V9.2 structure row closure drift")
        pair_uids = {str(row["canonical_pair_uid"]) for row in data["endpoints"]}
        observed = {
            "world": set(ordered_worlds),
            "seller": seller_uids,
            "item": {key[2] for key in item_keys},
            "pair": pair_uids,
        }
        if public_sellers != seller_uids:
            raise QualityAuditRunnerV92Error("Public-code seller universe drift")
        expected = manifests[split].get("uid_registries")
        if not isinstance(expected, Mapping):
            raise QualityAuditRunnerV92Error("Split UID registry is absent")
        for kind, values in observed.items():
            spec = expected.get(kind)
            if not isinstance(spec, Mapping) or spec != {
                "count": len(values),
                "sha256": _registry_sha256(values),
            }:
                raise QualityAuditRunnerV92Error(f"Split {kind} registry drift")
            registries[split][kind] = values
    root_registries = root_manifest.get("uid_registries")
    if not isinstance(root_registries, Mapping):
        raise QualityAuditRunnerV92Error("Root UID registry is absent")
    for kind in ("world", "seller", "item", "pair"):
        values = [registries[split][kind] for split in SPLITS]
        merged = set().union(*values)
        if len(merged) != sum(len(value) for value in values) or root_registries.get(kind) != {
            "count": len(merged),
            "sha256": _registry_sha256(merged),
        }:
            raise QualityAuditRunnerV92Error(f"Root {kind} registry drift")


def _source_tuple(
    *values: preparer_v9.SourceCommitment,
) -> tuple[preparer_v9.SourceCommitment, ...]:
    return tuple(sorted(values, key=lambda value: value.path.encode("utf-8")))


def _global_ordinals(policy: Mapping[str, Any], split: str, worlds: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = policy["design_scale"]["world_counts"]
    offset = sum(int(counts[name]) for name in SPLITS[: SPLITS.index(split)])
    return {
        str(row["world_uid"]): offset + int(row["split_ordinal"])
        for row in worlds
    }


def _freeze_train_development(
    *,
    loaded: Mapping[str, Mapping[str, Any]],
    policy: Mapping[str, Any],
    run_authorization: Mapping[str, Any],
) -> tuple[
    dict[str, preparer_v9_2.FrozenTextBundleV92],
    dict[str, tuple[preparer_v9.FrozenFeatureMatrix, ...]],
    dict[str, preparer_v9.FrozenTextEligibility],
]:
    text: dict[str, preparer_v9_2.FrozenTextBundleV92] = {}
    code: dict[str, tuple[preparer_v9.FrozenFeatureMatrix, ...]] = {}
    eligibility: dict[str, preparer_v9.FrozenTextEligibility] = {}
    document_key = bytes.fromhex(
        run_authorization["private_key_material"]["document_variation_key_hex"]
    )
    code_key = document_capacity.derive_code_key(document_key)
    id_key = str(run_authorization["private_key_material"]["id_key_hex"])
    policy_source = preparer_v9.SourceCommitment(
        path=policy_module.DEFAULT_POLICY_PATH.relative_to(ROOT).as_posix(),
        size_bytes=policy_module.DEFAULT_POLICY_PATH.stat().st_size,
        sha256=common.sha256_file(policy_module.DEFAULT_POLICY_PATH),
    )
    for split in ("train", "development"):
        data = loaded[split]
        endpoints = data["endpoints"]
        worlds = tuple(str(row["world_uid"]) for row in data["worlds"])
        sources = data["sources"]
        eligibility[split] = preparer_v9.freeze_text_eligibility(
            eligibility_rows=data["eligibility"],
            endpoints=endpoints,
            ordered_world_uids=worlds,
            sources=_source_tuple(sources[ELIGIBILITY_PATH], sources[ENDPOINT_PATH]),
        )
        text[split] = preparer_v9_2.freeze_all_text_surfaces_before_truth(
            surface_rows=data["surface_rows"],
            endpoints=endpoints,
            ordered_world_uids=worlds,
            sources_by_surface={
                surface: _source_tuple(sources[item_path], sources[profile_path])
                for surface, (item_path, profile_path) in SURFACE_FILES.items()
            },
            text_eligibility=eligibility[split],
        )
        public = preparer_v9.prepare_public_code_matrix(
            public_rows=data["public_code"],
            endpoints=endpoints,
            ordered_world_uids=worlds,
            sources=_source_tuple(sources[PUBLIC_CODE_PATH], sources[ENDPOINT_PATH]),
        )
        expected_slots = {
            (
                world_uid,
                structure.base_uid(
                    key_hex=id_key,
                    entity_kind="seller",
                    parent_uid_or_mode=world_uid,
                    ordinal=slot,
                ),
            ): slot
            for world_uid in worlds
            for slot in range(28)
        }
        decoded = preparer_v9.prepare_decoded_slot_matrix(
            public_rows=data["public_code"],
            endpoints=endpoints,
            ordered_world_uids=worlds,
            expected_mode_global_ordinal_by_world=_global_ordinals(
                policy, split, data["worlds"]
            ),
            expected_seller_slot_by_world_and_seller=expected_slots,
            decode_coordinate=lambda _world_uid, value: document_capacity.decode_code(
                code_key=code_key, code=value
            ),
            sources=_source_tuple(
                policy_source,
                sources[PUBLIC_CODE_PATH],
                sources[ENDPOINT_PATH],
                sources[WORLDS_PATH],
            ),
        )
        code[split] = (public, decoded)
    return text, code, eligibility


def _validate_root_claim_and_bindings(
    *,
    root_manifest: Mapping[str, Any],
    policy: Mapping[str, Any],
    run_authorization: Mapping[str, Any],
) -> None:
    key_material = run_authorization["private_key_material"]
    expected_commitments = {
        name.removesuffix("_hex") + "_sha256": hashlib.sha256(
            bytes.fromhex(value)
        ).hexdigest()
        for name, value in key_material.items()
    }
    if (
        root_manifest.get("version") != builder_v9_2.VERSION
        or root_manifest.get("status")
        != "PASS_DESIGN_BUILD_NOT_TRAINING_QUALIFIED"
        or root_manifest.get("execution_mode") != "method_qualification_1004"
        or root_manifest.get("scientific_use_forbidden") is not True
        or root_manifest.get("formal_seed_created") is not False
        or root_manifest.get("formal_rows_created") != 0
        or root_manifest.get("training_started") is not False
        or root_manifest.get("quality_policy_canonical_self_hash")
        != policy["canonical_self_hash"]
        or root_manifest.get("quality_policy_file")
        != _sha256_pin(policy_module.DEFAULT_POLICY_PATH)
        or root_manifest.get("quality_required_key_commitments")
        != expected_commitments
        or root_manifest.get("model_input_file_count") != 8
    ):
        raise QualityAuditRunnerV92Error("V9.2 root claim/policy/key binding drift")


def _root_pin(authorization: Mapping[str, Any]) -> tuple[Path, truth_capability.RootManifestPin]:
    spec = authorization["design_root_manifest"]
    path = _safe_repo_file(spec["path"], expected_name="root_manifest.json")
    return path.parent, truth_capability.RootManifestPin(
        path="root_manifest.json",
        size_bytes=int(spec["size_bytes"]),
        sha256=str(spec["sha256"]),
        canonical_self_hash=str(spec["canonical_self_hash"]),
    )


def _source_bundle_sha256(
    *,
    loaded: Mapping[str, Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> str:
    rows = [
        {
            "split": split,
            "path": source.path,
            "size_bytes": source.size_bytes,
            "sha256": source.sha256,
        }
        for split in SPLITS
        for source in sorted(
            loaded[split]["sources"].values(),
            key=lambda value: value.path.encode("utf-8"),
        )
    ]
    return common.canonical_sha256(
        {
            "implementation_source_pins": policy["source_pins"],
            "label_free_input_sources": rows,
        }
    )


def _snapshot_label_free_bytes(
    *,
    dataset_root: Path,
    loaded: Mapping[str, Mapping[str, Any]],
) -> dict[str, tuple[int, str]]:
    paths = {
        dataset_root / "root_manifest.json",
        *(dataset_root / split / "split_manifest.json" for split in SPLITS),
        *(
            path
            for split in SPLITS
            for path in loaded[split]["paths"].values()
        ),
    }
    output: dict[str, tuple[int, str]] = {}
    for path in sorted(paths, key=lambda value: value.as_posix().encode("utf-8")):
        relative = path.relative_to(dataset_root).as_posix()
        output[relative] = (path.stat().st_size, common.sha256_file(path))
    return output


def _reverify_label_free_bytes(
    *,
    dataset_root: Path,
    snapshots: Mapping[str, tuple[int, str]],
) -> None:
    for relative, (size_bytes, sha256) in snapshots.items():
        path = (dataset_root / relative).resolve()
        if (
            dataset_root.resolve() not in path.parents
            or not path.is_file()
            or path.stat().st_size != size_bytes
            or common.sha256_file(path) != sha256
        ):
            raise QualityAuditRunnerV92Error(
                "Label-free source changed after truth was opened"
            )


def _calculate_complete_evidence(
    *,
    policy: Mapping[str, Any],
    run_authorization: Mapping[str, Any],
    state: dict[str, str],
) -> dict[str, Any]:
    state["stage"] = "root_manifest_and_physical_universe"
    dataset_root, root_pin = _root_pin(run_authorization)
    root_manifest, manifests = _load_root_manifests(
        dataset_root=dataset_root, root_pin=root_pin
    )
    _validate_root_claim_and_bindings(
        root_manifest=root_manifest,
        policy=policy,
        run_authorization=run_authorization,
    )
    state["stage"] = "eight_input_label_free_loading"
    loaded = {
        split: _load_split_label_free(
            dataset_root=dataset_root, split=split, manifest=manifests[split]
        )
        for split in SPLITS
    }
    state["stage"] = "public_uid_and_structure_closure"
    _validate_public_closure(
        root_manifest=root_manifest, manifests=manifests, loaded=loaded
    )
    structure_receipt = structure_v9_2.aggregate_formal_structure(
        public_rows_by_split={split: loaded[split]["public_code"] for split in SPLITS},
        structure_rows_by_split={split: loaded[split]["structure_audit"] for split in SPLITS},
        eligibility_rows_by_split={split: loaded[split]["eligibility"] for split in SPLITS},
        model_surface_rows_by_split={split: loaded[split]["surface_rows"] for split in SPLITS},
        policy=policy,
        run_authorization=run_authorization,
    )
    state["stage"] = "freeze_28_text_2_code_slot_and_2_masks"
    text, code, eligibility = _freeze_train_development(
        loaded=loaded, policy=policy, run_authorization=run_authorization
    )
    label_free_snapshots = _snapshot_label_free_bytes(
        dataset_root=dataset_root, loaded=loaded
    )
    state["stage"] = "single_train_and_development_truth_open"
    numerical_receipt = validator_v9_2.evaluate_formal_probe_families(
        text_train_matrices=text["train"].matrices,
        text_development_matrices=text["development"].matrices,
        code_train_matrices=code["train"],
        code_development_matrices=code["development"],
        train_text_eligibility=eligibility["train"],
        development_text_eligibility=eligibility["development"],
        dataset_root=dataset_root,
        root_manifest_pin=root_pin,
        policy=policy,
        run_authorization=run_authorization,
        verify_label_free_bytes=lambda: _reverify_label_free_bytes(
            dataset_root=dataset_root,
            snapshots=label_free_snapshots,
        ),
    )
    state["stage"] = "assemble_all_registry_entries"
    return result_assembler.assemble_formal_complete_evidence(
        structure_receipt=structure_receipt,
        numerical_receipt=numerical_receipt,
        train_text_bundle=text["train"],
        development_text_bundle=text["development"],
        train_code_matrices=code["train"],
        development_code_matrices=code["development"],
        quality_policy=policy,
        root_manifest_sha256=root_pin.sha256,
        source_bundle_sha256=_source_bundle_sha256(
            loaded=loaded, policy=policy
        ),
    )


def _classified_failure(
    *, status: str, stage: str, exc: BaseException
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "version": VERSION,
        "status": status,
        "claim_boundary": "V9_2_DESIGN_QUALITY_ONLY_NOT_FORMAL_DATA_OR_TRAINING",
        "failure_stage": stage,
        "exception_type": type(exc).__name__,
        "exception_message_sha256": hashlib.sha256(
            str(exc).encode("utf-8")
        ).hexdigest(),
        "complete_quality_calculation": False,
        "row_level_labels_returned": 0,
        "row_level_predictions_returned": 0,
        "formal_500_by_4_generated": False,
        "training_started": False,
    }
    receipt["canonical_self_hash"] = common.canonical_sha256(receipt)
    return receipt


def _validate_outer_wrapper(
    *, evidence_path: Path, publication: Mapping[str, Any]
) -> dict[str, Any]:
    value = v9_runner._load_json(evidence_path)
    normalized = complete_evidence.validate_complete_quality_evidence(value)
    if (
        common.sha256_file(evidence_path) != publication["sha256"]
        or normalized["canonical_self_hash"]
        != publication["canonical_self_hash"]
    ):
        raise QualityAuditRunnerV92Error("Outer complete-evidence binding drift")
    return normalized


def run_formal_quality_audit() -> dict[str, Any]:
    """Consume one external receipt, publish complete evidence, then wrap it."""

    policy, authorization = load_run_authorization()
    consumed = _consume_authorization(AUTHORIZATION_PATH, authorization)
    state = {"stage": "authorization_consumed"}
    try:
        authorization = validate_run_authorization(
            authorization, policy=policy, verify_bound_files=True
        )
        evidence = _calculate_complete_evidence(
            policy=policy, run_authorization=authorization, state=state
        )
    except Exception as exc:
        mechanical = isinstance(
            exc,
            (
                AuditorExecutionV92Error,
                v9_runner.AuditorExecutionFailure,
                truth_capability.QualityTruthAuditorExecutionError,
            ),
        )
        status = (
            "DATASET_INVALIDATED"
            if not mechanical and state["stage"] in DATASET_GATE_STAGES
            else "AUDITOR_EXECUTION_FAILED_NO_DATASET_CONCLUSION"
        )
        return _classified_failure(status=status, stage=state["stage"], exc=exc)
    output_path = _safe_repo_file(authorization["complete_evidence_output_path"])
    state["stage"] = "exclusive_complete_evidence_publication"
    try:
        publication = complete_evidence.publish_complete_evidence_exclusive(
            output_path, evidence
        )
    except Exception as exc:
        return _classified_failure(
            status="AUDITOR_EXECUTION_FAILED_NO_DATASET_CONCLUSION",
            stage=state["stage"],
            exc=exc,
        )
    wrapper_error: BaseException | None = None
    try:
        _validate_outer_wrapper(evidence_path=output_path, publication=publication)
    except Exception as exc:
        wrapper_error = exc
    terminal = complete_evidence.wrapper_terminal_after_complete_evidence(
        evidence=evidence, wrapper_error=wrapper_error
    )
    result = {
        "version": VERSION,
        "status": terminal["status"],
        "complete_evidence_publication": publication,
        "terminal_wrapper": terminal,
        "authorization_consumed_path_sha256": hashlib.sha256(
            consumed.relative_to(ROOT).as_posix().encode("utf-8")
        ).hexdigest(),
        "row_level_labels_returned": 0,
        "row_level_predictions_returned": 0,
        "formal_500_by_4_generated": False,
        "training_started": False,
    }
    result["canonical_self_hash"] = common.canonical_sha256(result)
    return result


def main() -> None:
    result = run_formal_quality_audit()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
