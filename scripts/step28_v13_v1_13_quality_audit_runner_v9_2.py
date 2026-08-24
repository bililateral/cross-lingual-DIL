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
REQUIRED_REVIEW_FINAL_LINE = truth_capability.QUALITY_RUN_REVIEW_FINAL_LINE
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
CAPABILITIES = dict(truth_capability.QUALITY_RUN_CAPABILITIES)
ROOT_MANIFEST_FIELDS = {
    "version",
    "status",
    "execution_mode",
    "scientific_use_forbidden",
    "formal_seed_created",
    "formal_rows_created",
    "training_started",
    "quality_policy_canonical_self_hash",
    "quality_policy_file",
    "builder_source_file",
    "design_build_authorization",
    "random_authority",
    "quality_required_key_commitments",
    "model_input_file_count",
    "split_order",
    "world_count",
    "seller_count",
    "pair_count",
    "positive_pair_count",
    "negative_pair_count",
    "uid_registries",
    "item_document_registry_count",
    "item_document_registry_sha256",
    "seller_document_registry_count",
    "seller_document_registry_sha256",
    "item_code_registry_count",
    "item_code_registry_sha256",
    "identity_value_registry_count",
    "identity_value_registry_sha256",
    "historical_exclusion_counts",
    "split_manifest_self_hashes",
    "canonical_self_hash",
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


def _physical_row_count(path: Path) -> int:
    """Count framing only; never parse or materialize private truth rows."""

    line_count = 0
    size_bytes = 0
    last_byte = b""
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                size_bytes += len(chunk)
                line_count += chunk.count(b"\n")
                last_byte = chunk[-1:]
    except OSError as exc:
        raise AuditorExecutionV92Error(
            f"Physical payload scan failed: {path.name}"
        ) from exc
    if size_bytes and last_byte != b"\n":
        line_count += 1
    if path.suffix == ".csv":
        if line_count == 0:
            raise QualityAuditRunnerV92Error(
                f"Physical CSV framing is empty: {path.name}"
            )
        return line_count - 1
    return line_count


def _verify_all_manifest_payloads(
    *,
    dataset_root: Path,
    manifests: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Re-hash and row-count every one of the 20 files in every split."""

    output: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        records = _manifest_records(manifests[split])
        if set(records) != set(EXPECTED_SPLIT_DATA_PATHS):
            raise QualityAuditRunnerV92Error(
                "V9.2 complete split payload registry drift"
            )
        paths: dict[str, Path] = {}
        sources: dict[str, preparer_v9.SourceCommitment] = {}
        for relative in EXPECTED_SPLIT_DATA_PATHS:
            record = records[relative]
            row_count = record.get("row_count")
            if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count < 0:
                raise QualityAuditRunnerV92Error(
                    "V9.2 split payload row-count pin drift"
                )
            try:
                path, source = v9_runner._verified_source(
                    dataset_root=dataset_root,
                    split=split,
                    relative=relative,
                    record=record,
                )
            except v9_runner.QualityAuditRunnerError as exc:
                raise QualityAuditRunnerV92Error(str(exc)) from exc
            if _physical_row_count(path) != row_count:
                raise QualityAuditRunnerV92Error(
                    "V9.2 split payload physical row count drift"
                )
            paths[relative] = path
            sources[relative] = source
        output[split] = {
            "records": records,
            "paths": paths,
            "sources": sources,
        }
    return output


def _load_split_label_free(
    *,
    split: str,
    verified_payload: Mapping[str, Any],
) -> dict[str, Any]:
    records = verified_payload["records"]
    required = {
        WORLDS_PATH,
        ENDPOINT_PATH,
        PUBLIC_CODE_PATH,
        ELIGIBILITY_PATH,
        STRUCTURE_AUDIT_PATH,
        *(path for pair in SURFACE_FILES.values() for path in pair),
    }
    if set(records) != set(EXPECTED_SPLIT_DATA_PATHS) or not required <= set(records):
        raise QualityAuditRunnerV92Error("Required label-free V9.2 input is absent")
    paths = dict(verified_payload["paths"])
    sources = dict(verified_payload["sources"])
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
    run_capability: truth_capability.ConsumedQualityRunCapabilityV92,
) -> tuple[
    dict[str, preparer_v9_2.FrozenTextBundleV92],
    dict[str, tuple[preparer_v9.FrozenFeatureMatrix, ...]],
    dict[str, preparer_v9.FrozenTextEligibility],
]:
    text: dict[str, preparer_v9_2.FrozenTextBundleV92] = {}
    code: dict[str, tuple[preparer_v9.FrozenFeatureMatrix, ...]] = {}
    eligibility: dict[str, preparer_v9.FrozenTextEligibility] = {}
    document_key = bytes.fromhex(
        run_capability.private_key_hex("document_variation_key_hex")
    )
    code_key = document_capacity.derive_code_key(document_key)
    id_key = run_capability.private_key_hex("id_key_hex")
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
    run_capability: truth_capability.ConsumedQualityRunCapabilityV92,
) -> None:
    def validate_consumption_receipt(value: object, *, name: str) -> bool:
        return bool(
            isinstance(value, Mapping)
            and set(value)
            == {"path_sha256", "size_bytes", "sha256", "canonical_self_hash"}
            and isinstance(value.get("size_bytes"), int)
            and not isinstance(value.get("size_bytes"), bool)
            and int(value["size_bytes"]) > 0
            and all(
                isinstance(value.get(field), str)
                and HEX_64.fullmatch(str(value[field])) is not None
                for field in ("path_sha256", "sha256", "canonical_self_hash")
            )
        )

    build_lineage = root_manifest.get("design_build_authorization")
    random_lineage = root_manifest.get("random_authority")
    builder_source_path = Path(builder_v9_2.__file__).resolve()
    expected_builder_pin = _sha256_pin(builder_source_path)
    policy_builder_pins = [
        dict(value)
        for value in policy["source_pins"]
        if value.get("path") == expected_builder_pin["path"]
    ]
    expected_commitments = {
        name.removesuffix("_hex") + "_sha256": hashlib.sha256(
            bytes.fromhex(run_capability.private_key_hex(name))
        ).hexdigest()
        for name in ("id_key_hex", "document_variation_key_hex")
    }
    if (
        set(root_manifest) != ROOT_MANIFEST_FIELDS
        or root_manifest.get("version") != builder_v9_2.VERSION
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
        or root_manifest.get("builder_source_file") != expected_builder_pin
        or policy_builder_pins != [expected_builder_pin]
        or root_manifest.get("quality_required_key_commitments")
        != expected_commitments
        or root_manifest.get("model_input_file_count") != 8
        or root_manifest.get("split_order") != list(SPLITS)
        or root_manifest.get("world_count") != 1004
        or root_manifest.get("seller_count") != 1004 * 28
        or root_manifest.get("pair_count") != 1004 * 378
        or root_manifest.get("positive_pair_count") != 1004 * 20
        or root_manifest.get("negative_pair_count") != 1004 * 358
        or not isinstance(build_lineage, Mapping)
        or set(build_lineage)
        != {"status", "receipt", "review_response_sha256", "git_commit", "git_tree"}
        or build_lineage.get("status")
        != "CONSUMED_ONE_SHOT_BUILD_AUTHORIZATION"
        or not validate_consumption_receipt(
            build_lineage.get("receipt"), name="build"
        )
        or HEX_64.fullmatch(str(build_lineage.get("review_response_sha256", "")))
        is None
        or GIT_OBJECT.fullmatch(str(build_lineage.get("git_commit", ""))) is None
        or GIT_OBJECT.fullmatch(str(build_lineage.get("git_tree", ""))) is None
        or not isinstance(random_lineage, Mapping)
        or set(random_lineage) != {"status", "receipt", "authority_bundle_sha256"}
        or random_lineage.get("status")
        != "CONSUMED_FRESH_V9_2_RANDOM_AUTHORITY"
        or not validate_consumption_receipt(
            random_lineage.get("receipt"), name="random"
        )
        or HEX_64.fullmatch(str(random_lineage.get("authority_bundle_sha256", "")))
        is None
        or random_lineage.get("authority_bundle_sha256")
        in builder_v9_2.RETIRED_PREFLIGHT_AUTHORITIES
    ):
        raise QualityAuditRunnerV92Error("V9.2 root claim/policy/key binding drift")


def _root_pin(
    run_capability: truth_capability.ConsumedQualityRunCapabilityV92,
) -> tuple[Path, truth_capability.RootManifestPin]:
    spec = run_capability.design_root_binding()
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
    run_capability: truth_capability.ConsumedQualityRunCapabilityV92,
    state: dict[str, str],
) -> dict[str, Any]:
    state["stage"] = "root_manifest_and_physical_universe"
    dataset_root, root_pin = _root_pin(run_capability)
    root_manifest, manifests = _load_root_manifests(
        dataset_root=dataset_root, root_pin=root_pin
    )
    _validate_root_claim_and_bindings(
        root_manifest=root_manifest,
        policy=policy,
        run_capability=run_capability,
    )
    verified_payloads = _verify_all_manifest_payloads(
        dataset_root=dataset_root,
        manifests=manifests,
    )
    state["stage"] = "eight_input_label_free_loading"
    loaded = {
        split: _load_split_label_free(
            split=split,
            verified_payload=verified_payloads[split],
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
        run_capability=run_capability,
    )
    state["stage"] = "freeze_28_text_2_code_slot_and_2_masks"
    text, code, eligibility = _freeze_train_development(
        loaded=loaded, policy=policy, run_capability=run_capability
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
        run_capability=run_capability,
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
        "cleanup_required": False,
        "cleanup_requires_documented_failure_boundary": True,
    }
    receipt["canonical_self_hash"] = common.canonical_sha256(receipt)
    return receipt


def _publish_terminal_exclusive(
    path: Path, result: Mapping[str, Any]
) -> dict[str, Any]:
    normalized = json.loads(common.canonical_json_bytes(result).decode("utf-8"))
    if normalized.get("canonical_self_hash") != common.canonical_sha256(
        {key: value for key, value in normalized.items() if key != "canonical_self_hash"}
    ):
        raise QualityAuditRunnerV92Error("Terminal result self-hash drift")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = common.canonical_json_bytes(normalized) + b"\n"
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
    except OSError as exc:
        raise AuditorExecutionV92Error(
            "V9.2 terminal result could not be persisted exclusively"
        ) from exc
    if path.read_bytes() != payload:
        raise AuditorExecutionV92Error("V9.2 terminal result replay drift")
    return normalized


def _complete_run_result(
    *,
    evidence: Mapping[str, Any],
    publication: Mapping[str, Any] | None,
    publication_error: BaseException | None,
    consumed: Path,
) -> dict[str, Any]:
    terminal = complete_evidence.wrapper_terminal_after_complete_evidence(
        evidence=evidence,
        wrapper_error=publication_error,
    )
    invalidated = bool(terminal["dataset_invalidation_preserved"])
    result: dict[str, Any] = {
        "version": VERSION,
        "status": terminal["status"],
        "complete_evidence_publication": (
            None if publication is None else dict(publication)
        ),
        "complete_evidence_publication_error": (
            None
            if publication_error is None
            else {
                "exception_type": type(publication_error).__name__,
                "exception_message_sha256": hashlib.sha256(
                    str(publication_error).encode("utf-8")
                ).hexdigest(),
            }
        ),
        "terminal_wrapper": terminal,
        "authorization_consumed_path_sha256": hashlib.sha256(
            consumed.relative_to(ROOT).as_posix().encode("utf-8")
        ).hexdigest(),
        "cleanup_required": invalidated,
        "cleanup_requires_documented_failure_boundary": True,
        "row_level_labels_returned": 0,
        "row_level_predictions_returned": 0,
        "formal_500_by_4_generated": False,
        "training_started": False,
    }
    result["canonical_self_hash"] = common.canonical_sha256(result)
    return result


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
    output_path = _safe_repo_file(authorization["complete_evidence_output_path"])
    terminal_path = output_path.with_name("quality_audit_terminal.json")

    def persist(result: dict[str, Any]) -> dict[str, Any]:
        return _publish_terminal_exclusive(terminal_path, result)

    state = {"stage": "authorization_consumed"}
    try:
        authorization = validate_run_authorization(
            authorization, policy=policy, verify_bound_files=True
        )
        run_capability = (
            truth_capability.ConsumedQualityRunCapabilityV92._from_consumed_authorization(
                authorization=authorization,
                consumed_path=consumed,
            )
        )
        evidence = _calculate_complete_evidence(
            policy=policy, run_capability=run_capability, state=state
        )
    except Exception as exc:
        # A dataset conclusion is valid only after every registered observation
        # has been assembled.  Parse, schema, I/O, and uncomputable precondition
        # failures therefore never masquerade as dataset invalidation.
        return persist(
            _classified_failure(
                status="AUDITOR_EXECUTION_FAILED_NO_DATASET_CONCLUSION",
                stage=state["stage"],
                exc=exc,
            )
        )
    if output_path != _safe_repo_file(run_capability.complete_evidence_output_path()):
        raise QualityAuditRunnerV92Error("Capability/output path binding drift")
    state["stage"] = "exclusive_complete_evidence_publication"
    try:
        publication = complete_evidence.publish_complete_evidence_exclusive(
            output_path, evidence
        )
    except Exception as exc:
        return persist(
            _complete_run_result(
                evidence=evidence,
                publication=None,
                publication_error=exc,
                consumed=consumed,
            )
        )
    wrapper_error: BaseException | None = None
    try:
        _validate_outer_wrapper(evidence_path=output_path, publication=publication)
    except Exception as exc:
        wrapper_error = exc
    return persist(
        _complete_run_result(
            evidence=evidence,
            publication=publication,
            publication_error=wrapper_error,
            consumed=consumed,
        )
    )


def main() -> None:
    result = run_formal_quality_audit()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
