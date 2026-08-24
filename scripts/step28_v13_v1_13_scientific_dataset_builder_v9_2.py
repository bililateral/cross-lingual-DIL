#!/usr/bin/env python3
"""Minimal V9.2 dataset-writer layer; execution remains externally gated."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any

import step28_v13_common as common
import step28_v13_structure as structure
import step28_v13_v1_13_document_collision as collision
import step28_v13_v1_13_quality_policy_v9_2 as quality_policy_module
import step28_v13_v1_13_scientific_common_v9 as scientific
import step28_v13_v1_13_scientific_dataset_builder_v9 as v9_builder
import step28_v13_v1_13_scientific_world_v9_2 as world_v9_2


VERSION = "2026-08-23-step28-v13-v1-13-scientific-dataset-builder-v9-2"
COUNTERFACTUAL_ITEM_PATH = "observed/redacted_items.style_deranged.jsonl"
COUNTERFACTUAL_PROFILE_PATH = (
    "observed/model_seller_profiles.style_deranged.jsonl"
)
MODEL_INPUT_PATHS = (
    "observed/redacted_items.jsonl",
    "observed/model_seller_profiles.jsonl",
    "observed/redacted_items.code_masked.jsonl",
    "observed/model_seller_profiles.code_masked.jsonl",
    "observed/redacted_items.code_neutralized.jsonl",
    "observed/model_seller_profiles.code_neutralized.jsonl",
    COUNTERFACTUAL_ITEM_PATH,
    COUNTERFACTUAL_PROFILE_PATH,
)
EXPECTED_SPLIT_DATA_PATHS = tuple(
    sorted(
        (*v9_builder.EXPECTED_SPLIT_DATA_PATHS, *MODEL_INPUT_PATHS[6:]),
        key=lambda value: value.encode("utf-8"),
    )
)
ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "development", "audit_a", "audit_b")
EXECUTION_MODE = "method_qualification_1004"
WORLD_COUNTS = {"train": 500, "development": 500, "audit_a": 2, "audit_b": 2}
RANDOM_AUTHORITY_PATH = (
    ROOT / "private_custody" / "step28_v13_v1_13_v9_2_random_authority.json"
)
BUILD_AUTHORIZATION_PATH = (
    ROOT
    / "private_custody"
    / "step28_v13_v1_13_v9_2_design_build_authorization.json"
)
RANDOM_AUTHORITY_VERSION = (
    "2026-08-23-step28-v13-v1-13-random-authority-v9-2"
)
BUILD_AUTHORIZATION_VERSION = (
    "2026-08-23-step28-v13-v1-13-design-build-authorization-v9-2"
)
RANDOM_REVIEW_FINAL_LINE = "允许生成并冻结一次V9.2全新随机权威"
BUILD_REVIEW_FINAL_LINE = "允许运行一次V9.2方法资格根1004世界构建"
KEY_FIELDS = {
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
HEX_64 = re.compile(r"^[0-9a-f]{64}$")
GIT_OBJECT = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
RETIRED_PREFLIGHT_AUTHORITIES = frozenset(
    {
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
)


class DatasetBuilderV92Error(v9_builder.DatasetBuildError):
    """Raised when the eight-file V9.2 model-input contract drifts."""


def _self_hash(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("canonical_self_hash", None)
    return common.canonical_sha256(payload)


def _repo_pin(path: Path, *, include_self_hash: bool = False) -> dict[str, Any]:
    output: dict[str, Any] = {
        "path": path.resolve().relative_to(ROOT.resolve()).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": common.sha256_file(path),
    }
    if include_self_hash:
        value = common.load_json(path)
        if not isinstance(value, Mapping):
            raise DatasetBuilderV92Error("Pinned JSON root is not an object")
        output["canonical_self_hash"] = value.get("canonical_self_hash")
    return output


def _current_git_identity() -> tuple[str, str]:
    def run(*args: str) -> str:
        try:
            value = subprocess.run(
                ["git", *args],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise DatasetBuilderV92Error("Git identity could not be verified") from exc
        return value.stdout.strip()

    return run("rev-parse", "HEAD"), run("rev-parse", "HEAD^{tree}")


def _canonical_external_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DatasetBuilderV92Error("Required external authorization is absent")
    value = common.load_json(path)
    if not isinstance(value, dict):
        raise DatasetBuilderV92Error("External authorization must be an object")
    if path.read_bytes() != common.canonical_json_bytes(value) + b"\n":
        raise DatasetBuilderV92Error("External authorization bytes are noncanonical")
    return value


def _key_values(block: Mapping[str, Any]) -> tuple[str, ...]:
    if set(block) != KEY_FIELDS:
        raise DatasetBuilderV92Error("Random-authority key schema drift")
    rewires = block.get("rewire_key_hexes")
    if not isinstance(rewires, list) or len(rewires) != 5:
        raise DatasetBuilderV92Error("Random authority requires five rewire keys")
    scalar_names = sorted(KEY_FIELDS - {"rewire_key_hexes"})
    values = tuple(str(block[name]) for name in scalar_names) + tuple(
        str(value) for value in rewires
    )
    if (
        any(HEX_64.fullmatch(value) is None for value in values)
        or len(values) != len(set(values))
    ):
        raise DatasetBuilderV92Error("Random authority syntax/reuse drift")
    return values


def _load_parent_builder_policy(quality_policy: Mapping[str, Any]) -> dict[str, Any]:
    pin = quality_policy["parent_policies"]["scientific_builder_v9"]
    path = common.repo_path(str(pin["path"]))
    value = common.load_json(path)
    if not isinstance(value, dict):
        raise DatasetBuilderV92Error("Parent builder policy is not an object")
    scientific.validate_policy(value)
    return value


def validate_random_authority(
    value: Mapping[str, Any],
    *,
    quality_policy: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = json.loads(common.canonical_json_bytes(value).decode("utf-8"))
    if set(normalized) != {
        "version",
        "status",
        "canonical_self_hash",
        "created_by_repository_code",
        "single_use_for_method_qualification_root",
        "keys",
        "authority_bundle_sha256",
        "git_commit",
        "git_tree",
        "review_response_sha256",
        "review_final_line",
    }:
        raise DatasetBuilderV92Error("Random-authority receipt schema drift")
    keys = normalized.get("keys")
    if not isinstance(keys, Mapping):
        raise DatasetBuilderV92Error("Random-authority key block is absent")
    values = _key_values(keys)
    if (
        normalized.get("version") != RANDOM_AUTHORITY_VERSION
        or normalized.get("status") != "FROZEN_FRESH_V9_2_RANDOM_AUTHORITY"
        or normalized.get("canonical_self_hash") != _self_hash(normalized)
        or normalized.get("created_by_repository_code") is not False
        or normalized.get("single_use_for_method_qualification_root") is not True
        or normalized.get("authority_bundle_sha256")
        != common.canonical_sha256(sorted(values))
        or normalized.get("review_final_line") != RANDOM_REVIEW_FINAL_LINE
        or GIT_OBJECT.fullmatch(str(normalized.get("git_commit", ""))) is None
        or GIT_OBJECT.fullmatch(str(normalized.get("git_tree", ""))) is None
        or HEX_64.fullmatch(str(normalized.get("review_response_sha256", "")))
        is None
    ):
        raise DatasetBuilderV92Error("Random-authority receipt identity drift")
    parent = _load_parent_builder_policy(quality_policy)
    base_path = scientific._verify_pin(
        parent["base_dataset_policy"], label="base dataset policy"
    )
    base = common.load_json(base_path)
    if not isinstance(base, Mapping):
        raise DatasetBuilderV92Error("Base dataset policy is not an object")
    forbidden = scientific._collect_random_authorities(base.get("randomness", {}))
    forbidden.update(scientific._collect_random_authorities(parent["public_preflight_keys"]))
    retired = parent["retired_public_preflight_authorities"]
    if (
        len(RETIRED_PREFLIGHT_AUTHORITIES) != retired["count"]
        or common.canonical_sha256(sorted(RETIRED_PREFLIGHT_AUTHORITIES))
        != retired["sorted_values_sha256"]
    ):
        raise DatasetBuilderV92Error("Retired authority registry commitment drift")
    forbidden.update(RETIRED_PREFLIGHT_AUTHORITIES)
    if set(values) & forbidden:
        raise DatasetBuilderV92Error("V9.2 random authority reuses a prior authority")
    return normalized


def validate_build_authorization(
    value: Mapping[str, Any],
    *,
    random_authority: Mapping[str, Any],
    quality_policy: Mapping[str, Any],
    verify_git: bool,
) -> dict[str, Any]:
    normalized = json.loads(common.canonical_json_bytes(value).decode("utf-8"))
    if set(normalized) != {
        "version",
        "status",
        "canonical_self_hash",
        "single_use",
        "receipt_generation_by_repository_code_forbidden",
        "random_authority_file",
        "quality_policy",
        "execution_mode",
        "world_counts",
        "output_root",
        "git_commit",
        "git_tree",
        "review_response_sha256",
        "review_final_line",
        "formal_generation_authorized",
        "quality_audit_authorized",
        "training_authorized",
    }:
        raise DatasetBuilderV92Error("Design-build authorization schema drift")
    expected_random_pin = {
        **_repo_pin(RANDOM_AUTHORITY_PATH),
        "canonical_self_hash": random_authority["canonical_self_hash"],
    }
    expected_quality_pin = {
        **_repo_pin(quality_policy_module.DEFAULT_POLICY_PATH),
        "canonical_self_hash": quality_policy["canonical_self_hash"],
    }
    output_root = common.repo_path(str(normalized.get("output_root", "")))
    reports_root = (ROOT / "reports").resolve()
    if (
        normalized.get("version") != BUILD_AUTHORIZATION_VERSION
        or normalized.get("status")
        != "ONE_SHOT_V9_2_METHOD_QUALIFICATION_BUILD_AUTHORIZED"
        or normalized.get("canonical_self_hash") != _self_hash(normalized)
        or normalized.get("single_use") is not True
        or normalized.get("receipt_generation_by_repository_code_forbidden")
        is not True
        or normalized.get("random_authority_file") != expected_random_pin
        or normalized.get("quality_policy") != expected_quality_pin
        or normalized.get("git_commit") != random_authority["git_commit"]
        or normalized.get("git_tree") != random_authority["git_tree"]
        or normalized.get("execution_mode") != EXECUTION_MODE
        or normalized.get("world_counts") != WORLD_COUNTS
        or reports_root not in output_root.parents
        or output_root.name != "method_qualification_1004"
        or normalized.get("review_final_line") != BUILD_REVIEW_FINAL_LINE
        or normalized.get("formal_generation_authorized") is not False
        or normalized.get("quality_audit_authorized") is not False
        or normalized.get("training_authorized") is not False
        or GIT_OBJECT.fullmatch(str(normalized.get("git_commit", ""))) is None
        or GIT_OBJECT.fullmatch(str(normalized.get("git_tree", ""))) is None
        or HEX_64.fullmatch(str(normalized.get("review_response_sha256", "")))
        is None
    ):
        raise DatasetBuilderV92Error("Design-build authorization identity drift")
    if verify_git and _current_git_identity() != (
        normalized["git_commit"],
        normalized["git_tree"],
    ):
        raise DatasetBuilderV92Error("Reviewed build git identity drift")
    return normalized


def _consume_external(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    consumed = path.with_name(path.stem + ".consumed.json")
    if consumed.exists():
        raise DatasetBuilderV92Error("External authority was already consumed")
    try:
        path.replace(consumed)
    except OSError as exc:
        raise DatasetBuilderV92Error("External authority could not be consumed") from exc
    expected = common.canonical_json_bytes(value) + b"\n"
    if consumed.read_bytes() != expected:
        raise DatasetBuilderV92Error("Consumed external authority bytes drift")
    return {
        "path_sha256": hashlib.sha256(
            consumed.relative_to(ROOT).as_posix().encode("utf-8")
        ).hexdigest(),
        "size_bytes": len(expected),
        "sha256": hashlib.sha256(expected).hexdigest(),
        "canonical_self_hash": value["canonical_self_hash"],
    }


@dataclass
class SplitWritersV92:
    base: v9_builder._SplitWriters
    counterfactual_items: v9_builder._JsonlWriter
    counterfactual_profiles: v9_builder._JsonlWriter

    @classmethod
    def open(
        cls,
        root: Path,
        *,
        endpoint_fields: Sequence[str],
        identity_fields: Sequence[str],
    ) -> "SplitWritersV92":
        return cls(
            base=v9_builder._SplitWriters.open(
                root,
                endpoint_fields=endpoint_fields,
                identity_fields=identity_fields,
            ),
            counterfactual_items=v9_builder._JsonlWriter.open(
                root / COUNTERFACTUAL_ITEM_PATH
            ),
            counterfactual_profiles=v9_builder._JsonlWriter.open(
                root / COUNTERFACTUAL_PROFILE_PATH
            ),
        )

    def close(self) -> None:
        errors: list[BaseException] = []
        for writer in (self.counterfactual_items, self.counterfactual_profiles):
            try:
                writer.close()
            except BaseException as exc:  # pragma: no cover - mechanical cleanup
                errors.append(exc)
        try:
            self.base.close()
        except BaseException as exc:  # pragma: no cover - mechanical cleanup
            errors.append(exc)
        if errors:
            raise DatasetBuilderV92Error(
                "Failed to close one or more V9.2 split outputs"
            ) from errors[0]

    def model_input_row_counts(self) -> dict[str, int]:
        counts = {
            "observed/redacted_items.jsonl": self.base.redacted_items.row_count,
            "observed/model_seller_profiles.jsonl": (
                self.base.model_seller_profiles.row_count
            ),
            "observed/redacted_items.code_masked.jsonl": (
                self.base.masked_redacted_items.row_count
            ),
            "observed/model_seller_profiles.code_masked.jsonl": (
                self.base.masked_model_seller_profiles.row_count
            ),
            "observed/redacted_items.code_neutralized.jsonl": (
                self.base.neutral_redacted_items.row_count
            ),
            "observed/model_seller_profiles.code_neutralized.jsonl": (
                self.base.neutral_model_seller_profiles.row_count
            ),
            COUNTERFACTUAL_ITEM_PATH: self.counterfactual_items.row_count,
            COUNTERFACTUAL_PROFILE_PATH: self.counterfactual_profiles.row_count,
        }
        if tuple(counts) != MODEL_INPUT_PATHS:
            raise DatasetBuilderV92Error("V9.2 model-input path order drift")
        return counts


def write_world(
    writers: SplitWritersV92,
    accepted: world_v9_2.AcceptedScientificWorldV92,
) -> None:
    """Write six original-author and two deranged model inputs exactly once."""

    if not isinstance(writers, SplitWritersV92) or not isinstance(
        accepted, world_v9_2.AcceptedScientificWorldV92
    ):
        raise DatasetBuilderV92Error("V9.2 writer type drift")
    v9_builder._write_world(writers.base, accepted.base)
    for row in accepted.counterfactual_redacted_items:
        writers.counterfactual_items.write(v9_builder._project_model_redacted_item(row))
    for row in accepted.counterfactual_seller_profiles:
        writers.counterfactual_profiles.write(
            v9_builder._project_model_seller_profile(row)
        )


def validate_one_world_model_input_counts(
    writers: SplitWritersV92,
    *,
    expected_item_count: int,
    expected_seller_count: int = 28,
) -> dict[str, int]:
    counts = writers.model_input_row_counts()
    expected = {
        path: (
            expected_seller_count if "seller_profiles" in path else expected_item_count
        )
        for path in MODEL_INPUT_PATHS
    }
    if counts != expected:
        raise DatasetBuilderV92Error("V9.2 eight-file row counts did not close")
    return counts


def _build_execution_context(
    *,
    quality_policy: Mapping[str, Any],
    random_authority: Mapping[str, Any],
    build_authorization: Mapping[str, Any],
) -> tuple[scientific.ExecutionContext, dict[str, Any]]:
    parent = _load_parent_builder_policy(quality_policy)
    base_path = scientific._verify_pin(
        parent["base_dataset_policy"], label="base dataset policy"
    )
    base = common.load_json(base_path)
    if not isinstance(base, dict):
        raise DatasetBuilderV92Error("Base dataset policy is not an object")
    effective = json.loads(common.canonical_json_bytes(base).decode("utf-8"))
    keys = random_authority["keys"]
    scientific._replace_development_stream(effective, keys)
    effective["modes"]["development_smoke"]["world_counts"] = dict(WORLD_COUNTS)
    common.validate_policy(effective, mode="development_smoke")
    records = tuple(
        json.loads(common.canonical_json_bytes(row).decode("utf-8"))
        for row in structure.build_mode_world_pool(
            effective, mode="development_smoke"
        )
    )
    if (
        len(records) != 1004
        or Counter(str(row["split"]) for row in records) != Counter(WORLD_COUNTS)
    ):
        raise DatasetBuilderV92Error("V9.2 world authority cardinality drift")
    context = scientific.ExecutionContext(
        execution_mode=EXECUTION_MODE,
        base_mode="development_smoke",
        effective_policy=effective,
        world_records=records,
        document_variation_key=bytes.fromhex(keys["document_variation_key_hex"]),
        anonymous_handle_key=bytes.fromhex(keys["anonymous_handle_key_hex"]),
        output_root=common.repo_path(str(build_authorization["output_root"])),
        scientific_use_forbidden=True,
    )
    return context, parent


def _all_writers(
    writers: SplitWritersV92,
) -> tuple[v9_builder._JsonlWriter | v9_builder._CsvWriter, ...]:
    return (
        *writers.base.all_writers(),
        writers.counterfactual_items,
        writers.counterfactual_profiles,
    )


def _verify_output_tree(root: Path, root_manifest: Mapping[str, Any]) -> None:
    if common.load_json(root / "root_manifest.json") != root_manifest:
        raise DatasetBuilderV92Error("V9.2 root manifest replay drift")
    if _self_hash(root_manifest) != root_manifest.get("canonical_self_hash"):
        raise DatasetBuilderV92Error("V9.2 root manifest self-hash drift")
    expected_files = {"root_manifest.json"}
    for split in SPLITS:
        split_root = root / split
        manifest = common.load_json(split_root / "split_manifest.json")
        if (
            not isinstance(manifest, Mapping)
            or _self_hash(manifest) != manifest.get("canonical_self_hash")
            or root_manifest["split_manifest_self_hashes"].get(split)
            != manifest["canonical_self_hash"]
        ):
            raise DatasetBuilderV92Error("V9.2 split manifest replay drift")
        records = manifest.get("files")
        if not isinstance(records, list) or {
            str(row.get("path")) for row in records if isinstance(row, Mapping)
        } != set(EXPECTED_SPLIT_DATA_PATHS):
            raise DatasetBuilderV92Error("V9.2 split file universe drift")
        expected_files.add(f"{split}/split_manifest.json")
        for row in records:
            if not isinstance(row, Mapping) or set(row) != {
                "path",
                "size_bytes",
                "sha256",
                "row_count",
            }:
                raise DatasetBuilderV92Error("V9.2 split file record drift")
            path = (split_root / str(row["path"])).resolve()
            if (
                split_root.resolve() not in path.parents
                or not path.is_file()
                or path.stat().st_size != row["size_bytes"]
                or common.sha256_file(path) != row["sha256"]
                or v9_builder._count_file_rows(path) != row["row_count"]
            ):
                raise DatasetBuilderV92Error("V9.2 persisted file replay drift")
            expected_files.add(path.relative_to(root).as_posix())
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        raise DatasetBuilderV92Error("V9.2 physical output universe drift")


def _publish_verified_output(
    *,
    temp_root: Path,
    output_root: Path,
    root_manifest: Mapping[str, Any],
) -> None:
    """Publish once and replay the same verification at the final path."""

    _verify_output_tree(temp_root, root_manifest)
    temp_root.rename(output_root)
    try:
        _verify_output_tree(output_root, root_manifest)
    except Exception:
        # Put the failed publication back under the transaction-owned temporary
        # path so the existing finally block can remove it precisely.
        if output_root.exists() and not temp_root.exists():
            output_root.rename(temp_root)
        raise


def _run_transaction(
    *,
    context: scientific.ExecutionContext,
    parent_policy: Mapping[str, Any],
    quality_policy: Mapping[str, Any],
    random_authority: Mapping[str, Any],
    build_authorization: Mapping[str, Any],
    random_receipt: Mapping[str, Any],
    build_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    output_root = context.output_root
    temp_root = output_root.parent / f".{output_root.name}.building"
    if output_root.exists() or temp_root.exists():
        raise DatasetBuilderV92Error(
            "Immutable output or temporary root already exists; resume is forbidden"
        )
    temp_root.mkdir(parents=True, exist_ok=False)
    completed = False
    writers_by_split: dict[str, SplitWritersV92] = {}
    try:
        template, fixture, style_profile = scientific.load_release_inputs(context)
        historical = collision.load_historical_exclusion_registries()
        endpoint_fields = tuple(
            context.effective_policy["relational_integrity"][
                "pair_projection_contract"
            ]["complete_model_pair_endpoints_schema"]
        )
        identity_fields = (
            "canonical_pair_uid",
            "world_uid",
            *tuple(context.effective_policy["history_features"]["feature_names"]),
        )
        for split in SPLITS:
            writers_by_split[split] = SplitWritersV92.open(
                temp_root / split,
                endpoint_fields=endpoint_fields,
                identity_fields=identity_fields,
            )
        current_item_hashes: set[str] = set()
        current_seller_hashes: set[str] = set()
        current_identity_hashes: set[str] = set()
        current_item_codes: set[str] = set()
        seen_uids = {kind: set() for kind in v9_builder.GLOBAL_UID_KINDS}
        split_uid_sets = {
            split: {kind: set() for kind in v9_builder.GLOBAL_UID_KINDS}
            for split in SPLITS
        }
        split_item_hashes = {split: set() for split in SPLITS}
        split_seller_hashes = {split: set() for split in SPLITS}
        split_identity_hashes = {split: set() for split in SPLITS}
        split_item_codes = {split: set() for split in SPLITS}
        positive_counts: Counter[str] = Counter()
        candidate_histograms: dict[str, Counter[int]] = defaultdict(Counter)
        rejection_totals: dict[str, Counter[str]] = defaultdict(Counter)
        split_world_counts: Counter[str] = Counter()
        split_ordinals = {split: set() for split in SPLITS}
        records = sorted(
            context.world_records,
            key=lambda row: (
                SPLITS.index(str(row["split"])),
                int(row["split_ordinal"]),
            ),
        )
        for position, record in enumerate(records, start=1):
            split = str(record["split"])
            ordinal = record["split_ordinal"]
            if (
                type(ordinal) is not int
                or not 0 <= ordinal < WORLD_COUNTS[split]
                or ordinal in split_ordinals[split]
            ):
                raise DatasetBuilderV92Error("Split world ordinal drift")
            split_ordinals[split].add(ordinal)
            accepted = world_v9_2.build_scientific_world(
                policy=context.effective_policy,
                template=template,
                fixture=fixture,
                style_profile=style_profile,
                mode=context.base_mode,
                world_record=record,
                structure_key_hex=common.structure_key_for_split(
                    context.effective_policy,
                    mode=context.base_mode,
                    split=split,
                ),
                document_variation_key=context.document_variation_key,
                anonymous_handle_key=context.anonymous_handle_key,
                historical_item_hashes=historical.item_document_hashes,
                historical_seller_hashes=historical.seller_document_hashes,
                historical_identity_hashes=historical.identity_value_hashes,
                current_item_hashes=current_item_hashes,
                current_seller_hashes=current_seller_hashes,
                current_identity_hashes=current_identity_hashes,
                current_item_codes=current_item_codes,
                candidate_limit=int(
                    parent_policy["candidate_selection"]["candidate_limit"]
                ),
                identity_maximum_counter=int(
                    parent_policy["candidate_selection"][
                        "identity_value_maximum_counter"
                    ]
                ),
            )
            world_uid_sets = v9_builder._commit_unique_uid_sets(
                accepted, seen=seen_uids
            )
            for kind in v9_builder.GLOBAL_UID_KINDS:
                split_uid_sets[split][kind].update(world_uid_sets[kind])
            if (
                split_item_hashes[split] & set(accepted.item_registry_delta)
                or split_seller_hashes[split] & set(accepted.seller_registry_delta)
                or split_identity_hashes[split] & set(accepted.identity_registry_delta)
                or split_item_codes[split] & set(accepted.code_registry_delta)
            ):
                raise DatasetBuilderV92Error("Within-split registry delta reuse")
            split_item_hashes[split].update(accepted.item_registry_delta)
            split_seller_hashes[split].update(accepted.seller_registry_delta)
            split_identity_hashes[split].update(accepted.identity_registry_delta)
            split_item_codes[split].update(accepted.code_registry_delta)
            write_world(writers_by_split[split], accepted)
            split_world_counts[split] += 1
            positive_counts[split] += sum(row["label"] for row in accepted.pair_labels)
            candidate_histograms[split][accepted.candidate_index] += 1
            rejection_totals[split].update(accepted.rejection_counts)
            print(
                json.dumps(
                    {
                        "event": "world_complete",
                        "execution_mode": EXECUTION_MODE,
                        "position": position,
                        "total_worlds": 1004,
                        "split": split,
                        "split_ordinal": accepted.split_ordinal,
                        "candidate_index": accepted.candidate_index,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )
        split_manifests: dict[str, dict[str, Any]] = {}
        for split in SPLITS:
            writers = writers_by_split[split]
            if split_ordinals[split] != set(range(WORLD_COUNTS[split])):
                raise DatasetBuilderV92Error("Split world ordinals are not contiguous")
            v9_builder._validate_split_counts(
                split=split,
                world_count=WORLD_COUNTS[split],
                writers=writers.base,
                positive_count=positive_counts[split],
                expected_item_count=len(split_item_hashes[split]),
                item_document_hashes=split_item_hashes[split],
                seller_document_hashes=split_seller_hashes[split],
                identity_value_hashes=split_identity_hashes[split],
            )
            expected_counterfactual_counts = {
                builder_v9_2_path: (
                    WORLD_COUNTS[split] * 28
                    if "seller_profiles" in builder_v9_2_path
                    else len(split_item_hashes[split])
                )
                for builder_v9_2_path in MODEL_INPUT_PATHS[6:]
            }
            observed_counts = writers.model_input_row_counts()
            if any(
                observed_counts[path] != expected_counterfactual_counts[path]
                for path in MODEL_INPUT_PATHS[6:]
            ):
                raise DatasetBuilderV92Error("Counterfactual split row count drift")
            writers.close()
            files = [
                v9_builder._file_record(
                    writer.path,
                    root=temp_root / split,
                    row_count=writer.row_count,
                )
                for writer in _all_writers(writers)
            ]
            files.sort(key=lambda row: row["path"].encode("utf-8"))
            manifest: dict[str, Any] = {
                "version": VERSION,
                "status": "PASS_DESIGN_BUILD_NOT_TRAINING_QUALIFIED",
                "execution_mode": EXECUTION_MODE,
                "split": split,
                "world_count": WORLD_COUNTS[split],
                "world_ordinal_count": len(split_ordinals[split]),
                "world_ordinals_sha256": common.canonical_sha256(
                    sorted(split_ordinals[split])
                ),
                "seller_count": WORLD_COUNTS[split] * 28,
                "pair_count": WORLD_COUNTS[split] * 378,
                "positive_pair_count": WORLD_COUNTS[split] * 20,
                "negative_pair_count": WORLD_COUNTS[split] * 358,
                "item_count": len(split_item_hashes[split]),
                "item_code_registry_count": len(split_item_codes[split]),
                "item_code_registry_sha256": common.canonical_sha256(
                    sorted(split_item_codes[split])
                ),
                "item_document_registry_count": len(split_item_hashes[split]),
                "item_document_registry_sha256": common.canonical_sha256(
                    sorted(split_item_hashes[split])
                ),
                "seller_document_registry_count": len(split_seller_hashes[split]),
                "seller_document_registry_sha256": common.canonical_sha256(
                    sorted(split_seller_hashes[split])
                ),
                "identity_value_registry_count": len(split_identity_hashes[split]),
                "identity_value_registry_sha256": common.canonical_sha256(
                    sorted(split_identity_hashes[split])
                ),
                "uid_registries": {
                    kind: {
                        "count": len(split_uid_sets[split][kind]),
                        "sha256": common.canonical_sha256(
                            sorted(split_uid_sets[split][kind])
                        ),
                    }
                    for kind in v9_builder.GLOBAL_UID_KINDS
                },
                "candidate_index_histogram": {
                    str(key): value
                    for key, value in sorted(candidate_histograms[split].items())
                },
                "collision_rejection_totals": {
                    name: int(rejection_totals[split][name])
                    for name in world_v9_2.v9.COLLISION_CATEGORIES
                },
                "model_input_file_count": 8,
                "files": files,
            }
            manifest["canonical_self_hash"] = _self_hash(manifest)
            common.write_json(temp_root / split / "split_manifest.json", manifest)
            split_manifests[split] = manifest
        writers_by_split.clear()
        if split_world_counts != Counter(WORLD_COUNTS):
            raise DatasetBuilderV92Error("Root split world count drift")
        for kind in v9_builder.GLOBAL_UID_KINDS:
            if len(seen_uids[kind]) != sum(
                len(split_uid_sets[split][kind]) for split in SPLITS
            ):
                raise DatasetBuilderV92Error("Cross-split UID registry intersection")
        for global_values, split_values in (
            (current_item_hashes, split_item_hashes),
            (current_seller_hashes, split_seller_hashes),
            (current_identity_hashes, split_identity_hashes),
            (current_item_codes, split_item_codes),
        ):
            if len(global_values) != sum(
                len(split_values[split]) for split in SPLITS
            ):
                raise DatasetBuilderV92Error("Cross-split content registry intersection")
        key_values = random_authority["keys"]
        root_manifest: dict[str, Any] = {
            "version": VERSION,
            "status": "PASS_DESIGN_BUILD_NOT_TRAINING_QUALIFIED",
            "execution_mode": EXECUTION_MODE,
            "scientific_use_forbidden": True,
            "formal_seed_created": False,
            "formal_rows_created": 0,
            "training_started": False,
            "quality_policy_canonical_self_hash": quality_policy[
                "canonical_self_hash"
            ],
            "quality_policy_file": _repo_pin(
                quality_policy_module.DEFAULT_POLICY_PATH
            ),
            "builder_source_file": _repo_pin(Path(__file__)),
            "design_build_authorization": {
                "status": "CONSUMED_ONE_SHOT_BUILD_AUTHORIZATION",
                "receipt": dict(build_receipt),
                "review_response_sha256": build_authorization[
                    "review_response_sha256"
                ],
                "git_commit": build_authorization["git_commit"],
                "git_tree": build_authorization["git_tree"],
            },
            "random_authority": {
                "status": "CONSUMED_FRESH_V9_2_RANDOM_AUTHORITY",
                "receipt": dict(random_receipt),
                "authority_bundle_sha256": random_authority[
                    "authority_bundle_sha256"
                ],
            },
            "quality_required_key_commitments": {
                "id_key_sha256": hashlib.sha256(
                    bytes.fromhex(key_values["id_key_hex"])
                ).hexdigest(),
                "document_variation_key_sha256": hashlib.sha256(
                    bytes.fromhex(key_values["document_variation_key_hex"])
                ).hexdigest(),
            },
            "model_input_file_count": 8,
            "split_order": list(SPLITS),
            "world_count": 1004,
            "seller_count": 1004 * 28,
            "pair_count": 1004 * 378,
            "positive_pair_count": 1004 * 20,
            "negative_pair_count": 1004 * 358,
            "uid_registries": {
                kind: {
                    "count": len(seen_uids[kind]),
                    "sha256": common.canonical_sha256(sorted(seen_uids[kind])),
                }
                for kind in v9_builder.GLOBAL_UID_KINDS
            },
            "item_document_registry_count": len(current_item_hashes),
            "item_document_registry_sha256": common.canonical_sha256(
                sorted(current_item_hashes)
            ),
            "seller_document_registry_count": len(current_seller_hashes),
            "seller_document_registry_sha256": common.canonical_sha256(
                sorted(current_seller_hashes)
            ),
            "item_code_registry_count": len(current_item_codes),
            "item_code_registry_sha256": common.canonical_sha256(
                sorted(current_item_codes)
            ),
            "identity_value_registry_count": len(current_identity_hashes),
            "identity_value_registry_sha256": common.canonical_sha256(
                sorted(current_identity_hashes)
            ),
            "historical_exclusion_counts": {
                "item_documents": len(historical.item_document_hashes),
                "seller_documents": len(historical.seller_document_hashes),
                "identity_values": len(historical.identity_value_hashes),
            },
            "split_manifest_self_hashes": {
                split: split_manifests[split]["canonical_self_hash"]
                for split in SPLITS
            },
        }
        root_manifest["canonical_self_hash"] = _self_hash(root_manifest)
        common.write_json(temp_root / "root_manifest.json", root_manifest)
        _publish_verified_output(
            temp_root=temp_root,
            output_root=output_root,
            root_manifest=root_manifest,
        )
        completed = True
        return root_manifest
    finally:
        if not completed:
            for writers in writers_by_split.values():
                for writer in _all_writers(writers):
                    if not writer.handle.closed:
                        writer.handle.close()
            v9_builder._safe_remove_temp(temp_root, expected_output=output_root)


def run_design_preflight_once() -> dict[str, Any]:
    """Consume two independently reviewed receipts before creating any root byte."""

    quality_policy = quality_policy_module.load_policy()
    if not RANDOM_AUTHORITY_PATH.is_file() or not BUILD_AUTHORIZATION_PATH.is_file():
        raise DatasetBuilderV92Error(
            "V9.2 random ceremony/build remain unauthorized; external receipts are absent"
        )
    random_authority = validate_random_authority(
        _canonical_external_json(RANDOM_AUTHORITY_PATH),
        quality_policy=quality_policy,
    )
    build_authorization = validate_build_authorization(
        _canonical_external_json(BUILD_AUTHORIZATION_PATH),
        random_authority=random_authority,
        quality_policy=quality_policy,
        verify_git=True,
    )
    context, parent = _build_execution_context(
        quality_policy=quality_policy,
        random_authority=random_authority,
        build_authorization=build_authorization,
    )
    output_root = context.output_root
    temp_root = output_root.parent / f".{output_root.name}.building"
    if output_root.exists() or temp_root.exists():
        raise DatasetBuilderV92Error(
            "Immutable output or temporary root already exists; resume is forbidden"
        )
    build_receipt = _consume_external(
        BUILD_AUTHORIZATION_PATH, build_authorization
    )
    random_receipt = _consume_external(RANDOM_AUTHORITY_PATH, random_authority)
    return _run_transaction(
        context=context,
        parent_policy=parent,
        quality_policy=quality_policy,
        random_authority=random_authority,
        build_authorization=build_authorization,
        random_receipt=random_receipt,
        build_receipt=build_receipt,
    )


if __name__ == "__main__":
    run_design_preflight_once()
