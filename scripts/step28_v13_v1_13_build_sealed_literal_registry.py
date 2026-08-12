#!/usr/bin/env python3
"""Build the relation-free sealed literal registry for the v1.13 design audit.

This privileged design-only packer deterministically replays the already frozen
104 worlds.  It emits raw forbidden literals only under ignored
``private_custody/`` and publishes a commitment-only receipt.  It does not
change observed rows, candidates, labels, worlds, thresholds, or derangements.
"""

from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import importlib.util
import json
import os
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import step28_v13_common as common
import step28_v13_v1_13_blind_literal_scan as literal_scan
import step28_v13_v1_13_document_collision as collision
import step28_v13_v1_13_scientific_common as scientific
import step28_v13_v1_13_scientific_dataset_builder as dataset_builder
import step28_v13_v1_13_scientific_world as world_module


ROOT = Path(__file__).resolve().parents[1]
VERSION = "2026-08-12-step28-v13-v1-13-sealed-literal-registry-builder-v2"
TRANSACTION_VERSION = (
    "2026-08-12-step28-v13-v1-13-sealed-literal-registry-transaction-v1"
)
TRANSACTION_LOCK_VERSION = (
    "2026-08-12-step28-v13-v1-13-sealed-literal-registry-lock-v1"
)
DATASET_ROOT = (
    ROOT
    / "reports"
    / "step28_v13_v1_13_scientific_builder"
    / "design_preflight_v2_20260811"
)
EXPECTED_ROOT_MANIFEST_SELF_HASH = (
    "9baa90828cf459bcee3cc6101c166f6c1084353dd2997e40e5d3d85d29f49d48"
)
EXPECTED_ROOT_MANIFEST_SIZE_BYTES = 2_663
EXPECTED_ROOT_MANIFEST_SHA256 = (
    "c7b323d9d3b76a0795ce45452b4989926cae51fca7101bc6d04acf6fdf4a93f3"
)
EXPECTED_BUILDER_POLICY_SIZE_BYTES = 8_189
EXPECTED_BUILDER_POLICY_SHA256 = (
    "5c4ba22cbbd001efab521384a9a988410f30703bf2657a1d425bd8eba4d2629a"
)
EXPECTED_BUILDER_POLICY_SELF_HASH = (
    "8f40cc0b008e6447e5ace55b59159545346c6527a91d2129677e59ce087a7a47"
)
PRIVATE_OUTPUT = (
    ROOT
    / "private_custody"
    / "step28_v13_v1_13"
    / "sealed_private_literal_registry_design_v1_20260812.json"
)
PUBLIC_RECEIPT = (
    ROOT
    / "reports"
    / "step28_v13_v1_13_scientific_builder"
    / "sealed_private_literal_registry_receipt_v1_20260812.json"
)
TRANSACTION_INTENT = (
    ROOT
    / "private_custody"
    / "step28_v13_v1_13"
    / "sealed_private_literal_registry_design_v1_20260812.transaction.json"
)
TRANSACTION_LOCK = (
    ROOT
    / "private_custody"
    / "step28_v13_v1_13"
    / "sealed_private_literal_registry_design_v1_20260812.lock.json"
)
AUDIT_SPLITS = ("audit_a", "audit_b")
LITERAL_AUTHORITY_SOURCE = (
    ROOT / "scripts" / "step28_v13_v1_13_blind_literal_scan.py"
)
ONE_ROW_PRIVATE_REPLAY_FILES = (
    "private/world_generation_audit.jsonl",
    "private/document_collision_attempts.jsonl",
    "private/identity_allocation_receipts.jsonl",
)
MULTI_ROW_PRIVATE_REPLAY_FILES = (
    "private/controller_membership.jsonl",
    "private/qrels.jsonl",
)
PAIR_LABEL_FILE = "private/pair_labels.csv"
SEMANTIC_PRIVATE_FILES = (
    "private/controller_membership.jsonl",
    PAIR_LABEL_FILE,
    "private/qrels.jsonl",
    "private/world_generation_audit.jsonl",
    "private/document_collision_attempts.jsonl",
    "private/identity_allocation_receipts.jsonl",
)
class SealedLiteralRegistryBuildError(common.ContractError):
    """Raised without echoing any sealed literal."""


def _canonical_self_hash(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("canonical_self_hash", None)
    return common.canonical_sha256(payload)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _source_record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file() or ROOT not in path.parents:
        raise SealedLiteralRegistryBuildError("Runtime source path drift")
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": common.sha256_file(path),
    }


def _discover_repo_local_module_paths(
    *, entry_path: Path | None = None, scripts_root: Path | None = None
) -> dict[str, str]:
    """Recursively derive the complete repo-local import closure from source."""

    entry = Path(__file__).resolve() if entry_path is None else entry_path.resolve()
    local_root = (
        (ROOT / "scripts").resolve()
        if scripts_root is None
        else scripts_root.resolve()
    )
    if not entry.is_file() or (entry != local_root and local_root not in entry.parents):
        raise SealedLiteralRegistryBuildError("Source-closure entry path drift")
    pending = [entry]
    discovered_paths: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in discovered_paths:
            continue
        discovered_paths.add(path)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeError, SyntaxError) as exc:
            raise SealedLiteralRegistryBuildError(
                "Source-closure import graph cannot be parsed"
            ) from exc
        imported_names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_names.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported_names.add(node.module.split(".", 1)[0])
            elif isinstance(node, ast.Call) and node.args:
                function_name = ""
                if isinstance(node.func, ast.Name):
                    function_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    function_name = node.func.attr
                first = node.args[0]
                if (
                    function_name in {"__import__", "import_module"}
                    and isinstance(first, ast.Constant)
                    and isinstance(first.value, str)
                ):
                    imported_names.add(first.value.split(".", 1)[0])
        for module_name in imported_names:
            candidate = (local_root / f"{module_name}.py").resolve()
            if candidate.is_file() and candidate not in discovered_paths:
                pending.append(candidate)
    output: dict[str, str] = {}
    for path in sorted(discovered_paths):
        key = "packer" if path == entry else path.stem
        if key in output:
            raise SealedLiteralRegistryBuildError(
                "Source-closure module-name collision"
            )
        output[key] = path.relative_to(ROOT if entry_path is None else local_root.parent).as_posix()
    return dict(sorted(output.items()))


def _official_cli_context() -> bool:
    try:
        return Path(sys.argv[0]).resolve() == Path(__file__).resolve()
    except (OSError, RuntimeError):
        return False


def _capture_runtime_source_closure() -> dict[str, dict[str, Any]]:
    discovered = _discover_repo_local_module_paths()
    closure: dict[str, dict[str, Any]] = {}
    for module_name, relative in discovered.items():
        if module_name == "packer":
            closure[module_name] = _source_record(Path(__file__))
            continue
        module = sys.modules.get(module_name)
        expected = (ROOT / relative).resolve()
        if module is not None:
            source = getattr(module, "__file__", None)
            observed = None if source is None else Path(source).resolve()
        else:
            spec = importlib.util.find_spec(module_name)
            observed = (
                None
                if spec is None or spec.origin is None
                else Path(spec.origin).resolve()
            )
        if observed != expected:
            raise SealedLiteralRegistryBuildError(
                "Runtime source import path drift"
            )
        closure[module_name] = _source_record(expected)
    if _official_cli_context():
        discovered_paths = {
            (ROOT / relative).resolve() for relative in discovered.values()
        }
        for module in tuple(sys.modules.values()):
            source = getattr(module, "__file__", None)
            if source is None:
                continue
            path = Path(source).resolve()
            if (
                path.suffix == ".py"
                and (ROOT / "scripts").resolve() in path.parents
                and path not in discovered_paths
            ):
                raise SealedLiteralRegistryBuildError(
                    "Undeclared repo-local runtime module"
                )
    return dict(sorted(closure.items()))


def _verify_runtime_source_closure(
    expected: Mapping[str, Mapping[str, Any]],
) -> None:
    if _capture_runtime_source_closure() != expected:
        raise SealedLiteralRegistryBuildError(
            "Runtime source closure changed during replay"
        )


def _root_manifest_record() -> dict[str, Any]:
    path = DATASET_ROOT / "root_manifest.json"
    record = _source_record(path)
    if (
        record["size_bytes"] != EXPECTED_ROOT_MANIFEST_SIZE_BYTES
        or record["sha256"] != EXPECTED_ROOT_MANIFEST_SHA256
    ):
        raise SealedLiteralRegistryBuildError(
            "Frozen root manifest raw bytes drift"
        )
    return record


def _builder_policy_record(policy: Mapping[str, Any]) -> dict[str, Any]:
    record = {
        **_source_record(scientific.DEFAULT_POLICY_PATH),
        "canonical_self_hash": policy.get("canonical_self_hash"),
    }
    if record != {
        "path": scientific.DEFAULT_POLICY_PATH.relative_to(ROOT).as_posix(),
        "size_bytes": EXPECTED_BUILDER_POLICY_SIZE_BYTES,
        "sha256": EXPECTED_BUILDER_POLICY_SHA256,
        "canonical_self_hash": EXPECTED_BUILDER_POLICY_SELF_HASH,
    }:
        raise SealedLiteralRegistryBuildError(
            "Frozen builder policy raw bytes drift"
        )
    return record


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for line in handle:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise SealedLiteralRegistryBuildError(
                        "Sealed replay row is not an object"
                    )
                rows.append(row)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SealedLiteralRegistryBuildError(
            "Sealed replay input cannot be read"
        ) from exc
    if not rows:
        raise SealedLiteralRegistryBuildError("Sealed replay input is empty")
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if (
                reader.fieldnames
                != ["canonical_pair_uid", "world_uid", "label"]
                or len(reader.fieldnames) != len(set(reader.fieldnames))
            ):
                raise SealedLiteralRegistryBuildError(
                    "Sealed pair-label header drift"
                )
            rows = [dict(row) for row in reader]
    except (OSError, UnicodeError, csv.Error) as exc:
        raise SealedLiteralRegistryBuildError(
            "Sealed pair-label replay input cannot be read"
        ) from exc
    if not rows:
        raise SealedLiteralRegistryBuildError("Sealed pair-label input is empty")
    return rows


def _index_one_row_per_world(
    rows: Sequence[Mapping[str, Any]], *, split: str, label: str
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for source in rows:
        row = dict(source)
        world_uid = str(row.get("world_uid", ""))
        if (
            not world_uid
            or str(row.get("split", "")) != split
            or world_uid in output
        ):
            raise SealedLiteralRegistryBuildError(
                f"Sealed {label} world index drift"
            )
        output[world_uid] = row
    return output


def _group_rows_per_world(
    rows: Sequence[Mapping[str, Any]], *, label: str
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in rows:
        row = dict(source)
        world_uid = str(row.get("world_uid", ""))
        if not world_uid:
            raise SealedLiteralRegistryBuildError(
                f"Sealed {label} world grouping drift"
            )
        output[world_uid].append(row)
    return dict(output)


def _expected_pair_label_rows(
    accepted: world_module.AcceptedScientificWorld,
) -> list[dict[str, str]]:
    return [
        {
            "canonical_pair_uid": str(row["canonical_pair_uid"]),
            "world_uid": str(row["world_uid"]),
            "label": str(int(row["label"])),
        }
        for row in accepted.pair_labels
    ]


def _pair_label_literal_projection(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    projected: list[dict[str, str]] = []
    for row in rows:
        if set(row) != {"canonical_pair_uid", "world_uid", "label"} or str(
            row["label"]
        ) not in {"0", "1"}:
            raise SealedLiteralRegistryBuildError(
                "Pair-label literal projection schema drift"
            )
        projected.append(
            {
                "canonical_pair_uid": str(row["canonical_pair_uid"]),
                "world_uid": str(row["world_uid"]),
            }
        )
    return projected


def _expected_collision_row(
    accepted: world_module.AcceptedScientificWorld,
) -> dict[str, Any]:
    return {
        "world_uid": accepted.world_uid,
        "split": accepted.split,
        "split_ordinal": accepted.split_ordinal,
        "accepted_candidate_index": accepted.candidate_index,
        "candidates_examined": accepted.candidates_examined,
        "rejection_counts": accepted.rejection_counts,
        "item_registry_delta_count": len(accepted.item_registry_delta),
        "item_registry_delta_sha256": common.canonical_sha256(
            accepted.item_registry_delta
        ),
        "seller_registry_delta_count": len(accepted.seller_registry_delta),
        "seller_registry_delta_sha256": common.canonical_sha256(
            accepted.seller_registry_delta
        ),
        "natural_output_sha256": accepted.natural_output_sha256,
    }


def _expected_identity_allocation_row(
    accepted: world_module.AcceptedScientificWorld,
) -> dict[str, Any]:
    return {
        "world_uid": accepted.world_uid,
        "split": accepted.split,
        "split_ordinal": accepted.split_ordinal,
        "identity_registry_delta_count": len(accepted.identity_registry_delta),
        "identity_registry_delta_sha256": common.canonical_sha256(
            accepted.identity_registry_delta
        ),
        "receipt": accepted.identity_allocation_receipt,
    }


def _add_literal_rows(
    categories: dict[str, set[str]],
    *,
    category: str,
    value: Any,
) -> None:
    for _source_category, literal in literal_scan._collect_literal_fields(value):
        if literal:
            categories[category].add(str(literal))


def _collect_audit_world_literals(
    accepted: world_module.AcceptedScientificWorld,
    *,
    persisted_world_audit: Mapping[str, Any],
    persisted_collision: Mapping[str, Any],
    persisted_identity_allocation: Mapping[str, Any],
    persisted_controller_membership: Sequence[Mapping[str, Any]],
    persisted_qrels: Sequence[Mapping[str, Any]],
    persisted_pair_labels: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, set[str]], set[str]]:
    categories: dict[str, set[str]] = defaultdict(set)
    categories["full_world_forbidden"].update(
        literal_scan.collect_complete_world_forbidden_literals(
            world_uid=accepted.world_uid,
            public_sellers=accepted.world["public"]["sellers"],
            public_items=accepted.world["public"]["items"],
            public_pair_endpoints=accepted.world["public"][
                "complete_model_pair_endpoints"
            ],
            qrels=accepted.qrels,
            private_world=accepted.world["private"],
            persisted_private_world_audit=persisted_world_audit,
        )
    )
    _add_literal_rows(
        categories,
        category="persisted_world_audit_string",
        value=persisted_world_audit,
    )
    _add_literal_rows(
        categories,
        category="document_collision_receipt_string",
        value=persisted_collision,
    )
    _add_literal_rows(
        categories,
        category="identity_allocation_receipt_string",
        value=persisted_identity_allocation,
    )
    _add_literal_rows(
        categories,
        category="controller_membership_string",
        value=list(persisted_controller_membership),
    )
    _add_literal_rows(
        categories,
        category="qrels_string",
        value=list(persisted_qrels),
    )
    _add_literal_rows(
        categories,
        category="pair_label_string",
        value=_pair_label_literal_projection(persisted_pair_labels),
    )
    allowed_noise_surfaces = {
        str(row.get("raw_surface", ""))
        for row in accepted.world["private"].get("noise_slots_audit", ())
        if str(row.get("raw_surface", ""))
    }
    noise_uids = {
        str(row.get("noise_slot_uid", ""))
        for row in accepted.world["private"].get("noise_slots_audit", ())
        if str(row.get("noise_slot_uid", ""))
    }
    if not noise_uids or not noise_uids <= categories["full_world_forbidden"]:
        raise SealedLiteralRegistryBuildError(
            "Noise-slot UID forbidden registry is incomplete"
        )
    # A raw nuisance surface is not globally exempt: if it independently
    # equals a forbidden UID/value it remains forbidden.  The allowlist exists
    # only to prove the packer did not ban the entire noise object by default.
    return dict(categories), allowed_noise_surfaces


def _merge_sets(
    target: dict[str, set[str]], source: Mapping[str, set[str]]
) -> None:
    for category, values in source.items():
        target.setdefault(category, set()).update(values)


def _validate_split_projection(
    *,
    split: str,
    categories: Mapping[str, Sequence[str]],
    allowed_noise: Sequence[str],
    world_count: int,
    expected_world_count: int,
) -> None:
    if split not in AUDIT_SPLITS:
        raise SealedLiteralRegistryBuildError("Audit split projection drift")
    if tuple(categories) != literal_scan.SEALED_REGISTRY_CATEGORIES:
        raise SealedLiteralRegistryBuildError(
            "Forbidden category universe or order drift"
        )
    if any(not values for values in categories.values()):
        raise SealedLiteralRegistryBuildError("Empty forbidden category")
    if not allowed_noise:
        raise SealedLiteralRegistryBuildError("Empty allowed-noise projection")
    if world_count != expected_world_count:
        raise SealedLiteralRegistryBuildError("Audit world count drift")


def _replay_and_collect() -> tuple[dict[str, Any], dict[str, Any]]:
    source_closure = _capture_runtime_source_closure()
    root_manifest_source = _root_manifest_record()
    if Path(str(literal_scan.__file__)).resolve() != LITERAL_AUTHORITY_SOURCE.resolve():
        raise SealedLiteralRegistryBuildError("Literal authority import path drift")
    literal_authority_source = copy.deepcopy(
        source_closure["step28_v13_v1_13_blind_literal_scan"]
    )
    policy = scientific.load_policy()
    builder_policy_source = _builder_policy_record(policy)
    dataset_builder._validate_model_mount_contract(policy)
    context = scientific.build_execution_context(
        policy, execution_mode="design_preflight"
    )
    if context.output_root.resolve() != DATASET_ROOT.resolve():
        raise SealedLiteralRegistryBuildError("Frozen design root drift")
    root_manifest = common.load_json(DATASET_ROOT / "root_manifest.json")
    if (
        not isinstance(root_manifest, dict)
        or _canonical_self_hash(root_manifest)
        != root_manifest.get("canonical_self_hash")
        or root_manifest.get("canonical_self_hash")
        != EXPECTED_ROOT_MANIFEST_SELF_HASH
        or root_manifest.get("builder_policy_canonical_self_hash")
        != policy.get("canonical_self_hash")
    ):
        raise SealedLiteralRegistryBuildError("Frozen root manifest drift")
    dataset_builder._verify_output_tree(DATASET_ROOT, root_manifest)

    replay: dict[str, dict[str, Any]] = {}
    for split in scientific.SPLITS:
        replay[split] = {}
        for relative in ONE_ROW_PRIVATE_REPLAY_FILES:
            label = Path(relative).stem
            replay[split][label] = _index_one_row_per_world(
                _read_jsonl(DATASET_ROOT / split / relative),
                split=split,
                label=label,
            )
        for relative in MULTI_ROW_PRIVATE_REPLAY_FILES:
            label = Path(relative).stem
            replay[split][label] = _group_rows_per_world(
                _read_jsonl(DATASET_ROOT / split / relative),
                label=label,
            )
        replay[split][Path(PAIR_LABEL_FILE).stem] = _group_rows_per_world(
            _read_csv(DATASET_ROOT / split / PAIR_LABEL_FILE),
            label=Path(PAIR_LABEL_FILE).stem,
        )

    template, fixture, style_profile = scientific.load_release_inputs(context)
    historical = collision.load_historical_exclusion_registries()
    current_item_hashes: set[str] = set()
    current_seller_hashes: set[str] = set()
    current_identity_hashes: set[str] = set()
    seen_uids: dict[str, set[str]] = {
        kind: set() for kind in dataset_builder.GLOBAL_UID_KINDS
    }
    ordinals: dict[str, set[int]] = {
        split: set() for split in scientific.SPLITS
    }
    audit_categories: dict[str, dict[str, set[str]]] = {
        split: {} for split in AUDIT_SPLITS
    }
    audit_allowed_noise: dict[str, set[str]] = {
        split: set() for split in AUDIT_SPLITS
    }
    audit_world_counts = {split: 0 for split in AUDIT_SPLITS}
    records = sorted(
        context.world_records,
        key=lambda row: (
            scientific.SPLITS.index(str(row["split"])),
            int(row["split_ordinal"]),
        ),
    )
    for position, record in enumerate(records, start=1):
        split = str(record["split"])
        ordinal = int(record["split_ordinal"])
        if ordinal in ordinals[split]:
            raise SealedLiteralRegistryBuildError("Replay ordinal collision")
        ordinals[split].add(ordinal)
        structure_key = common.structure_key_for_split(
            context.effective_policy,
            mode=context.base_mode,
            split=split,
        )
        accepted = world_module.build_scientific_world(
            policy=context.effective_policy,
            template=template,
            fixture=fixture,
            style_profile=style_profile,
            mode=context.base_mode,
            world_record=record,
            structure_key_hex=structure_key,
            document_variation_key=context.document_variation_key,
            anonymous_handle_key=context.anonymous_handle_key,
            historical_item_hashes=historical.item_document_hashes,
            historical_seller_hashes=historical.seller_document_hashes,
            historical_identity_hashes=historical.identity_value_hashes,
            current_item_hashes=current_item_hashes,
            current_seller_hashes=current_seller_hashes,
            current_identity_hashes=current_identity_hashes,
            candidate_limit=int(policy["candidate_selection"]["candidate_limit"]),
            identity_maximum_counter=int(
                policy["candidate_selection"]["identity_value_maximum_counter"]
            ),
        )
        dataset_builder._commit_unique_uid_sets(accepted, seen=seen_uids)
        world_uid = accepted.world_uid
        source_world = replay[split]["world_generation_audit"].get(world_uid)
        source_collision = replay[split]["document_collision_attempts"].get(
            world_uid
        )
        source_identity = replay[split]["identity_allocation_receipts"].get(
            world_uid
        )
        source_controller_membership = replay[split]["controller_membership"].get(
            world_uid
        )
        source_qrels = replay[split]["qrels"].get(world_uid)
        source_pair_labels = replay[split]["pair_labels"].get(world_uid)
        if (
            source_world is None
            or source_collision is None
            or source_identity is None
            or source_controller_membership is None
            or source_qrels is None
            or source_pair_labels is None
            or common.canonical_json_bytes(
                dataset_builder._private_world_audit_row(accepted)
            )
            != common.canonical_json_bytes(source_world)
            or common.canonical_json_bytes(_expected_collision_row(accepted))
            != common.canonical_json_bytes(source_collision)
            or common.canonical_json_bytes(
                _expected_identity_allocation_row(accepted)
            )
            != common.canonical_json_bytes(source_identity)
            or common.canonical_json_bytes(list(accepted.controller_membership))
            != common.canonical_json_bytes(source_controller_membership)
            or common.canonical_json_bytes(list(accepted.qrels))
            != common.canonical_json_bytes(source_qrels)
            or common.canonical_json_bytes(_expected_pair_label_rows(accepted))
            != common.canonical_json_bytes(source_pair_labels)
        ):
            raise SealedLiteralRegistryBuildError(
                "Frozen private world replay drift"
            )
        if split in AUDIT_SPLITS:
            categories, allowed_noise = _collect_audit_world_literals(
                accepted,
                persisted_world_audit=source_world,
                persisted_collision=source_collision,
                persisted_identity_allocation=source_identity,
                persisted_controller_membership=source_controller_membership,
                persisted_qrels=source_qrels,
                persisted_pair_labels=source_pair_labels,
            )
            _merge_sets(audit_categories[split], categories)
            audit_allowed_noise[split].update(allowed_noise)
            audit_world_counts[split] += 1
        print(
            json.dumps(
                {
                    "event": "sealed_registry_world_replayed",
                    "position": position,
                    "total_worlds": len(records),
                    "split": split,
                    "split_ordinal": ordinal,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )

    for split in scientific.SPLITS:
        expected_count = int(
            policy["execution_modes"]["design_preflight"]["world_counts"][split]
        )
        if ordinals[split] != set(range(expected_count)):
            raise SealedLiteralRegistryBuildError("Replay ordinal universe drift")
        expected_world_uids = set(replay[split]["world_generation_audit"])
        if len(expected_world_uids) != expected_count:
            raise SealedLiteralRegistryBuildError("Replay world universe drift")
        if any(
            set(replay[split][label]) != expected_world_uids
            for label in (
                "document_collision_attempts",
                "identity_allocation_receipts",
                "controller_membership",
                "qrels",
                "pair_labels",
            )
        ):
            raise SealedLiteralRegistryBuildError("Replay receipt universe drift")
    pair_label_rows_replayed = sum(
        len(rows)
        for split in scientific.SPLITS
        for rows in replay[split]["pair_labels"].values()
    )
    if pair_label_rows_replayed != len(records) * 378:
        raise SealedLiteralRegistryBuildError("Pair-label replay cardinality drift")
    expected_registries = {
        "item": (current_item_hashes, "item_document_registry"),
        "seller": (current_seller_hashes, "seller_document_registry"),
        "identity": (current_identity_hashes, "identity_value_registry"),
    }
    for values, prefix in expected_registries.values():
        if (
            len(values) != int(root_manifest[f"{prefix}_count"])
            or common.canonical_sha256(sorted(values))
            != root_manifest[f"{prefix}_sha256"]
        ):
            raise SealedLiteralRegistryBuildError("Root registry replay drift")
    for kind, values in seen_uids.items():
        expected = root_manifest["uid_registries"][kind]
        if (
            len(values) != int(expected["count"])
            or common.canonical_sha256(sorted(values)) != expected["sha256"]
        ):
            raise SealedLiteralRegistryBuildError("Root UID replay drift")

    # Revalidate both the complete frozen tree and every imported source byte
    # after the long replay.  A mid-run source/input edit cannot produce a
    # successful registry receipt.
    dataset_builder._verify_output_tree(DATASET_ROOT, root_manifest)
    if _root_manifest_record() != root_manifest_source:
        raise SealedLiteralRegistryBuildError(
            "Frozen root manifest changed during replay"
        )
    _verify_runtime_source_closure(source_closure)

    split_registries: dict[str, Any] = {}
    public_split_commitments: dict[str, Any] = {}
    for split in AUDIT_SPLITS:
        categories = {
            category: sorted(values, key=lambda value: value.encode("utf-8"))
            for category, values in sorted(audit_categories[split].items())
        }
        allowed_noise = sorted(
            audit_allowed_noise[split], key=lambda value: value.encode("utf-8")
        )
        expected_audit_world_count = int(
            policy["execution_modes"]["design_preflight"]["world_counts"][split]
        )
        _validate_split_projection(
            split=split,
            categories=categories,
            allowed_noise=allowed_noise,
            world_count=audit_world_counts[split],
            expected_world_count=expected_audit_world_count,
        )
        split_value = {
            "world_count": audit_world_counts[split],
            "categories": categories,
            "allowed_noise_raw_surfaces": allowed_noise,
            "forbidden_literal_count": len(
                set().union(*(set(values) for values in categories.values()))
            ),
            "category_commitments": {
                category: {
                    "count": len(values),
                    "sha256": common.canonical_sha256(values),
                }
                for category, values in categories.items()
            },
            "allowed_noise_raw_surface_commitment": {
                "count": len(allowed_noise),
                "sha256": common.canonical_sha256(allowed_noise),
            },
        }
        split_registries[split] = split_value
        public_split_commitments[split] = {
            key: copy.deepcopy(split_value[key])
            for key in (
                "world_count",
                "forbidden_literal_count",
                "category_commitments",
                "allowed_noise_raw_surface_commitment",
            )
        }
    sidecar = {
        "version": VERSION,
        "status": "SEALED_RELATION_FREE_LITERAL_REGISTRY",
        "dataset_root_manifest": {
            **root_manifest_source,
            "canonical_self_hash": root_manifest["canonical_self_hash"],
        },
        "literal_authority_source": copy.deepcopy(literal_authority_source),
        "builder_policy": copy.deepcopy(builder_policy_source),
        "split_order": list(AUDIT_SPLITS),
        "private_relations_persisted": False,
        "labels_persisted": False,
        "labels_opened_for_exact_replay": True,
        "labels_used_for_candidate_selection": False,
        "labels_used_for_literal_selection": False,
        "pair_label_rows_replayed": pair_label_rows_replayed,
        "qrels_persisted": False,
        "observed_rows_modified": 0,
        "private_input_files_semantically_replayed": list(SEMANTIC_PRIVATE_FILES),
        "split_registries": split_registries,
        "canonical_self_hash": None,
    }
    sidecar["canonical_self_hash"] = _canonical_self_hash(sidecar)
    receipt_base = {
        "version": VERSION,
        "status": "PASS_SEALED_LITERAL_REGISTRY_BUILD_NOT_AUDIT_GO",
        "dataset_root_manifest": copy.deepcopy(sidecar["dataset_root_manifest"]),
        "literal_authority_source": copy.deepcopy(literal_authority_source),
        "builder_policy": copy.deepcopy(builder_policy_source),
        "worlds_replayed": len(records),
        "audit_worlds_projected": sum(audit_world_counts.values()),
        "private_input_files_semantically_replayed": list(SEMANTIC_PRIVATE_FILES),
        "private_input_file_count": len(SEMANTIC_PRIVATE_FILES),
        "split_commitments": public_split_commitments,
        "private_values_returned": 0,
        "private_relations_returned": 0,
        "labels_returned": 0,
        "labels_opened_for_exact_replay": True,
        "labels_used_for_candidate_selection": False,
        "labels_used_for_literal_selection": False,
        "pair_label_rows_replayed": pair_label_rows_replayed,
        "qrels_returned": 0,
        "observed_rows_modified": 0,
        "candidate_selection_changed": False,
        "derangement_changed": False,
        "quality_probe_run": False,
        "formal_generation_authorized": False,
        "model_training_authorized": False,
        "formal_seed_authorized": False,
        "audit_truth_release_authorized": False,
        "quality_audit_run_authorized": False,
        "formal_500x4_generation_authorized": False,
        "design_dataset_training_qualified": False,
        "source_closure": copy.deepcopy(source_closure),
    }
    return sidecar, receipt_base


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.building")


def _fsync_parent_directory(path: Path) -> None:
    """Persist rename metadata where the host exposes directory fsync."""

    if os.name == "nt":
        # CPython cannot portably open a Windows directory for os.fsync.
        # Recovery is therefore provided by the immutable transaction intent.
        return
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_once(
    path: Path,
    value: Mapping[str, Any],
    owned_paths: list[Path] | None = None,
    *,
    retain_temporary_marker: bool = False,
) -> None:
    if retain_temporary_marker and owned_paths is None:
        raise SealedLiteralRegistryBuildError(
            "Retained build marker requires an ownership ledger"
        )
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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    if path.exists() or temporary.exists():
        raise SealedLiteralRegistryBuildError("Immutable registry output exists")
    temporary_owned = False
    final_owned = False
    try:
        with temporary.open("xb") as handle:
            temporary_owned = True
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # Same-directory hard-link publication is atomic and refuses to
            # overwrite a final path created by another invocation.
            os.link(temporary, path)
        except FileExistsError as exc:
            raise SealedLiteralRegistryBuildError(
                "Immutable registry output appeared during publication"
            ) from exc
        final_owned = True
        if owned_paths is not None:
            owned_paths.append(path)
        if retain_temporary_marker:
            if owned_paths is not None:
                owned_paths.append(temporary)
        else:
            temporary.unlink()
            temporary_owned = False
        _fsync_parent_directory(path)
    except BaseException:
        # A signal can arrive after os.link() returns but before Python records
        # final_owned.  While our exclusive temporary link still exists,
        # samefile() is the ownership proof.  Never delete a merely planned
        # target or another invocation's fixed .building path.
        if (
            not final_owned
            and temporary_owned
            and temporary.is_file()
            and path.is_file()
        ):
            try:
                final_owned = os.path.samefile(temporary, path)
            except OSError:
                final_owned = False
        cleanup_paths: list[Path] = []
        if final_owned or (owned_paths is not None and path in owned_paths):
            cleanup_paths.append(path)
        if temporary_owned:
            cleanup_paths.append(temporary)
        for owned_path in cleanup_paths:
            if not owned_path.exists():
                continue
            if not owned_path.is_file():
                continue
            owned_path.unlink()
            try:
                _fsync_parent_directory(owned_path)
            except BaseException:
                pass
        raise


def _repo_relative(path: Path) -> str:
    path = path.resolve()
    if ROOT not in path.parents:
        raise SealedLiteralRegistryBuildError("Transaction target path drift")
    return path.relative_to(ROOT).as_posix()


def _transaction_targets() -> dict[str, str]:
    return {
        "private_sidecar": _repo_relative(PRIVATE_OUTPUT),
        "public_receipt": _repo_relative(PUBLIC_RECEIPT),
    }


def _derive_transaction_id(
    *,
    dataset_root_manifest: Mapping[str, Any],
    builder_policy: Mapping[str, Any],
    source_closure: Mapping[str, Any],
    targets: Mapping[str, str],
) -> str:
    return common.canonical_sha256(
        {
            "version": TRANSACTION_VERSION,
            "dataset_root_manifest": dataset_root_manifest,
            "builder_policy": builder_policy,
            "source_closure": source_closure,
            "targets": targets,
        }
    )


def _build_transaction_lock_contract() -> dict[str, Any]:
    source_closure = _capture_runtime_source_closure()
    policy = scientific.load_policy()
    builder_policy = _builder_policy_record(policy)
    root_manifest = common.load_json(DATASET_ROOT / "root_manifest.json")
    root_pin = {
        **_root_manifest_record(),
        "canonical_self_hash": root_manifest.get("canonical_self_hash"),
    }
    if (
        not isinstance(root_manifest, dict)
        or root_manifest.get("canonical_self_hash")
        != EXPECTED_ROOT_MANIFEST_SELF_HASH
        or root_manifest.get("canonical_self_hash")
        != _canonical_self_hash(root_manifest)
        or root_manifest.get("builder_policy_canonical_self_hash")
        != policy.get("canonical_self_hash")
    ):
        raise SealedLiteralRegistryBuildError(
            "Frozen lock input contract drift"
        )
    targets = _transaction_targets()
    transaction_id = _derive_transaction_id(
        dataset_root_manifest=root_pin,
        builder_policy=builder_policy,
        source_closure=source_closure,
        targets=targets,
    )
    lock = {
        "version": TRANSACTION_LOCK_VERSION,
        "status": "SEALED_LITERAL_REGISTRY_EXCLUSIVE_REPLAY_LOCK",
        "transaction_id": transaction_id,
        "dataset_root_manifest": root_pin,
        "builder_policy": builder_policy,
        "source_closure": source_closure,
        "targets": targets,
        "canonical_self_hash": None,
    }
    lock["canonical_self_hash"] = _canonical_self_hash(lock)
    return lock


def _validate_transaction_lock(
    value: Mapping[str, Any], *, require_current_inputs: bool = False
) -> dict[str, Any]:
    lock = dict(value)
    if (
        set(lock)
        != {
            "version",
            "status",
            "transaction_id",
            "dataset_root_manifest",
            "builder_policy",
            "source_closure",
            "targets",
            "canonical_self_hash",
        }
        or lock.get("version") != TRANSACTION_LOCK_VERSION
        or lock.get("status")
        != "SEALED_LITERAL_REGISTRY_EXCLUSIVE_REPLAY_LOCK"
        or lock.get("canonical_self_hash") != _canonical_self_hash(lock)
        or lock.get("targets") != _transaction_targets()
    ):
        raise SealedLiteralRegistryBuildError("Transaction lock drift")
    root_pin = lock.get("dataset_root_manifest")
    builder_policy = lock.get("builder_policy")
    source_closure = lock.get("source_closure")
    targets = lock.get("targets")
    if (
        not isinstance(root_pin, dict)
        or root_pin
        != {
            "path": DATASET_ROOT.relative_to(ROOT).as_posix()
            + "/root_manifest.json",
            "size_bytes": EXPECTED_ROOT_MANIFEST_SIZE_BYTES,
            "sha256": EXPECTED_ROOT_MANIFEST_SHA256,
            "canonical_self_hash": EXPECTED_ROOT_MANIFEST_SELF_HASH,
        }
        or not isinstance(builder_policy, dict)
        or not isinstance(source_closure, dict)
        or not isinstance(targets, dict)
        or lock.get("transaction_id")
        != _derive_transaction_id(
            dataset_root_manifest=root_pin,
            builder_policy=builder_policy,
            source_closure=source_closure,
            targets=targets,
        )
    ):
        raise SealedLiteralRegistryBuildError("Transaction lock binding drift")
    _validate_builder_policy_pin(builder_policy)
    _validate_source_closure_shape(
        source_closure, require_current_graph=require_current_inputs
    )
    if require_current_inputs:
        if _capture_runtime_source_closure() != source_closure:
            raise SealedLiteralRegistryBuildError(
                "Transaction lock source bytes drift"
            )
        if _builder_policy_record(scientific.load_policy()) != builder_policy:
            raise SealedLiteralRegistryBuildError(
                "Transaction lock builder policy drift"
            )
        current_root = common.load_json(DATASET_ROOT / "root_manifest.json")
        if {
            **_root_manifest_record(),
            "canonical_self_hash": current_root.get("canonical_self_hash"),
        } != root_pin:
            raise SealedLiteralRegistryBuildError(
                "Transaction lock root manifest drift"
            )
    return lock


def _validate_builder_policy_pin(value: Any) -> dict[str, Any]:
    expected = {
        "path": scientific.DEFAULT_POLICY_PATH.relative_to(ROOT).as_posix(),
        "size_bytes": EXPECTED_BUILDER_POLICY_SIZE_BYTES,
        "sha256": EXPECTED_BUILDER_POLICY_SHA256,
        "canonical_self_hash": EXPECTED_BUILDER_POLICY_SELF_HASH,
    }
    if (
        not isinstance(value, dict)
        or value != expected
    ):
        raise SealedLiteralRegistryBuildError(
            "Transaction builder-policy pin drift"
        )
    return value


def _validate_source_closure_shape(
    value: Any, *, require_current_graph: bool = False
) -> dict[str, Any]:
    expected_paths = (
        _discover_repo_local_module_paths() if require_current_graph else None
    )
    required_names = {
        "packer",
        "step3_build_seller_profiles",
        "step7_v3_1_source_data",
        "step7_v4_common",
        "step28_v13_v1_13_blind_literal_scan",
        "step28_v13_v1_13_scientific_dataset_builder",
        "step28_v13_v1_13_scientific_world",
    }
    if (
        not isinstance(value, dict)
        or not required_names <= set(value)
        or (expected_paths is not None and set(value) != set(expected_paths))
    ):
        raise SealedLiteralRegistryBuildError(
            "Transaction source-closure universe drift"
        )
    for name in sorted(value):
        expected_path = (
            expected_paths[name]
            if expected_paths is not None
            else (
                "scripts/step28_v13_v1_13_build_sealed_literal_registry.py"
                if name == "packer"
                else f"scripts/{name}.py"
            )
        )
        record = value.get(name)
        if (
            not isinstance(record, dict)
            or set(record) != {"path", "size_bytes", "sha256"}
            or record.get("path") != expected_path
            or isinstance(record.get("size_bytes"), bool)
            or not isinstance(record.get("size_bytes"), int)
            or record.get("size_bytes", 0) <= 0
            or not isinstance(record.get("sha256"), str)
            or len(record.get("sha256", "")) != 64
            or any(
                character not in "0123456789abcdef"
                for character in record.get("sha256", "")
            )
        ):
            raise SealedLiteralRegistryBuildError(
                "Transaction source-closure record drift"
            )
    return value


def _prepare_transaction(
    sidecar_source: Mapping[str, Any], receipt_source: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    sidecar = copy.deepcopy(dict(sidecar_source))
    receipt = copy.deepcopy(dict(receipt_source))
    root_pin = sidecar.get("dataset_root_manifest")
    builder_policy = receipt.get("builder_policy")
    source_closure = receipt.get("source_closure")
    if not isinstance(root_pin, dict):
        raise SealedLiteralRegistryBuildError(
            "Transaction input contract is incomplete"
        )
    _validate_source_closure_shape(source_closure, require_current_graph=True)
    _validate_builder_policy_pin(builder_policy)
    if sidecar.get("builder_policy") != builder_policy:
        raise SealedLiteralRegistryBuildError(
            "Sidecar/receipt builder-policy binding drift"
        )
    targets = _transaction_targets()
    transaction_id = _derive_transaction_id(
        dataset_root_manifest=root_pin,
        builder_policy=builder_policy,
        source_closure=source_closure,
        targets=targets,
    )
    sidecar["transaction_id"] = transaction_id
    sidecar["canonical_self_hash"] = _canonical_self_hash(sidecar)
    receipt["transaction_id"] = transaction_id
    intent = {
        "version": TRANSACTION_VERSION,
        "status": "SEALED_LITERAL_REGISTRY_TRANSACTION_STARTED",
        "transaction_id": transaction_id,
        "dataset_root_manifest": copy.deepcopy(root_pin),
        "builder_policy": copy.deepcopy(builder_policy),
        "source_closure": copy.deepcopy(source_closure),
        "targets": targets,
        "sidecar_canonical_self_hash": sidecar["canonical_self_hash"],
        "canonical_self_hash": None,
    }
    intent["canonical_self_hash"] = _canonical_self_hash(intent)
    return sidecar, receipt, intent


def _read_contract_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SealedLiteralRegistryBuildError(
            "Transaction artifact cannot be read"
        ) from exc
    if not isinstance(value, dict):
        raise SealedLiteralRegistryBuildError(
            "Transaction artifact schema drift"
        )
    return value


def _validate_transaction_intent(value: Mapping[str, Any]) -> dict[str, Any]:
    intent = dict(value)
    if (
        set(intent)
        != {
            "version",
            "status",
            "transaction_id",
            "dataset_root_manifest",
            "builder_policy",
            "source_closure",
            "targets",
            "sidecar_canonical_self_hash",
            "canonical_self_hash",
        }
        or intent.get("version") != TRANSACTION_VERSION
        or intent.get("status")
        != "SEALED_LITERAL_REGISTRY_TRANSACTION_STARTED"
        or intent.get("canonical_self_hash") != _canonical_self_hash(intent)
        or intent.get("targets") != _transaction_targets()
    ):
        raise SealedLiteralRegistryBuildError("Transaction intent drift")
    root_pin = intent.get("dataset_root_manifest")
    builder_policy = intent.get("builder_policy")
    source_closure = intent.get("source_closure")
    targets = intent.get("targets")
    if (
        not isinstance(root_pin, dict)
        or set(root_pin)
        != {"path", "size_bytes", "sha256", "canonical_self_hash"}
        or root_pin.get("path")
        != DATASET_ROOT.relative_to(ROOT).as_posix() + "/root_manifest.json"
        or root_pin.get("size_bytes") != EXPECTED_ROOT_MANIFEST_SIZE_BYTES
        or root_pin.get("sha256") != EXPECTED_ROOT_MANIFEST_SHA256
        or root_pin.get("canonical_self_hash")
        != EXPECTED_ROOT_MANIFEST_SELF_HASH
        or not isinstance(source_closure, dict)
        or not isinstance(targets, dict)
        or intent.get("transaction_id")
        != _derive_transaction_id(
            dataset_root_manifest=root_pin,
            builder_policy=builder_policy,
            source_closure=source_closure,
            targets=targets,
        )
    ):
        raise SealedLiteralRegistryBuildError("Transaction binding drift")
    _validate_source_closure_shape(source_closure)
    _validate_builder_policy_pin(builder_policy)
    return intent


def _validate_private_sidecar_against_intent(
    intent: Mapping[str, Any], path: Path | None = None
) -> dict[str, Any]:
    path = PRIVATE_OUTPUT if path is None else path
    sidecar = _read_contract_json(path)
    if (
        sidecar.get("canonical_self_hash") != _canonical_self_hash(sidecar)
        or sidecar.get("canonical_self_hash")
        != intent.get("sidecar_canonical_self_hash")
        or sidecar.get("transaction_id") != intent.get("transaction_id")
        or sidecar.get("dataset_root_manifest")
        != intent.get("dataset_root_manifest")
        or sidecar.get("builder_policy") != intent.get("builder_policy")
    ):
        raise SealedLiteralRegistryBuildError("Private sidecar transaction drift")
    return sidecar


def _validate_completed_transaction(
    intent: Mapping[str, Any],
) -> dict[str, Any]:
    sidecar = _validate_private_sidecar_against_intent(intent)
    receipt = _read_contract_json(PUBLIC_RECEIPT)
    try:
        literal_scan.validate_sealed_registry_public_receipt_structure(receipt)
    except literal_scan.BlindLiteralInputError as exc:
        raise SealedLiteralRegistryBuildError(
            "Completed public receipt contract drift"
        ) from exc
    sealed_pin = receipt.get("sealed_registry")
    if (
        receipt.get("canonical_self_hash") != _canonical_self_hash(receipt)
        or receipt.get("status")
        != "PASS_SEALED_LITERAL_REGISTRY_BUILD_NOT_AUDIT_GO"
        or receipt.get("transaction_id") != intent.get("transaction_id")
        or receipt.get("dataset_root_manifest")
        != intent.get("dataset_root_manifest")
        or receipt.get("builder_policy") != intent.get("builder_policy")
        or receipt.get("source_closure") != intent.get("source_closure")
        or not isinstance(sealed_pin, dict)
        or sealed_pin
        != {
            "path": _repo_relative(PRIVATE_OUTPUT),
            "size_bytes": PRIVATE_OUTPUT.stat().st_size,
            "sha256": common.sha256_file(PRIVATE_OUTPUT),
            "canonical_self_hash": sidecar["canonical_self_hash"],
        }
    ):
        raise SealedLiteralRegistryBuildError(
            "Completed transaction binding drift"
        )
    return receipt


def _verify_runtime_input_closure(intent: Mapping[str, Any]) -> None:
    _verify_runtime_source_closure(intent["source_closure"])
    builder_policy = scientific.load_policy()
    observed_policy_pin = _builder_policy_record(builder_policy)
    if observed_policy_pin != intent.get("builder_policy"):
        raise SealedLiteralRegistryBuildError(
            "Builder policy changed during transaction"
        )
    root_manifest = common.load_json(DATASET_ROOT / "root_manifest.json")
    observed_root_pin = {
        **_root_manifest_record(),
        "canonical_self_hash": root_manifest.get("canonical_self_hash"),
    }
    if (
        observed_root_pin != intent.get("dataset_root_manifest")
        or root_manifest.get("canonical_self_hash")
        != _canonical_self_hash(root_manifest)
        or root_manifest.get("builder_policy_canonical_self_hash")
        != builder_policy.get("canonical_self_hash")
    ):
        raise SealedLiteralRegistryBuildError(
            "Frozen dataset binding changed during transaction"
        )
    dataset_builder._verify_output_tree(DATASET_ROOT, root_manifest)


def _unlink_exact_file(path: Path) -> None:
    if not path.exists():
        return
    if not path.is_file():
        raise SealedLiteralRegistryBuildError(
            "Transaction cleanup target is not a file"
        )
    path.unlink()
    _fsync_parent_directory(path)


def _prune_empty_private_parents() -> None:
    custody_root = (ROOT / "private_custody").resolve()
    parent = PRIVATE_OUTPUT.resolve().parent
    if parent != custody_root and custody_root not in parent.parents:
        return
    for path in (parent, custody_root):
        if path.is_dir():
            try:
                path.rmdir()
            except OSError:
                # A nonempty directory is outside this transaction's cleanup
                # authority and must remain untouched.
                pass


def _transaction_artifact_paths() -> tuple[Path, ...]:
    final_paths = (
        TRANSACTION_LOCK,
        TRANSACTION_INTENT,
        PRIVATE_OUTPUT,
        PUBLIC_RECEIPT,
    )
    return (*final_paths, *(_temporary_path(path) for path in final_paths))


def _assert_clean_transaction_start() -> None:
    if any(path.exists() for path in _transaction_artifact_paths()):
        raise SealedLiteralRegistryBuildError(
            "Immutable output or interrupted transaction exists"
        )


def recover_interrupted_transaction() -> dict[str, Any]:
    """Fail closed without reading, validating, or deleting any artifact."""

    raise SealedLiteralRegistryBuildError(
        "Automatic registry recovery is disabled; preserve artifacts for audit"
    )


def run_build() -> dict[str, Any]:
    _assert_clean_transaction_start()
    lock = _build_transaction_lock_contract()
    lock_owned_paths: list[Path] = []
    completed = False
    created_paths: list[Path] = []
    try:
        # Enter the cleanup boundary before publishing the exclusive lock.
        # This closes the catchable-signal window between a successful lock
        # publication and the start of the replay transaction.
        _write_once(
            TRANSACTION_LOCK,
            lock,
            lock_owned_paths,
            retain_temporary_marker=True,
        )
        _validate_transaction_lock(lock, require_current_inputs=True)
        _verify_runtime_input_closure(lock)
        sidecar_source, receipt_source = _replay_and_collect()
        sidecar, receipt, intent = _prepare_transaction(
            sidecar_source, receipt_source
        )
        if (
            intent.get("transaction_id") != lock.get("transaction_id")
            or intent.get("dataset_root_manifest")
            != lock.get("dataset_root_manifest")
            or intent.get("builder_policy") != lock.get("builder_policy")
            or intent.get("source_closure") != lock.get("source_closure")
            or intent.get("targets") != lock.get("targets")
        ):
            raise SealedLiteralRegistryBuildError(
                "Replay transaction differs from exclusive lock"
            )
        _verify_runtime_input_closure(intent)
        _write_once(TRANSACTION_INTENT, intent, created_paths)
        _write_once(PRIVATE_OUTPUT, sidecar, created_paths)
        receipt = {
            **receipt,
            "sealed_registry": {
                "path": PRIVATE_OUTPUT.relative_to(ROOT).as_posix(),
                "size_bytes": PRIVATE_OUTPUT.stat().st_size,
                "sha256": common.sha256_file(PRIVATE_OUTPUT),
                "canonical_self_hash": sidecar["canonical_self_hash"],
            },
            "builder_source": {
                **receipt["source_closure"]["packer"],
            },
            "canonical_self_hash": None,
        }
        receipt["canonical_self_hash"] = _canonical_self_hash(receipt)
        _write_once(PUBLIC_RECEIPT, receipt, created_paths)
        _verify_runtime_input_closure(intent)
        _validate_completed_transaction(intent)
        _unlink_exact_file(TRANSACTION_INTENT)
        created_paths.remove(TRANSACTION_INTENT)
        completed = True
        return receipt
    finally:
        if not completed:
            for path in reversed(created_paths):
                _unlink_exact_file(path)
        for path in reversed(lock_owned_paths):
            _unlink_exact_file(path)
        _prune_empty_private_parents()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the v1.13 sealed literal registry exactly once"
    )
    parser.parse_args()
    try:
        receipt = run_build()
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "SEALED_LITERAL_REGISTRY_BUILD_FAILED",
                    "exception_type": type(exc).__name__,
                    "private_values_returned": 0,
                    "private_relations_returned": 0,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
        raise SystemExit(2) from None
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "canonical_self_hash": receipt["canonical_self_hash"],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
