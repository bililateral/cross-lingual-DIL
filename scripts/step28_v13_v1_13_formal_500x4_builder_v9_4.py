#!/usr/bin/env python3
"""Build the one-time V9.4 formal 500x4 Chinese synthetic dataset."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_balanced_world_schedule_v9_4 as schedule_v94
import step28_v13_v1_13_document_collision as collision
import step28_v13_v1_13_formal_identity_exclusion_validator_v9_4 as identity_exclusion
import step28_v13_v1_13_method_root_builder_v9_4 as engine


VERSION = "2026-08-29-step28-v13-v1-13-formal-500x4-builder-v9-4-v1"
POLICY_PATH = ROOT / "schema" / "step28_v13_v1_13_v9_4_formal_500x4_policy.json"
SPLITS = ("train", "development", "audit_a", "audit_b")
FORMAL_WORLD_COUNTS = {split: 500 for split in SPLITS}
SMOKE_WORLD_COUNTS = {split: 1 for split in SPLITS}
AUTHORIZATION_STATUS = "AUTHORIZED_ONCE_NOT_CONSUMED"
AUTHORIZATION_VERSION = (
    "2026-08-29-step28-v13-v1-13-v9-4-formal-500x4-authority-v1"
)
ISSUANCE_CLAIM_VERSION = (
    "2026-08-29-step28-v13-v1-13-v9-4-formal-500x4-"
    "authority-issuance-claim-v1"
)
POLICY_READY_STATUS = "READY_FOR_ONE_TIME_FORMAL_AUTHORIZATION"
QUALITY_STATUS = "PASSED_METHOD_ROOT_QUALITY_ELIGIBLE_FOR_FORMAL_500X4_APPLICATION"
HEX64 = frozenset("0123456789abcdef")


class Formal500x4BuildError(ValueError):
    """Raised when the formal dataset contract cannot close."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise Formal500x4BuildError(f"JSON object required: {path}")
    return value


def write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True, encoding="utf-8",
    ).stdout.strip()


def git_tree() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=ROOT, check=True,
        capture_output=True, text=True, encoding="utf-8",
    ).stdout.strip()


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(HEX64)
    )


def require_self_hash(value: Mapping[str, Any], *, label: str) -> None:
    payload = dict(value)
    claimed = payload.pop("canonical_self_hash", None)
    if not is_sha256(claimed) or claimed != canonical_sha256(payload):
        raise Formal500x4BuildError(f"{label} canonical self-hash drift")


def validate_pinned_file(spec: Mapping[str, Any], *, label: str) -> Path:
    if set(spec) != {"path", "sha256"}:
        raise Formal500x4BuildError(f"{label} pin schema drift")
    path = ROOT / str(spec["path"])
    if not path.is_file() or sha256_file(path) != spec["sha256"]:
        raise Formal500x4BuildError(f"{label} pinned bytes drift")
    return path


def validate_manifest_payloads(
    *, root: Path, expected_rows: Any, label: str,
    excluded_actual_paths: Sequence[str] = (),
) -> None:
    if not isinstance(expected_rows, list):
        raise Formal500x4BuildError(f"{label} file manifest is not a list")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in expected_rows:
        if not isinstance(row, dict) or set(row) != {
            "path", "size_bytes", "sha256",
        }:
            raise Formal500x4BuildError(f"{label} file manifest schema drift")
        relative = row["path"]
        pure = PurePosixPath(relative) if isinstance(relative, str) else None
        if (
            pure is None
            or pure.is_absolute()
            or "\\" in relative
            or not pure.parts
            or any(part in ("", ".", "..") for part in pure.parts)
            or relative in seen
            or not isinstance(row["size_bytes"], int)
            or row["size_bytes"] < 0
            or not is_sha256(row["sha256"])
        ):
            raise Formal500x4BuildError(f"{label} file manifest entry drift")
        seen.add(relative)
        normalized.append(dict(row))
    excluded = set(excluded_actual_paths)
    actual = [
        row for row in engine.file_manifest(root)
        if row["path"] not in excluded
    ]
    if set(excluded).intersection(seen) or normalized != actual:
        raise Formal500x4BuildError(f"{label} committed payload bytes drift")


def validate_method_root_payloads(
    policy: Mapping[str, Any], manifest: Mapping[str, Any],
) -> None:
    method = policy["method_qualification"]
    validate_manifest_payloads(
        root=ROOT / str(method["public_root"]),
        expected_rows=manifest.get("public_files"),
        label="method-root public",
        excluded_actual_paths=("root_manifest.json",),
    )
    validate_manifest_payloads(
        root=ROOT / str(method["private_root"]),
        expected_rows=manifest.get("private_file_commitments"),
        label="method-root private",
    )


def validate_policy(*, formal: bool) -> dict[str, Any]:
    policy = read_json(POLICY_PATH)
    if tuple(policy.get("split_order", ())) != SPLITS:
        raise Formal500x4BuildError("Formal split order drift")
    if policy.get("world_counts") != FORMAL_WORLD_COUNTS:
        raise Formal500x4BuildError("Formal 500x4 world counts drift")
    if policy.get("world_contract") != {
        "seller_count": 28,
        "controller_count": 12,
        "triad_count": 4,
        "dyad_count": 8,
        "pair_count": 378,
        "positive_pair_count": 20,
        "negative_pair_count": 358,
    }:
        raise Formal500x4BuildError("Formal per-world contract drift")
    if policy.get("authorization") != {
        "formal_build": False,
        "training_qualification": False,
        "audit_truth_unsealing": False,
        "model_training": False,
    }:
        raise Formal500x4BuildError("Pre-build authorization boundary drift")
    if policy.get("private_authority") != {
        "key_names": ["text", "identity", "style", "schedule", "uid", "time"],
        "key_size_bytes": 32,
        "all_commitments_distinct": True,
        "derivation_from_method_root_forbidden": True,
        "print_key_material_forbidden": True,
    }:
        raise Formal500x4BuildError("Formal private-authority contract drift")
    identity_boundary = policy.get("truth_access", {}).get(
        "method_root_identity_exclusion_validator", {}
    )
    if (
        policy.get("truth_access", {}).get(
            "build_time_audit_a_semantic_reads"
        ) != 0
        or policy.get("truth_access", {}).get(
            "build_time_audit_b_semantic_reads"
        ) != 0
        or identity_boundary.get("allowed_private_files") != [
            f"{split}/identity_plan.jsonl" for split in SPLITS
        ]
        or identity_boundary.get("allowed_projected_fields")
        != ["asset_uid", "value_sha256"]
        or identity_boundary.get("private_rows_or_values_returned") != 0
        or identity_boundary.get(
            "controller_membership_pair_labels_qrels_forbidden"
        ) is not True
    ):
        raise Formal500x4BuildError("Sealed identity exclusion boundary drift")
    identity_contract = policy.get("identity_contract", {})
    if (
        identity_contract.get("mechanism_graph_by_split") != {
            "train": "G_A", "development": "G_A",
            "audit_a": "G_A", "audit_b": "G_B",
        }
        or identity_contract.get("feature_count") != 33
        or identity_contract.get("production_parser_required") is not True
        or identity_contract.get(
            "controller_or_label_direct_feature_construction_forbidden"
        ) is not True
        or identity_contract.get("cross_world_identity_value_reuse_forbidden")
        is not True
        or identity_contract.get("method_root_identity_value_reuse_forbidden")
        is not True
        or policy.get("publication_status") != "BUILT_NOT_TRAINING_QUALIFIED"
    ):
        raise Formal500x4BuildError("Formal identity or publication contract drift")
    for name in ("scientific_contract", "reconciliation_contract"):
        validate_pinned_file(policy[name], label=name)
    for name, spec in policy["reuse_sources"].items():
        validate_pinned_file(spec, label=f"reuse_sources.{name}")
    if policy.get("collision_contract") != {
        "historical_policy_path": (
            "schema/step28_v13_v1_13_document_collision_policy.json"
        ),
        "item_surface": "production_redacted_title_description",
        "seller_surface": "frozen_five_field_model_profile",
        "exclude_successful_v1_2": True,
        "exclude_failed_v1_12": True,
        "exclude_v9_4_method_root": True,
        "historical_uid_hash_method": "sha256_utf8_exact_uid",
        "historical_uid_hash_types_required": [
            "canonical_pair_uid", "controller_uid", "item_uid", "query_uid",
            "seller_uid", "world_uid",
        ],
        "method_root_pair_uid_exclusion_required": True,
        "same_world_duplicate_maximum": 0,
        "cross_world_duplicate_maximum": 0,
        "cross_split_duplicate_maximum": 0,
        "candidate_retry_forbidden": True,
    }:
        raise Formal500x4BuildError("Formal collision contract drift")
    schedule_spec = policy["public_balanced_schedule"]
    schedule_path = ROOT / str(schedule_spec["source_path"])
    if (
        not schedule_path.is_file()
        or sha256_file(schedule_path) != schedule_spec["source_sha256"]
        or schedule_spec.get("basis_by_split") != {
            "train": "train",
            "development": "development",
            "audit_a": "train",
            "audit_b": "development",
        }
        or schedule_spec.get("formal_namespace") != "v9_4_formal_500x4"
        or any(
            schedule_spec.get(name) is not True
            for name in (
                "private_transform_authority_required",
                "split_specific_world_order",
                "split_specific_seller_slot_permutation",
                "split_specific_noise_slot_permutation",
                "noise_permutation_preserves_market_equivalence",
                "model_visible_identifiers_forbidden",
                "method_root_identifier_reuse_forbidden",
            )
        )
    ):
        raise Formal500x4BuildError("Balanced schedule contract drift")
    quality_spec = policy["method_qualification"]
    manifest_path = ROOT / str(quality_spec["root_manifest_path"])
    quality_path = ROOT / str(quality_spec["quality_result_path"])
    method_authorization_path = ROOT / str(
        quality_spec["method_root_authorization_path"]
    )
    if (
        not manifest_path.is_file()
        or sha256_file(manifest_path) != quality_spec["root_manifest_sha256"]
        or not quality_path.is_file()
        or sha256_file(quality_path) != quality_spec["quality_result_sha256"]
        or not method_authorization_path.is_file()
        or sha256_file(method_authorization_path)
        != quality_spec["method_root_authorization_sha256"]
    ):
        raise Formal500x4BuildError("Method-qualification evidence drift")
    manifest = read_json(manifest_path)
    quality = read_json(quality_path)
    require_self_hash(manifest, label="method-root manifest")
    require_self_hash(quality, label="method-root quality result")
    truth = quality.get("truth_access", {})
    if (
        manifest.get("canonical_self_hash")
        != quality_spec["root_manifest_canonical_self_hash"]
        or quality.get("canonical_self_hash")
        != quality_spec["quality_result_canonical_self_hash"]
        or quality.get("status") != QUALITY_STATUS
        or quality.get("method_root_quality_passed") is not True
        or quality.get("eligible_for_formal_500x4_generation_application") is not True
        or quality.get("formal_500x4_generated") is not False
        or quality.get("training_qualified") is not False
        or quality.get("m0_m1_m2_m3_training_authorized") is not False
        or truth.get("audit_a_semantic_reads") != 0
        or truth.get("audit_b_semantic_reads") != 0
    ):
        raise Formal500x4BuildError("Method-root quality claim boundary drift")
    validate_method_root_payloads(policy, manifest)
    authority_root = ROOT / str(policy["formal_authority_root"])
    receipt_paths = []
    for name in (
        "formal_issuance_claim_path", "formal_consumption_path",
        "formal_failure_path", "formal_completion_path",
    ):
        path = ROOT / str(policy[name])
        receipt_paths.append(path)
        if path.parent != authority_root:
            raise Formal500x4BuildError("Formal receipt path boundary drift")
    if len(set(receipt_paths)) != len(receipt_paths):
        raise Formal500x4BuildError("Formal receipt paths are not distinct")
    if formal:
        if policy.get("status") != POLICY_READY_STATUS:
            raise Formal500x4BuildError("Formal policy is not frozen for authorization")
        validate_pinned_file(policy["formal_build_contract"], label="formal_build_contract")
        require_self_hash(policy, label="formal policy")
    return policy


def forbidden_authority_commitments(policy: Mapping[str, Any]) -> set[str]:
    method_spec = policy["method_qualification"]
    method_authorization = read_json(
        ROOT / str(method_spec["method_root_authorization_path"])
    )
    commitments = {
        str(spec["commitment_sha256"])
        for spec in method_authorization["key_files"].values()
    }
    commitments.add(str(method_authorization["time_key"]["commitment_sha256"]))
    historical = collision.load_historical_exclusion_registries(
        ROOT / str(policy["collision_contract"]["historical_policy_path"])
    )
    commitments.update(historical.consumed_capability_commitments)
    if not commitments or any(not is_sha256(value) for value in commitments):
        raise Formal500x4BuildError("Forbidden authority commitment registry drift")
    return commitments


def _ranked_ints(
    values: Sequence[int], key: bytes, *parts: object,
) -> list[int]:
    return [int(value) for value in engine.ranked(
        (str(value) for value in values), key, *parts,
    )]


def _market_preserving_noise_permutation(
    schedule_key: bytes, split: str,
) -> list[int]:
    market_count = len(engine.MARKETS)
    output: list[int | None] = [None] * 28
    for market_index in range(market_count):
        slots = [
            slot for slot in range(28) if slot % market_count == market_index
        ]
        permuted = _ranked_ints(
            slots, schedule_key, "noise-slot-permutation", split, market_index,
        )
        for source, target in zip(slots, permuted, strict=True):
            output[source] = target
    if (
        any(value is None for value in output)
        or {int(value) for value in output} != set(range(28))
        or any(
            source % market_count != int(target) % market_count
            for source, target in enumerate(output)
        )
    ):
        raise Formal500x4BuildError(
            "Formal noise permutation changes market equivalence"
        )
    return [int(value) for value in output]


@lru_cache(maxsize=2)
def _load_basis_schedule(split: str):
    return schedule_v94.build_split_schedule(split)


def _transform_schedule(
    *, split: str, schedule_key: bytes, policy: Mapping[str, Any],
) -> tuple[tuple[engine.PublicWorld, ...], dict[str, str]]:
    schedule_spec = policy["public_balanced_schedule"]
    basis_name = str(schedule_spec["basis_by_split"][split])
    basis = _load_basis_schedule(basis_name)
    world_order = _ranked_ints(range(500), schedule_key, "world-order", split)
    seller_permutation = _ranked_ints(
        range(28), schedule_key, "seller-slot-permutation", split,
    )
    noise_permutation = _market_preserving_noise_permutation(
        schedule_key, split,
    )
    namespace = str(schedule_spec["formal_namespace"])
    output: list[engine.PublicWorld] = []
    commitment_rows: list[dict[str, Any]] = []
    for ordinal, basis_index in enumerate(world_order):
        public = basis.public_worlds[basis_index]
        groups = basis.controller_groups_by_world[basis_index]
        old_uids = tuple(str(value) for value in public["seller_uids"])
        old_slot = {seller_uid: index for index, seller_uid in enumerate(old_uids)}
        world_uid = f"{namespace}_{split}_world_{ordinal:03d}"
        new_uids = tuple(
            f"{world_uid}_seller_{slot:02d}" for slot in range(28)
        )
        noise_slots: list[int | None] = [None] * 28
        for source_slot, source_noise in enumerate(
            public["noise_slot_by_seller_slot"]
        ):
            target_slot = seller_permutation[source_slot]
            noise_slots[target_slot] = noise_permutation[int(source_noise)]
        if any(value is None for value in noise_slots):
            raise Formal500x4BuildError("Formal noise permutation is incomplete")
        transformed_groups = tuple(
            tuple(sorted(
                (
                    new_uids[seller_permutation[old_slot[str(seller_uid)]]]
                    for seller_uid in group
                ),
                key=lambda value: value.encode("utf-8"),
            ))
            for group in groups
        )
        world = engine.PublicWorld(
            split=split,
            ordinal=ordinal,
            world_uid=world_uid,
            seller_uids=new_uids,
            noise_slots=tuple(int(value) for value in noise_slots),
            controller_groups=transformed_groups,
        )
        output.append(world)
        commitment_rows.append({
            "world_uid": world_uid,
            "seller_uids": list(new_uids),
            "noise_slots": list(world.noise_slots),
            "controller_groups": [list(group) for group in transformed_groups],
        })
    commitment = {
        "world_order_sha256": canonical_sha256(world_order),
        "seller_slot_permutation_sha256": canonical_sha256(
            seller_permutation
        ),
        "noise_slot_permutation_sha256": canonical_sha256(noise_permutation),
        "transformed_schedule_sha256": canonical_sha256({
            "version": VERSION,
            "split": split,
            "basis": basis_name,
            "worlds": commitment_rows,
        }),
    }
    return tuple(output), commitment


def build_world_schedules(
    *, formal: bool, authorities: engine.Authorities, policy: Mapping[str, Any],
) -> tuple[
    dict[str, tuple[engine.PublicWorld, ...]], dict[str, dict[str, str]],
]:
    if not formal:
        schedules: dict[str, tuple[engine.PublicWorld, ...]] = {}
        for split in SPLITS:
            source = engine._smoke_world(split)
            world_uid = f"v9_4_formal_500x4_smoke_{split}_world_000"
            seller_uids = tuple(
                f"{world_uid}_seller_{slot:02d}" for slot in range(28)
            )
            source_slot = {
                seller_uid: slot
                for slot, seller_uid in enumerate(source.seller_uids)
            }
            groups = tuple(
                tuple(
                    seller_uids[source_slot[seller_uid]] for seller_uid in group
                )
                for group in source.controller_groups
            )
            schedules[split] = (engine.PublicWorld(
                split=split,
                ordinal=0,
                world_uid=world_uid,
                seller_uids=seller_uids,
                noise_slots=source.noise_slots,
                controller_groups=groups,
            ),)
        return schedules, {
            split: {
                name: canonical_sha256({
                    "split": split,
                    "world_uid": schedules[split][0].world_uid,
                    "commitment_type": name,
                })
                for name in (
                    "world_order_sha256",
                    "seller_slot_permutation_sha256",
                    "noise_slot_permutation_sha256",
                    "transformed_schedule_sha256",
                )
            }
            for split in SPLITS
        }
    schedules: dict[str, tuple[engine.PublicWorld, ...]] = {}
    commitments: dict[str, dict[str, str]] = {}
    for split in SPLITS:
        schedules[split], commitments[split] = _transform_schedule(
            split=split,
            schedule_key=authorities.audit_schedule,
            policy=policy,
        )
    for name in (
        "world_order_sha256", "seller_slot_permutation_sha256",
        "noise_slot_permutation_sha256", "transformed_schedule_sha256",
    ):
        if len({commitments[split][name] for split in SPLITS}) != len(SPLITS):
            raise Formal500x4BuildError(
                f"Formal split schedule transform commitments collide: {name}"
            )
    return schedules, commitments


def _count_histogram(values: Sequence[int]) -> dict[int, int]:
    return dict(sorted(Counter(int(value) for value in values).items()))


def audit_formal_schedule_balance(
    schedules: Mapping[str, Sequence[engine.PublicWorld]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    expected_pair_histogram = {26: 206, 27: 172}
    expected_triad_histogram = {214: 20, 215: 8}
    expected_assignment_histogram = {17: 4, 18: 24}
    for split in SPLITS:
        worlds = schedules[split]
        if len(worlds) != 500:
            raise Formal500x4BuildError(f"Formal schedule size drift: {split}")
        seller_pair = [[0] * 28 for _ in range(28)]
        noise_pair = [[0] * 28 for _ in range(28)]
        seller_triad = [0] * 28
        noise_triad = [0] * 28
        assignment = [[0] * 28 for _ in range(28)]
        for world in worlds:
            slot_by_uid = {
                seller_uid: slot
                for slot, seller_uid in enumerate(world.seller_uids)
            }
            if (
                len(slot_by_uid) != 28
                or len(world.noise_slots) != 28
                or set(world.noise_slots) != set(range(28))
            ):
                raise Formal500x4BuildError(
                    f"Formal schedule world cardinality drift: {world.world_uid}"
                )
            for seller_slot, noise_slot in enumerate(world.noise_slots):
                assignment[seller_slot][noise_slot] += 1
            for group in world.controller_groups:
                seller_slots = [slot_by_uid[seller_uid] for seller_uid in group]
                noise_slots = [world.noise_slots[slot] for slot in seller_slots]
                if len(group) == 3:
                    for slot in seller_slots:
                        seller_triad[slot] += 1
                    for slot in noise_slots:
                        noise_triad[slot] += 1
                for left_index in range(len(group)):
                    for right_index in range(left_index + 1, len(group)):
                        left, right = sorted((
                            seller_slots[left_index], seller_slots[right_index],
                        ))
                        seller_pair[left][right] += 1
                        left, right = sorted((
                            noise_slots[left_index], noise_slots[right_index],
                        ))
                        noise_pair[left][right] += 1
        seller_pair_histogram = _count_histogram([
            seller_pair[left][right]
            for left in range(28) for right in range(left + 1, 28)
        ])
        noise_pair_histogram = _count_histogram([
            noise_pair[left][right]
            for left in range(28) for right in range(left + 1, 28)
        ])
        assignment_row_histograms = [
            _count_histogram(row) for row in assignment
        ]
        assignment_column_histograms = [
            _count_histogram([assignment[row][column] for row in range(28)])
            for column in range(28)
        ]
        if (
            seller_pair_histogram != expected_pair_histogram
            or noise_pair_histogram != expected_pair_histogram
            or _count_histogram(seller_triad) != expected_triad_histogram
            or _count_histogram(noise_triad) != expected_triad_histogram
            or any(
                value != expected_assignment_histogram
                for value in assignment_row_histograms
            )
            or any(
                value != expected_assignment_histogram
                for value in assignment_column_histograms
            )
        ):
            raise Formal500x4BuildError(
                f"Formal balanced schedule invariant drift: {split}"
            )
        output[split] = {
            "world_count": 500,
            "seller_pair_histogram": seller_pair_histogram,
            "noise_pair_histogram": noise_pair_histogram,
            "seller_triad_histogram": _count_histogram(seller_triad),
            "noise_triad_histogram": _count_histogram(noise_triad),
            "noise_assignment_histogram_per_row_and_column": (
                expected_assignment_histogram
            ),
        }
    return output


def smoke_authorities() -> engine.Authorities:
    def key(name: str) -> bytes:
        return hashlib.sha256(f"{VERSION}::smoke::{name}".encode("ascii")).digest()
    return engine.Authorities(
        text=key("text"), identity=key("identity"), style=key("style"),
        audit_schedule=key("schedule"), uid=key("uid"), time=key("time"),
    )


def load_formal_authorities(
    policy: Mapping[str, Any],
) -> tuple[engine.Authorities, dict[str, str]]:
    auth_path = ROOT / str(policy["formal_authorization_path"])
    if not auth_path.is_file():
        raise Formal500x4BuildError("Formal 500x4 authorization is absent")
    auth = read_json(auth_path)
    required = {
        "version", "status", "implementation_commit", "implementation_tree",
        "policy_sha256", "quality_result_sha256", "root_manifest_sha256",
        "issuance_claim_sha256", "output_root", "private_root", "key_files",
        "canonical_self_hash",
    }
    if (
        set(auth) != required
        or auth["version"] != AUTHORIZATION_VERSION
        or auth["status"] != AUTHORIZATION_STATUS
    ):
        raise Formal500x4BuildError("Formal authorization schema or status drift")
    require_self_hash(auth, label="formal authorization")
    quality_spec = policy["method_qualification"]
    claim_path = ROOT / str(policy["formal_issuance_claim_path"])
    if (
        not claim_path.is_file()
        or sha256_file(claim_path) != auth["issuance_claim_sha256"]
    ):
        raise Formal500x4BuildError("Formal issuance claim bytes drift")
    claim = read_json(claim_path)
    claim_required = {
        "version", "status", "issuance_ordinal", "implementation_commit",
        "implementation_tree", "policy_path", "policy_sha256",
        "authorization_path", "authority_root", "output_root", "private_root",
        "key_names", "key_size_bytes", "candidate_draws_at_claim",
        "rerun_authorized", "canonical_self_hash",
    }
    if set(claim) != claim_required:
        raise Formal500x4BuildError("Formal issuance claim schema drift")
    require_self_hash(claim, label="formal issuance claim")
    if (
        auth["implementation_commit"] != git_head()
        or auth["implementation_tree"] != git_tree()
        or auth["policy_sha256"] != sha256_file(POLICY_PATH)
        or auth["quality_result_sha256"] != quality_spec["quality_result_sha256"]
        or auth["root_manifest_sha256"] != quality_spec["root_manifest_sha256"]
        or auth["output_root"] != policy["formal_output_root"]
        or auth["private_root"] != policy["formal_private_root"]
        or claim["status"] != "FORMAL_500X4_AUTHORITY_ISSUANCE_CLAIMED"
        or claim["version"] != ISSUANCE_CLAIM_VERSION
        or claim["issuance_ordinal"] != 1
        or claim["implementation_commit"] != auth["implementation_commit"]
        or claim["implementation_tree"] != auth["implementation_tree"]
        or claim["policy_path"] != POLICY_PATH.relative_to(ROOT).as_posix()
        or claim["policy_sha256"] != auth["policy_sha256"]
        or claim["authorization_path"] != policy["formal_authorization_path"]
        or claim["authority_root"] != policy["formal_authority_root"]
        or claim["output_root"] != auth["output_root"]
        or claim["private_root"] != auth["private_root"]
        or claim["key_names"] != list(policy["private_authority"]["key_names"])
        or claim["key_size_bytes"] != 32
        or claim["candidate_draws_at_claim"] != 0
        or claim["rerun_authorized"] is not False
    ):
        raise Formal500x4BuildError("Formal authorization binding drift")
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=ROOT, check=True, capture_output=True, text=True, encoding="utf-8",
    ).stdout.splitlines()
    expected_untracked = f"?? {auth_path.relative_to(ROOT).as_posix()}"
    if status != [expected_untracked]:
        raise Formal500x4BuildError(
            "Formal build requires the frozen commit plus only its authorization file"
        )
    for name in (
        "formal_consumption_path", "formal_failure_path",
        "formal_completion_path",
    ):
        if (ROOT / str(policy[name])).exists():
            raise Formal500x4BuildError(
                "Formal authorization already consumed or terminal"
            )
    expected_names = tuple(policy["private_authority"]["key_names"])
    if set(auth["key_files"]) != set(expected_names):
        raise Formal500x4BuildError("Formal authority key set drift")
    keys: dict[str, bytes] = {}
    commitments: dict[str, str] = {}
    for name in expected_names:
        spec = auth["key_files"][name]
        expected_path = (
            ROOT / str(policy["formal_authority_root"]) / f"{name}_key.bin"
        )
        if (
            not isinstance(spec, dict)
            or set(spec) != {"path", "commitment_sha256"}
            or ROOT / str(spec["path"]) != expected_path
            or not is_sha256(spec["commitment_sha256"])
        ):
            raise Formal500x4BuildError(
                f"Formal authority key-file schema drift: {name}"
            )
        path = ROOT / str(spec["path"])
        data = path.read_bytes() if path.is_file() else b""
        observed = hashlib.sha256(data).hexdigest()
        if len(data) != 32 or observed != spec["commitment_sha256"]:
            raise Formal500x4BuildError(f"Formal authority drift: {name}")
        keys[name] = data
        commitments[name] = observed
    if len(set(commitments.values())) != len(expected_names):
        raise Formal500x4BuildError("Formal authority commitments are not distinct")
    if set(commitments.values()).intersection(forbidden_authority_commitments(policy)):
        raise Formal500x4BuildError("Formal authority reuses a forbidden commitment")
    return engine.Authorities(
        text=keys["text"], identity=keys["identity"], style=keys["style"],
        audit_schedule=keys["schedule"], uid=keys["uid"], time=keys["time"],
    ), commitments


def consume_authorization(policy: Mapping[str, Any]) -> dict[str, Any]:
    auth_path = ROOT / str(policy["formal_authorization_path"])
    auth = read_json(auth_path)
    marker_path = ROOT / str(policy["formal_consumption_path"])
    payload: dict[str, Any] = {
        "version": "2026-08-29-step28-v13-v1-13-v9-4-formal-500x4-consumption-v1",
        "status": "FORMAL_500X4_BUILD_AUTHORITY_CONSUMED",
        "authorization_sha256": sha256_file(auth_path),
        "authorization_canonical_self_hash": auth["canonical_self_hash"],
        "implementation_commit": auth["implementation_commit"],
        "implementation_tree": auth["implementation_tree"],
        "output_root": auth["output_root"],
        "private_root": auth["private_root"],
        "rerun_authorized": False,
    }
    payload["canonical_self_hash"] = canonical_sha256(payload)
    write_json_exclusive(marker_path, payload)
    return {
        "path": marker_path.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(marker_path),
        "canonical_self_hash": payload["canonical_self_hash"],
    }


def write_post_consumption_failure(
    policy: Mapping[str, Any], *, consumption: Mapping[str, Any],
    stage: str, exc: BaseException, worlds_completed: int,
    path_presence_before_cleanup: Mapping[str, bool],
) -> None:
    path = ROOT / str(policy["formal_failure_path"])
    auth_path = ROOT / str(policy["formal_authorization_path"])
    payload: dict[str, Any] = {
        "version": (
            "2026-08-29-step28-v13-v1-13-v9-4-formal-500x4-"
            "build-failure-v1"
        ),
        "status": "FORMAL_500X4_BUILD_FAILED_NO_DATASET_CONCLUSION_NO_RERUN",
        "authorization_sha256": sha256_file(auth_path),
        "consumption": dict(consumption),
        "implementation_commit": git_head(),
        "implementation_tree": git_tree(),
        "policy_sha256": sha256_file(POLICY_PATH),
        "failure_stage": stage,
        "exception_type": type(exc).__name__,
        "exception_message_sha256": hashlib.sha256(
            str(exc).encode("utf-8")
        ).hexdigest(),
        "worlds_completed": worlds_completed,
        "path_presence_before_cleanup": dict(path_presence_before_cleanup),
        "claim_boundary": "MECHANICAL_FAILURE_NO_DATASET_CONCLUSION",
        "rerun_authorized": False,
        "training_authorized": False,
    }
    payload["canonical_self_hash"] = canonical_sha256(payload)
    write_json_exclusive(path, payload)


def publish_dual_roots(
    *, temporary: Path, root: Path, private_temporary: Path, private: Path,
) -> None:
    private_temporary.rename(private)
    try:
        temporary.rename(root)
    except BaseException:
        if private.exists() and not private_temporary.exists():
            private.rename(private_temporary)
        raise


def write_completion_receipt(
    policy: Mapping[str, Any], *, consumption: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    root = ROOT / str(policy["formal_output_root"])
    private = ROOT / str(policy["formal_private_root"])
    manifest_path = root / "root_manifest.json"
    observed_manifest = read_json(manifest_path) if manifest_path.is_file() else None
    if (
        not root.is_dir()
        or not private.is_dir()
        or observed_manifest is None
        or canonical_bytes(observed_manifest) != canonical_bytes(manifest)
    ):
        raise Formal500x4BuildError("Formal dual-root completion boundary drift")
    payload: dict[str, Any] = {
        "version": (
            "2026-08-29-step28-v13-v1-13-v9-4-formal-500x4-"
            "build-completion-v1"
        ),
        "status": "FORMAL_500X4_DUAL_ROOT_BUILD_COMPLETED",
        "output_root": policy["formal_output_root"],
        "private_root": policy["formal_private_root"],
        "root_manifest_sha256": sha256_file(manifest_path),
        "root_manifest_canonical_self_hash": manifest["canonical_self_hash"],
        "private_file_commitments_sha256": canonical_sha256(
            manifest["private_file_commitments"]
        ),
        "consumption": dict(consumption),
        "publication_status": manifest["status"],
        "training_qualified": False,
        "rerun_authorized": False,
    }
    payload["canonical_self_hash"] = canonical_sha256(payload)
    path = ROOT / str(policy["formal_completion_path"])
    write_json_exclusive(path, payload)
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(path),
        "canonical_self_hash": payload["canonical_self_hash"],
    }


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise Formal500x4BuildError(f"Blank JSONL line: {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise Formal500x4BuildError(f"JSONL object required: {path}:{line_number}")
            yield value


@dataclass
class CollisionState:
    blocked_item_documents: set[str]
    blocked_seller_documents: set[str]
    blocked_world_uids: set[str]
    blocked_seller_uids: set[str]
    blocked_item_uids: set[str]
    blocked_pair_uids: set[str]
    blocked_historical_uid_hashes: dict[str, set[str]]
    blocked_identity_values: set[str]
    formal_item_documents: set[str] = field(default_factory=set)
    formal_seller_documents: set[str] = field(default_factory=set)
    formal_identity_values: set[str] = field(default_factory=set)
    formal_world_uids: set[str] = field(default_factory=set)
    formal_seller_uids: set[str] = field(default_factory=set)
    formal_item_uids: set[str] = field(default_factory=set)
    formal_pair_uids: set[str] = field(default_factory=set)
    formal_asset_uids: set[str] = field(default_factory=set)
    split_item_documents: dict[str, set[str]] = field(
        default_factory=lambda: {split: set() for split in SPLITS}
    )
    split_seller_documents: dict[str, set[str]] = field(
        default_factory=lambda: {split: set() for split in SPLITS}
    )


def load_collision_state(policy: Mapping[str, Any]) -> CollisionState:
    historical = collision.load_historical_exclusion_registries(
        ROOT / str(policy["collision_contract"]["historical_policy_path"])
    )
    method_spec = policy["method_qualification"]
    public_root = ROOT / str(method_spec["public_root"])
    item_documents: set[str] = set(historical.item_document_hashes)
    seller_documents: set[str] = set(historical.seller_document_hashes)
    identity_values: set[str] = set(historical.identity_value_hashes)
    world_uids: set[str] = set()
    seller_uids: set[str] = set()
    item_uids: set[str] = set()
    pair_uids: set[str] = set()
    expected_historical_uid_types = {
        "canonical_pair_uid", "controller_uid", "item_uid", "query_uid",
        "seller_uid", "world_uid",
    }
    if set(historical.uid_hashes) != expected_historical_uid_types:
        raise Formal500x4BuildError("Historical UID exclusion types drift")
    method_manifest = read_json(
        ROOT / str(method_spec["root_manifest_path"])
    )
    for split in SPLITS:
        observed = public_root / split / "observed"
        for row in iter_jsonl(observed / "worlds.jsonl"):
            world_uids.add(str(row["world_uid"]))
        for row in iter_jsonl(observed / "sellers.jsonl"):
            seller_uids.add(str(row["seller_uid"]))
        for row in iter_jsonl(observed / "redacted_items.jsonl"):
            item_uids.add(str(row["item_uid"]))
            item_documents.add(collision.item_document_hash(
                title=str(row["title"]), description=str(row["description"]),
            ))
        for row in iter_jsonl(observed / "model_seller_profiles.jsonl"):
            seller_documents.add(collision.seller_document_hash(row))
        endpoint_path = observed / "complete_model_pair_endpoints.csv"
        endpoint_count = 0
        with endpoint_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            expected_fields = [
                "canonical_pair_uid", "world_uid", "seller_uid_left",
                "seller_uid_right",
            ]
            if reader.fieldnames != expected_fields:
                raise Formal500x4BuildError(
                    f"Method-root endpoint schema drift: {split}"
                )
            for row in reader:
                pair_uid = str(row["canonical_pair_uid"])
                if pair_uid != (
                    f"{row['seller_uid_left']}||{row['seller_uid_right']}"
                ):
                    raise Formal500x4BuildError(
                        f"Method-root canonical pair UID drift: {split}"
                    )
                if pair_uid in pair_uids:
                    raise Formal500x4BuildError(
                        "Method-root canonical pair UID collision"
                    )
                pair_uids.add(pair_uid)
                endpoint_count += 1
        expected_endpoint_count = (
            int(method_manifest["world_counts"][split]) * 378
        )
        if endpoint_count != expected_endpoint_count:
            raise Formal500x4BuildError(
                f"Method-root endpoint count drift: {split}"
            )
    return CollisionState(
        blocked_item_documents=item_documents,
        blocked_seller_documents=seller_documents,
        blocked_world_uids=world_uids,
        blocked_seller_uids=seller_uids,
        blocked_item_uids=item_uids,
        blocked_pair_uids=pair_uids,
        blocked_historical_uid_hashes={
            name: set(values) for name, values in historical.uid_hashes.items()
        },
        blocked_identity_values=identity_values,
    )


def _require_new_unique(
    values: Sequence[str], blocked: set[str], accepted: set[str], *, label: str,
) -> None:
    materialized = list(values)
    if len(materialized) != len(set(materialized)):
        raise Formal500x4BuildError(f"Within-world {label} collision")
    current = set(materialized)
    if current.intersection(blocked) or current.intersection(accepted):
        raise Formal500x4BuildError(f"Historical or formal {label} collision")
    accepted.update(current)


def _require_new_uid_unique(
    values: Sequence[str], blocked: set[str], accepted: set[str], *,
    historical_hashes: set[str], label: str,
) -> None:
    materialized = list(values)
    if len(materialized) != len(set(materialized)):
        raise Formal500x4BuildError(f"Within-world {label} collision")
    current = set(materialized)
    if current.intersection(blocked) or current.intersection(accepted):
        raise Formal500x4BuildError(f"Historical or formal {label} collision")
    observed_hashes = _exact_uid_hashes(materialized)
    if observed_hashes.intersection(historical_hashes):
        raise Formal500x4BuildError(f"Historical hashed {label} collision")
    accepted.update(current)


def _exact_uid_hashes(values: Sequence[str] | set[str]) -> set[str]:
    return {
        hashlib.sha256(value.encode("utf-8")).hexdigest()
        for value in values
    }


def _uid_exclusion_intersection_count(
    values: set[str], *, blocked_raw: set[str], blocked_hashes: set[str],
) -> int:
    return len(values.intersection(blocked_raw)) + len(
        _exact_uid_hashes(values).intersection(blocked_hashes)
    )


def register_world(
    *,
    split: str,
    world: Mapping[str, Any],
    sellers: Sequence[Mapping[str, Any]],
    redacted_items: Sequence[Mapping[str, Any]],
    model_profiles: Sequence[Mapping[str, Any]],
    endpoints: Sequence[Mapping[str, Any]],
    identity_projection: Sequence[Mapping[str, Any]],
    state: CollisionState,
    sealed_method_identities: identity_exclusion.SealedMethodIdentityExclusion,
) -> None:
    world_uid = str(world["world_uid"])
    seller_uids = [str(row["seller_uid"]) for row in sellers]
    item_uids = [str(row["item_uid"]) for row in redacted_items]
    expected_endpoint_fields = {
        "canonical_pair_uid", "world_uid", "seller_uid_left",
        "seller_uid_right",
    }
    if any(set(row) != expected_endpoint_fields for row in endpoints):
        raise Formal500x4BuildError("Pair endpoint collision schema drift")
    pair_uids: list[str] = []
    seller_uid_set = set(seller_uids)
    for row in endpoints:
        left = str(row["seller_uid_left"])
        right = str(row["seller_uid_right"])
        pair_uid = str(row["canonical_pair_uid"])
        if (
            str(row["world_uid"]) != world_uid
            or left not in seller_uid_set
            or right not in seller_uid_set
            or left >= right
            or pair_uid != f"{left}||{right}"
        ):
            raise Formal500x4BuildError("Canonical pair UID collision input drift")
        pair_uids.append(pair_uid)
    if len(pair_uids) != 378:
        raise Formal500x4BuildError("Pair UID cardinality drift")
    if any(
        set(row) != {"asset_uid", "value_sha256"}
        for row in identity_projection
    ):
        raise Formal500x4BuildError("Identity collision projection schema drift")
    asset_uids = [str(row["asset_uid"]) for row in identity_projection]
    identity_values = [str(row["value_sha256"]) for row in identity_projection]
    item_documents = [collision.item_document_hash(
        title=str(row["title"]), description=str(row["description"]),
    ) for row in redacted_items]
    seller_documents = [
        collision.seller_document_hash(row) for row in model_profiles
    ]
    sealed_method_identities.require_disjoint(
        value_hashes=identity_values,
        asset_uids=asset_uids,
    )
    _require_new_uid_unique(
        [world_uid], state.blocked_world_uids, state.formal_world_uids,
        historical_hashes=state.blocked_historical_uid_hashes["world_uid"],
        label="world UID",
    )
    _require_new_uid_unique(
        seller_uids, state.blocked_seller_uids, state.formal_seller_uids,
        historical_hashes=state.blocked_historical_uid_hashes["seller_uid"],
        label="seller UID",
    )
    _require_new_uid_unique(
        item_uids, state.blocked_item_uids, state.formal_item_uids,
        historical_hashes=state.blocked_historical_uid_hashes["item_uid"],
        label="item UID",
    )
    _require_new_uid_unique(
        pair_uids, state.blocked_pair_uids, state.formal_pair_uids,
        historical_hashes=(
            state.blocked_historical_uid_hashes["canonical_pair_uid"]
        ),
        label="canonical pair UID",
    )
    _require_new_unique(
        asset_uids, set(), state.formal_asset_uids,
        label="identity asset UID",
    )
    _require_new_unique(
        identity_values, state.blocked_identity_values,
        state.formal_identity_values, label="identity value",
    )
    _require_new_unique(
        item_documents, state.blocked_item_documents,
        state.formal_item_documents, label="redacted item document",
    )
    _require_new_unique(
        seller_documents, state.blocked_seller_documents,
        state.formal_seller_documents, label="five-field seller document",
    )
    state.split_item_documents[split].update(item_documents)
    state.split_seller_documents[split].update(seller_documents)


def hash_set_digest(values: set[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(values)) + "\n").encode("ascii")).hexdigest()


def write_public_collision_hashes(
    path: Path, state: CollisionState,
) -> dict[str, Any]:
    writer = engine.JsonlWriter(path)
    rows = 0
    try:
        for document_type, values in (
            ("item_document", state.formal_item_documents),
            ("seller_document", state.formal_seller_documents),
        ):
            for value in sorted(values):
                writer.write({
                    "document_type": document_type,
                    "sha256": value,
                })
                rows += 1
    finally:
        writer.close()
    return {
        "path": path.name,
        "row_count": rows,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def public_collision_registry(
    state: CollisionState, *, hash_list: Mapping[str, Any],
) -> dict[str, Any]:
    intersection_counts = {
        "item_documents": len(
            state.formal_item_documents.intersection(
                state.blocked_item_documents
            )
        ),
        "seller_documents": len(
            state.formal_seller_documents.intersection(
                state.blocked_seller_documents
            )
        ),
        "world_uids": _uid_exclusion_intersection_count(
            state.formal_world_uids,
            blocked_raw=state.blocked_world_uids,
            blocked_hashes=state.blocked_historical_uid_hashes["world_uid"],
        ),
        "seller_uids": _uid_exclusion_intersection_count(
            state.formal_seller_uids,
            blocked_raw=state.blocked_seller_uids,
            blocked_hashes=state.blocked_historical_uid_hashes["seller_uid"],
        ),
        "item_uids": _uid_exclusion_intersection_count(
            state.formal_item_uids,
            blocked_raw=state.blocked_item_uids,
            blocked_hashes=state.blocked_historical_uid_hashes["item_uid"],
        ),
        "canonical_pair_uids": _uid_exclusion_intersection_count(
            state.formal_pair_uids,
            blocked_raw=state.blocked_pair_uids,
            blocked_hashes=(
                state.blocked_historical_uid_hashes["canonical_pair_uid"]
            ),
        ),
    }
    if any(intersection_counts.values()):
        raise Formal500x4BuildError(
            "Formal public collision registry contains an exclusion hit"
        )
    value: dict[str, Any] = {
        "version": "2026-08-29-step28-v13-v1-13-v9-4-formal-public-collision-registry-v1",
        "hash_list": dict(hash_list),
        "counts": {
            "item_documents": len(state.formal_item_documents),
            "seller_documents": len(state.formal_seller_documents),
            "world_uids": len(state.formal_world_uids),
            "seller_uids": len(state.formal_seller_uids),
            "item_uids": len(state.formal_item_uids),
            "canonical_pair_uids": len(state.formal_pair_uids),
        },
        "digests": {
            "item_documents": hash_set_digest(state.formal_item_documents),
            "seller_documents": hash_set_digest(state.formal_seller_documents),
            "world_uids": hash_set_digest(state.formal_world_uids),
            "seller_uids": hash_set_digest(state.formal_seller_uids),
            "item_uids": hash_set_digest(state.formal_item_uids),
            "canonical_pair_uids": hash_set_digest(state.formal_pair_uids),
        },
        "split_counts": {
            split: {
                "item_documents": len(state.split_item_documents[split]),
                "seller_documents": len(state.split_seller_documents[split]),
            }
            for split in SPLITS
        },
        "historical_and_method_root_intersection_counts": intersection_counts,
        "historical_uid_hash_registry_counts_loaded": {
            name: len(values)
            for name, values in sorted(
                state.blocked_historical_uid_hashes.items()
            )
        },
        "formal_uid_values_checked": {
            "world_uid": len(state.formal_world_uids),
            "seller_uid": len(state.formal_seller_uids),
            "item_uid": len(state.formal_item_uids),
            "canonical_pair_uid": len(state.formal_pair_uids),
        },
    }
    value["canonical_self_hash"] = canonical_sha256(value)
    return value


def private_identity_collision_registry(state: CollisionState) -> dict[str, Any]:
    identity_value_intersections = len(
        state.formal_identity_values.intersection(
            state.blocked_identity_values
        )
    )
    if identity_value_intersections:
        raise Formal500x4BuildError(
            "Formal private collision registry contains an identity hit"
        )
    value: dict[str, Any] = {
        "version": "2026-08-29-step28-v13-v1-13-v9-4-formal-private-identity-collision-registry-v1",
        "identity_value_hashes": sorted(state.formal_identity_values),
        "identity_asset_uids": sorted(state.formal_asset_uids),
        "counts": {
            "identity_values": len(state.formal_identity_values),
            "identity_asset_uids": len(state.formal_asset_uids),
        },
        "digests": {
            "identity_values": hash_set_digest(state.formal_identity_values),
            "identity_asset_uids": hashlib.sha256(
                ("\n".join(sorted(state.formal_asset_uids)) + "\n").encode("utf-8")
            ).hexdigest(),
        },
        "historical_and_method_root_intersection_counts": {
            "identity_values": identity_value_intersections,
            "identity_asset_uids": 0,
        },
    }
    value["canonical_self_hash"] = canonical_sha256(value)
    return value


def build_dataset(*, formal: bool, output_root: Path | None = None) -> dict[str, Any]:
    policy = validate_policy(formal=formal)
    base_policy = read_json(ROOT / str(policy["reuse_sources"]["base_dataset_policy"]["path"]))
    template = read_json(ROOT / str(policy["reuse_sources"]["template_library"]["path"]))
    if formal:
        authorities, authority_commitments = load_formal_authorities(policy)
        root = ROOT / str(policy["formal_output_root"])
        private = ROOT / str(policy["formal_private_root"])
    else:
        authorities = smoke_authorities()
        authority_commitments = {
            name: hashlib.sha256(getattr(
                authorities, "audit_schedule" if name == "schedule" else name,
            )).hexdigest()
            for name in policy["private_authority"]["key_names"]
        }
        root = output_root or (
            ROOT / "reports" / "step28_synthetic_chinese_dataset"
            / "_v9_4_formal_500x4_smoke"
        )
        private = root.parent / f".{root.name}.private"
    temporary = root.parent / f".{root.name}.building"
    private_temporary = private.parent / f".{private.name}.building"
    if any(path.exists() for path in (root, private, temporary, private_temporary)):
        raise Formal500x4BuildError("Formal output or temporary path already exists")
    collision_state = load_collision_state(policy)
    method_manifest = read_json(
        ROOT / str(policy["method_qualification"]["root_manifest_path"])
    )
    sealed_method_identities = identity_exclusion.open_validator(
        method_private_root=(
            ROOT / str(policy["method_qualification"]["private_root"])
        ),
        root_manifest=method_manifest,
    )
    schedules, schedule_commitments = build_world_schedules(
        formal=formal, authorities=authorities, policy=policy,
    )
    if {
        split: len(schedules[split]) for split in SPLITS
    } != (FORMAL_WORLD_COUNTS if formal else SMOKE_WORLD_COUNTS):
        raise Formal500x4BuildError("Constructed world-count closure drift")
    schedule_balance_audit = (
        audit_formal_schedule_balance(schedules) if formal else None
    )
    total = sum(len(schedules[split]) for split in SPLITS)
    consumption = consume_authorization(policy) if formal else None
    writers: dict[str, engine.SplitWriters] = {}
    completed = 0
    item_count = 0
    stage = "creating_temporary_roots"
    try:
        temporary.mkdir(parents=True)
        private_temporary.mkdir(parents=True)
        stage = "preparing_generation_dependencies"
        signatures = [
            dict(row) for row in engine.noise_v94.build_noise_signatures().rows
        ]
        identity_fields = list(base_policy["history_features"]["feature_names"])
        stage = "opening_split_writers"
        for split in SPLITS:
            writers[split] = engine.open_writers(
                temporary, private_temporary, split, identity_fields,
            )
        for split in SPLITS:
            for world in schedules[split]:
                stage = f"generating_{split}_world_{world.ordinal:03d}"
                value = engine.build_one_world(
                    world=world,
                    auth=authorities,
                    base_policy=base_policy,
                    template=template,
                    signatures=signatures,
                )
                register_world(
                    split=split,
                    world=value["world"],
                    sellers=value["sellers"],
                    redacted_items=value["redacted_items"],
                    model_profiles=value["model_profiles"],
                    endpoints=value["endpoints"],
                    identity_projection=tuple({
                        "asset_uid": row["asset_uid"],
                        "value_sha256": row["value_sha256"],
                    } for row in value["identity_plan"]),
                    state=collision_state,
                    sealed_method_identities=sealed_method_identities,
                )
                engine.write_world(writers[split], value)
                item_count += len(value["items"])
                completed += 1
                if completed % 25 == 0 or completed == total:
                    print(json.dumps({
                        "event": "progress",
                        "worlds_completed": completed,
                        "worlds_total": total,
                        "split": split,
                    }, ensure_ascii=False), flush=True)
        stage = "closing_split_writers"
        for writer in writers.values():
            writer.close()
        writers.clear()
        stage = "writing_collision_registries"
        collision_hash_list = write_public_collision_hashes(
            temporary / "document_collision_hashes.jsonl",
            collision_state,
        )
        registry = public_collision_registry(
            collision_state, hash_list=collision_hash_list,
        )
        private_registry = private_identity_collision_registry(collision_state)
        engine.write_json(temporary / "document_collision_registry.json", registry)
        engine.write_json(
            private_temporary / "identity_collision_registry.json",
            private_registry,
        )
        stage = "building_dual_root_manifests"
        public_files = engine.file_manifest(temporary)
        private_files = engine.file_manifest(private_temporary)
        manifest: dict[str, Any] = {
            "version": VERSION,
            "status": policy["publication_status"],
            "formal": formal,
            "world_counts": {
                split: len(schedules[split]) for split in SPLITS
            },
            "world_count": total,
            "seller_count": total * 28,
            "item_count": item_count,
            "pair_count": total * 378,
            "positive_pair_count": total * 20,
            "negative_pair_count": total * 358,
            "policy_sha256": sha256_file(POLICY_PATH),
            "method_quality_result_sha256": policy["method_qualification"][
                "quality_result_sha256"
            ],
            "method_root_manifest_sha256": policy["method_qualification"][
                "root_manifest_sha256"
            ],
            "formal_authorization_sha256": (
                sha256_file(ROOT / str(policy["formal_authorization_path"]))
                if formal else None
            ),
            "formal_authorization_consumption": consumption,
            "formal_completion_receipt": ({
                "path": policy["formal_completion_path"],
                "required": True,
            } if formal else None),
            "authority_commitments": authority_commitments,
            "schedule_commitments": schedule_commitments,
            "schedule_balance_audit": schedule_balance_audit,
            "collision_registry_sha256": sha256_file(
                temporary / "document_collision_registry.json"
            ),
            "collision_registry_canonical_self_hash": registry[
                "canonical_self_hash"
            ],
            "private_identity_collision_registry_sha256": sha256_file(
                private_temporary / "identity_collision_registry.json"
            ),
            "private_identity_collision_registry_canonical_self_hash": (
                private_registry["canonical_self_hash"]
            ),
            "sealed_method_identity_exclusion_audit": (
                sealed_method_identities.public_audit()
            ),
            "public_files": public_files,
            "private_file_commitments": private_files,
            "audit_truth_read_counts": {"audit_a": 0, "audit_b": 0},
            "training_qualified": False,
            "m0_m1_m2_m3_training_authorized": False,
        }
        manifest["canonical_self_hash"] = canonical_sha256(manifest)
        engine.write_json(temporary / "root_manifest.json", manifest)
        root.parent.mkdir(parents=True, exist_ok=True)
        private.parent.mkdir(parents=True, exist_ok=True)
        stage = "publishing_private_root"
        publish_dual_roots(
            temporary=temporary,
            root=root,
            private_temporary=private_temporary,
            private=private,
        )
        if formal:
            stage = "writing_dual_root_completion_receipt"
            if consumption is None:
                raise Formal500x4BuildError("Formal consumption receipt absent")
            write_completion_receipt(
                policy, consumption=consumption, manifest=manifest,
            )
        return manifest
    except BaseException as exc:
        for writer in writers.values():
            try:
                writer.close()
            except Exception:
                pass
        receipt_error: BaseException | None = None
        if formal and consumption is not None:
            completion_path = ROOT / str(policy["formal_completion_path"])
            if completion_path.exists():
                completion_path.unlink()
            try:
                write_post_consumption_failure(
                    policy,
                    consumption=consumption,
                    stage=stage,
                    exc=exc,
                    worlds_completed=completed,
                    path_presence_before_cleanup={
                        "public_temporary": temporary.exists(),
                        "private_temporary": private_temporary.exists(),
                        "public_final": root.exists(),
                        "private_final": private.exists(),
                    },
                )
            except BaseException as failure_exc:
                receipt_error = failure_exc
        for path in (temporary, private_temporary, root, private):
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
        if receipt_error is not None:
            raise Formal500x4BuildError(
                "Post-consumption failure receipt could not be persisted"
            ) from receipt_error
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--formal", action="store_true")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    if args.formal and args.output_root is not None:
        raise SystemExit("--output-root is smoke-only")
    result = build_dataset(formal=args.formal, output_root=args.output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
