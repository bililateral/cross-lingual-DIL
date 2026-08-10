#!/usr/bin/env python3
"""Design-smoke-only exact document-collision candidate selection for v1.13."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import step28_v13_common as common
import step28_v13_v1_13_candidate_parent as candidate_parent
import step28_v13_v1_13_document_collision as collision
import step28_v13_v1_13_natural_variation as natural


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_PATH = (
    ROOT / "schema" / "step28_v13_v1_13_candidate_selection_policy.json"
)
POLICY_VERSION = "2026-08-10-step28-v13-v1-13-candidate-selection-policy-v1"
POLICY_STATUS = "DESIGN_SMOKE_ONLY_CANDIDATE_SELECTION_NO_FORMAL_AUTHORIZATION"
CLAIM_BOUNDARY = (
    "One pinned development-smoke world may search exact document-collision "
    "candidates in memory. No formal seed, capability, candidate, row, "
    "transaction, split state, dataset, model, metric, or release is authorized."
)
ALLOWED_MODE = "development_smoke"
ALLOWED_SPLIT = "audit_a"
CANDIDATE_LIMIT = 32
HASH_ROW_FIELDS = ("row_ordinal", "item_uid", "document_sha256")
SELLER_HASH_ROW_FIELDS = ("row_ordinal", "seller_uid", "document_sha256")
COLLISION_CATEGORIES = (
    "same_world_item_document",
    "same_world_seller_document",
    "historical_item_document",
    "historical_seller_document",
    "current_split_item_document",
    "current_split_seller_document",
    "predecessor_item_document",
    "predecessor_seller_document",
)
FORMAL_AUTHORIZATION_KEYS = frozenset(
    {
        "audit_truth_access",
        "formal_capability_derivation",
        "formal_candidate_generation",
        "formal_dataset_generation",
        "formal_model_training",
        "formal_seed_ceremony",
    }
)
FROZEN_INPUT_PATHS = {
    "collision_contract": (
        "docs/STEP28_V13_V1_13_VISIBLE_DOCUMENT_COLLISION_CONTRACT_20260809.zh.md"
    ),
    "collision_policy": "schema/step28_v13_v1_13_document_collision_policy.json",
    "candidate_parent_policy": "schema/step28_v13_v1_13_candidate_parent_policy.json",
    "natural_variation_policy": "schema/step28_v13_v1_13_natural_variation_policy.json",
    "base_dataset_policy": "schema/step28_v13_synthetic_chinese_dataset_policy.json",
    "common_contract": "scripts/step28_v13_common.py",
    "collision_primitives": "scripts/step28_v13_v1_13_document_collision.py",
    "candidate_parent": "scripts/step28_v13_v1_13_candidate_parent.py",
    "natural_variation": "scripts/step28_v13_v1_13_natural_variation.py",
    "production_chain": "scripts/step28_v13_production_chain.py",
    "profiles": "scripts/step28_v13_profiles.py",
    "step3_profiles": "scripts/step3_build_seller_profiles.py",
    "candidate_selection": "scripts/step28_v13_v1_13_candidate_selection.py",
    "candidate_selection_tests": (
        "tests/test_step28_v13_v1_13_candidate_selection_contracts.py"
    ),
}
FROZEN_INPUT_KEYS = frozenset(FROZEN_INPUT_PATHS)
HEX = frozenset("0123456789abcdef")


class CandidateSelectionError(ValueError):
    """Raised when the candidate selector must fail closed."""


def _canonical_bytes(value: Any) -> bytes:
    return common.canonical_json_bytes(value)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in HEX for character in value)
    ):
        raise CandidateSelectionError(f"{label} is not a lowercase SHA-256")
    return value


def _require_plain_int(value: Any, *, label: str) -> int:
    if type(value) is not int:
        raise CandidateSelectionError(f"{label} must be a plain integer")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: Sequence[str] | set[str], *, label: str
) -> None:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise CandidateSelectionError(f"{label} keyset drift")


def _decode_canonical(payload: bytes, *, label: str) -> Any:
    if not isinstance(payload, bytes):
        raise CandidateSelectionError(f"{label} must be immutable bytes")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CandidateSelectionError(f"{label} is not UTF-8 JSON") from exc
    if _canonical_bytes(value) != payload:
        raise CandidateSelectionError(f"{label} is not canonical JSON")
    return value


def _bytes_commitment(payload: bytes, *, label: str) -> dict[str, Any]:
    if not isinstance(payload, bytes):
        raise CandidateSelectionError(f"{label} must be immutable bytes")
    return {"size_bytes": len(payload), "sha256": _sha256_bytes(payload)}


def _sorted_hash_tuple(values: Sequence[str], *, label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, Mapping)):
        raise CandidateSelectionError(f"{label} must be a hash sequence")
    materialized = tuple(values)
    if any(_require_sha256(value, label=label) != value for value in materialized):
        raise AssertionError("unreachable")
    if materialized != tuple(sorted(materialized)) or len(materialized) != len(
        set(materialized)
    ):
        raise CandidateSelectionError(f"{label} must be strictly sorted and unique")
    return materialized


def _hash_tuple_digest(values: tuple[str, ...]) -> str:
    return common.canonical_sha256(list(values))


def _verify_self_hash(document: Mapping[str, Any], *, label: str) -> None:
    expected = _require_sha256(document.get("canonical_self_hash"), label=label)
    unsigned = dict(document)
    unsigned.pop("canonical_self_hash")
    if common.canonical_sha256(unsigned) != expected:
        raise CandidateSelectionError(f"{label} canonical self-hash drift")


def _verify_pin(spec: Mapping[str, Any], *, label: str) -> Path:
    _require_exact_keys(spec, {"path", "sha256", "size_bytes"}, label=label)
    path = common.repo_path(str(spec["path"]))
    size = _require_plain_int(spec["size_bytes"], label=f"{label}.size_bytes")
    if (
        size < 0
        or not path.is_file()
        or path.stat().st_size != size
        or common.sha256_file(path) != _require_sha256(spec["sha256"], label=label)
    ):
        raise CandidateSelectionError(f"{label} exact file pin drift")
    return path


def _validate_policy(policy: Mapping[str, Any]) -> None:
    required = {
        "version",
        "status",
        "claim_boundary",
        "formal_authorizations",
        "allowed_mode",
        "allowed_split",
        "candidate_limit",
        "collision_categories",
        "frozen_inputs",
        "smoke_collision_context",
        "expected_smoke_selection",
        "canonical_self_hash",
    }
    _require_exact_keys(policy, required, label="candidate-selection policy")
    _verify_self_hash(policy, label="candidate-selection policy")
    if (
        policy["version"] != POLICY_VERSION
        or policy["status"] != POLICY_STATUS
        or policy["claim_boundary"] != CLAIM_BOUNDARY
        or policy["allowed_mode"] != ALLOWED_MODE
        or policy["allowed_split"] != ALLOWED_SPLIT
        or policy["candidate_limit"] != CANDIDATE_LIMIT
        or tuple(policy["collision_categories"]) != COLLISION_CATEGORIES
    ):
        raise CandidateSelectionError("Candidate-selection policy identity drift")
    authorizations = policy["formal_authorizations"]
    if (
        not isinstance(authorizations, dict)
        or set(authorizations) != FORMAL_AUTHORIZATION_KEYS
        or any(value is not False for value in authorizations.values())
    ):
        raise CandidateSelectionError("Formal authorization must remain false")
    frozen = policy["frozen_inputs"]
    if not isinstance(frozen, dict) or set(frozen) != FROZEN_INPUT_KEYS:
        raise CandidateSelectionError("Candidate-selection source closure drift")
    for key, expected_path in FROZEN_INPUT_PATHS.items():
        spec = frozen[key]
        if not isinstance(spec, Mapping) or spec.get("path") != expected_path:
            raise CandidateSelectionError(
                f"Candidate-selection frozen input {key} canonical path drift"
            )
        _verify_pin(spec, label=f"candidate-selection frozen input {key}")
    context = policy["smoke_collision_context"]
    _require_exact_keys(
        context,
        {
            "world_ordinal",
            "historical_item_count",
            "historical_item_hashes_sha256",
            "historical_seller_count",
            "historical_seller_hashes_sha256",
            "current_item_count",
            "current_item_hashes_sha256",
            "current_seller_count",
            "current_seller_hashes_sha256",
            "predecessor_item_count",
            "predecessor_item_hashes_sha256",
            "predecessor_seller_count",
            "predecessor_seller_hashes_sha256",
            "previous_world_marker_sha256",
            "predecessor_seal_pin_count",
            "predecessor_seal_pins_sha256",
        },
        label="smoke collision context",
    )
    for name in (
        "world_ordinal",
        "historical_item_count",
        "historical_seller_count",
        "current_item_count",
        "current_seller_count",
        "predecessor_item_count",
        "predecessor_seller_count",
        "predecessor_seal_pin_count",
    ):
        if _require_plain_int(context[name], label=name) < 0:
            raise CandidateSelectionError(f"{name} cannot be negative")
    for name in context:
        if name.endswith("sha256"):
            _require_sha256(context[name], label=name)
    expected = policy["expected_smoke_selection"]
    expected_fields = {
        "world_uid",
        "accepted_candidate_index",
        "candidates_examined",
        "rejected_candidate_count",
        "item_count",
        "seller_count",
        "selection_context_sha256",
        "exact_title_clone_qualification_sha256",
        "redacted_items_sha256",
        "profiles_sha256",
        "item_registry_delta_sha256",
        "seller_registry_delta_sha256",
        "allocation_delta_count",
        "allocation_delta_sha256",
        "accepted_state_sha256",
    }
    _require_exact_keys(expected, expected_fields, label="expected smoke selection")
    for name in (
        "accepted_candidate_index",
        "candidates_examined",
        "rejected_candidate_count",
        "item_count",
        "seller_count",
        "allocation_delta_count",
    ):
        if _require_plain_int(expected[name], label=name) < 0:
            raise CandidateSelectionError(f"Expected {name} cannot be negative")
    for name in expected:
        if name.endswith("sha256"):
            _require_sha256(expected[name], label=name)


def load_policy(path: Path = DEFAULT_POLICY_PATH) -> dict[str, Any]:
    if path.resolve() != DEFAULT_POLICY_PATH.resolve():
        raise CandidateSelectionError("Only the canonical selection policy may pass")
    try:
        policy = common.load_json(path)
    except common.ContractError as exc:
        raise CandidateSelectionError("Candidate-selection policy is invalid") from exc
    if not isinstance(policy, dict):
        raise CandidateSelectionError("Candidate-selection policy must be an object")
    _validate_policy(policy)
    return policy


@dataclass(frozen=True)
class PredecessorSealPin:
    split: str
    split_seal_complete_sha256: str
    document_registry_sha256: str
    item_count: int
    item_hashes_sha256: str
    seller_count: int
    seller_hashes_sha256: str


@dataclass(frozen=True)
class FrozenCollisionContext:
    mode: str
    split: str
    world_ordinal: int
    historical_item_hashes: tuple[str, ...]
    historical_seller_hashes: tuple[str, ...]
    current_item_hashes: tuple[str, ...]
    current_seller_hashes: tuple[str, ...]
    predecessor_item_hashes: tuple[str, ...]
    predecessor_seller_hashes: tuple[str, ...]
    previous_world_marker_sha256: str
    predecessor_seal_pins: tuple[PredecessorSealPin, ...]


@dataclass(frozen=True)
class CollisionClassification:
    categories: tuple[str, ...]
    hit_counts: tuple[tuple[str, int], ...]

    @property
    def has_collision(self) -> bool:
        return bool(self.categories)


def _validate_collision_classification(
    value: CollisionClassification,
) -> CollisionClassification:
    if not isinstance(value, CollisionClassification):
        raise CandidateSelectionError("Collision classifier result type drift")
    if (
        value.categories
        != tuple(name for name in COLLISION_CATEGORIES if name in value.categories)
        or len(value.categories) != len(set(value.categories))
    ):
        raise CandidateSelectionError("Collision category order or membership drift")
    if (
        not isinstance(value.hit_counts, tuple)
        or tuple(name for name, _count in value.hit_counts) != COLLISION_CATEGORIES
        or any(type(count) is not int or count < 0 for _name, count in value.hit_counts)
    ):
        raise CandidateSelectionError("Collision hit-count schema drift")
    positive = tuple(name for name, count in value.hit_counts if count > 0)
    if positive != value.categories:
        raise CandidateSelectionError("Collision categories disagree with hit counts")
    return value


@dataclass(frozen=True)
class _FinalDocumentObservation:
    redacted_items_bytes: bytes
    redacted_items_sha256: str
    profiles_bytes: bytes
    profiles_sha256: str
    item_hash_rows_bytes: bytes
    item_hash_rows_sha256: str
    seller_hash_rows_bytes: bytes
    seller_hash_rows_sha256: str


@dataclass(frozen=True)
class AcceptedDevelopmentCandidate:
    version: str
    mode: str
    split: str
    world_uid: str
    design_smoke_only: bool
    committable: bool
    candidate_index: int
    candidates_examined: int
    rejected_candidate_count: int
    rejection_counts_bytes: bytes
    rejection_counts_sha256: str
    candidate_parent_full_state_sha256: str
    frozen_trial_identity_full_state_sha256: str
    candidate_invariant_sha256: str
    identity_parent_sha256: str
    identity33_bytes: bytes
    identity33_sha256: str
    profile_provenance_bytes: bytes
    profile_provenance_sha256: str
    natural_output_sha256: str
    world_bytes: bytes
    world_sha256: str
    redacted_items_bytes: bytes
    redacted_items_sha256: str
    profiles_bytes: bytes
    profiles_sha256: str
    item_hash_rows_bytes: bytes
    item_hash_rows_sha256: str
    seller_hash_rows_bytes: bytes
    seller_hash_rows_sha256: str
    item_registry_delta: tuple[str, ...]
    item_registry_delta_count: int
    item_registry_delta_sha256: str
    seller_registry_delta: tuple[str, ...]
    seller_registry_delta_count: int
    seller_registry_delta_sha256: str
    allocation_delta: tuple[str, ...]
    allocation_delta_count: int
    allocation_delta_sha256: str
    selection_context_bytes: bytes
    selection_context_sha256: str
    exact_title_clone_qualification_bytes: bytes
    exact_title_clone_qualification_sha256: str
    accepted_state_sha256: str


def _predecessor_pin_value(pin: PredecessorSealPin) -> dict[str, Any]:
    if not isinstance(pin, PredecessorSealPin):
        raise CandidateSelectionError("Predecessor seal pin type drift")
    if pin.split not in collision.SPLITS:
        raise CandidateSelectionError("Predecessor seal split drift")
    for name in (
        "split_seal_complete_sha256",
        "document_registry_sha256",
        "item_hashes_sha256",
        "seller_hashes_sha256",
    ):
        _require_sha256(getattr(pin, name), label=f"predecessor.{name}")
    if (
        _require_plain_int(pin.item_count, label="predecessor.item_count") < 0
        or _require_plain_int(pin.seller_count, label="predecessor.seller_count") < 0
    ):
        raise CandidateSelectionError("Predecessor seal count drift")
    return {
        "split": pin.split,
        "split_seal_complete_sha256": pin.split_seal_complete_sha256,
        "document_registry_sha256": pin.document_registry_sha256,
        "item_count": pin.item_count,
        "item_hashes_sha256": pin.item_hashes_sha256,
        "seller_count": pin.seller_count,
        "seller_hashes_sha256": pin.seller_hashes_sha256,
    }


def _selection_context_value(context: FrozenCollisionContext) -> dict[str, Any]:
    if not isinstance(context, FrozenCollisionContext):
        raise CandidateSelectionError("Collision context type drift")
    if (
        context.mode != ALLOWED_MODE
        or context.split != ALLOWED_SPLIT
        or _require_plain_int(context.world_ordinal, label="world_ordinal") != 0
    ):
        raise CandidateSelectionError("Collision context identity drift")
    groups = {}
    for name in (
        "historical_item_hashes",
        "historical_seller_hashes",
        "current_item_hashes",
        "current_seller_hashes",
        "predecessor_item_hashes",
        "predecessor_seller_hashes",
    ):
        values = _sorted_hash_tuple(getattr(context, name), label=name)
        groups[name.removesuffix("_hashes")] = {
            "count": len(values),
            "hashes_sha256": _hash_tuple_digest(values),
        }
    previous = _require_sha256(
        context.previous_world_marker_sha256,
        label="previous_world_marker_sha256",
    )
    pins = [_predecessor_pin_value(pin) for pin in context.predecessor_seal_pins]
    if [row["split"] for row in pins] != [
        split for split in collision.SPLITS if split != ALLOWED_SPLIT
    ][: len(pins)]:
        if pins:
            raise CandidateSelectionError("Predecessor seal order drift")
    return {
        "version": "2026-08-10-step28-v13-v1-13-collision-context-v1",
        "mode": context.mode,
        "split": context.split,
        "world_ordinal": context.world_ordinal,
        **groups,
        "previous_world_marker_sha256": previous,
        "predecessor_seal_pins": pins,
        "predecessor_seal_pin_count": len(pins),
        "predecessor_seal_pins_sha256": common.canonical_sha256(pins),
    }


def _build_smoke_collision_context() -> FrozenCollisionContext:
    policy = load_policy()
    expected = policy["smoke_collision_context"]
    registries = collision.load_historical_exclusion_registries()
    empty: tuple[str, ...] = ()
    context = FrozenCollisionContext(
        mode=ALLOWED_MODE,
        split=ALLOWED_SPLIT,
        world_ordinal=0,
        historical_item_hashes=tuple(sorted(registries.item_document_hashes)),
        historical_seller_hashes=tuple(sorted(registries.seller_document_hashes)),
        current_item_hashes=empty,
        current_seller_hashes=empty,
        predecessor_item_hashes=empty,
        predecessor_seller_hashes=empty,
        previous_world_marker_sha256=common.canonical_sha256(
            {"state": "NO_PREVIOUS_WORLD_IN_DEVELOPMENT_SMOKE"}
        ),
        predecessor_seal_pins=(),
    )
    value = _selection_context_value(context)
    facts = {
        "world_ordinal": context.world_ordinal,
        "historical_item_count": value["historical_item"]["count"],
        "historical_item_hashes_sha256": value["historical_item"]["hashes_sha256"],
        "historical_seller_count": value["historical_seller"]["count"],
        "historical_seller_hashes_sha256": value["historical_seller"]["hashes_sha256"],
        "current_item_count": value["current_item"]["count"],
        "current_item_hashes_sha256": value["current_item"]["hashes_sha256"],
        "current_seller_count": value["current_seller"]["count"],
        "current_seller_hashes_sha256": value["current_seller"]["hashes_sha256"],
        "predecessor_item_count": value["predecessor_item"]["count"],
        "predecessor_item_hashes_sha256": value["predecessor_item"]["hashes_sha256"],
        "predecessor_seller_count": value["predecessor_seller"]["count"],
        "predecessor_seller_hashes_sha256": value["predecessor_seller"]["hashes_sha256"],
        "previous_world_marker_sha256": value["previous_world_marker_sha256"],
        "predecessor_seal_pin_count": value["predecessor_seal_pin_count"],
        "predecessor_seal_pins_sha256": value["predecessor_seal_pins_sha256"],
    }
    if facts != expected:
        raise CandidateSelectionError("Pinned smoke collision context drift")
    return context


def _hash_rows(payload: bytes, *, fields: tuple[str, ...], label: str) -> list[dict[str, Any]]:
    rows = _decode_canonical(payload, label=label)
    if not isinstance(rows, list):
        raise CandidateSelectionError(f"{label} is not a row list")
    output = []
    identifier_field = fields[1]
    for ordinal, source_row in enumerate(rows):
        if not isinstance(source_row, dict):
            raise CandidateSelectionError(f"{label} row is not an object")
        _require_exact_keys(source_row, fields, label=f"{label} row")
        if (
            _require_plain_int(source_row["row_ordinal"], label=f"{label}.row_ordinal")
            != ordinal
            or not isinstance(source_row[identifier_field], str)
            or not source_row[identifier_field]
        ):
            raise CandidateSelectionError(f"{label} row identity drift")
        _require_sha256(source_row["document_sha256"], label=label)
        output.append(dict(source_row))
    return output


def _replay_final_document_observation(
    candidate: natural.AssembledDevelopmentCandidate,
) -> _FinalDocumentObservation:
    if not isinstance(candidate, natural.AssembledDevelopmentCandidate):
        raise CandidateSelectionError("Selector received a non-assembled candidate")
    world = candidate.thaw_world()
    base_policy, template, _fixture, _style_profile = (
        candidate_parent._load_validated_base_inputs()
    )
    profiles, provenance, context = candidate_parent._build_profiles_and_provenance(
        base_policy=base_policy,
        mode=ALLOWED_MODE,
        split=ALLOWED_SPLIT,
        world=world,
        template=template,
    )
    profiles = sorted(
        (dict(row) for row in profiles),
        key=lambda row: str(row["seller_uid"]).encode("utf-8"),
    )
    profiles_bytes = _canonical_bytes(profiles)
    if (
        profiles_bytes != candidate.profiles_bytes
        or _sha256_bytes(profiles_bytes) != candidate.profiles_sha256
        or _canonical_bytes(provenance) != candidate.profile_provenance_bytes
        or _sha256_bytes(_canonical_bytes(provenance))
        != candidate.profile_provenance_sha256
    ):
        raise CandidateSelectionError("Independent final profile replay drift")
    processed = context.get("processed")
    if not isinstance(processed, dict):
        raise CandidateSelectionError("Production replay lacks its processed output")
    redacted = processed.get("public", {}).get("redacted_items")
    if not isinstance(redacted, list):
        raise CandidateSelectionError("Production replay lacks redacted item rows")
    redacted = sorted(
        (dict(row) for row in redacted),
        key=lambda row: str(row["item_uid"]).encode("utf-8"),
    )
    redacted_schema = tuple(
        candidate_parent.production._schema(base_policy, "redacted_items.jsonl")
    )
    if (
        len(redacted) != 105
        or len({str(row.get("item_uid", "")) for row in redacted}) != len(redacted)
        or any(set(row) != set(redacted_schema) for row in redacted)
        or len(profiles) != 28
        or len({str(row.get("seller_uid", "")) for row in profiles}) != len(profiles)
    ):
        raise CandidateSelectionError("Final document observation cardinality drift")
    item_hash_rows = []
    for ordinal, row in enumerate(redacted):
        item_uid = row.get("item_uid")
        title = row.get("title")
        description = row.get("description")
        if (
            not isinstance(item_uid, str)
            or not item_uid
            or not isinstance(title, str)
            or not isinstance(description, str)
        ):
            raise CandidateSelectionError("Redacted item document schema drift")
        item_hash_rows.append(
            {
                "row_ordinal": ordinal,
                "item_uid": item_uid,
                "document_sha256": collision.item_document_hash(
                    title=title,
                    description=description,
                ),
            }
        )
    seller_hash_rows = []
    for ordinal, row in enumerate(profiles):
        seller_uid = row.get("seller_uid")
        if not isinstance(seller_uid, str) or not seller_uid:
            raise CandidateSelectionError("Final seller profile identity drift")
        seller_hash_rows.append(
            {
                "row_ordinal": ordinal,
                "seller_uid": seller_uid,
                "document_sha256": collision.seller_document_hash(row),
            }
        )
    redacted_bytes = _canonical_bytes(redacted)
    item_hash_rows_bytes = _canonical_bytes(item_hash_rows)
    seller_hash_rows_bytes = _canonical_bytes(seller_hash_rows)
    return _FinalDocumentObservation(
        redacted_items_bytes=redacted_bytes,
        redacted_items_sha256=_sha256_bytes(redacted_bytes),
        profiles_bytes=profiles_bytes,
        profiles_sha256=_sha256_bytes(profiles_bytes),
        item_hash_rows_bytes=item_hash_rows_bytes,
        item_hash_rows_sha256=_sha256_bytes(item_hash_rows_bytes),
        seller_hash_rows_bytes=seller_hash_rows_bytes,
        seller_hash_rows_sha256=_sha256_bytes(seller_hash_rows_bytes),
    )


def _classify_document_collisions(
    *,
    item_hash_rows_bytes: bytes,
    seller_hash_rows_bytes: bytes,
    context: FrozenCollisionContext,
) -> CollisionClassification:
    item_rows = _hash_rows(
        item_hash_rows_bytes, fields=HASH_ROW_FIELDS, label="item hash rows"
    )
    seller_rows = _hash_rows(
        seller_hash_rows_bytes,
        fields=SELLER_HASH_ROW_FIELDS,
        label="seller hash rows",
    )
    _selection_context_value(context)
    item_hashes = [row["document_sha256"] for row in item_rows]
    seller_hashes = [row["document_sha256"] for row in seller_rows]
    item_set = set(item_hashes)
    seller_set = set(seller_hashes)
    item_counts = Counter(item_hashes)
    seller_counts = Counter(seller_hashes)
    hits = {
        "same_world_item_document": sum(value > 1 for value in item_counts.values()),
        "same_world_seller_document": sum(
            value > 1 for value in seller_counts.values()
        ),
        "historical_item_document": len(
            item_set.intersection(context.historical_item_hashes)
        ),
        "historical_seller_document": len(
            seller_set.intersection(context.historical_seller_hashes)
        ),
        "current_split_item_document": len(
            item_set.intersection(context.current_item_hashes)
        ),
        "current_split_seller_document": len(
            seller_set.intersection(context.current_seller_hashes)
        ),
        "predecessor_item_document": len(
            item_set.intersection(context.predecessor_item_hashes)
        ),
        "predecessor_seller_document": len(
            seller_set.intersection(context.predecessor_seller_hashes)
        ),
    }
    if any(type(value) is not int or value < 0 for value in hits.values()):
        raise CandidateSelectionError("Collision hit count drift")
    categories = tuple(name for name in COLLISION_CATEGORIES if hits[name] > 0)
    return CollisionClassification(
        categories=categories,
        hit_counts=tuple((name, hits[name]) for name in COLLISION_CATEGORIES),
    )


def _accepted_state_projection(value: AcceptedDevelopmentCandidate) -> dict[str, Any]:
    if not isinstance(value, AcceptedDevelopmentCandidate):
        raise CandidateSelectionError("Accepted candidate type drift")
    return {
        "version": value.version,
        "mode": value.mode,
        "split": value.split,
        "world_uid": value.world_uid,
        "design_smoke_only": value.design_smoke_only,
        "committable": value.committable,
        "candidate_index": value.candidate_index,
        "candidates_examined": value.candidates_examined,
        "rejected_candidate_count": value.rejected_candidate_count,
        "rejection_counts_bytes": _bytes_commitment(
            value.rejection_counts_bytes, label="rejection counts"
        ),
        "rejection_counts_sha256": value.rejection_counts_sha256,
        "candidate_parent_full_state_sha256": value.candidate_parent_full_state_sha256,
        "frozen_trial_identity_full_state_sha256": value.frozen_trial_identity_full_state_sha256,
        "candidate_invariant_sha256": value.candidate_invariant_sha256,
        "identity_parent_sha256": value.identity_parent_sha256,
        "identity33_bytes": _bytes_commitment(value.identity33_bytes, label="identity33"),
        "identity33_sha256": value.identity33_sha256,
        "profile_provenance_bytes": _bytes_commitment(
            value.profile_provenance_bytes, label="profile provenance"
        ),
        "profile_provenance_sha256": value.profile_provenance_sha256,
        "natural_output_sha256": value.natural_output_sha256,
        "world_bytes": _bytes_commitment(value.world_bytes, label="accepted world"),
        "world_sha256": value.world_sha256,
        "redacted_items_bytes": _bytes_commitment(
            value.redacted_items_bytes, label="redacted items"
        ),
        "redacted_items_sha256": value.redacted_items_sha256,
        "profiles_bytes": _bytes_commitment(value.profiles_bytes, label="profiles"),
        "profiles_sha256": value.profiles_sha256,
        "item_hash_rows_bytes": _bytes_commitment(
            value.item_hash_rows_bytes, label="item hash rows"
        ),
        "item_hash_rows_sha256": value.item_hash_rows_sha256,
        "seller_hash_rows_bytes": _bytes_commitment(
            value.seller_hash_rows_bytes, label="seller hash rows"
        ),
        "seller_hash_rows_sha256": value.seller_hash_rows_sha256,
        "item_registry_delta": list(value.item_registry_delta),
        "item_registry_delta_count": value.item_registry_delta_count,
        "item_registry_delta_sha256": value.item_registry_delta_sha256,
        "seller_registry_delta": list(value.seller_registry_delta),
        "seller_registry_delta_count": value.seller_registry_delta_count,
        "seller_registry_delta_sha256": value.seller_registry_delta_sha256,
        "allocation_delta": list(value.allocation_delta),
        "allocation_delta_count": value.allocation_delta_count,
        "allocation_delta_sha256": value.allocation_delta_sha256,
        "selection_context_bytes": _bytes_commitment(
            value.selection_context_bytes, label="selection context"
        ),
        "selection_context_sha256": value.selection_context_sha256,
        "exact_title_clone_qualification_bytes": _bytes_commitment(
            value.exact_title_clone_qualification_bytes,
            label="title-clone qualification",
        ),
        "exact_title_clone_qualification_sha256": value.exact_title_clone_qualification_sha256,
    }


def _validate_accepted_candidate(
    value: AcceptedDevelopmentCandidate,
    *,
    context: FrozenCollisionContext,
    material: natural.TrustedDevelopmentSelectionMaterial,
    trusted_candidate: natural.AssembledDevelopmentCandidate,
) -> None:
    if (
        value.version
        != "2026-08-10-step28-v13-v1-13-accepted-development-candidate-v1"
        or value.mode != ALLOWED_MODE
        or value.split != ALLOWED_SPLIT
        or value.design_smoke_only is not True
        or value.committable is not False
        or type(value.candidate_index) is not int
        or not 0 <= value.candidate_index < CANDIDATE_LIMIT
        or value.candidates_examined != value.candidate_index + 1
        or value.rejected_candidate_count != value.candidate_index
    ):
        raise CandidateSelectionError("Accepted candidate envelope drift")
    if (
        not isinstance(material, natural.TrustedDevelopmentSelectionMaterial)
        or value.mode != material.mode
        or value.split != material.split
        or value.world_uid != material.world_uid
        or value.candidate_parent_full_state_sha256
        != material.candidate_parent_full_state_sha256
        or value.frozen_trial_identity_full_state_sha256
        != material.frozen_trial_identity_full_state_sha256
        or value.candidate_invariant_sha256
        != material.candidate_invariant_sha256
        or value.identity_parent_sha256 != material.identity_parent_sha256
        or value.identity33_sha256 != material.identity33_sha256
        or value.profile_provenance_sha256
        != material.profile_provenance_sha256
        or value.allocation_delta != material.allocation_delta
        or value.exact_title_clone_qualification_bytes
        != material.exact_title_clone_qualification_bytes
        or value.exact_title_clone_qualification_sha256
        != material.exact_title_clone_qualification_sha256
    ):
        raise CandidateSelectionError("Accepted candidate trusted authority drift")
    byte_pairs = (
        (value.rejection_counts_bytes, value.rejection_counts_sha256, "rejection counts"),
        (value.identity33_bytes, value.identity33_sha256, "identity33"),
        (
            value.profile_provenance_bytes,
            value.profile_provenance_sha256,
            "profile provenance",
        ),
        (value.world_bytes, value.world_sha256, "world"),
        (value.redacted_items_bytes, value.redacted_items_sha256, "redacted items"),
        (value.profiles_bytes, value.profiles_sha256, "profiles"),
        (value.item_hash_rows_bytes, value.item_hash_rows_sha256, "item hash rows"),
        (
            value.seller_hash_rows_bytes,
            value.seller_hash_rows_sha256,
            "seller hash rows",
        ),
        (
            value.selection_context_bytes,
            value.selection_context_sha256,
            "selection context",
        ),
        (
            value.exact_title_clone_qualification_bytes,
            value.exact_title_clone_qualification_sha256,
            "title-clone qualification",
        ),
    )
    for payload, claimed, label in byte_pairs:
        _decode_canonical(payload, label=label)
        if _sha256_bytes(payload) != _require_sha256(claimed, label=label):
            raise CandidateSelectionError(f"Accepted {label} hash drift")
    _require_sha256(value.natural_output_sha256, label="accepted natural output")
    replay_candidate = natural.AssembledDevelopmentCandidate(
        candidate_index=value.candidate_index,
        world_bytes=value.world_bytes,
        world_sha256=value.world_sha256,
        profiles_bytes=value.profiles_bytes,
        profiles_sha256=value.profiles_sha256,
        profile_provenance_bytes=value.profile_provenance_bytes,
        profile_provenance_sha256=value.profile_provenance_sha256,
        identity33_bytes=value.identity33_bytes,
        identity33_sha256=value.identity33_sha256,
        natural_output_sha256=value.natural_output_sha256,
        candidate_invariant_sha256=value.candidate_invariant_sha256,
        identity_parent_sha256=value.identity_parent_sha256,
    )
    if (
        not isinstance(trusted_candidate, natural.AssembledDevelopmentCandidate)
        or replay_candidate != trusted_candidate
    ):
        raise CandidateSelectionError(
            "Accepted candidate differs from the selector's trusted assembled candidate"
        )
    replayed = _replay_final_document_observation(replay_candidate)
    for field_name in (
        "redacted_items_bytes",
        "redacted_items_sha256",
        "profiles_bytes",
        "profiles_sha256",
        "item_hash_rows_bytes",
        "item_hash_rows_sha256",
        "seller_hash_rows_bytes",
        "seller_hash_rows_sha256",
    ):
        if getattr(replayed, field_name) != getattr(value, field_name):
            raise CandidateSelectionError(
                f"Accepted {field_name} disagrees with production replay"
            )
    classification = _validate_collision_classification(
        _classify_document_collisions(
            item_hash_rows_bytes=replayed.item_hash_rows_bytes,
            seller_hash_rows_bytes=replayed.seller_hash_rows_bytes,
            context=context,
        )
    )
    if classification.has_collision:
        raise CandidateSelectionError("Accepted candidate still has a document collision")
    rejection_counts = _decode_canonical(
        value.rejection_counts_bytes, label="rejection counts"
    )
    if (
        not isinstance(rejection_counts, dict)
        or tuple(rejection_counts) != tuple(sorted(COLLISION_CATEGORIES))
        or set(rejection_counts) != set(COLLISION_CATEGORIES)
        or any(type(count) is not int or count < 0 for count in rejection_counts.values())
    ):
        raise CandidateSelectionError("Accepted rejection-count schema drift")
    context_bytes = _canonical_bytes(_selection_context_value(context))
    if context_bytes != value.selection_context_bytes:
        raise CandidateSelectionError("Accepted candidate collision context drift")
    item_rows = _hash_rows(
        value.item_hash_rows_bytes,
        fields=HASH_ROW_FIELDS,
        label="accepted item hash rows",
    )
    seller_rows = _hash_rows(
        value.seller_hash_rows_bytes,
        fields=SELLER_HASH_ROW_FIELDS,
        label="accepted seller hash rows",
    )
    item_delta = _sorted_hash_tuple(
        value.item_registry_delta, label="accepted item registry delta"
    )
    seller_delta = _sorted_hash_tuple(
        value.seller_registry_delta, label="accepted seller registry delta"
    )
    allocation_delta = _sorted_hash_tuple(
        value.allocation_delta, label="accepted allocation delta"
    )
    expected_counts = load_policy()["expected_smoke_selection"]
    if (
        len(item_rows) != expected_counts["item_count"]
        or len(seller_rows) != expected_counts["seller_count"]
        or value.item_registry_delta_count != expected_counts["item_count"]
        or value.seller_registry_delta_count != expected_counts["seller_count"]
        or value.item_registry_delta_count != len(item_delta)
        or value.item_registry_delta_sha256 != _hash_tuple_digest(item_delta)
        or value.seller_registry_delta_count != len(seller_delta)
        or value.seller_registry_delta_sha256 != _hash_tuple_digest(seller_delta)
        or value.allocation_delta_count != len(allocation_delta)
        or value.allocation_delta_sha256 != _hash_tuple_digest(allocation_delta)
    ):
        raise CandidateSelectionError("Accepted registry delta commitment drift")
    collision.validate_row_hash_multiplicity(
        row_count=len(item_rows),
        row_hashes=[row["document_sha256"] for row in item_rows],
        registry_hashes=item_delta,
        label="accepted item documents",
    )
    collision.validate_row_hash_multiplicity(
        row_count=len(seller_rows),
        row_hashes=[row["document_sha256"] for row in seller_rows],
        registry_hashes=seller_delta,
        label="accepted seller documents",
    )
    if common.canonical_sha256(_accepted_state_projection(value)) != value.accepted_state_sha256:
        raise CandidateSelectionError("Accepted full-state commitment drift")


def _build_accepted_candidate(
    *,
    candidate: natural.AssembledDevelopmentCandidate,
    observation: _FinalDocumentObservation,
    material: natural.TrustedDevelopmentSelectionMaterial,
    context: FrozenCollisionContext,
    rejection_counts: Mapping[str, int],
) -> AcceptedDevelopmentCandidate:
    item_rows = _hash_rows(
        observation.item_hash_rows_bytes,
        fields=HASH_ROW_FIELDS,
        label="candidate item hash rows",
    )
    seller_rows = _hash_rows(
        observation.seller_hash_rows_bytes,
        fields=SELLER_HASH_ROW_FIELDS,
        label="candidate seller hash rows",
    )
    item_delta = tuple(sorted(row["document_sha256"] for row in item_rows))
    seller_delta = tuple(sorted(row["document_sha256"] for row in seller_rows))
    allocation_delta = _sorted_hash_tuple(
        material.allocation_delta, label="trusted allocation delta"
    )
    collision.validate_row_hash_multiplicity(
        row_count=len(item_rows),
        row_hashes=[row["document_sha256"] for row in item_rows],
        registry_hashes=item_delta,
        label="selected item documents",
    )
    collision.validate_row_hash_multiplicity(
        row_count=len(seller_rows),
        row_hashes=[row["document_sha256"] for row in seller_rows],
        registry_hashes=seller_delta,
        label="selected seller documents",
    )
    rejection_value = {
        name: _require_plain_int(rejection_counts[name], label=f"rejection.{name}")
        for name in sorted(COLLISION_CATEGORIES)
    }
    rejection_bytes = _canonical_bytes(rejection_value)
    context_bytes = _canonical_bytes(_selection_context_value(context))
    kwargs = {
        "version": "2026-08-10-step28-v13-v1-13-accepted-development-candidate-v1",
        "mode": material.mode,
        "split": material.split,
        "world_uid": material.world_uid,
        "design_smoke_only": True,
        "committable": False,
        "candidate_index": candidate.candidate_index,
        "candidates_examined": candidate.candidate_index + 1,
        "rejected_candidate_count": candidate.candidate_index,
        "rejection_counts_bytes": rejection_bytes,
        "rejection_counts_sha256": _sha256_bytes(rejection_bytes),
        "candidate_parent_full_state_sha256": material.candidate_parent_full_state_sha256,
        "frozen_trial_identity_full_state_sha256": material.frozen_trial_identity_full_state_sha256,
        "candidate_invariant_sha256": material.candidate_invariant_sha256,
        "identity_parent_sha256": material.identity_parent_sha256,
        "identity33_bytes": candidate.identity33_bytes,
        "identity33_sha256": candidate.identity33_sha256,
        "profile_provenance_bytes": candidate.profile_provenance_bytes,
        "profile_provenance_sha256": candidate.profile_provenance_sha256,
        "natural_output_sha256": candidate.natural_output_sha256,
        "world_bytes": candidate.world_bytes,
        "world_sha256": candidate.world_sha256,
        "redacted_items_bytes": observation.redacted_items_bytes,
        "redacted_items_sha256": observation.redacted_items_sha256,
        "profiles_bytes": observation.profiles_bytes,
        "profiles_sha256": observation.profiles_sha256,
        "item_hash_rows_bytes": observation.item_hash_rows_bytes,
        "item_hash_rows_sha256": observation.item_hash_rows_sha256,
        "seller_hash_rows_bytes": observation.seller_hash_rows_bytes,
        "seller_hash_rows_sha256": observation.seller_hash_rows_sha256,
        "item_registry_delta": item_delta,
        "item_registry_delta_count": len(item_delta),
        "item_registry_delta_sha256": _hash_tuple_digest(item_delta),
        "seller_registry_delta": seller_delta,
        "seller_registry_delta_count": len(seller_delta),
        "seller_registry_delta_sha256": _hash_tuple_digest(seller_delta),
        "allocation_delta": allocation_delta,
        "allocation_delta_count": len(allocation_delta),
        "allocation_delta_sha256": _hash_tuple_digest(allocation_delta),
        "selection_context_bytes": context_bytes,
        "selection_context_sha256": _sha256_bytes(context_bytes),
        "exact_title_clone_qualification_bytes": material.exact_title_clone_qualification_bytes,
        "exact_title_clone_qualification_sha256": material.exact_title_clone_qualification_sha256,
    }
    provisional = AcceptedDevelopmentCandidate(
        **kwargs,
        accepted_state_sha256="0" * 64,
    )
    accepted = AcceptedDevelopmentCandidate(
        **kwargs,
        accepted_state_sha256=common.canonical_sha256(
            _accepted_state_projection(provisional)
        ),
    )
    _validate_accepted_candidate(
        accepted,
        context=context,
        material=material,
        trusted_candidate=candidate,
    )
    return accepted


class DevelopmentSmokeCandidateSelector:
    """Single-use selector with no caller-supplied screening authority."""

    def __init__(self) -> None:
        self._policy = load_policy()
        self._context = _build_smoke_collision_context()
        self._session = natural.DevelopmentSmokeVariationSession()
        self._material = self._session.trusted_selection_material()
        self._failed = False
        self._completed = False
        self._trusted_accepted_candidate: (
            natural.AssembledDevelopmentCandidate | None
        ) = None
        self._trusted_accepted_state_sha256: str | None = None

    def select(self) -> AcceptedDevelopmentCandidate:
        if self._failed:
            raise CandidateSelectionError("Candidate selector is permanently failed")
        if self._completed:
            raise CandidateSelectionError("Candidate selector has already completed")
        completed = False
        rejection_counts = {name: 0 for name in COLLISION_CATEGORIES}
        try:
            for candidate_index in range(CANDIDATE_LIMIT):
                candidate = self._session.render(candidate_index)
                observation = _replay_final_document_observation(candidate)
                classification = _validate_collision_classification(
                    _classify_document_collisions(
                        item_hash_rows_bytes=observation.item_hash_rows_bytes,
                        seller_hash_rows_bytes=observation.seller_hash_rows_bytes,
                        context=self._context,
                    )
                )
                if classification.has_collision:
                    for category in classification.categories:
                        rejection_counts[category] += 1
                    continue
                accepted = _build_accepted_candidate(
                    candidate=candidate,
                    observation=observation,
                    material=self._material,
                    context=self._context,
                    rejection_counts=rejection_counts,
                )
                self._trusted_accepted_candidate = candidate
                self._trusted_accepted_state_sha256 = accepted.accepted_state_sha256
                completed = True
                self._completed = True
                return accepted
            raise CandidateSelectionError("All 32 document candidates collided")
        finally:
            if not completed:
                self._failed = True

    def validate_completed_candidate(
        self, value: AcceptedDevelopmentCandidate
    ) -> None:
        """Revalidate only against this selector's retained trusted candidate."""

        if self._failed:
            raise CandidateSelectionError("Candidate selector is permanently failed")
        if (
            not self._completed
            or self._trusted_accepted_candidate is None
            or self._trusted_accepted_state_sha256 is None
        ):
            self._failed = True
            raise CandidateSelectionError("Candidate selector has no completed authority")
        try:
            if (
                _require_sha256(
                    value.accepted_state_sha256,
                    label="completed accepted state",
                )
                != self._trusted_accepted_state_sha256
            ):
                raise CandidateSelectionError(
                    "Completed candidate differs from the selector's retained state root"
                )
            _validate_accepted_candidate(
                value,
                context=self._context,
                material=self._material,
                trusted_candidate=self._trusted_accepted_candidate,
            )
        except BaseException:
            self._failed = True
            raise


def main() -> None:
    accepted = DevelopmentSmokeCandidateSelector().select()
    print(
        json.dumps(
            {
                "status": "PASS_DEVELOPMENT_SMOKE_CANDIDATE_SELECTION_ONLY",
                "formal_authorizations": load_policy()["formal_authorizations"],
                "formal_seeds_generated": 0,
                "formal_rows_generated": 0,
                "formal_transactions_written": 0,
                "formal_models_trained": 0,
                "design_smoke_only": accepted.design_smoke_only,
                "committable": accepted.committable,
                "world_uid": accepted.world_uid,
                "accepted_candidate_index": accepted.candidate_index,
                "candidates_examined": accepted.candidates_examined,
                "rejected_candidate_count": accepted.rejected_candidate_count,
                "item_registry_delta_count": accepted.item_registry_delta_count,
                "seller_registry_delta_count": accepted.seller_registry_delta_count,
                "allocation_delta_count": accepted.allocation_delta_count,
                "selection_context_sha256": accepted.selection_context_sha256,
                "exact_title_clone_qualification_sha256": (
                    accepted.exact_title_clone_qualification_sha256
                ),
                "accepted_state_sha256": accepted.accepted_state_sha256,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
