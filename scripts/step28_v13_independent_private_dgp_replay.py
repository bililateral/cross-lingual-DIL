#!/usr/bin/env python3
"""Independent typed-plan replay for the Step 28-v13 private DGP.

This implementation intentionally imports no Step28 producer, parser, redactor,
or shared helper module.  It receives a public policy, one observed seller UID
pool, and one split structure key.  It independently reconstructs the private
controller graph through the typed identity-asset plan.

It also reconstructs the registered high-semantic and exact-title-clone side
and item choices.  It does not replay identity values, occurrence-to-item
flow, rendered text, or parser output.  Those remain outside this gate and
must not be represented as independently certified by its receipt.
"""

from __future__ import annotations

import hashlib
import hmac
import itertools
import json
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from decimal import Decimal, ROUND_FLOOR, ROUND_HALF_EVEN, localcontext
from typing import Any


FIELD_SEPARATOR = b"\x1f"
TUPLE_SEPARATOR = "\x1e"
STRUCTURE_PREFIX = b"step28-v13-structure"
POLICY_VERSION = (
    "2026-07-29-step28-v13-synthetic-chinese-dataset-v13-draft"
)
REPLAY_LEDGER_VERSION = (
    "2026-07-28-step28-v13-independent-typed-replay-v2-draft"
)
REPLAY_SCOPE = (
    "membership_market_style_mechanism_typed_identity_assets_"
    "positive_negative_targets_repeat_and_registered_overrides"
)
SPLIT_ORDER = ("train", "development", "audit_a", "audit_b")
ENTITY_PREFIXES = {
    "world": "w",
    "controller": "ctl",
    "seller": "sel",
    "identity_asset": "ias",
}
EVIDENCE_LEVEL = (
    "INDEPENDENT_TYPED_DGP_REPLAY_DEVELOPMENT_INTEGRATION_"
    "NOT_FORMAL_CUSTODY_SEAL"
)
PROBABILITY_SCALE = 10**12


class IndependentReplayError(ValueError):
    """Fail-closed error from the independent replay implementation."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _utf8_sorted(values: Sequence[str] | set[str]) -> list[str]:
    return sorted((str(value) for value in values), key=lambda value: value.encode("utf-8"))


def _key_bytes(key_hex: str) -> bytes:
    value = str(key_hex)
    if len(value) != 64 or value != value.lower():
        raise IndependentReplayError("REPLAY_STRUCTURE_KEY_ENCODING_INVALID")
    try:
        raw = bytes.fromhex(value)
    except ValueError as exc:
        raise IndependentReplayError(
            "REPLAY_STRUCTURE_KEY_ENCODING_INVALID"
        ) from exc
    if len(raw) != 32:
        raise IndependentReplayError("REPLAY_STRUCTURE_KEY_LENGTH_INVALID")
    return raw


def _base_uid(
    *,
    id_key_hex: str,
    entity_kind: str,
    parent_uid: str,
    ordinal: int,
) -> str:
    if entity_kind not in ENTITY_PREFIXES or not parent_uid or ordinal < 0:
        raise IndependentReplayError("REPLAY_BASE_UID_INPUT_INVALID")
    message = FIELD_SEPARATOR.join(
        (
            b"step28-v13",
            entity_kind.encode("ascii"),
            parent_uid.encode("utf-8"),
            str(ordinal).encode("ascii"),
        )
    )
    digest = hmac.new(_key_bytes(id_key_hex), message, hashlib.sha256).hexdigest()
    return f"{ENTITY_PREFIXES[entity_kind]}_{digest}"


def _pair(left: str, right: str) -> tuple[str, str]:
    if left == right or "|" in left or "|" in right:
        raise IndependentReplayError("REPLAY_PAIR_INPUT_INVALID")
    ordered = _utf8_sorted([left, right])
    return ordered[0], ordered[1]


def _pair_uid(left: str, right: str) -> str:
    ordered = _pair(left, right)
    return f"{ordered[0]}||{ordered[1]}"


def _validate_atom(value: str, *, label: str) -> str:
    output = str(value)
    if (
        not output
        or FIELD_SEPARATOR.decode("ascii") in output
        or TUPLE_SEPARATOR in output
    ):
        raise IndependentReplayError(f"REPLAY_STRUCTURE_ATOM_INVALID:{label}")
    return output


def _candidate_bytes(
    draw: Mapping[str, Any],
    candidate: str,
    prefix_atoms: Mapping[str, str] | None,
) -> bytes:
    value = str(candidate)
    if not value or FIELD_SEPARATOR.decode("ascii") in value:
        raise IndependentReplayError("REPLAY_STRUCTURE_CANDIDATE_INVALID")
    tuple_atoms = value.split(TUPLE_SEPARATOR)
    if any(not atom for atom in tuple_atoms):
        raise IndependentReplayError("REPLAY_STRUCTURE_CANDIDATE_INVALID")
    for atom in tuple_atoms:
        _validate_atom(atom, label="candidate")

    registered = list(draw.get("candidate_prefix_atoms", []))
    supplied = dict(prefix_atoms or {})
    required = {
        name for name in registered if not str(name).startswith("literal:")
    }
    if set(supplied) != required:
        raise IndependentReplayError("REPLAY_STRUCTURE_PREFIX_SCHEMA_MISMATCH")
    atoms: list[str] = []
    for name in registered:
        if str(name).startswith("literal:"):
            atom = str(name).split(":", 1)[1]
        else:
            atom = supplied[str(name)]
        atoms.append(_validate_atom(atom, label=str(name)))
    if atoms:
        value = TUPLE_SEPARATOR.join([*atoms, value])
    return value.encode("utf-8")


def _rank_structure(
    policy: Mapping[str, Any],
    *,
    structure_key_hex: str,
    world_uid: str,
    draw_name: str,
    candidates: Sequence[str],
    prefix_atoms: Mapping[str, str] | None = None,
) -> list[str]:
    try:
        draw = policy["randomness"]["structure_draw_registry"]["draws"][draw_name]
    except KeyError as exc:
        raise IndependentReplayError(
            f"REPLAY_DRAW_UNREGISTERED:{draw_name}"
        ) from exc
    if draw.get("operation") not in {"choice", "permutation"}:
        raise IndependentReplayError(f"REPLAY_DRAW_OPERATION_INVALID:{draw_name}")
    values = [str(value) for value in candidates]
    if not values or len(values) != len(set(values)):
        raise IndependentReplayError(
            f"REPLAY_CANDIDATE_POOL_INVALID:{draw_name}"
        )
    world = _validate_atom(world_uid, label="world_uid")
    key = _key_bytes(structure_key_hex)
    ranked: list[tuple[bytes, bytes, str]] = []
    for candidate in values:
        material = _candidate_bytes(draw, candidate, prefix_atoms)
        message = FIELD_SEPARATOR.join(
            (
                STRUCTURE_PREFIX,
                world.encode("utf-8"),
                draw_name.encode("ascii"),
                material,
            )
        )
        ranked.append(
            (
                hmac.new(key, message, hashlib.sha256).digest(),
                candidate.encode("utf-8"),
                candidate,
            )
        )
    return [candidate for _digest, _raw, candidate in sorted(ranked)]


def _choose_structure(
    policy: Mapping[str, Any],
    **kwargs: Any,
) -> str:
    draw_name = str(kwargs["draw_name"])
    operation = policy["randomness"]["structure_draw_registry"]["draws"][
        draw_name
    ].get("operation")
    if operation != "choice":
        raise IndependentReplayError(f"REPLAY_DRAW_NOT_CHOICE:{draw_name}")
    return _rank_structure(policy, **kwargs)[0]


def _bernoulli(
    policy: Mapping[str, Any],
    *,
    structure_key_hex: str,
    world_uid: str,
    draw_name: str,
    subject_uid: str,
    probability: float | str | Decimal,
) -> bool:
    draw = policy["randomness"]["structure_draw_registry"]["draws"].get(
        draw_name
    )
    if draw is None or draw.get("operation") != "bernoulli":
        raise IndependentReplayError(f"REPLAY_DRAW_NOT_BERNOULLI:{draw_name}")
    exact = Decimal(str(probability))
    if not exact.is_finite() or exact < 0 or exact > 1:
        raise IndependentReplayError("REPLAY_BERNOULLI_PROBABILITY_INVALID")
    message = FIELD_SEPARATOR.join(
        (
            STRUCTURE_PREFIX,
            _validate_atom(world_uid, label="world_uid").encode("utf-8"),
            draw_name.encode("ascii"),
            _validate_atom(subject_uid, label="subject_uid").encode("utf-8"),
        )
    )
    draw_value = int.from_bytes(
        hmac.new(
            _key_bytes(structure_key_hex), message, hashlib.sha256
        ).digest()[:8],
        "big",
    )
    with localcontext() as context:
        context.prec = 80
        threshold = int(
            (exact * Decimal(1 << 64)).to_integral_value(rounding=ROUND_FLOOR)
        )
    return draw_value < threshold


def _text_rank(
    *,
    text_key_hex: str,
    context: Sequence[str],
    candidates: Sequence[str],
) -> list[str]:
    values = [str(value) for value in candidates]
    if not values or len(values) != len(set(values)):
        raise IndependentReplayError("REPLAY_STYLE_CANDIDATE_POOL_INVALID")
    prefix = FIELD_SEPARATOR.join(str(part).encode("utf-8") for part in context)
    key = _key_bytes(text_key_hex)
    ranked = [
        (
            hmac.new(
                key,
                prefix + FIELD_SEPARATOR + candidate.encode("utf-8"),
                hashlib.sha256,
            ).digest(),
            candidate.encode("utf-8"),
            candidate,
        )
        for candidate in values
    ]
    return [candidate for _digest, _raw, candidate in sorted(ranked)]


class _ReplayRng:
    """Independent implementation of the registered counter-HMAC stream."""

    def __init__(self, key_hex: str, *context: str) -> None:
        if not context or any(
            FIELD_SEPARATOR in str(part).encode("utf-8") for part in context
        ):
            raise IndependentReplayError("REPLAY_RNG_CONTEXT_INVALID")
        self._key = _key_bytes(key_hex)
        self._context = FIELD_SEPARATOR.join(
            str(part).encode("utf-8") for part in context
        )
        self._counter = 0

    def _block(self) -> bytes:
        counter = self._counter
        self._counter += 1
        message = (
            self._context
            + FIELD_SEPARATOR
            + counter.to_bytes(16, "big")
        )
        return hmac.new(self._key, message, hashlib.sha256).digest()

    def randbelow(self, upper: int) -> int:
        if upper <= 0:
            raise IndependentReplayError("REPLAY_RNG_UPPER_INVALID")
        limit = (1 << 256) - ((1 << 256) % upper)
        while True:
            value = int.from_bytes(self._block(), "big")
            if value < limit:
                return value % upper

    def choice(self, values: Sequence[Any]) -> Any:
        if not values:
            raise IndependentReplayError("REPLAY_RNG_CHOICE_POOL_EMPTY")
        return values[self.randbelow(len(values))]


def _integerized_probabilities(probabilities: Sequence[float]) -> list[int]:
    if not probabilities:
        raise IndependentReplayError("REPLAY_CATEGORY_PROBABILITIES_EMPTY")
    output: list[int] = []
    with localcontext() as context:
        context.prec = 80
        for raw in probabilities[:-1]:
            probability = Decimal(str(raw))
            if not probability.is_finite() or probability < 0:
                raise IndependentReplayError(
                    "REPLAY_CATEGORY_PROBABILITY_INVALID"
                )
            output.append(
                int(
                    (
                        probability * Decimal(PROBABILITY_SCALE)
                    ).to_integral_value(rounding=ROUND_HALF_EVEN)
                )
            )
    final = PROBABILITY_SCALE - sum(output)
    if final < 0:
        raise IndependentReplayError(
            "REPLAY_CATEGORY_PROBABILITY_SUM_INVALID"
        )
    output.append(final)
    return output


def _categorical_choice(
    rng: _ReplayRng,
    values: Sequence[Any],
    probabilities: Sequence[float],
) -> Any:
    if len(values) != len(probabilities):
        raise IndependentReplayError(
            "REPLAY_CATEGORY_VALUE_PROBABILITY_MISMATCH"
        )
    draw = rng.randbelow(PROBABILITY_SCALE)
    cursor = 0
    for value, weight in zip(
        values, _integerized_probabilities(probabilities), strict=True
    ):
        cursor += weight
        if draw < cursor:
            return value
    raise IndependentReplayError("REPLAY_CATEGORY_DRAW_OUTSIDE_MASS")


def registered_world_uids_for_split(
    policy: Mapping[str, Any],
    *,
    mode: str,
    split: str,
) -> list[str]:
    """Independently reconstruct the complete public world set for a split."""

    if policy.get("version") != POLICY_VERSION:
        raise IndependentReplayError("REPLAY_POLICY_VERSION_UNSUPPORTED")
    if mode not in {"development_smoke", "training_ready", "formal"}:
        raise IndependentReplayError("REPLAY_MODE_INVALID")
    if split not in SPLIT_ORDER:
        raise IndependentReplayError("REPLAY_SPLIT_INVALID")
    try:
        stream = policy["randomness"][mode]
        raw_counts = policy["modes"][mode]["world_counts"]
    except (KeyError, TypeError) as exc:
        raise IndependentReplayError(
            "REPLAY_WORLD_REGISTRY_SCHEMA_INVALID"
        ) from exc
    if not isinstance(raw_counts, Mapping) or set(raw_counts) != set(
        SPLIT_ORDER
    ):
        raise IndependentReplayError("REPLAY_WORLD_COUNT_SCHEMA_INVALID")
    counts: dict[str, int] = {}
    for split_name in SPLIT_ORDER:
        raw_count = raw_counts[split_name]
        if isinstance(raw_count, bool):
            raise IndependentReplayError("REPLAY_WORLD_COUNT_INVALID")
        try:
            count = int(raw_count)
        except (TypeError, ValueError) as exc:
            raise IndependentReplayError(
                "REPLAY_WORLD_COUNT_INVALID"
            ) from exc
        if count <= 0 or str(count) != str(raw_count):
            raise IndependentReplayError("REPLAY_WORLD_COUNT_INVALID")
        counts[split_name] = count
    world_pool = [
        _base_uid(
            id_key_hex=str(stream["id_key_hex"]),
            entity_kind="world",
            parent_uid=mode,
            ordinal=ordinal,
        )
        for ordinal in range(sum(counts.values()))
    ]
    namespace_key = _key_bytes(str(stream["id_namespace_key_hex"]))
    ranked_worlds = sorted(
        world_pool,
        key=lambda candidate: (
            hmac.new(
                namespace_key,
                b"world_split_assignment"
                + FIELD_SEPARATOR
                + candidate.encode("utf-8"),
                hashlib.sha256,
            ).digest(),
            candidate.encode("utf-8"),
        ),
    )
    cursor = 0
    registered_by_split: dict[str, list[str]] = {}
    for split_name in SPLIT_ORDER:
        count = counts[split_name]
        registered_by_split[split_name] = _utf8_sorted(
            ranked_worlds[cursor : cursor + count]
        )
        cursor += count
    if cursor != len(ranked_worlds):
        raise IndependentReplayError(
            "REPLAY_WORLD_SPLIT_SLICING_INVALID"
        )
    return registered_by_split[split]


def _validate_replay_inputs(
    policy: Mapping[str, Any],
    *,
    mode: str,
    split: str,
    world_uid: str,
    observed_seller_uids: Sequence[str],
    structure_key_hex: str,
) -> tuple[list[str], str]:
    registered_worlds = registered_world_uids_for_split(
        policy,
        mode=mode,
        split=split,
    )
    _validate_atom(world_uid, label="world_uid")
    key = _key_bytes(structure_key_hex)
    stream = policy["randomness"][mode]
    if mode in {"development_smoke", "training_ready"}:
        if structure_key_hex != str(stream["structure_key_hex"]):
            raise IndependentReplayError("REPLAY_SPLIT_KEY_MISMATCH")
    else:
        if policy.get("status") != "FROZEN" or not policy.get(
            "formal_generation_enabled"
        ):
            raise IndependentReplayError("REPLAY_FORMAL_RELEASE_NOT_FROZEN")
        commitment = stream["label_bearing_structure_keys"][split].get(
            "sha256_commitment"
        )
        if (
            not isinstance(commitment, str)
            or len(commitment) != 64
            or hashlib.sha256(key).hexdigest() != commitment
        ):
            raise IndependentReplayError("REPLAY_SPLIT_KEY_COMMITMENT_MISMATCH")

    if world_uid not in set(registered_worlds):
        raise IndependentReplayError(
            "REPLAY_WORLD_UID_NOT_REGISTERED_FOR_SPLIT"
        )

    observed = [str(value) for value in observed_seller_uids]
    if len(observed) != 28 or len(set(observed)) != 28:
        raise IndependentReplayError("REPLAY_OBSERVED_SELLER_UID_POOL_INVALID")
    expected = [
        _base_uid(
            id_key_hex=str(stream["id_key_hex"]),
            entity_kind="seller",
            parent_uid=world_uid,
            ordinal=ordinal,
        )
        for ordinal in range(28)
    ]
    if _utf8_sorted(observed) != _utf8_sorted(expected):
        raise IndependentReplayError("REPLAY_OBSERVED_SELLER_UID_POOL_MISMATCH")
    return _utf8_sorted(observed), str(
        policy["identity_design"]["mechanism_by_split"][split]
    )


def _membership(
    policy: Mapping[str, Any],
    *,
    mode: str,
    world_uid: str,
    seller_uids: Sequence[str],
    structure_key_hex: str,
) -> dict[str, Any]:
    id_key = str(policy["randomness"][mode]["id_key_hex"])
    controller_uids = [
        _base_uid(
            id_key_hex=id_key,
            entity_kind="controller",
            parent_uid=world_uid,
            ordinal=ordinal,
        )
        for ordinal in range(12)
    ]
    controller_order = _rank_structure(
        policy,
        structure_key_hex=structure_key_hex,
        world_uid=world_uid,
        draw_name="controller_partition_order",
        candidates=controller_uids,
    )
    seller_order = _rank_structure(
        policy,
        structure_key_hex=structure_key_hex,
        world_uid=world_uid,
        draw_name="seller_partition_order",
        candidates=seller_uids,
    )
    members: dict[str, list[str]] = {}
    seller_to_controller: dict[str, str] = {}
    cursor = 0
    for controller_index, controller_uid in enumerate(controller_order):
        size = 3 if controller_index < 4 else 2
        selected = _utf8_sorted(seller_order[cursor : cursor + size])
        cursor += size
        if len(selected) != size:
            raise IndependentReplayError("REPLAY_MEMBERSHIP_SLICE_UNDERFLOW")
        members[controller_uid] = selected
        for seller_uid in selected:
            if seller_uid in seller_to_controller:
                raise IndependentReplayError(
                    "REPLAY_MEMBERSHIP_DUPLICATE_SELLER"
                )
            seller_to_controller[seller_uid] = controller_uid
    if cursor != 28 or set(seller_to_controller) != set(seller_uids):
        raise IndependentReplayError("REPLAY_MEMBERSHIP_PARTITION_INVALID")
    return {
        "controller_uids": _utf8_sorted(controller_uids),
        "seller_uids": _utf8_sorted(list(seller_uids)),
        "controller_members": members,
        "seller_to_controller": seller_to_controller,
    }


def _markets(
    policy: Mapping[str, Any],
    *,
    world_uid: str,
    structure_key_hex: str,
    membership: Mapping[str, Any],
) -> tuple[dict[str, str], int]:
    choices = [str(value) for value in policy["world_design"]["markets"]]
    maximum = int(
        policy["identity_design"]["controller_size_mechanism_assignment"][
            "seller_markets"
        ]["maximum_proposals"]
    )
    for proposal in range(maximum):
        assigned = {
            seller_uid: _choose_structure(
                policy,
                structure_key_hex=structure_key_hex,
                world_uid=world_uid,
                draw_name="market_proposal",
                candidates=choices,
                prefix_atoms={
                    "proposal_counter_unpadded_decimal": str(proposal),
                    "seller_uid": seller_uid,
                },
            )
            for seller_uid in membership["seller_uids"]
        }
        multi_market_sizes = {
            len(controller_sellers)
            for controller_sellers in membership["controller_members"].values()
            if len({assigned[seller] for seller in controller_sellers}) >= 2
        }
        if {2, 3}.issubset(multi_market_sizes):
            return assigned, proposal
    raise IndependentReplayError("REPLAY_MARKET_PROPOSAL_EXHAUSTED")


def _mechanism_name(slot_uid: str) -> str:
    name, separator, ordinal = str(slot_uid).partition("#")
    if (
        separator != "#"
        or not ordinal.isdigit()
        or str(int(ordinal)) != ordinal
    ):
        raise IndependentReplayError("REPLAY_MECHANISM_SLOT_UID_INVALID")
    return name


def _mechanisms(
    policy: Mapping[str, Any],
    *,
    world_uid: str,
    graph_name: str,
    structure_key_hex: str,
    membership: Mapping[str, Any],
    markets: Mapping[str, str],
) -> dict[str, dict[str, str]]:
    design = policy["identity_design"]["controller_size_mechanism_assignment"]
    graph = design[graph_name]
    by_size = {
        size: [
            controller_uid
            for controller_uid, sellers in membership[
                "controller_members"
            ].items()
            if len(sellers) == size
        ]
        for size in (2, 3)
    }
    assigned: dict[str, dict[str, str]] = {}
    cross_slots = (
        [(3, "cross_market_stable_reuse#0"), (2, "cross_market_stable_reuse#1")]
        if graph_name == "G_A"
        else [(2, "cross_market_stable_reuse#0")]
    )
    for size, slot_uid in cross_slots:
        eligible = [
            controller_uid
            for controller_uid in by_size[size]
            if controller_uid not in assigned
            and len(
                {
                    markets[seller_uid]
                    for seller_uid in membership["controller_members"][
                        controller_uid
                    ]
                }
            )
            >= 2
        ]
        ordered = _rank_structure(
            policy,
            structure_key_hex=structure_key_hex,
            world_uid=world_uid,
            draw_name="mechanism_cross_market_controller_order",
            candidates=eligible,
            prefix_atoms={
                "graph_name_ascii": graph_name,
                "controller_size_unpadded_decimal": str(size),
            },
        )
        assigned[ordered[0]] = {
            "mechanism": "cross_market_stable_reuse",
            "mechanism_slot_uid": slot_uid,
        }
    for size, slot_key in (
        (3, "expanded_triad_slot_ids_before_hmac_permutation"),
        (2, "expanded_dyad_slot_ids_before_hmac_permutation"),
    ):
        remaining = [
            controller_uid
            for controller_uid in by_size[size]
            if controller_uid not in assigned
        ]
        slots = [str(value) for value in graph[slot_key]]
        if len(remaining) != len(slots):
            raise IndependentReplayError(
                "REPLAY_MECHANISM_CONTROLLER_COUNT_MISMATCH"
            )
        atoms = {
            "graph_name_ascii": graph_name,
            "controller_size_unpadded_decimal": str(size),
        }
        controller_order = _rank_structure(
            policy,
            structure_key_hex=structure_key_hex,
            world_uid=world_uid,
            draw_name="mechanism_remaining_controller_order",
            candidates=remaining,
            prefix_atoms=atoms,
        )
        slot_order = _rank_structure(
            policy,
            structure_key_hex=structure_key_hex,
            world_uid=world_uid,
            draw_name="mechanism_remaining_slot_order",
            candidates=slots,
            prefix_atoms=atoms,
        )
        for controller_uid, slot_uid in zip(
            controller_order, slot_order, strict=True
        ):
            assigned[controller_uid] = {
                "mechanism": _mechanism_name(slot_uid),
                "mechanism_slot_uid": slot_uid,
            }
    if set(assigned) != set(membership["controller_uids"]):
        raise IndependentReplayError("REPLAY_MECHANISM_COVERAGE_INVALID")
    return assigned


def _controller_order(
    policy: Mapping[str, Any],
    *,
    world_uid: str,
    controller_uid: str,
    structure_key_hex: str,
    members: Sequence[str],
) -> list[str]:
    return _rank_structure(
        policy,
        structure_key_hex=structure_key_hex,
        world_uid=world_uid,
        draw_name="controller_member_left_middle_right",
        candidates=members,
        prefix_atoms={"controller_uid": controller_uid},
    )


def _cross_market_pair(
    policy: Mapping[str, Any],
    *,
    world_uid: str,
    controller_uid: str,
    structure_key_hex: str,
    members: Sequence[str],
    markets: Mapping[str, str],
) -> tuple[str, str]:
    candidates = [
        TUPLE_SEPARATOR.join(_pair(left, right))
        for index, left in enumerate(members)
        for right in members[index + 1 :]
        if markets[left] != markets[right]
    ]
    selected = _choose_structure(
        policy,
        structure_key_hex=structure_key_hex,
        world_uid=world_uid,
        draw_name="cross_market_seller_pair",
        candidates=candidates,
        prefix_atoms={"controller_uid": controller_uid},
    )
    left, right = selected.split(TUPLE_SEPARATOR)
    return left, right


def _style_groups(
    policy: Mapping[str, Any],
    *,
    mode: str,
    world_uid: str,
    structure_key_hex: str,
    controller_uids: Sequence[str],
) -> dict[str, str]:
    style_ids = [
        str(value)
        for value in policy["template_library"]["style_prototype_ids"]
    ]
    if (
        len(style_ids)
        != int(policy["template_library"]["expected_style_prototypes"])
        or len(style_ids) != len(set(style_ids))
    ):
        raise IndependentReplayError("REPLAY_STYLE_POLICY_POOL_INVALID")
    selected = _text_rank(
        text_key_hex=str(policy["randomness"][mode]["text_key_hex"]),
        context=(world_uid, "selected_style"),
        candidates=style_ids,
    )[:4]
    controller_order = _rank_structure(
        policy,
        structure_key_hex=structure_key_hex,
        world_uid=world_uid,
        draw_name="controller_style_assignment",
        candidates=controller_uids,
    )
    assigned = {
        controller_uid: selected[index // 3]
        for index, controller_uid in enumerate(controller_order)
    }
    if len(assigned) != 12 or any(
        sum(style_id == value for value in assigned.values()) != 3
        for style_id in selected
    ):
        raise IndependentReplayError("REPLAY_STYLE_GROUP_ALLOCATION_INVALID")
    return assigned


def _asset(
    *,
    descriptor_kind: str,
    descriptor_index: str,
    role: str,
    sellers: Sequence[str],
    occurrence_counts: Mapping[str, int],
    allowed_types: Sequence[str],
    fixed_type: str | None = None,
    distinct_groups: Sequence[str] = (),
    repeat_draw_name: str | None = None,
    repeat_probability: float | None = None,
) -> dict[str, Any]:
    seller_list = _utf8_sorted(list(sellers))
    if (
        not seller_list
        or set(seller_list) != set(occurrence_counts)
        or any(int(occurrence_counts[seller]) <= 0 for seller in seller_list)
        or (repeat_draw_name is None) != (repeat_probability is None)
    ):
        raise IndependentReplayError("REPLAY_IDENTITY_ASSET_PLAN_INVALID")
    return {
        "descriptor_kind": descriptor_kind,
        "descriptor_index": descriptor_index,
        "descriptor_uid": _canonical_sha256(
            {
                "descriptor_kind": descriptor_kind,
                "descriptor_index": descriptor_index,
            }
        ),
        "role": role,
        "sellers": seller_list,
        "occurrence_counts": {
            seller: int(occurrence_counts[seller]) for seller in seller_list
        },
        "allowed_types": [str(value) for value in allowed_types],
        "fixed_type": fixed_type,
        "distinct_groups": [str(value) for value in distinct_groups],
        "repeat_draw_name": repeat_draw_name,
        "repeat_probability": repeat_probability,
    }


def _background_assets(
    policy: Mapping[str, Any],
    *,
    world_uid: str,
    structure_key_hex: str,
    seller_uids: Sequence[str],
) -> list[dict[str, Any]]:
    allowed = [
        str(value)
        for value in policy["identity_design"]["background_private_scaffold"][
            "allowed_types"
        ]
    ]
    offset = int(
        _choose_structure(
            policy,
            structure_key_hex=structure_key_hex,
            world_uid=world_uid,
            draw_name="background_type_offset",
            candidates=[str(value) for value in range(len(allowed))],
            prefix_atoms={},
        )
    )
    output: list[dict[str, Any]] = []
    edge_ordinal = 0
    for seller_uid in _utf8_sorted(list(seller_uids)):
        for background_slot in range(2):
            identity_type = allowed[(edge_ordinal + offset) % len(allowed)]
            occurrence_count = 1 + (edge_ordinal % 2)
            output.append(
                _asset(
                    descriptor_kind="background_private",
                    descriptor_index=(
                        f"{seller_uid}{TUPLE_SEPARATOR}{background_slot}"
                    ),
                    role="direct_or_private",
                    sellers=[seller_uid],
                    occurrence_counts={seller_uid: occurrence_count},
                    allowed_types=[identity_type],
                    fixed_type=identity_type,
                    distinct_groups=[f"background::{seller_uid}"],
                )
            )
            edge_ordinal += 1
    counts: defaultdict[str, int] = defaultdict(int)
    type_count_counts: defaultdict[tuple[str, int], int] = defaultdict(int)
    for row in output:
        counts[str(row["fixed_type"])] += 1
        type_count_counts[
            (
                str(row["fixed_type"]),
                int(row["occurrence_counts"][str(row["sellers"][0])]),
            )
        ] += 1
    if (
        len(output) != 56
        or set(counts.values()) != {8}
        or set(type_count_counts.values()) != {4}
        or len(type_count_counts) != 14
    ):
        raise IndependentReplayError("REPLAY_BACKGROUND_BALANCE_INVALID")
    seller_count_multisets: defaultdict[str, list[int]] = defaultdict(list)
    for row in output:
        seller_uid = str(row["sellers"][0])
        seller_count_multisets[seller_uid].append(
            int(row["occurrence_counts"][seller_uid])
        )
    if any(
        sorted(seller_counts) != [1, 2]
        for seller_counts in seller_count_multisets.values()
    ):
        raise IndependentReplayError(
            "REPLAY_BACKGROUND_SELLER_COUNT_BALANCE_INVALID"
        )
    return output


def _positive_assets(
    policy: Mapping[str, Any],
    *,
    world_uid: str,
    graph_name: str,
    structure_key_hex: str,
    membership: Mapping[str, Any],
    markets: Mapping[str, str],
    mechanisms: Mapping[str, Mapping[str, str]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    direct_types = [
        str(value)
        for value in policy["identity_design"]["parser_observable_role_types"][
            "direct_or_private"
        ]
    ]
    stable_probability = float(
        policy["identity_design"]["stable_identity_repeat_probability"][
            graph_name
        ]
    )
    rotation = policy["identity_design"]["rotation_occurrence_probabilities"][
        graph_name
    ]
    traversal = [
        str(value)
        for value in policy["identity_design"]["mechanism_traversal_order"]
    ]
    ordered_controllers = sorted(
        mechanisms,
        key=lambda controller_uid: (
            traversal.index(str(mechanisms[controller_uid]["mechanism"])),
            controller_uid.encode("utf-8"),
        ),
    )
    assets: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    repeat_rows: list[dict[str, Any]] = []
    for controller_uid in ordered_controllers:
        mechanism = str(mechanisms[controller_uid]["mechanism"])
        members = list(membership["controller_members"][controller_uid])
        designated: tuple[str, str] | None = (
            _pair(*members) if len(members) == 2 else None
        )
        if mechanism == "single_identity_stable_reuse":
            assets.append(
                _asset(
                    descriptor_kind=mechanism,
                    descriptor_index=(
                        f"{controller_uid}{TUPLE_SEPARATOR}0"
                    ),
                    role="direct_or_private",
                    sellers=members,
                    occurrence_counts={seller: 1 for seller in members},
                    allowed_types=direct_types,
                    repeat_draw_name="stable_identity_repeat",
                    repeat_probability=stable_probability,
                )
            )
        elif mechanism == "multi_type_identity_reuse":
            group = f"multi::{controller_uid}"
            for asset_index in range(2):
                assets.append(
                    _asset(
                        descriptor_kind=mechanism,
                        descriptor_index=(
                            f"{controller_uid}{TUPLE_SEPARATOR}{asset_index}"
                        ),
                        role="direct_or_private",
                        sellers=members,
                        occurrence_counts={seller: 1 for seller in members},
                        allowed_types=direct_types,
                        distinct_groups=[group],
                        repeat_draw_name="stable_identity_repeat",
                        repeat_probability=stable_probability,
                    )
                )
        elif mechanism == "cross_market_stable_reuse":
            selected = _cross_market_pair(
                policy,
                world_uid=world_uid,
                controller_uid=controller_uid,
                structure_key_hex=structure_key_hex,
                members=members,
                markets=markets,
            )
            designated = selected
            assets.append(
                _asset(
                    descriptor_kind=mechanism,
                    descriptor_index=(
                        f"{controller_uid}{TUPLE_SEPARATOR}0"
                    ),
                    role="direct_or_private",
                    sellers=selected,
                    occurrence_counts={seller: 1 for seller in selected},
                    allowed_types=direct_types,
                    repeat_draw_name="stable_identity_repeat",
                    repeat_probability=stable_probability,
                )
            )
        elif mechanism == "single_hop_rotation":
            left, middle, right = _controller_order(
                policy,
                world_uid=world_uid,
                controller_uid=controller_uid,
                structure_key_hex=structure_key_hex,
                members=members,
            )
            designated = _pair(left, right)
            repeats = _bernoulli(
                policy,
                structure_key_hex=structure_key_hex,
                world_uid=world_uid,
                draw_name="single_hop_path_repeat",
                subject_uid=controller_uid,
                probability=rotation[
                    "single_hop_any_one_side_repeat_probability"
                ],
            )
            side = (
                _choose_structure(
                    policy,
                    structure_key_hex=structure_key_hex,
                    world_uid=world_uid,
                    draw_name="single_hop_repeat_side",
                    candidates=["left_middle", "middle_right"],
                    prefix_atoms={"controller_uid": controller_uid},
                )
                if repeats
                else ""
            )
            repeat_rows.extend(
                (
                    {
                        "decision_kind": "single_hop_path_repeat",
                        "subject_uid": controller_uid,
                        "decision": repeats,
                    },
                    {
                        "decision_kind": "single_hop_repeat_side",
                        "subject_uid": controller_uid,
                        "decision": side,
                    },
                )
            )
            group = f"single_hop::{controller_uid}"
            edges = (
                ("left_middle", (left, middle)),
                ("middle_right", (middle, right)),
            )
            for asset_index, (edge_name, endpoints) in enumerate(edges):
                count = 2 if side == edge_name else 1
                assets.append(
                    _asset(
                        descriptor_kind=mechanism,
                        descriptor_index=(
                            f"{controller_uid}{TUPLE_SEPARATOR}{asset_index}"
                        ),
                        role="direct_or_private",
                        sellers=endpoints,
                        occurrence_counts={
                            seller: count for seller in endpoints
                        },
                        allowed_types=direct_types,
                        distinct_groups=[group],
                    )
                )
        elif mechanism == "corroborated_two_hop_rotation":
            left, middle, right = _controller_order(
                policy,
                world_uid=world_uid,
                controller_uid=controller_uid,
                structure_key_hex=structure_key_hex,
                members=members,
            )
            designated = _pair(left, right)
            endpoints = (
                (left, middle),
                (left, middle),
                (middle, right),
                (middle, right),
            )
            for asset_index, sellers in enumerate(endpoints):
                assets.append(
                    _asset(
                        descriptor_kind=mechanism,
                        descriptor_index=(
                            f"{controller_uid}{TUPLE_SEPARATOR}{asset_index}"
                        ),
                        role="direct_or_private",
                        sellers=sellers,
                        occurrence_counts={seller: 1 for seller in sellers},
                        allowed_types=direct_types,
                        distinct_groups=[
                            f"corroborated::{controller_uid}::"
                            f"{'left' if asset_index < 2 else 'right'}"
                        ],
                    )
                )
        elif mechanism == "sparse_history":
            assets.append(
                _asset(
                    descriptor_kind=mechanism,
                    descriptor_index=(
                        f"{controller_uid}{TUPLE_SEPARATOR}0"
                    ),
                    role="direct_or_private",
                    sellers=members,
                    occurrence_counts={seller: 1 for seller in members},
                    allowed_types=direct_types,
                )
            )
        elif mechanism not in {
            "same_controller_no_direct_share",
            "zero_visible_identity_history",
        }:
            raise IndependentReplayError(
                f"REPLAY_POSITIVE_MECHANISM_UNKNOWN:{mechanism}"
            )
        if designated is None:
            raise IndependentReplayError(
                f"REPLAY_POSITIVE_TARGET_MISSING:{mechanism}"
            )
        targets.append(
            {
                "controller_uid": controller_uid,
                "mechanism": mechanism,
                "mechanism_slot_uid": mechanisms[controller_uid][
                    "mechanism_slot_uid"
                ],
                "seller_uid_left": designated[0],
                "seller_uid_right": designated[1],
                "canonical_pair_uid": _pair_uid(*designated),
            }
        )
    return assets, targets, repeat_rows


def _hard_digest(
    structure_key_hex: str,
    *,
    world_uid: str,
    asset_kind: str,
    asset_index: int,
    candidate: str,
) -> bytes:
    message = FIELD_SEPARATOR.join(
        (
            world_uid.encode("utf-8"),
            b"hard_negative",
            asset_kind.encode("ascii"),
            str(asset_index).encode("ascii"),
            candidate.encode("utf-8"),
        )
    )
    return hmac.new(
        _key_bytes(structure_key_hex), message, hashlib.sha256
    ).digest()


def _hard_rank(
    structure_key_hex: str,
    *,
    world_uid: str,
    asset_kind: str,
    asset_index: int,
    candidates: Sequence[str],
) -> list[str]:
    values = [str(value) for value in candidates]
    if not values or len(values) != len(set(values)):
        raise IndependentReplayError(
            f"REPLAY_HARD_CANDIDATE_POOL_INVALID:{asset_kind}"
        )
    ranked = [
        (
            _hard_digest(
                structure_key_hex,
                world_uid=world_uid,
                asset_kind=asset_kind,
                asset_index=asset_index,
                candidate=candidate,
            ),
            candidate.encode("utf-8"),
            candidate,
        )
        for candidate in values
    ]
    return [candidate for _digest, _raw, candidate in sorted(ranked)]


def _hub_seller(
    structure_key_hex: str,
    *,
    world_uid: str,
    asset_kind: str,
    asset_index: int,
    controller_uid: str,
    members: Sequence[str],
) -> str:
    key = _key_bytes(structure_key_hex)

    def rank(seller_uid: str) -> tuple[bytes, bytes]:
        message = FIELD_SEPARATOR.join(
            (
                world_uid.encode("utf-8"),
                b"hard_negative_hub_seller",
                asset_kind.encode("ascii"),
                str(asset_index).encode("ascii"),
                controller_uid.encode("utf-8"),
                seller_uid.encode("utf-8"),
            )
        )
        return (
            hmac.new(key, message, hashlib.sha256).digest(),
            seller_uid.encode("utf-8"),
        )

    return min((str(value) for value in members), key=rank)


def _iter_hard_negative_leaves(
    policy: Mapping[str, Any],
    *,
    world_uid: str,
    graph_name: str,
    structure_key_hex: str,
    membership: Mapping[str, Any],
    mechanisms: Mapping[str, Mapping[str, str]],
) -> Iterator[
    tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]
]:
    design = policy["identity_design"]["hard_negative_dgp"][graph_name]
    maximum_membership_nodes = int(
        policy["identity_design"]["hard_negative_generator_contract"][
            "common"
        ]["membership_solver"]["maximum_search_nodes"]
    )
    if maximum_membership_nodes <= 0:
        raise IndependentReplayError(
            "REPLAY_HARD_MEMBERSHIP_SEARCH_BUDGET_INVALID"
        )
    controller_members = membership["controller_members"]
    zero_controllers = {
        controller_uid
        for controller_uid, row in mechanisms.items()
        if row["mechanism"] == "zero_visible_identity_history"
    }
    zero_sellers = {
        seller_uid
        for controller_uid in zero_controllers
        for seller_uid in controller_members[controller_uid]
    }
    eligible_controllers = [
        controller_uid
        for controller_uid in _utf8_sorted(
            list(membership["controller_uids"])
        )
        if controller_uid not in zero_controllers
    ]
    eligible_sellers = [
        seller_uid
        for seller_uid in membership["seller_uids"]
        if seller_uid not in zero_sellers
    ]
    seller_to_controller = membership["seller_to_controller"]
    direct_types = [
        str(value)
        for value in policy["identity_design"]["parser_observable_role_types"][
            "direct_or_private"
        ]
    ]
    support_types = [
        str(value)
        for value in policy["identity_design"]["parser_observable_role_types"][
            "public_support"
        ]
    ]
    risky_types = [
        str(value)
        for value in policy["identity_design"]["parser_observable_role_types"][
            "risky_product"
        ]
    ]
    assets: list[dict[str, Any]] = []
    flags: list[dict[str, Any]] = []

    def select_hub_sellers(
        kind: str, asset_index: int, degree: int
    ) -> list[str]:
        controller_subsets = [
            TUPLE_SEPARATOR.join(subset)
            for subset in itertools.combinations(
                eligible_controllers, int(degree)
            )
        ]
        selected_controllers = _hard_rank(
            structure_key_hex,
            world_uid=world_uid,
            asset_kind=kind,
            asset_index=asset_index,
            candidates=controller_subsets,
        )[0].split(TUPLE_SEPARATOR)
        return [
            _hub_seller(
                structure_key_hex,
                world_uid=world_uid,
                asset_kind=kind,
                asset_index=asset_index,
                controller_uid=controller_uid,
                members=controller_members[controller_uid],
            )
            for controller_uid in selected_controllers
        ]

    for asset_index, degree in enumerate(design["support_hub_degrees"]):
        sellers = select_hub_sellers("support_hub", asset_index, int(degree))
        assets.append(
            _asset(
                descriptor_kind="support_hub",
                descriptor_index=str(asset_index),
                role="public_support",
                sellers=sellers,
                occurrence_counts={seller: 1 for seller in sellers},
                allowed_types=support_types,
            )
        )
        for left, right in itertools.combinations(sellers, 2):
            flags.append(
                {
                    "canonical_pair_uid": _pair_uid(left, right),
                    "flag": "support_hub_pair",
                    "asset_index": asset_index,
                }
            )

    background_for_capacity = _background_assets(
        policy,
        world_uid=world_uid,
        structure_key_hex=structure_key_hex,
        seller_uids=membership["seller_uids"],
    )
    fixed_background_demand: defaultdict[tuple[str, str], int] = (
        defaultdict(int)
    )
    for background_asset in background_for_capacity:
        fixed_type = str(background_asset["fixed_type"])
        for seller_uid, count in background_asset[
            "occurrence_counts"
        ].items():
            fixed_background_demand[(seller_uid, fixed_type)] += int(
                count
            )
    fixed_type_capacity = int(
        policy["identity_design"]["slot_feasibility"]["type_assignment"][
            "fixed_per_seller_type_capacity"
        ]
    )

    def direct_hub_has_capacity(seller_uids: Sequence[str]) -> bool:
        for identity_type in direct_types:
            if all(
                fixed_background_demand[(seller_uid, identity_type)] + 1
                <= fixed_type_capacity
                for seller_uid in seller_uids
            ):
                return True
        return False

    for asset_index, degree in enumerate(
        design["high_frequency_direct_hub_degrees"]
    ):
        controller_subsets = [
            TUPLE_SEPARATOR.join(subset)
            for subset in itertools.combinations(
                eligible_controllers,
                int(degree),
            )
        ]
        ranked_subsets = _hard_rank(
            structure_key_hex,
            world_uid=world_uid,
            asset_kind="high_frequency_direct_hub",
            asset_index=asset_index,
            candidates=controller_subsets,
        )
        sellers: list[str] | None = None
        for serialized_subset in ranked_subsets:
            selected_controllers = serialized_subset.split(TUPLE_SEPARATOR)
            primary = [
                _hub_seller(
                    structure_key_hex,
                    world_uid=world_uid,
                    asset_kind="high_frequency_direct_hub",
                    asset_index=asset_index,
                    controller_uid=controller_uid,
                    members=controller_members[controller_uid],
                )
                for controller_uid in selected_controllers
            ]
            if direct_hub_has_capacity(primary):
                sellers = primary
                break
            feasible_serializations = [
                TUPLE_SEPARATOR.join(_utf8_sorted(list(proposal)))
                for proposal in itertools.product(
                    *(
                        controller_members[controller_uid]
                        for controller_uid in selected_controllers
                    )
                )
                if direct_hub_has_capacity(proposal)
            ]
            if feasible_serializations:
                sellers = _hard_rank(
                    structure_key_hex,
                    world_uid=world_uid,
                    asset_kind=(
                        "high_frequency_direct_hub_capacity_fallback"
                    ),
                    asset_index=asset_index,
                    candidates=feasible_serializations,
                )[0].split(TUPLE_SEPARATOR)
                break
        if sellers is None:
            raise IndependentReplayError(
                "REPLAY_HIGH_FREQUENCY_HUB_CAPACITY_INFEASIBLE"
            )
        assets.append(
            _asset(
                descriptor_kind="high_frequency_direct_hub",
                descriptor_index=str(asset_index),
                role="high_frequency_direct",
                sellers=sellers,
                occurrence_counts={seller: 1 for seller in sellers},
                allowed_types=direct_types,
            )
        )
        for left, right in itertools.combinations(sellers, 2):
            flags.append(
                {
                    "canonical_pair_uid": _pair_uid(left, right),
                    "flag": "high_frequency_direct_hub_pair",
                    "asset_index": asset_index,
                }
            )

    risky_needed = int(
        policy["identity_design"]["risk_seller_count_per_world"][graph_name]
    )
    risky_controller_order = _hard_rank(
        structure_key_hex,
        world_uid=world_uid,
        asset_kind="risky_shared_scaffold",
        asset_index=0,
        candidates=eligible_controllers,
    )[:risky_needed]
    if len(risky_controller_order) != risky_needed:
        raise IndependentReplayError("REPLAY_RISKY_CONTROLLER_COUNT_INVALID")
    risky_sellers = [
        _hub_seller(
            structure_key_hex,
            world_uid=world_uid,
            asset_kind="risky_shared_scaffold",
            asset_index=0,
            controller_uid=controller_uid,
            members=controller_members[controller_uid],
        )
        for controller_uid in risky_controller_order
    ]
    risky_sets = (
        [risky_sellers[:3], risky_sellers[3:6]]
        if graph_name == "G_A"
        else [
            risky_sellers[0:4],
            risky_sellers[3:7],
            risky_sellers[6:10],
        ]
    )
    for asset_index, sellers in enumerate(risky_sets):
        assets.append(
            _asset(
                descriptor_kind="risky_shared_token",
                descriptor_index=str(asset_index),
                role="risky_product",
                sellers=sellers,
                occurrence_counts={seller: 1 for seller in sellers},
                allowed_types=risky_types,
            )
        )
        for left, right in itertools.combinations(sellers, 2):
            flags.append(
                {
                    "canonical_pair_uid": _pair_uid(left, right),
                    "flag": "risky_shared_token_pair",
                    "asset_index": asset_index,
                }
            )
    if len(set(risky_sellers)) != risky_needed:
        raise IndependentReplayError("REPLAY_RISKY_SELLER_UNION_INVALID")

    typed_membership_kinds = (
        ["private_collision"] * int(design["private_collision_edges"])
        + ["false_rotation"] * int(design["false_rotation_paths"])
    )
    observed_override_kinds = (
        ["exact_title_clone"]
        * int(design["cross_controller_exact_title_clone_pairs"])
        + ["high_semantic_similarity"]
        * int(design["cross_controller_high_semantic_similarity_pairs"])
    )
    selected_specs: list[dict[str, Any]] = []
    designated_pairs: set[str] = set()
    override_sellers: set[str] = set()
    membership_node_count = 0
    complete_leaf_count = 0
    candidate_cache: dict[tuple[str, int], list[dict[str, Any]]] = {}

    def candidates_for(kind: str, index: int) -> list[dict[str, Any]]:
        cache_key = (kind, index)
        cached = candidate_cache.get(cache_key)
        if cached is not None:
            return cached
        if kind in {
            "private_collision",
            "exact_title_clone",
            "high_semantic_similarity",
        }:
            pool = (
                eligible_sellers
                if kind == "private_collision"
                else list(membership["seller_uids"])
            )
            candidates = [
                {
                    "serialization": TUPLE_SEPARATOR.join(_pair(left, right)),
                    "endpoints": _pair(left, right),
                }
                for left, right in itertools.combinations(pool, 2)
                if seller_to_controller[left] != seller_to_controller[right]
            ]
        elif kind == "false_rotation":
            candidates = []
            for controllers in itertools.combinations(
                eligible_controllers, 3
            ):
                chosen = [
                    _hub_seller(
                        structure_key_hex,
                        world_uid=world_uid,
                        asset_kind="false_rotation",
                        asset_index=index,
                        controller_uid=controller_uid,
                        members=controller_members[controller_uid],
                    )
                    for controller_uid in controllers
                ]
                for middle in _utf8_sorted(chosen):
                    left, right = _utf8_sorted(
                        [seller for seller in chosen if seller != middle]
                    )
                    candidates.append(
                        {
                            "serialization": TUPLE_SEPARATOR.join(
                                (left, middle, right)
                            ),
                            "endpoints": _pair(left, right),
                            "ordered": (left, middle, right),
                        }
                    )
        else:
            raise IndependentReplayError(
                f"REPLAY_HARD_REQUEST_KIND_UNKNOWN:{kind}"
            )
        order = _hard_rank(
            structure_key_hex,
            world_uid=world_uid,
            asset_kind=kind,
            asset_index=index,
            candidates=[row["serialization"] for row in candidates],
        )
        indexed = {row["serialization"]: row for row in candidates}
        ranked = [indexed[serialization] for serialization in order]
        candidate_cache[cache_key] = ranked
        return ranked

    def accept_candidate(
        *,
        kind: str,
        index: int,
        candidate: Mapping[str, Any],
        is_override: bool,
    ) -> bool:
        target_uid = _pair_uid(*candidate["endpoints"])
        if target_uid in designated_pairs:
            return False
        endpoint_set = set(candidate["endpoints"])
        if is_override and endpoint_set & override_sellers:
            return False
        designated_pairs.add(target_uid)
        if is_override:
            override_sellers.update(endpoint_set)
        selected_specs.append(
            {"kind": kind, "index": index, **dict(candidate)}
        )
        return True

    def undo_candidate(*, is_override: bool) -> None:
        selected = selected_specs.pop()
        designated_pairs.remove(_pair_uid(*selected["endpoints"]))
        if is_override:
            override_sellers.difference_update(selected["endpoints"])

    def choose_first_override(position: int) -> bool:
        nonlocal membership_node_count, complete_leaf_count
        if position == len(observed_override_kinds):
            return True
        kind = observed_override_kinds[position]
        index = sum(
            prior == kind for prior in observed_override_kinds[:position]
        )
        for candidate in candidates_for(kind, index):
            membership_node_count += 1
            if membership_node_count > maximum_membership_nodes:
                raise IndependentReplayError(
                    "REPLAY_HARD_MEMBERSHIP_SOLVER_EXHAUSTED"
                )
            if not accept_candidate(
                kind=kind,
                index=index,
                candidate=candidate,
                is_override=True,
            ):
                continue
            if choose_first_override(position + 1):
                return True
            undo_candidate(is_override=True)
        return False

    def iter_typed_memberships(
        position: int,
    ) -> Iterator[list[dict[str, Any]]]:
        nonlocal membership_node_count, complete_leaf_count
        if position == len(typed_membership_kinds):
            typed_prefix_length = len(selected_specs)
            if not choose_first_override(0):
                return
            complete_leaf_count += 1
            snapshot = [dict(row) for row in selected_specs]
            while len(selected_specs) > typed_prefix_length:
                undo_candidate(is_override=True)
            yield snapshot
            return
        kind = typed_membership_kinds[position]
        index = sum(
            prior == kind for prior in typed_membership_kinds[:position]
        )
        for candidate in candidates_for(kind, index):
            membership_node_count += 1
            if membership_node_count > maximum_membership_nodes:
                raise IndependentReplayError(
                    "REPLAY_HARD_MEMBERSHIP_SOLVER_EXHAUSTED"
                )
            if not accept_candidate(
                kind=kind,
                index=index,
                candidate=candidate,
                is_override=False,
            ):
                continue
            try:
                yield from iter_typed_memberships(position + 1)
            finally:
                undo_candidate(is_override=False)

    base_assets = [dict(row) for row in assets]
    base_flags = [dict(row) for row in flags]
    emitted = False
    for selected_snapshot in iter_typed_memberships(0):
        emitted = True
        materialized_assets = [dict(row) for row in base_assets]
        materialized_flags = [dict(row) for row in base_flags]
        for spec in selected_snapshot:
            kind = str(spec["kind"])
            asset_index = int(spec["index"])
            left, right = spec["endpoints"]
            if kind == "private_collision":
                materialized_assets.append(
                    _asset(
                        descriptor_kind=kind,
                        descriptor_index=str(asset_index),
                        role="direct_or_private",
                        sellers=[left, right],
                        occurrence_counts={left: 1, right: 1},
                        allowed_types=direct_types,
                    )
                )
                flag = "private_collision_target"
            elif kind == "false_rotation":
                left, middle, right = spec["ordered"]
                for token_index, endpoints in enumerate(
                    ((left, middle), (middle, right))
                ):
                    materialized_assets.append(
                        _asset(
                            descriptor_kind=kind,
                            descriptor_index=(
                                f"{asset_index}{TUPLE_SEPARATOR}"
                                f"{token_index}"
                            ),
                            role="direct_or_private",
                            sellers=endpoints,
                            occurrence_counts={
                                seller: 1 for seller in endpoints
                            },
                            allowed_types=direct_types,
                        )
                    )
                flag = "false_rotation_target"
            elif kind == "exact_title_clone":
                flag = "exact_title_clone_target"
            elif kind == "high_semantic_similarity":
                flag = "high_semantic_similarity_target"
            else:  # pragma: no cover - guarded above
                raise IndependentReplayError(
                    "REPLAY_HARD_KIND_INTERNAL_ERROR"
                )
            materialized_flags.append(
                {
                    "canonical_pair_uid": _pair_uid(left, right),
                    "flag": flag,
                    "asset_index": asset_index,
                }
            )
        selected_ordinal = complete_leaf_count - 1
        yield materialized_assets, materialized_flags, {
            "zero_visible_seller_uids": _utf8_sorted(
                list(zero_sellers)
            ),
            "risk_seller_uids": _utf8_sorted(list(set(risky_sellers))),
            "membership_solver_node_count": membership_node_count,
            "membership_complete_assignments_examined": (
                complete_leaf_count
            ),
            "selected_membership_complete_assignment_ordinal": (
                selected_ordinal
            ),
        }
    if not emitted:
        raise IndependentReplayError(
            "REPLAY_HARD_MEMBERSHIP_ASSIGNMENT_NOT_FOUND"
        )


def _hard_negative_leaf(
    policy: Mapping[str, Any],
    *,
    world_uid: str,
    graph_name: str,
    structure_key_hex: str,
    membership: Mapping[str, Any],
    mechanisms: Mapping[str, Mapping[str, str]],
    complete_assignment_ordinal: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Return one replay topology while preserving the private test API."""

    if complete_assignment_ordinal < 0:
        raise IndependentReplayError(
            "REPLAY_MEMBERSHIP_LEAF_ORDINAL_INVALID"
        )
    for ordinal, result in enumerate(
        _iter_hard_negative_leaves(
            policy,
            world_uid=world_uid,
            graph_name=graph_name,
            structure_key_hex=structure_key_hex,
            membership=membership,
            mechanisms=mechanisms,
        )
    ):
        if ordinal == complete_assignment_ordinal:
            return result
    raise IndependentReplayError(
        "REPLAY_HARD_MEMBERSHIP_ASSIGNMENT_NOT_FOUND"
    )


def _assign_asset_uids(
    policy: Mapping[str, Any],
    *,
    mode: str,
    world_uid: str,
    graph_name: str,
    structure_key_hex: str,
    assets: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    uid_contract = policy["identity_design"]["slot_feasibility"][
        "identity_asset_uid_pool"
    ]
    positive_order = [str(value) for value in uid_contract["positive_mechanism_order"]]
    negative_order = [
        str(value)
        for value in policy["identity_design"][
            "hard_negative_generator_contract"
        ]["common"]["identity_asset_kind_order"]
    ]

    def descriptor_key(asset: Mapping[str, Any]) -> tuple[Any, ...]:
        kind = str(asset["descriptor_kind"])
        parts = str(asset["descriptor_index"]).split(TUPLE_SEPARATOR)
        if kind == "background_private":
            if len(parts) != 2 or not parts[1].isdigit():
                raise IndependentReplayError(
                    "REPLAY_BACKGROUND_DESCRIPTOR_INVALID"
                )
            return (0, parts[0].encode("utf-8"), int(parts[1]))
        if kind in positive_order:
            if len(parts) != 2 or not parts[1].isdigit():
                raise IndependentReplayError(
                    "REPLAY_POSITIVE_DESCRIPTOR_INVALID"
                )
            return (
                1,
                positive_order.index(kind),
                parts[0].encode("utf-8"),
                int(parts[1]),
            )
        if kind in negative_order:
            if not parts or not all(part.isdigit() for part in parts):
                raise IndependentReplayError(
                    "REPLAY_NEGATIVE_DESCRIPTOR_INVALID"
                )
            return (
                2,
                negative_order.index(kind),
                *(int(part) for part in parts),
            )
        raise IndependentReplayError(
            f"REPLAY_ASSET_KIND_UNREGISTERED:{kind}"
        )

    descriptors = [
        (str(row["descriptor_kind"]), str(row["descriptor_index"]))
        for row in assets
    ]
    if len(descriptors) != len(set(descriptors)):
        raise IndependentReplayError("REPLAY_ASSET_DESCRIPTOR_DUPLICATE")
    ordered = sorted((dict(row) for row in assets), key=descriptor_key)
    expected_count = int(uid_contract["expected_used_count_by_graph"][graph_name])
    if len(ordered) != expected_count:
        raise IndependentReplayError("REPLAY_ASSET_USED_COUNT_MISMATCH")
    pool = [
        _base_uid(
            id_key_hex=str(policy["randomness"][mode]["id_key_hex"]),
            entity_kind="identity_asset",
            parent_uid=world_uid,
            ordinal=ordinal,
        )
        for ordinal in range(int(uid_contract["count_per_world"]))
    ]
    key = _key_bytes(structure_key_hex)
    ranked_pool = sorted(
        pool,
        key=lambda asset_uid: (
            hmac.new(
                key,
                FIELD_SEPARATOR.join(
                    (
                        world_uid.encode("utf-8"),
                        b"identity_asset_assignment",
                        asset_uid.encode("utf-8"),
                    )
                ),
                hashlib.sha256,
            ).digest(),
            asset_uid.encode("utf-8"),
        ),
    )
    return [
        {**asset, "identity_asset_uid": asset_uid}
        for asset, asset_uid in zip(
            ordered, ranked_pool[: len(ordered)], strict=True
        )
    ]


def _materialize_asset_repeats(
    policy: Mapping[str, Any],
    *,
    world_uid: str,
    structure_key_hex: str,
    assets: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output: list[dict[str, Any]] = []
    repeat_rows: list[dict[str, Any]] = []
    for row in assets:
        asset = dict(row)
        asset["occurrence_counts"] = dict(row["occurrence_counts"])
        draw_name = asset.get("repeat_draw_name")
        probability = asset.get("repeat_probability")
        if draw_name is None:
            if probability is not None:
                raise IndependentReplayError(
                    "REPLAY_REPEAT_CONTRACT_INCOMPLETE"
                )
            asset["asset_repeat_decision"] = None
        else:
            if probability is None or any(
                int(value) != 1
                for value in asset["occurrence_counts"].values()
            ):
                raise IndependentReplayError(
                    "REPLAY_REPEAT_CONTRACT_INCOMPLETE"
                )
            decision = _bernoulli(
                policy,
                structure_key_hex=structure_key_hex,
                world_uid=world_uid,
                draw_name=str(draw_name),
                subject_uid=str(asset["identity_asset_uid"]),
                probability=probability,
            )
            asset["occurrence_counts"] = {
                seller_uid: 2 if decision else 1
                for seller_uid in asset["sellers"]
            }
            asset["asset_repeat_decision"] = decision
            repeat_rows.append(
                {
                    "decision_kind": str(draw_name),
                    "subject_uid": str(asset["identity_asset_uid"]),
                    "decision": decision,
                }
            )
        output.append(asset)
    return output, repeat_rows


def _rank_identity_types(
    structure_key_hex: str,
    *,
    world_uid: str,
    asset_uid: str,
    allowed_types: Sequence[str],
    global_order: Sequence[str],
) -> list[str]:
    key = _key_bytes(structure_key_hex)
    return sorted(
        (str(value) for value in allowed_types),
        key=lambda identity_type: (
            hmac.new(
                key,
                FIELD_SEPARATOR.join(
                    (
                        world_uid.encode("utf-8"),
                        b"identity_type",
                        asset_uid.encode("utf-8"),
                        identity_type.encode("ascii"),
                    )
                ),
                hashlib.sha256,
            ).digest(),
            global_order.index(identity_type),
        ),
    )


def _assign_identity_types(
    policy: Mapping[str, Any],
    *,
    world_uid: str,
    structure_key_hex: str,
    assets: Sequence[Mapping[str, Any]],
    maximum_nodes: int,
) -> tuple[list[dict[str, Any]] | None, int]:
    global_types = [
        str(value) for value in policy["identity_design"]["identity_types"]
    ]
    capacity = int(
        policy["identity_design"]["slot_feasibility"]["type_assignment"][
            "fixed_per_seller_type_capacity"
        ]
    )
    assigned: dict[str, str] = {}
    used_by_group: defaultdict[str, set[str]] = defaultdict(set)
    seller_type_demand: defaultdict[tuple[str, str], int] = defaultdict(int)

    fixed_assets = [row for row in assets if row["fixed_type"] is not None]
    for row in fixed_assets:
        identity_type = str(row["fixed_type"])
        if identity_type not in row["allowed_types"]:
            raise IndependentReplayError("REPLAY_FIXED_TYPE_OUTSIDE_DOMAIN")
        if any(
            identity_type in used_by_group[group]
            for group in row["distinct_groups"]
        ):
            raise IndependentReplayError(
                "REPLAY_FIXED_TYPE_DISTINCT_GROUP_VIOLATION"
            )
        asset_uid = str(row["identity_asset_uid"])
        assigned[asset_uid] = identity_type
        for group in row["distinct_groups"]:
            used_by_group[str(group)].add(identity_type)
        for seller_uid in row["sellers"]:
            key = (str(seller_uid), identity_type)
            seller_type_demand[key] += int(
                row["occurrence_counts"][seller_uid]
            )
            if seller_type_demand[key] > capacity:
                raise IndependentReplayError(
                    "REPLAY_FIXED_TYPE_CAPACITY_VIOLATION"
                )

    variable_rows = sorted(
        (dict(row) for row in assets if row["fixed_type"] is None),
        key=lambda row: str(row["identity_asset_uid"]).encode("utf-8"),
    )
    variable = {
        str(row["identity_asset_uid"]): row for row in variable_rows
    }
    if len(variable) != len(variable_rows) or maximum_nodes <= 0:
        raise IndependentReplayError("REPLAY_TYPE_SOLVER_INPUT_INVALID")
    type_bit = {
        identity_type: 1 << index
        for index, identity_type in enumerate(global_types)
    }
    sellers = _utf8_sorted(
        list(
            {
                str(seller_uid)
                for row in assets
                for seller_uid in row["sellers"]
            }
        )
    )
    groups = _utf8_sorted(
        list(
            {
                str(group)
                for row in variable_rows
                for group in row["distinct_groups"]
            }
        )
    )
    memo_limit = int(
        policy["identity_design"]["slot_feasibility"]["type_assignment"][
            "maximum_memoized_states"
        ]
    )
    failed_states: set[tuple[Any, ...]] = set()
    node_count = 0

    def feasible(row: Mapping[str, Any], identity_type: str) -> bool:
        return not any(
            identity_type in used_by_group[str(group)]
            for group in row["distinct_groups"]
        ) and all(
            seller_type_demand[(str(seller_uid), identity_type)]
            + int(row["occurrence_counts"][seller_uid])
            <= capacity
            for seller_uid in row["sellers"]
        )

    def options(asset_uid: str) -> list[str]:
        row = variable[asset_uid]
        return [
            identity_type
            for identity_type in _rank_identity_types(
                structure_key_hex,
                world_uid=world_uid,
                asset_uid=asset_uid,
                allowed_types=row["allowed_types"],
                global_order=global_types,
            )
            if feasible(row, identity_type)
        ]

    def mutate(
        row: Mapping[str, Any], identity_type: str, direction: int
    ) -> None:
        for group in row["distinct_groups"]:
            group = str(group)
            if direction > 0:
                used_by_group[group].add(identity_type)
            else:
                used_by_group[group].remove(identity_type)
        for seller_uid in row["sellers"]:
            seller_type_demand[(str(seller_uid), identity_type)] += (
                direction * int(row["occurrence_counts"][seller_uid])
            )

    def state_key(remaining: frozenset[str]) -> tuple[Any, ...]:
        return (
            tuple(sorted(remaining, key=lambda value: value.encode("utf-8"))),
            tuple(
                seller_type_demand[(seller_uid, identity_type)]
                for seller_uid in sellers
                for identity_type in global_types
            ),
            tuple(
                sum(type_bit[value] for value in used_by_group[group])
                for group in groups
            ),
        )

    def propagation(
        remaining: frozenset[str],
        option_map: Mapping[str, list[str]],
    ) -> bool:
        for group in groups:
            members = [
                asset_uid
                for asset_uid in remaining
                if group in variable[asset_uid]["distinct_groups"]
            ]
            for subset_size in range(1, len(members) + 1):
                for subset in itertools.combinations(members, subset_size):
                    union = {
                        identity_type
                        for asset_uid in subset
                        for identity_type in option_map[asset_uid]
                    }
                    if len(union) < subset_size:
                        return False
        for seller_uid in sellers:
            incident = [
                asset_uid
                for asset_uid in remaining
                if seller_uid
                in variable[asset_uid]["occurrence_counts"]
            ]
            if not incident:
                continue
            domains = {
                asset_uid: sum(
                    type_bit[identity_type]
                    for identity_type in option_map[asset_uid]
                )
                for asset_uid in incident
            }
            relevant_masks = {0}
            for domain_mask in domains.values():
                relevant_masks.update(
                    mask | domain_mask for mask in tuple(relevant_masks)
                )
            for subset_mask in sorted(relevant_masks - {0}):
                residual = sum(
                    capacity
                    - seller_type_demand[(seller_uid, identity_type)]
                    for identity_type in global_types
                    if subset_mask & type_bit[identity_type]
                )
                forced = sum(
                    int(
                        variable[asset_uid]["occurrence_counts"][seller_uid]
                    )
                    for asset_uid in incident
                    if domains[asset_uid] & ~subset_mask == 0
                )
                if forced > residual:
                    return False
        return True

    def search(remaining: frozenset[str]) -> bool:
        nonlocal node_count
        if not remaining:
            return True
        key = state_key(remaining)
        if key in failed_states:
            return False
        option_map = {asset_uid: options(asset_uid) for asset_uid in remaining}
        if any(not values for values in option_map.values()) or not propagation(
            remaining, option_map
        ):
            failed_states.add(key)
            return False
        selected = min(
            remaining,
            key=lambda asset_uid: (
                len(option_map[asset_uid]),
                -max(
                    int(
                        variable[asset_uid]["occurrence_counts"][seller_uid]
                    )
                    for seller_uid in variable[asset_uid]["sellers"]
                ),
                -len(variable[asset_uid]["sellers"]),
                asset_uid.encode("utf-8"),
            ),
        )
        row = variable[selected]
        for identity_type in option_map[selected]:
            node_count += 1
            if node_count > maximum_nodes:
                raise IndependentReplayError("REPLAY_TYPE_SOLVER_EXHAUSTED")
            assigned[selected] = identity_type
            mutate(row, identity_type, 1)
            if search(remaining - {selected}):
                return True
            mutate(row, identity_type, -1)
            del assigned[selected]
        failed_states.add(key)
        if len(failed_states) > memo_limit:
            raise IndependentReplayError("REPLAY_TYPE_MEMO_BUDGET_EXHAUSTED")
        return False

    if not search(frozenset(variable)):
        return None, node_count
    output = [
        {**dict(row), "identity_type": assigned[str(row["identity_asset_uid"])]}
        for row in assets
    ]
    return output, node_count


def _validate_observed_item_uid_pools(
    *,
    world_uid: str,
    seller_uids: Sequence[str],
    all_item_rows: Sequence[Mapping[str, str]],
    nonempty_title_rows: Sequence[Mapping[str, str]],
    nonempty_description_rows: Sequence[Mapping[str, str]],
    fixed_type_capacity: int,
) -> dict[str, Any]:
    fields = {"world_uid", "seller_uid", "item_uid"}
    sellers = set(seller_uids)

    def normalize(
        rows: Sequence[Mapping[str, str]], *, label: str
    ) -> list[dict[str, str]]:
        output: list[dict[str, str]] = []
        for raw in rows:
            if set(raw) != fields:
                raise IndependentReplayError(
                    f"REPLAY_{label}_UID_POOL_SCHEMA_INVALID"
                )
            row = {name: str(raw[name]) for name in fields}
            if (
                row["world_uid"] != world_uid
                or row["seller_uid"] not in sellers
                or not row["item_uid"]
            ):
                raise IndependentReplayError(
                    f"REPLAY_{label}_UID_POOL_FOREIGN_KEY_INVALID"
                )
            output.append(row)
        output.sort(
            key=lambda row: (
                row["seller_uid"].encode("utf-8"),
                row["item_uid"].encode("utf-8"),
            )
        )
        keys = {
            (row["world_uid"], row["seller_uid"], row["item_uid"])
            for row in output
        }
        if len(keys) != len(output):
            raise IndependentReplayError(
                f"REPLAY_{label}_UID_POOL_DUPLICATE"
            )
        return output

    all_rows = normalize(all_item_rows, label="ALL_ITEM")
    title_rows = normalize(nonempty_title_rows, label="NONEMPTY_TITLE_ITEM")
    description_rows = normalize(
        nonempty_description_rows,
        label="NONEMPTY_DESCRIPTION_ITEM",
    )
    all_keys = {
        (row["world_uid"], row["seller_uid"], row["item_uid"])
        for row in all_rows
    }
    title_keys = {
        (row["world_uid"], row["seller_uid"], row["item_uid"])
        for row in title_rows
    }
    description_keys = {
        (row["world_uid"], row["seller_uid"], row["item_uid"])
        for row in description_rows
    }
    if not title_keys.issubset(all_keys) or not description_keys.issubset(
        all_keys
    ):
        raise IndependentReplayError("REPLAY_ITEM_MASK_NOT_SUBSET_OF_ALL_ITEMS")
    title_counts: defaultdict[str, int] = defaultdict(int)
    description_counts: defaultdict[str, int] = defaultdict(int)
    for row in title_rows:
        title_counts[row["seller_uid"]] += 1
    for row in description_rows:
        description_counts[row["seller_uid"]] += 1
    if any(title_counts[seller_uid] < 1 for seller_uid in sellers):
        raise IndependentReplayError(
            "REPLAY_NONEMPTY_TITLE_CAPACITY_INSUFFICIENT"
        )
    if any(
        description_counts[seller_uid] < fixed_type_capacity
        for seller_uid in sellers
    ):
        raise IndependentReplayError(
            "REPLAY_NONEMPTY_DESCRIPTION_CAPACITY_INSUFFICIENT"
        )
    return {
        "all_item_rows": all_rows,
        "nonempty_title_rows": title_rows,
        "nonempty_description_rows": description_rows,
        "all_item_uid_pool_sha256": _canonical_sha256(all_rows),
        "nonempty_title_item_uid_pool_sha256": _canonical_sha256(title_rows),
        "nonempty_description_item_uid_pool_sha256": _canonical_sha256(
            description_rows
        ),
    }


def _override_rank(
    structure_key_hex: str,
    *,
    world_uid: str,
    draw_name: str,
    asset_index: int,
    candidates: Sequence[str],
    prefix_atoms: Sequence[str] = (),
) -> list[str]:
    values = [str(value) for value in candidates]
    if not values or len(values) != len(set(values)):
        raise IndependentReplayError(
            f"REPLAY_OVERRIDE_CANDIDATE_POOL_INVALID:{draw_name}"
        )
    prefix = (
        world_uid.encode("utf-8"),
        draw_name.encode("ascii"),
        str(asset_index).encode("ascii"),
        *(str(value).encode("utf-8") for value in prefix_atoms),
    )
    key = _key_bytes(structure_key_hex)
    ranked = [
        (
            hmac.new(
                key,
                FIELD_SEPARATOR.join(
                    (*prefix, candidate.encode("utf-8"))
                ),
                hashlib.sha256,
            ).digest(),
            candidate.encode("utf-8"),
            candidate,
        )
        for candidate in values
    ]
    return [candidate for _digest, _raw, candidate in sorted(ranked)]


def _registered_override_decisions(
    policy: Mapping[str, Any],
    *,
    split: str,
    world_uid: str,
    structure_key_hex: str,
    negative_flags: Sequence[Mapping[str, Any]],
    nonempty_title_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    by_seller: defaultdict[str, list[str]] = defaultdict(list)
    for row in nonempty_title_rows:
        by_seller[str(row["seller_uid"])].append(str(row["item_uid"]))
    reserved: defaultdict[str, set[str]] = defaultdict(set)

    def ordered_endpoints(
        row: Mapping[str, Any], *, draw_name: str
    ) -> tuple[str, str]:
        pair_uid = str(row["canonical_pair_uid"])
        endpoints = pair_uid.split("||")
        if len(endpoints) != 2 or _pair_uid(*endpoints) != pair_uid:
            raise IndependentReplayError(
                "REPLAY_OVERRIDE_CANONICAL_PAIR_INVALID"
            )
        order = _override_rank(
            structure_key_hex,
            world_uid=world_uid,
            draw_name=draw_name,
            asset_index=int(row["asset_index"]),
            candidates=endpoints,
        )
        return order[0], order[1]

    def select_item(
        seller_uid: str,
        *,
        draw_name: str,
        asset_index: int,
        side_name: str,
    ) -> str:
        candidates = [
            item_uid
            for item_uid in by_seller[seller_uid]
            if item_uid not in reserved[seller_uid]
        ]
        if not candidates:
            raise IndependentReplayError(
                "REPLAY_OVERRIDE_ITEM_CAPACITY_EXHAUSTED"
            )
        selected = _override_rank(
            structure_key_hex,
            world_uid=world_uid,
            draw_name=draw_name,
            asset_index=asset_index,
            candidates=candidates,
            prefix_atoms=(side_name,),
        )[0]
        reserved[seller_uid].add(selected)
        return selected

    public_domains = policy["independent_replay_public_domains"]
    categories = [
        str(value)
        for value in public_domains["categories_in_registered_order"]
    ]
    probabilities = [
        float(value)
        for value in public_domains[
            "anonymous_category_rank_probability"
        ]
    ]
    category_products = {
        str(category): [str(value) for value in products]
        for category, products in public_domains["category_products"].items()
    }
    attributes = [str(value) for value in public_domains["attributes"]]
    skeleton_count = int(
        public_domains["title_skeleton_count_by_split"][split]
    )
    if (
        set(category_products) != set(categories)
        or len(categories) != len(probabilities)
        or skeleton_count < 2
    ):
        raise IndependentReplayError(
            "REPLAY_OVERRIDE_PUBLIC_DOMAIN_INVALID"
        )

    output: list[dict[str, Any]] = []
    semantic_flags = sorted(
        (
            row
            for row in negative_flags
            if row["flag"] == "high_semantic_similarity_target"
        ),
        key=lambda row: (
            int(row["asset_index"]),
            str(row["canonical_pair_uid"]).encode("utf-8"),
        ),
    )
    for flag in semantic_flags:
        asset_index = int(flag["asset_index"])
        left_seller, right_seller = ordered_endpoints(
            flag,
            draw_name="high_semantic_side",
        )
        left_item = select_item(
            left_seller,
            draw_name="high_semantic_item",
            asset_index=asset_index,
            side_name="left",
        )
        right_item = select_item(
            right_seller,
            draw_name="high_semantic_item",
            asset_index=asset_index,
            side_name="right",
        )
        category = str(
            _categorical_choice(
                _ReplayRng(
                    structure_key_hex,
                    world_uid,
                    "high_semantic_category",
                    str(asset_index),
                ),
                categories,
                probabilities,
            )
        )
        product = str(
            _ReplayRng(
                structure_key_hex,
                world_uid,
                "high_semantic_product",
                str(asset_index),
            ).choice(category_products[category])
        )
        attribute = str(
            _ReplayRng(
                structure_key_hex,
                world_uid,
                "high_semantic_attribute",
                str(asset_index),
            ).choice(attributes)
        )
        skeleton_candidates = [
            TUPLE_SEPARATOR.join((str(left), str(right)))
            for left in range(skeleton_count)
            for right in range(skeleton_count)
            if left != right
        ]
        skeleton_pair = _override_rank(
            structure_key_hex,
            world_uid=world_uid,
            draw_name="high_semantic_skeleton_pair",
            asset_index=asset_index,
            candidates=skeleton_candidates,
        )[0]
        left_skeleton, right_skeleton = (
            int(value)
            for value in skeleton_pair.split(TUPLE_SEPARATOR)
        )
        output.append(
            {
                "override_kind": "high_semantic_similarity",
                "asset_index": asset_index,
                "canonical_pair_uid": str(flag["canonical_pair_uid"]),
                "seller_uid_left": left_seller,
                "seller_uid_right": right_seller,
                "item_uid_left": left_item,
                "item_uid_right": right_item,
                "category": category,
                "product": product,
                "attribute": attribute,
                "title_skeleton_index_left": left_skeleton,
                "title_skeleton_index_right": right_skeleton,
            }
        )

    clone_flags = sorted(
        (
            row
            for row in negative_flags
            if row["flag"] == "exact_title_clone_target"
        ),
        key=lambda row: (
            int(row["asset_index"]),
            str(row["canonical_pair_uid"]).encode("utf-8"),
        ),
    )
    for flag in clone_flags:
        asset_index = int(flag["asset_index"])
        source_seller, destination_seller = ordered_endpoints(
            flag,
            draw_name="exact_clone_side",
        )
        source_item = select_item(
            source_seller,
            draw_name="exact_clone_item",
            asset_index=asset_index,
            side_name="source",
        )
        destination_item = select_item(
            destination_seller,
            draw_name="exact_clone_item",
            asset_index=asset_index,
            side_name="destination",
        )
        output.append(
            {
                "override_kind": "exact_title_clone",
                "asset_index": asset_index,
                "canonical_pair_uid": str(flag["canonical_pair_uid"]),
                "seller_uid_left": source_seller,
                "seller_uid_right": destination_seller,
                "item_uid_left": source_item,
                "item_uid_right": destination_item,
                "category": None,
                "product": None,
                "attribute": None,
                "title_skeleton_index_left": None,
                "title_skeleton_index_right": None,
            }
        )
    if (
        len(output) != 6
        or len(
            {
                row[key]
                for row in output
                for key in ("seller_uid_left", "seller_uid_right")
            }
        )
        != 12
        or len(
            {
                row[key]
                for row in output
                for key in ("item_uid_left", "item_uid_right")
            }
        )
        != 12
    ):
        raise IndependentReplayError(
            "REPLAY_OVERRIDE_ENDPOINT_UNIQUENESS_INVALID"
        )
    return output


def replay_typed_dgp(
    policy: Mapping[str, Any],
    *,
    mode: str,
    split: str,
    world_uid: str,
    observed_seller_uids: Sequence[str],
    observed_all_item_uid_rows: Sequence[Mapping[str, str]],
    observed_nonempty_title_item_uid_rows: Sequence[Mapping[str, str]],
    observed_nonempty_description_item_uid_rows: Sequence[
        Mapping[str, str]
    ],
    structure_key_hex: str,
) -> dict[str, Any]:
    """Reconstruct the registered typed DGP without producer intermediates."""

    seller_uids, graph_name = _validate_replay_inputs(
        policy,
        mode=mode,
        split=split,
        world_uid=world_uid,
        observed_seller_uids=observed_seller_uids,
        structure_key_hex=structure_key_hex,
    )
    fixed_capacity = int(
        policy["identity_design"]["slot_feasibility"]["type_assignment"][
            "fixed_per_seller_type_capacity"
        ]
    )
    item_pools = _validate_observed_item_uid_pools(
        world_uid=world_uid,
        seller_uids=seller_uids,
        all_item_rows=observed_all_item_uid_rows,
        nonempty_title_rows=observed_nonempty_title_item_uid_rows,
        nonempty_description_rows=(
            observed_nonempty_description_item_uid_rows
        ),
        fixed_type_capacity=fixed_capacity,
    )
    membership = _membership(
        policy,
        mode=mode,
        world_uid=world_uid,
        seller_uids=seller_uids,
        structure_key_hex=structure_key_hex,
    )
    markets, market_proposal_counter = _markets(
        policy,
        world_uid=world_uid,
        structure_key_hex=structure_key_hex,
        membership=membership,
    )
    mechanisms = _mechanisms(
        policy,
        world_uid=world_uid,
        graph_name=graph_name,
        structure_key_hex=structure_key_hex,
        membership=membership,
        markets=markets,
    )
    style_groups = _style_groups(
        policy,
        mode=mode,
        world_uid=world_uid,
        structure_key_hex=structure_key_hex,
        controller_uids=membership["controller_uids"],
    )
    background_assets = _background_assets(
        policy,
        world_uid=world_uid,
        structure_key_hex=structure_key_hex,
        seller_uids=seller_uids,
    )
    positive_assets, positive_targets, path_repeat_rows = _positive_assets(
        policy,
        world_uid=world_uid,
        graph_name=graph_name,
        structure_key_hex=structure_key_hex,
        membership=membership,
        markets=markets,
        mechanisms=mechanisms,
    )

    type_contract = policy["identity_design"]["slot_feasibility"][
        "type_assignment"
    ]
    maximum_type_nodes = int(type_contract["maximum_search_nodes"])
    maximum_leaves = int(type_contract["maximum_membership_complete_assignments"])
    total_type_nodes = 0
    typed_assets: list[dict[str, Any]] | None = None
    final_negative_flags: list[dict[str, Any]] | None = None
    final_membership_audit: dict[str, Any] | None = None
    selected_leaf: int | None = None
    stable_repeat_rows: list[dict[str, Any]] | None = None
    leaves = _iter_hard_negative_leaves(
        policy,
        world_uid=world_uid,
        graph_name=graph_name,
        structure_key_hex=structure_key_hex,
        membership=membership,
        mechanisms=mechanisms,
    )
    for leaf_ordinal, leaf in enumerate(
        itertools.islice(leaves, maximum_leaves)
    ):
        negative_assets, negative_flags, membership_audit = leaf
        if (
            membership_audit[
                "selected_membership_complete_assignment_ordinal"
            ]
            != leaf_ordinal
            or membership_audit[
                "membership_complete_assignments_examined"
            ]
            != leaf_ordinal + 1
        ):
            raise IndependentReplayError(
                "REPLAY_HARD_MEMBERSHIP_ITERATOR_ORDINAL_DRIFT"
            )
        uid_assets = _assign_asset_uids(
            policy,
            mode=mode,
            world_uid=world_uid,
            graph_name=graph_name,
            structure_key_hex=structure_key_hex,
            assets=[
                *background_assets,
                *positive_assets,
                *negative_assets,
            ],
        )
        repeat_assets, candidate_repeat_rows = _materialize_asset_repeats(
            policy,
            world_uid=world_uid,
            structure_key_hex=structure_key_hex,
            assets=uid_assets,
        )
        remaining_nodes = maximum_type_nodes - total_type_nodes
        if remaining_nodes <= 0:
            raise IndependentReplayError(
                "REPLAY_TYPE_CROSS_LEAF_BUDGET_EXHAUSTED"
            )
        candidate_typed, leaf_nodes = _assign_identity_types(
            policy,
            world_uid=world_uid,
            structure_key_hex=structure_key_hex,
            assets=repeat_assets,
            maximum_nodes=remaining_nodes,
        )
        if (
            isinstance(leaf_nodes, bool)
            or not isinstance(leaf_nodes, int)
            or leaf_nodes < 0
            or leaf_nodes > remaining_nodes
        ):
            raise IndependentReplayError(
                "REPLAY_TYPE_SOLVER_NODE_COUNT_INVALID"
            )
        total_type_nodes += leaf_nodes
        if candidate_typed is None:
            continue
        typed_assets = candidate_typed
        final_negative_flags = negative_flags
        final_membership_audit = membership_audit
        selected_leaf = leaf_ordinal
        stable_repeat_rows = candidate_repeat_rows
        break
    if (
        typed_assets is None
        or final_negative_flags is None
        or final_membership_audit is None
        or selected_leaf is None
        or stable_repeat_rows is None
    ):
        raise IndependentReplayError(
            "REPLAY_NO_TYPE_FEASIBLE_MEMBERSHIP_LEAF"
        )

    membership_rows = [
        {
            "world_uid": world_uid,
            "controller_uid": controller_uid,
            "seller_uid": seller_uid,
        }
        for controller_uid in _utf8_sorted(
            list(membership["controller_members"])
        )
        for seller_uid in membership["controller_members"][controller_uid]
    ]
    market_rows = [
        {
            "world_uid": world_uid,
            "seller_uid": seller_uid,
            "market": markets[seller_uid],
        }
        for seller_uid in seller_uids
    ]
    style_rows = [
        {
            "world_uid": world_uid,
            "controller_uid": controller_uid,
            "style_id": style_groups[controller_uid],
        }
        for controller_uid in _utf8_sorted(list(style_groups))
    ]
    mechanism_rows = [
        {
            "world_uid": world_uid,
            "controller_uid": controller_uid,
            "mechanism": mechanisms[controller_uid]["mechanism"],
            "mechanism_slot_uid": mechanisms[controller_uid][
                "mechanism_slot_uid"
            ],
        }
        for controller_uid in _utf8_sorted(list(mechanisms))
    ]
    repeat_rows = [*path_repeat_rows, *stable_repeat_rows]
    repeat_rows.sort(
        key=lambda row: (
            str(row["decision_kind"]).encode("utf-8"),
            str(row["subject_uid"]).encode("utf-8"),
        )
    )
    override_decisions = _registered_override_decisions(
        policy,
        split=split,
        world_uid=world_uid,
        structure_key_hex=structure_key_hex,
        negative_flags=final_negative_flags,
        nonempty_title_rows=item_pools["nonempty_title_rows"],
    )
    solver_trace = {
        "world_uid": world_uid,
        "split": split,
        "graph_name": graph_name,
        "market_proposal_counter": market_proposal_counter,
        **final_membership_audit,
        "selected_membership_complete_assignment_ordinal": selected_leaf,
        "membership_complete_assignments_type_tested": selected_leaf + 1,
        "type_solver_node_count": total_type_nodes,
        "identity_asset_count": len(typed_assets),
        "unused_identity_asset_uid_count": int(
            policy["identity_design"]["slot_feasibility"][
                "identity_asset_uid_pool"
            ]["count_per_world"]
        )
        - len(typed_assets),
    }
    tables = {
        "controller_membership": membership_rows,
        "seller_markets": market_rows,
        "controller_style_groups": style_rows,
        "mechanism_assignments": mechanism_rows,
        "identity_asset_decisions": typed_assets,
        "positive_targets": positive_targets,
        "negative_flags": final_negative_flags,
        "repeat_decisions": repeat_rows,
        "registered_override_decisions": override_decisions,
        "solver_trace": solver_trace,
    }
    ledger = {
        "version": REPLAY_LEDGER_VERSION,
        "scope": REPLAY_SCOPE,
        "mode": mode,
        "world_uid": world_uid,
        "split": split,
        "graph_name": graph_name,
        "observed_uid_pool_audit": {
            "seller_uid_pool_sha256": _canonical_sha256(seller_uids),
            "seller_count": len(seller_uids),
            "all_item_count": len(item_pools["all_item_rows"]),
            "nonempty_title_item_count": len(
                item_pools["nonempty_title_rows"]
            ),
            "nonempty_description_item_count": len(
                item_pools["nonempty_description_rows"]
            ),
            "all_item_uid_pool_sha256": item_pools[
                "all_item_uid_pool_sha256"
            ],
            "nonempty_title_item_uid_pool_sha256": item_pools[
                "nonempty_title_item_uid_pool_sha256"
            ],
            "nonempty_description_item_uid_pool_sha256": item_pools[
                "nonempty_description_item_uid_pool_sha256"
            ],
        },
        "tables": tables,
        "typed_replay_sha256": _canonical_sha256(tables),
        "secret_serialized": False,
        "producer_private_input_used": False,
    }
    ledger["canonical_self_hash"] = _canonical_sha256(ledger)
    return ledger
