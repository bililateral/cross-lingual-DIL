#!/usr/bin/env python3
"""Normative deterministic world structure primitives for Step 28-v13."""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Mapping, Sequence
from decimal import Decimal, ROUND_FLOOR, localcontext
from typing import Any

import step28_v13_common as common


TUPLE_SEPARATOR = "\x1e"
STRUCTURE_PREFIX = b"step28-v13-structure"
ENTITY_PREFIXES = {
    "world": "w",
    "controller": "ctl",
    "seller": "sel",
    "identity_asset": "ias",
    "item": "itm",
    "identity_slot": "slt",
    "noise_slot": "nsl",
}
SPLIT_ORDER = ("train", "development", "audit_a", "audit_b")


def _raw_key(key_hex: str) -> bytes:
    text = str(key_hex)
    if len(text) != 64 or text != text.lower():
        raise common.ContractError("Structure key must be canonical lowercase hex")
    try:
        output = bytes.fromhex(text)
    except ValueError as exc:
        raise common.ContractError("Structure key is not hexadecimal") from exc
    if len(output) != 32:
        raise common.ContractError("Structure key is not 32 bytes")
    return output


def base_uid(
    *,
    key_hex: str,
    entity_kind: str,
    parent_uid_or_mode: str,
    ordinal: int,
) -> str:
    if entity_kind not in ENTITY_PREFIXES:
        raise common.ContractError(f"Unknown Step28-v13 entity kind: {entity_kind}")
    if ordinal < 0 or not parent_uid_or_mode:
        raise common.ContractError("Invalid base UID parent or ordinal")
    message = common.FIELD_SEPARATOR.join(
        (
            b"step28-v13",
            entity_kind.encode("ascii"),
            parent_uid_or_mode.encode("utf-8"),
            str(ordinal).encode("ascii"),
        )
    )
    digest = hmac.new(_raw_key(key_hex), message, hashlib.sha256).hexdigest()
    return f"{ENTITY_PREFIXES[entity_kind]}_{digest}"


def _rank_digest(key_hex: str, message: bytes) -> bytes:
    return hmac.new(_raw_key(key_hex), message, hashlib.sha256).digest()


def _validated_atom(value: str, *, label: str) -> str:
    output = str(value)
    if (
        not output
        or common.FIELD_SEPARATOR.decode("ascii") in output
        or TUPLE_SEPARATOR in output
    ):
        raise common.ContractError(f"Invalid structure atom for {label}")
    return output


def _validated_candidate(value: str) -> str:
    """Validate a scalar or canonical U+001E-delimited candidate."""

    output = str(value)
    if not output or common.FIELD_SEPARATOR.decode("ascii") in output:
        raise common.ContractError("Invalid structure candidate")
    parts = output.split(TUPLE_SEPARATOR)
    if any(not part for part in parts):
        raise common.ContractError("Structure candidate has an empty tuple atom")
    for part in parts:
        _validated_atom(part, label="candidate tuple atom")
    return output


def _candidate_material(
    draw: Mapping[str, Any],
    candidate: str,
    atoms: Mapping[str, str] | None,
) -> bytes:
    candidate_value = _validated_candidate(candidate)
    expected_atoms = list(draw.get("candidate_prefix_atoms", []))
    supplied = dict(atoms or {})
    if set(supplied) != {
        atom for atom in expected_atoms if not atom.startswith("literal:")
    }:
        raise common.ContractError(
            "Structure draw prefix atoms do not match the registered schema"
        )
    values: list[str] = []
    for atom in expected_atoms:
        if atom.startswith("literal:"):
            value = atom.split(":", 1)[1]
        else:
            value = supplied[atom]
        values.append(_validated_atom(value, label=atom))
    if values:
        values.append(candidate_value)
        candidate_value = TUPLE_SEPARATOR.join(values)
    return candidate_value.encode("utf-8")


def rank_candidates(
    policy: Mapping[str, Any],
    *,
    structure_key_hex: str,
    world_uid: str,
    draw_name: str,
    candidates: Sequence[str],
    prefix_atoms: Mapping[str, str] | None = None,
) -> list[str]:
    registry = policy["randomness"]["structure_draw_registry"]
    try:
        draw = registry["draws"][draw_name]
    except KeyError as exc:
        raise common.ContractError(f"Unregistered structure draw: {draw_name}") from exc
    if draw["operation"] not in {"choice", "permutation"}:
        raise common.ContractError(f"Draw is not rank based: {draw_name}")
    values = [str(value) for value in candidates]
    if not values or len(values) != len(set(values)):
        raise common.ContractError(f"Structure candidates are empty or duplicated: {draw_name}")
    world = _validated_atom(world_uid, label="world_uid")
    ranked: list[tuple[bytes, bytes, str]] = []
    for candidate in values:
        material = _candidate_material(draw, candidate, prefix_atoms)
        message = common.FIELD_SEPARATOR.join(
            (
                STRUCTURE_PREFIX,
                world.encode("utf-8"),
                draw_name.encode("ascii"),
                material,
            )
        )
        ranked.append(
            (
                _rank_digest(structure_key_hex, message),
                candidate.encode("utf-8"),
                candidate,
            )
        )
    return [candidate for _digest, _bytes, candidate in sorted(ranked)]


def choose_candidate(
    policy: Mapping[str, Any],
    *,
    structure_key_hex: str,
    world_uid: str,
    draw_name: str,
    candidates: Sequence[str],
    prefix_atoms: Mapping[str, str] | None = None,
) -> str:
    draw = policy["randomness"]["structure_draw_registry"]["draws"][draw_name]
    if draw["operation"] != "choice":
        raise common.ContractError(f"Draw is not a choice: {draw_name}")
    return rank_candidates(
        policy,
        structure_key_hex=structure_key_hex,
        world_uid=world_uid,
        draw_name=draw_name,
        candidates=candidates,
        prefix_atoms=prefix_atoms,
    )[0]


def structure_bernoulli(
    policy: Mapping[str, Any],
    *,
    structure_key_hex: str,
    world_uid: str,
    draw_name: str,
    subject_uid: str,
    probability: float | str | Decimal,
) -> bool:
    draw = policy["randomness"]["structure_draw_registry"]["draws"].get(draw_name)
    if draw is None or draw.get("operation") != "bernoulli":
        raise common.ContractError(f"Draw is not a Bernoulli: {draw_name}")
    try:
        exact_probability = Decimal(str(probability))
    except Exception as exc:
        raise common.ContractError("Invalid structure Bernoulli probability") from exc
    if (
        not exact_probability.is_finite()
        or exact_probability < 0
        or exact_probability > 1
    ):
        raise common.ContractError("Invalid structure Bernoulli probability")
    message = common.FIELD_SEPARATOR.join(
        (
            STRUCTURE_PREFIX,
            _validated_atom(world_uid, label="world_uid").encode("utf-8"),
            draw_name.encode("ascii"),
            _validated_atom(subject_uid, label="subject_uid").encode("utf-8"),
        )
    )
    value = int.from_bytes(
        _rank_digest(structure_key_hex, message)[:8],
        "big",
        signed=False,
    )
    with localcontext() as context:
        context.prec = 80
        threshold = int(
            (exact_probability * Decimal(1 << 64)).to_integral_value(
                rounding=ROUND_FLOOR
            )
        )
    return value < threshold


def build_mode_world_pool(
    policy: Mapping[str, Any], *, mode: str
) -> list[dict[str, Any]]:
    stream = policy["randomness"][mode]
    counts = policy["modes"][mode]["world_counts"]
    total = sum(int(counts[split]) for split in SPLIT_ORDER)
    worlds = [
        {
            "world_uid": base_uid(
                key_hex=stream["id_key_hex"],
                entity_kind="world",
                parent_uid_or_mode=mode,
                ordinal=ordinal,
            ),
            "mode_global_ordinal": ordinal,
        }
        for ordinal in range(total)
    ]
    namespace_key = _raw_key(stream["id_namespace_key_hex"])
    ranked = sorted(
        worlds,
        key=lambda row: (
            hmac.new(
                namespace_key,
                b"world_split_assignment"
                + common.FIELD_SEPARATOR
                + row["world_uid"].encode("utf-8"),
                hashlib.sha256,
            ).digest(),
            row["world_uid"].encode("utf-8"),
        ),
    )
    cursor = 0
    output: list[dict[str, Any]] = []
    for split in SPLIT_ORDER:
        count = int(counts[split])
        for split_ordinal, row in enumerate(ranked[cursor : cursor + count]):
            output.append(
                {
                    **row,
                    "split": split,
                    "split_ordinal": split_ordinal,
                }
            )
        cursor += count
    if cursor != total:
        raise common.ContractError("World split slicing did not consume the pool")
    return sorted(output, key=lambda row: row["world_uid"].encode("utf-8"))


def build_world_membership(
    policy: Mapping[str, Any],
    *,
    mode: str,
    world_uid: str,
    structure_key_hex: str,
) -> dict[str, Any]:
    id_key = policy["randomness"][mode]["id_key_hex"]
    controller_uids = [
        base_uid(
            key_hex=id_key,
            entity_kind="controller",
            parent_uid_or_mode=world_uid,
            ordinal=index,
        )
        for index in range(12)
    ]
    seller_uids = [
        base_uid(
            key_hex=id_key,
            entity_kind="seller",
            parent_uid_or_mode=world_uid,
            ordinal=index,
        )
        for index in range(28)
    ]
    ranked_controllers = rank_candidates(
        policy,
        structure_key_hex=structure_key_hex,
        world_uid=world_uid,
        draw_name="controller_partition_order",
        candidates=controller_uids,
    )
    ranked_sellers = rank_candidates(
        policy,
        structure_key_hex=structure_key_hex,
        world_uid=world_uid,
        draw_name="seller_partition_order",
        candidates=seller_uids,
    )
    memberships: dict[str, list[str]] = {}
    seller_to_controller: dict[str, str] = {}
    cursor = 0
    for controller_index, controller_uid in enumerate(ranked_controllers):
        size = 3 if controller_index < 4 else 2
        members = ranked_sellers[cursor : cursor + size]
        cursor += size
        if len(members) != size:
            raise common.ContractError("Seller membership slice underflow")
        memberships[controller_uid] = common.utf8_sort(members)
        for seller_uid in members:
            if seller_uid in seller_to_controller:
                raise common.ContractError("Seller assigned to multiple controllers")
            seller_to_controller[seller_uid] = controller_uid
    if cursor != 28 or set(seller_to_controller) != set(seller_uids):
        raise common.ContractError("World membership is not an exact 28-seller partition")
    return {
        "controller_uids": common.utf8_sort(controller_uids),
        "seller_uids": common.utf8_sort(seller_uids),
        "controller_partition_order": ranked_controllers,
        "seller_partition_order": ranked_sellers,
        "controller_members": memberships,
        "seller_to_controller": seller_to_controller,
    }


def assign_markets(
    policy: Mapping[str, Any],
    *,
    world_uid: str,
    structure_key_hex: str,
    membership: Mapping[str, Any],
) -> tuple[dict[str, str], int]:
    markets = list(policy["world_design"]["markets"])
    maximum = int(
        policy["identity_design"]["controller_size_mechanism_assignment"][
            "seller_markets"
        ]["maximum_proposals"]
    )
    seller_uids = common.utf8_sort(membership["seller_uids"])
    controller_members = membership["controller_members"]
    for proposal in range(maximum):
        assignments = {
            seller_uid: choose_candidate(
                policy,
                structure_key_hex=structure_key_hex,
                world_uid=world_uid,
                draw_name="market_proposal",
                candidates=markets,
                prefix_atoms={
                    "proposal_counter_unpadded_decimal": str(proposal),
                    "seller_uid": seller_uid,
                },
            )
            for seller_uid in seller_uids
        }
        multi_sizes = {
            len(members)
            for members in controller_members.values()
            if len({assignments[seller] for seller in members}) >= 2
        }
        if {2, 3}.issubset(multi_sizes):
            return assignments, proposal
    raise common.ContractError(f"Market proposal exhaustion for {world_uid}")


def _mechanism_name(slot_uid: str) -> str:
    name, marker, ordinal = slot_uid.partition("#")
    if marker != "#" or not ordinal.isdigit() or str(int(ordinal)) != ordinal:
        raise common.ContractError(f"Invalid expanded mechanism slot UID: {slot_uid}")
    return name


def assign_controller_mechanisms(
    policy: Mapping[str, Any],
    *,
    world_uid: str,
    graph_name: str,
    structure_key_hex: str,
    membership: Mapping[str, Any],
    markets: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    design = policy["identity_design"]["controller_size_mechanism_assignment"]
    graph = design[graph_name]
    by_size = {
        2: [
            controller
            for controller, members in membership["controller_members"].items()
            if len(members) == 2
        ],
        3: [
            controller
            for controller, members in membership["controller_members"].items()
            if len(members) == 3
        ],
    }
    output: dict[str, dict[str, Any]] = {}
    cross_slots = (
        [(3, "cross_market_stable_reuse#0"), (2, "cross_market_stable_reuse#1")]
        if graph_name == "G_A"
        else [(2, "cross_market_stable_reuse#0")]
    )
    for size, slot_uid in cross_slots:
        eligible = [
            controller
            for controller in by_size[size]
            if len(
                {
                    markets[seller]
                    for seller in membership["controller_members"][controller]
                }
            )
            >= 2
            and controller not in output
        ]
        ranked = rank_candidates(
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
        if not ranked:
            raise common.ContractError("No eligible cross-market controller")
        controller = ranked[0]
        output[controller] = {
            "mechanism": "cross_market_stable_reuse",
            "mechanism_slot_uid": slot_uid,
        }

    for size, slot_key in (
        (3, "expanded_triad_slot_ids_before_hmac_permutation"),
        (2, "expanded_dyad_slot_ids_before_hmac_permutation"),
    ):
        remaining_controllers = [
            controller for controller in by_size[size] if controller not in output
        ]
        slots = list(graph[slot_key])
        if len(remaining_controllers) != len(slots):
            raise common.ContractError(
                f"Mechanism/controller count mismatch for {graph_name} size {size}"
            )
        atoms = {
            "graph_name_ascii": graph_name,
            "controller_size_unpadded_decimal": str(size),
        }
        controller_order = rank_candidates(
            policy,
            structure_key_hex=structure_key_hex,
            world_uid=world_uid,
            draw_name="mechanism_remaining_controller_order",
            candidates=remaining_controllers,
            prefix_atoms=atoms,
        )
        slot_order = rank_candidates(
            policy,
            structure_key_hex=structure_key_hex,
            world_uid=world_uid,
            draw_name="mechanism_remaining_slot_order",
            candidates=slots,
            prefix_atoms=atoms,
        )
        for controller, slot_uid in zip(controller_order, slot_order, strict=True):
            output[controller] = {
                "mechanism": _mechanism_name(slot_uid),
                "mechanism_slot_uid": slot_uid,
            }
    if set(output) != set(membership["controller_uids"]):
        raise common.ContractError("Not every controller received one mechanism")
    return output


def controller_member_order(
    policy: Mapping[str, Any],
    *,
    world_uid: str,
    controller_uid: str,
    structure_key_hex: str,
    members: Sequence[str],
) -> list[str]:
    return rank_candidates(
        policy,
        structure_key_hex=structure_key_hex,
        world_uid=world_uid,
        draw_name="controller_member_left_middle_right",
        candidates=members,
        prefix_atoms={"controller_uid": controller_uid},
    )


def cross_market_pair(
    policy: Mapping[str, Any],
    *,
    world_uid: str,
    controller_uid: str,
    structure_key_hex: str,
    members: Sequence[str],
    markets: Mapping[str, str],
) -> tuple[str, str]:
    candidates = [
        TUPLE_SEPARATOR.join(common.utf8_sort((left, right)))
        for index, left in enumerate(members)
        for right in members[index + 1 :]
        if markets[left] != markets[right]
    ]
    chosen = choose_candidate(
        policy,
        structure_key_hex=structure_key_hex,
        world_uid=world_uid,
        draw_name="cross_market_seller_pair",
        candidates=candidates,
        prefix_atoms={"controller_uid": controller_uid},
    )
    left, right = chosen.split(TUPLE_SEPARATOR)
    return left, right
