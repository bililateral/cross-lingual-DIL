#!/usr/bin/env python3
"""Construct Step 28-v13 identity assets, hard negatives, types, and item slots."""

from __future__ import annotations

import hashlib
import hmac
import itertools
from collections import defaultdict, deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

import step28_v13_common as common
import step28_v13_identity_values as identity_values
import step28_v13_structure as structure


DIRECT_TYPES = (
    "telegram",
    "email",
    "bat",
    "qq",
    "wechat",
    "phone",
    "crypto_wallet",
)


def pair(left: str, right: str) -> tuple[str, str]:
    return tuple(common.utf8_sort((left, right)))  # type: ignore[return-value]


def pair_uid(left: str, right: str) -> str:
    return common.canonical_pair_uid(left, right)


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
    seller_list = common.utf8_sort(sellers)
    if (
        not seller_list
        or set(seller_list) != set(occurrence_counts)
        or any(int(occurrence_counts[seller]) <= 0 for seller in seller_list)
    ):
        raise common.ContractError("Invalid identity asset seller/occurrence plan")
    if (repeat_draw_name is None) != (repeat_probability is None):
        raise common.ContractError("Identity asset repeat contract is incomplete")
    return {
        "descriptor_kind": descriptor_kind,
        "descriptor_index": descriptor_index,
        "descriptor_uid": common.canonical_sha256(
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
        "allowed_types": list(allowed_types),
        "fixed_type": fixed_type,
        "distinct_groups": list(distinct_groups),
        "repeat_draw_name": repeat_draw_name,
        "repeat_probability": repeat_probability,
    }


def build_background_assets(
    policy: dict[str, Any],
    *,
    world_uid: str,
    structure_key_hex: str,
    seller_uids: Sequence[str],
) -> list[dict[str, Any]]:
    allowed = list(
        policy["identity_design"]["background_private_scaffold"]["allowed_types"]
    )
    offset = int(
        structure.choose_candidate(
            policy,
            structure_key_hex=structure_key_hex,
            world_uid=world_uid,
            draw_name="background_type_offset",
            candidates=[str(value) for value in range(7)],
            prefix_atoms={},
        )
    )
    output: list[dict[str, Any]] = []
    edge_ordinal = 0
    for seller_uid in common.utf8_sort(seller_uids):
        for background_slot in range(2):
            identity_type = allowed[(edge_ordinal + offset) % len(allowed)]
            occurrence_count = 1 + (edge_ordinal % 2)
            output.append(
                _asset(
                    descriptor_kind="background_private",
                    descriptor_index=f"{seller_uid}{structure.TUPLE_SEPARATOR}{background_slot}",
                    role="direct_or_private",
                    sellers=[seller_uid],
                    occurrence_counts={seller_uid: occurrence_count},
                    allowed_types=[identity_type],
                    fixed_type=identity_type,
                    distinct_groups=[f"background::{seller_uid}"],
                )
            )
            edge_ordinal += 1
    if len(output) != 56:
        raise common.ContractError("Background identity asset count drift")
    type_counts = defaultdict(int)
    type_count_counts = defaultdict(int)
    for asset in output:
        type_counts[asset["fixed_type"]] += 1
        type_count_counts[
            (
                asset["fixed_type"],
                int(asset["occurrence_counts"][asset["sellers"][0]]),
            )
        ] += 1
    if (
        set(type_counts.values()) != {8}
        or set(type_count_counts.values()) != {4}
        or len(type_count_counts) != 14
    ):
        raise common.ContractError("Background identity type balance drift")
    seller_count_multisets = defaultdict(list)
    for asset in output:
        seller_uid = str(asset["sellers"][0])
        seller_count_multisets[seller_uid].append(
            int(asset["occurrence_counts"][seller_uid])
        )
    if any(
        sorted(seller_counts) != [1, 2]
        for seller_counts in seller_count_multisets.values()
    ):
        raise common.ContractError(
            "Background seller occurrence-count balance drift"
        )
    return output


def build_positive_assets(
    policy: dict[str, Any],
    *,
    world_uid: str,
    graph_name: str,
    structure_key_hex: str,
    membership: Mapping[str, Any],
    markets: Mapping[str, str],
    mechanisms: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    assets: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    stable_probability = policy["identity_design"]["stable_identity_repeat_probability"][
        graph_name
    ]
    rotation_probabilities = policy["identity_design"][
        "rotation_occurrence_probabilities"
    ][graph_name]
    target_mechanism_order = policy["identity_design"][
        "mechanism_traversal_order"
    ]
    controllers = sorted(
        mechanisms,
        key=lambda controller: (
            target_mechanism_order.index(mechanisms[controller]["mechanism"]),
            controller.encode("utf-8"),
        ),
    )
    for controller_uid in controllers:
        mechanism = str(mechanisms[controller_uid]["mechanism"])
        members = list(membership["controller_members"][controller_uid])
        base_index = controller_uid
        if len(members) == 2:
            designated = pair(*members)
        else:
            designated = None

        if mechanism == "single_identity_stable_reuse":
            assets.append(
                _asset(
                    descriptor_kind=mechanism,
                    descriptor_index=f"{base_index}{structure.TUPLE_SEPARATOR}0",
                    role="direct_or_private",
                    sellers=members,
                    occurrence_counts={seller: 1 for seller in members},
                    allowed_types=DIRECT_TYPES,
                    repeat_draw_name="stable_identity_repeat",
                    repeat_probability=float(stable_probability),
                )
            )
        elif mechanism == "multi_type_identity_reuse":
            group = f"multi::{controller_uid}"
            for asset_index in range(2):
                subject = f"{controller_uid}{structure.TUPLE_SEPARATOR}{asset_index}"
                assets.append(
                    _asset(
                        descriptor_kind=mechanism,
                        descriptor_index=subject,
                        role="direct_or_private",
                        sellers=members,
                        occurrence_counts={seller: 1 for seller in members},
                        allowed_types=DIRECT_TYPES,
                        distinct_groups=[group],
                        repeat_draw_name="stable_identity_repeat",
                        repeat_probability=float(stable_probability),
                    )
                )
        elif mechanism == "cross_market_stable_reuse":
            selected = structure.cross_market_pair(
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
                    descriptor_index=f"{base_index}{structure.TUPLE_SEPARATOR}0",
                    role="direct_or_private",
                    sellers=selected,
                    occurrence_counts={seller: 1 for seller in selected},
                    allowed_types=DIRECT_TYPES,
                    repeat_draw_name="stable_identity_repeat",
                    repeat_probability=float(stable_probability),
                )
            )
        elif mechanism == "single_hop_rotation":
            ordered = structure.controller_member_order(
                policy,
                world_uid=world_uid,
                controller_uid=controller_uid,
                structure_key_hex=structure_key_hex,
                members=members,
            )
            left, middle, right = ordered
            designated = pair(left, right)
            repeat_path = structure.structure_bernoulli(
                policy,
                structure_key_hex=structure_key_hex,
                world_uid=world_uid,
                draw_name="single_hop_path_repeat",
                subject_uid=controller_uid,
                probability=rotation_probabilities[
                    "single_hop_any_one_side_repeat_probability"
                ],
            )
            repeat_side = (
                structure.choose_candidate(
                    policy,
                    structure_key_hex=structure_key_hex,
                    world_uid=world_uid,
                    draw_name="single_hop_repeat_side",
                    candidates=["left_middle", "middle_right"],
                    prefix_atoms={"controller_uid": controller_uid},
                )
                if repeat_path
                else ""
            )
            group = f"single_hop::{controller_uid}"
            for asset_index, (side, endpoints) in enumerate(
                (
                    ("left_middle", (left, middle)),
                    ("middle_right", (middle, right)),
                )
            ):
                count = 2 if repeat_side == side else 1
                assets.append(
                    _asset(
                        descriptor_kind=mechanism,
                        descriptor_index=f"{base_index}{structure.TUPLE_SEPARATOR}{asset_index}",
                        role="direct_or_private",
                        sellers=endpoints,
                        occurrence_counts={seller: count for seller in endpoints},
                        allowed_types=DIRECT_TYPES,
                        distinct_groups=[group],
                    )
                )
        elif mechanism == "corroborated_two_hop_rotation":
            ordered = structure.controller_member_order(
                policy,
                world_uid=world_uid,
                controller_uid=controller_uid,
                structure_key_hex=structure_key_hex,
                members=members,
            )
            left, middle, right = ordered
            designated = pair(left, right)
            endpoints_by_index = (
                (left, middle),
                (left, middle),
                (middle, right),
                (middle, right),
            )
            for asset_index, endpoints in enumerate(endpoints_by_index):
                subject = f"{controller_uid}{structure.TUPLE_SEPARATOR}{asset_index}"
                assets.append(
                    _asset(
                        descriptor_kind=mechanism,
                        descriptor_index=subject,
                        role="direct_or_private",
                        sellers=endpoints,
                        occurrence_counts={seller: 1 for seller in endpoints},
                        allowed_types=DIRECT_TYPES,
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
                    descriptor_index=f"{base_index}{structure.TUPLE_SEPARATOR}0",
                    role="direct_or_private",
                    sellers=members,
                    occurrence_counts={seller: 1 for seller in members},
                    allowed_types=DIRECT_TYPES,
                )
            )
        elif mechanism in {
            "same_controller_no_direct_share",
            "zero_visible_identity_history",
        }:
            pass
        else:
            raise common.ContractError(f"Unknown positive mechanism: {mechanism}")

        if designated is None:
            raise common.ContractError(f"Mechanism lacks a designated target: {mechanism}")
        targets.append(
            {
                "controller_uid": controller_uid,
                "mechanism": mechanism,
                "mechanism_slot_uid": mechanisms[controller_uid]["mechanism_slot_uid"],
                "seller_uid_left": designated[0],
                "seller_uid_right": designated[1],
                "canonical_pair_uid": pair_uid(*designated),
            }
        )
    return assets, targets


def _hard_digest(
    structure_key_hex: str,
    *,
    world_uid: str,
    asset_kind: str,
    asset_index: int,
    candidate: str,
) -> bytes:
    message = common.FIELD_SEPARATOR.join(
        (
            world_uid.encode("utf-8"),
            b"hard_negative",
            asset_kind.encode("ascii"),
            str(asset_index).encode("ascii"),
            candidate.encode("utf-8"),
        )
    )
    return hmac.new(bytes.fromhex(structure_key_hex), message, hashlib.sha256).digest()


def _rank_hard_candidates(
    structure_key_hex: str,
    *,
    world_uid: str,
    asset_kind: str,
    asset_index: int,
    candidates: Sequence[str],
) -> list[str]:
    return [
        candidate
        for _digest, _bytes, candidate in sorted(
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
            for candidate in candidates
        )
    ]


def _hub_seller(
    structure_key_hex: str,
    *,
    world_uid: str,
    asset_kind: str,
    asset_index: int,
    controller_uid: str,
    members: Sequence[str],
) -> str:
    return min(
        members,
        key=lambda seller: (
            hmac.new(
                bytes.fromhex(structure_key_hex),
                common.FIELD_SEPARATOR.join(
                    (
                        world_uid.encode("utf-8"),
                        b"hard_negative_hub_seller",
                        asset_kind.encode("ascii"),
                        str(asset_index).encode("ascii"),
                        controller_uid.encode("utf-8"),
                        seller.encode("utf-8"),
                    )
                ),
                hashlib.sha256,
            ).digest(),
            seller.encode("utf-8"),
        ),
    )


def _iter_hard_negative_plans(
    policy: dict[str, Any],
    *,
    world_uid: str,
    graph_name: str,
    structure_key_hex: str,
    membership: Mapping[str, Any],
    mechanisms: Mapping[str, Mapping[str, Any]],
) -> Iterator[
    tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]
]:
    dgp = policy["identity_design"]["hard_negative_dgp"][graph_name]
    maximum_membership_nodes = int(
        policy["identity_design"]["hard_negative_generator_contract"][
            "common"
        ]["membership_solver"]["maximum_search_nodes"]
    )
    if maximum_membership_nodes <= 0:
        raise common.ContractError(
            "Hard-negative membership search budget must be positive"
        )
    controller_members = membership["controller_members"]
    zero_controllers = {
        controller
        for controller, record in mechanisms.items()
        if record["mechanism"] == "zero_visible_identity_history"
    }
    zero_sellers = {
        seller
        for controller in zero_controllers
        for seller in controller_members[controller]
    }
    eligible_controllers = [
        controller
        for controller in common.utf8_sort(controller_members)
        if controller not in zero_controllers
    ]
    eligible_sellers = [
        seller
        for seller in membership["seller_uids"]
        if seller not in zero_sellers
    ]
    seller_to_controller = membership["seller_to_controller"]
    assets: list[dict[str, Any]] = []
    flags: list[dict[str, Any]] = []

    for asset_index, degree in enumerate(dgp["support_hub_degrees"]):
        candidates = [
            structure.TUPLE_SEPARATOR.join(subset)
            for subset in itertools.combinations(eligible_controllers, int(degree))
        ]
        chosen = _rank_hard_candidates(
            structure_key_hex,
            world_uid=world_uid,
            asset_kind="support_hub",
            asset_index=asset_index,
            candidates=candidates,
        )[0].split(structure.TUPLE_SEPARATOR)
        sellers = [
            _hub_seller(
                structure_key_hex,
                world_uid=world_uid,
                asset_kind="support_hub",
                asset_index=asset_index,
                controller_uid=controller,
                members=controller_members[controller],
            )
            for controller in chosen
        ]
        assets.append(
            _asset(
                descriptor_kind="support_hub",
                descriptor_index=str(asset_index),
                role="public_support",
                sellers=sellers,
                occurrence_counts={seller: 1 for seller in sellers},
                allowed_types=["email", "phone", "external_url"],
            )
        )
        for left, right in itertools.combinations(sellers, 2):
            flags.append(
                {
                    "canonical_pair_uid": pair_uid(left, right),
                    "flag": "support_hub_pair",
                    "asset_index": asset_index,
                }
            )

    background_assets = build_background_assets(
        policy,
        world_uid=world_uid,
        structure_key_hex=structure_key_hex,
        seller_uids=membership["seller_uids"],
    )
    fixed_background_demand: defaultdict[tuple[str, str], int] = (
        defaultdict(int)
    )
    for background_asset in background_assets:
        identity_type = str(background_asset["fixed_type"])
        for seller_uid, count in background_asset[
            "occurrence_counts"
        ].items():
            fixed_background_demand[(seller_uid, identity_type)] += int(
                count
            )
    fixed_type_capacity = int(
        policy["identity_design"]["slot_feasibility"]["type_assignment"][
            "fixed_per_seller_type_capacity"
        ]
    )

    def direct_hub_is_background_feasible(
        seller_uids: Sequence[str],
    ) -> bool:
        return any(
            all(
                fixed_background_demand[(seller_uid, identity_type)] + 1
                <= fixed_type_capacity
                for seller_uid in seller_uids
            )
            for identity_type in DIRECT_TYPES
        )

    for asset_index, degree in enumerate(
        dgp["high_frequency_direct_hub_degrees"]
    ):
        controller_candidates = [
            structure.TUPLE_SEPARATOR.join(subset)
            for subset in itertools.combinations(
                eligible_controllers,
                int(degree),
            )
        ]
        ranked_controller_candidates = _rank_hard_candidates(
            structure_key_hex,
            world_uid=world_uid,
            asset_kind="high_frequency_direct_hub",
            asset_index=asset_index,
            candidates=controller_candidates,
        )
        sellers: list[str] | None = None
        for serialized_controllers in ranked_controller_candidates:
            chosen_controllers = serialized_controllers.split(
                structure.TUPLE_SEPARATOR
            )
            primary_sellers = [
                _hub_seller(
                    structure_key_hex,
                    world_uid=world_uid,
                    asset_kind="high_frequency_direct_hub",
                    asset_index=asset_index,
                    controller_uid=controller,
                    members=controller_members[controller],
                )
                for controller in chosen_controllers
            ]
            if direct_hub_is_background_feasible(primary_sellers):
                sellers = primary_sellers
                break
            fallback_candidates = [
                structure.TUPLE_SEPARATOR.join(
                    common.utf8_sort(candidate_sellers)
                )
                for candidate_sellers in itertools.product(
                    *(
                        controller_members[controller]
                        for controller in chosen_controllers
                    )
                )
                if direct_hub_is_background_feasible(candidate_sellers)
            ]
            if fallback_candidates:
                sellers = _rank_hard_candidates(
                    structure_key_hex,
                    world_uid=world_uid,
                    asset_kind=(
                        "high_frequency_direct_hub_capacity_fallback"
                    ),
                    asset_index=asset_index,
                    candidates=fallback_candidates,
                )[0].split(structure.TUPLE_SEPARATOR)
                break
        if sellers is None:
            raise common.ContractError(
                "No background-capacity-feasible high-frequency direct hub"
            )
        assets.append(
            _asset(
                descriptor_kind="high_frequency_direct_hub",
                descriptor_index=str(asset_index),
                role="high_frequency_direct",
                sellers=sellers,
                occurrence_counts={seller: 1 for seller in sellers},
                allowed_types=DIRECT_TYPES,
            )
        )
        for left, right in itertools.combinations(sellers, 2):
            flags.append(
                {
                    "canonical_pair_uid": pair_uid(left, right),
                    "flag": "high_frequency_direct_hub_pair",
                    "asset_index": asset_index,
                }
            )

    risky_needed = int(policy["identity_design"]["risk_seller_count_per_world"][graph_name])
    controller_order = _rank_hard_candidates(
        structure_key_hex,
        world_uid=world_uid,
        asset_kind="risky_shared_scaffold",
        asset_index=0,
        candidates=eligible_controllers,
    )[:risky_needed]
    if len(controller_order) != risky_needed:
        raise common.ContractError("Insufficient risky-shared eligible controllers")
    risky_sellers = [
        _hub_seller(
            structure_key_hex,
            world_uid=world_uid,
            asset_kind="risky_shared_scaffold",
            asset_index=0,
            controller_uid=controller,
            members=controller_members[controller],
        )
        for controller in controller_order
    ]
    risky_sets = (
        [risky_sellers[0:3], risky_sellers[3:6]]
        if graph_name == "G_A"
        else [risky_sellers[0:4], risky_sellers[3:7], risky_sellers[6:10]]
    )
    for asset_index, sellers in enumerate(risky_sets):
        assets.append(
            _asset(
                descriptor_kind="risky_shared_token",
                descriptor_index=str(asset_index),
                role="risky_product",
                sellers=sellers,
                occurrence_counts={seller: 1 for seller in sellers},
                allowed_types=list(policy["identity_design"]["identity_types"]),
            )
        )
        for left, right in itertools.combinations(sellers, 2):
            flags.append(
                {
                    "canonical_pair_uid": pair_uid(left, right),
                    "flag": "risky_shared_token_pair",
                    "asset_index": asset_index,
                }
            )
    if len(set(risky_sellers)) != risky_needed:
        raise common.ContractError("Risk seller union count drift")

    designated_used: set[str] = set()
    override_sellers_used: set[str] = set()
    selected_specs: list[dict[str, Any]] = []
    identity_request_kinds = (
        ["private_collision"] * int(dgp["private_collision_edges"])
        + ["false_rotation"] * int(dgp["false_rotation_paths"])
    )
    override_request_kinds = (
        ["exact_title_clone"]
        * int(dgp["cross_controller_exact_title_clone_pairs"])
        + ["high_semantic_similarity"]
        * int(dgp["cross_controller_high_semantic_similarity_pairs"])
    )
    node_count = 0
    complete_assignment_count = 0
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
            sellers = (
                eligible_sellers
                if kind == "private_collision"
                else list(membership["seller_uids"])
            )
            candidates = [
                {
                    "serialization": structure.TUPLE_SEPARATOR.join(pair(left, right)),
                    "endpoints": pair(left, right),
                }
                for left, right in itertools.combinations(sellers, 2)
                if seller_to_controller[left] != seller_to_controller[right]
            ]
        else:
            candidates = []
            for controllers in itertools.combinations(eligible_controllers, 3):
                chosen = [
                    _hub_seller(
                        structure_key_hex,
                        world_uid=world_uid,
                        asset_kind="false_rotation",
                        asset_index=index,
                        controller_uid=controller,
                        members=controller_members[controller],
                    )
                    for controller in controllers
                ]
                for middle in common.utf8_sort(chosen):
                    left, right = common.utf8_sort(
                        seller for seller in chosen if seller != middle
                    )
                    candidates.append(
                        {
                            "serialization": structure.TUPLE_SEPARATOR.join(
                                (left, middle, right)
                            ),
                            "endpoints": pair(left, right),
                            "ordered": (left, middle, right),
                        }
                    )
        rank = _rank_hard_candidates(
            structure_key_hex,
            world_uid=world_uid,
            asset_kind=kind,
            asset_index=index,
            candidates=[row["serialization"] for row in candidates],
        )
        by_serialization = {row["serialization"]: row for row in candidates}
        ranked = [by_serialization[value] for value in rank]
        candidate_cache[cache_key] = ranked
        return ranked

    def push_candidate(
        *,
        kind: str,
        index: int,
        candidate: Mapping[str, Any],
        observed_override: bool,
    ) -> bool:
        target_uid = pair_uid(*candidate["endpoints"])
        if target_uid in designated_used:
            return False
        endpoint_set = set(candidate["endpoints"])
        if observed_override and endpoint_set & override_sellers_used:
            return False
        designated_used.add(target_uid)
        if observed_override:
            override_sellers_used.update(endpoint_set)
        selected_specs.append(
            {"kind": kind, "index": index, **dict(candidate)}
        )
        return True

    def pop_candidate(*, observed_override: bool) -> None:
        candidate = selected_specs.pop()
        designated_used.remove(pair_uid(*candidate["endpoints"]))
        if observed_override:
            override_sellers_used.difference_update(
                candidate["endpoints"]
            )

    def find_first_override_completion(position: int) -> bool:
        nonlocal node_count, complete_assignment_count
        if position == len(override_request_kinds):
            return True
        kind = override_request_kinds[position]
        index = sum(
            prior == kind for prior in override_request_kinds[:position]
        )
        for candidate in candidates_for(kind, index):
            node_count += 1
            if node_count > maximum_membership_nodes:
                raise common.ContractError("Hard-negative membership solver exhausted")
            if not push_candidate(
                kind=kind,
                index=index,
                candidate=candidate,
                observed_override=True,
            ):
                continue
            if find_first_override_completion(position + 1):
                return True
            pop_candidate(observed_override=True)
        return False

    def iter_identity_topologies(
        position: int,
    ) -> Iterator[list[dict[str, Any]]]:
        nonlocal node_count, complete_assignment_count
        if position == len(identity_request_kinds):
            identity_spec_count = len(selected_specs)
            if not find_first_override_completion(0):
                return
            complete_assignment_count += 1
            snapshot = [dict(row) for row in selected_specs]
            while len(selected_specs) > identity_spec_count:
                pop_candidate(observed_override=True)
            yield snapshot
            return
        kind = identity_request_kinds[position]
        index = sum(
            prior == kind for prior in identity_request_kinds[:position]
        )
        for candidate in candidates_for(kind, index):
            node_count += 1
            if node_count > maximum_membership_nodes:
                raise common.ContractError(
                    "Hard-negative membership solver exhausted"
                )
            if not push_candidate(
                kind=kind,
                index=index,
                candidate=candidate,
                observed_override=False,
            ):
                continue
            try:
                yield from iter_identity_topologies(position + 1)
            finally:
                pop_candidate(observed_override=False)

    base_assets = [dict(row) for row in assets]
    base_flags = [dict(row) for row in flags]
    emitted = False
    for selected_snapshot in iter_identity_topologies(0):
        emitted = True
        materialized_assets = [dict(row) for row in base_assets]
        materialized_flags = [dict(row) for row in base_flags]
        for spec in selected_snapshot:
            kind = spec["kind"]
            index = int(spec["index"])
            left, right = spec["endpoints"]
            if kind == "private_collision":
                materialized_assets.append(
                    _asset(
                        descriptor_kind=kind,
                        descriptor_index=str(index),
                        role="direct_or_private",
                        sellers=[left, right],
                        occurrence_counts={left: 1, right: 1},
                        allowed_types=DIRECT_TYPES,
                    )
                )
                materialized_flags.append(
                    {
                        "canonical_pair_uid": pair_uid(left, right),
                        "flag": "private_collision_target",
                        "asset_index": index,
                    }
                )
            elif kind == "false_rotation":
                left, middle, right = spec["ordered"]
                for token_index, endpoints in enumerate(
                    ((left, middle), (middle, right))
                ):
                    materialized_assets.append(
                        _asset(
                            descriptor_kind=kind,
                            descriptor_index=(
                                f"{index}{structure.TUPLE_SEPARATOR}"
                                f"{token_index}"
                            ),
                            role="direct_or_private",
                            sellers=endpoints,
                            occurrence_counts={
                                seller: 1 for seller in endpoints
                            },
                            allowed_types=DIRECT_TYPES,
                        )
                    )
                materialized_flags.append(
                    {
                        "canonical_pair_uid": pair_uid(left, right),
                        "flag": "false_rotation_target",
                        "asset_index": index,
                    }
                )
            elif kind == "exact_title_clone":
                materialized_flags.append(
                    {
                        "canonical_pair_uid": pair_uid(left, right),
                        "flag": "exact_title_clone_target",
                        "asset_index": index,
                    }
                )
            elif kind == "high_semantic_similarity":
                materialized_flags.append(
                    {
                        "canonical_pair_uid": pair_uid(left, right),
                        "flag": "high_semantic_similarity_target",
                        "asset_index": index,
                    }
                )
        selected_ordinal = complete_assignment_count - 1
        yield materialized_assets, materialized_flags, {
            "zero_visible_seller_uids": common.utf8_sort(zero_sellers),
            "risk_seller_uids": common.utf8_sort(set(risky_sellers)),
            "membership_solver_node_count": node_count,
            "membership_complete_assignments_examined": (
                complete_assignment_count
            ),
            "selected_membership_complete_assignment_ordinal": (
                selected_ordinal
            ),
        }
    if not emitted:
        raise common.ContractError(
            "Hard-negative membership solver found no assignment"
        )


def build_hard_negative_plan(
    policy: dict[str, Any],
    *,
    world_uid: str,
    graph_name: str,
    structure_key_hex: str,
    membership: Mapping[str, Any],
    mechanisms: Mapping[str, Mapping[str, Any]],
    complete_assignment_ordinal: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Return one deterministic topology while preserving the public API."""

    if complete_assignment_ordinal < 0:
        raise common.ContractError(
            "Negative membership solution ordinal is negative"
        )
    for ordinal, result in enumerate(
        _iter_hard_negative_plans(
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
    raise common.ContractError(
        "Hard-negative membership solver found no assignment"
    )


def assign_asset_uids(
    policy: dict[str, Any],
    *,
    mode: str,
    world_uid: str,
    graph_name: str,
    structure_key_hex: str,
    assets_in_descriptor_order: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    positive_order = list(
        policy["identity_design"]["slot_feasibility"]["identity_asset_uid_pool"][
            "positive_mechanism_order"
        ]
    )
    negative_order = list(
        policy["identity_design"]["hard_negative_generator_contract"]["common"][
            "identity_asset_kind_order"
        ]
    )

    def descriptor_key(asset: Mapping[str, Any]) -> tuple[Any, ...]:
        kind = str(asset["descriptor_kind"])
        parts = str(asset["descriptor_index"]).split(structure.TUPLE_SEPARATOR)
        if kind == "background_private":
            if len(parts) != 2 or not parts[1].isdigit():
                raise common.ContractError("Malformed background asset descriptor")
            return (0, parts[0].encode("utf-8"), int(parts[1]))
        if kind in positive_order:
            if len(parts) != 2 or not parts[1].isdigit():
                raise common.ContractError("Malformed positive asset descriptor")
            return (
                1,
                positive_order.index(kind),
                parts[0].encode("utf-8"),
                int(parts[1]),
            )
        if kind in negative_order:
            if not parts or not all(part.isdigit() for part in parts):
                raise common.ContractError("Malformed negative asset descriptor")
            numeric_parts = tuple(int(part) for part in parts)
            return (2, negative_order.index(kind), *numeric_parts)
        raise common.ContractError(f"Unregistered identity asset kind: {kind}")

    descriptors = [
        (str(asset["descriptor_kind"]), str(asset["descriptor_index"]))
        for asset in assets_in_descriptor_order
    ]
    if len(descriptors) != len(set(descriptors)):
        raise common.ContractError("Duplicate structural identity asset descriptor")
    ordered_assets = sorted(assets_in_descriptor_order, key=descriptor_key)
    uid_contract = policy["identity_design"]["slot_feasibility"][
        "identity_asset_uid_pool"
    ]
    expected_used = int(uid_contract["expected_used_count_by_graph"][graph_name])
    if len(ordered_assets) != expected_used:
        raise common.ContractError(
            f"{graph_name} identity asset count drift: "
            f"{len(ordered_assets)} != {expected_used}"
        )
    id_key = policy["randomness"][mode]["id_key_hex"]
    pool_count = int(uid_contract["count_per_world"])
    pool = [
        (
            structure.base_uid(
                key_hex=id_key,
                entity_kind="identity_asset",
                parent_uid_or_mode=world_uid,
                ordinal=index,
            ),
            index,
        )
        for index in range(pool_count)
    ]
    ranked = sorted(
        pool,
        key=lambda row: (
            hmac.new(
                bytes.fromhex(structure_key_hex),
                world_uid.encode("utf-8")
                + common.FIELD_SEPARATOR
                + b"identity_asset_assignment"
                + common.FIELD_SEPARATOR
                + row[0].encode("utf-8"),
                hashlib.sha256,
            ).digest(),
            row[0].encode("utf-8"),
        ),
    )
    if len(ordered_assets) > len(ranked):
        raise common.ContractError("Identity asset UID pool exhausted")
    ordinal_by_uid = {uid: ordinal for uid, ordinal in pool}
    output: list[dict[str, Any]] = []
    for asset, (asset_uid, _ordinal) in zip(
        ordered_assets, ranked[: len(ordered_assets)], strict=True
    ):
        output.append({**asset, "identity_asset_uid": asset_uid})
    if len({asset["identity_asset_uid"] for asset in output}) != len(output):
        raise common.ContractError("Duplicate assigned identity asset UID")
    return output, ordinal_by_uid


def materialize_asset_repeat_decisions(
    policy: dict[str, Any],
    *,
    world_uid: str,
    structure_key_hex: str,
    assets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Apply asset-level repeat draws only after the public asset UID exists."""

    output: list[dict[str, Any]] = []
    for asset in assets:
        draw_name = asset.get("repeat_draw_name")
        probability = asset.get("repeat_probability")
        updated = dict(asset)
        updated["occurrence_counts"] = dict(asset["occurrence_counts"])
        if draw_name is None:
            if probability is not None:
                raise common.ContractError("Repeat probability exists without draw name")
            updated["asset_repeat_decision"] = None
            output.append(updated)
            continue
        if probability is None:
            raise common.ContractError("Repeat draw exists without probability")
        if any(int(value) != 1 for value in asset["occurrence_counts"].values()):
            raise common.ContractError(
                "Asset-level repeat draw requires one base occurrence per seller"
            )
        asset_uid = str(asset["identity_asset_uid"])
        repeat = structure.structure_bernoulli(
            policy,
            structure_key_hex=structure_key_hex,
            world_uid=world_uid,
            draw_name=str(draw_name),
            subject_uid=asset_uid,
            probability=probability,
        )
        updated["occurrence_counts"] = {
            seller_uid: 2 if repeat else 1 for seller_uid in asset["sellers"]
        }
        updated["asset_repeat_decision"] = repeat
        output.append(updated)
    return output


def _rank_types(
    structure_key_hex: str,
    *,
    world_uid: str,
    asset_uid: str,
    allowed_types: Sequence[str],
    global_order: Sequence[str],
) -> list[str]:
    return sorted(
        allowed_types,
        key=lambda identity_type: (
            hmac.new(
                bytes.fromhex(structure_key_hex),
                common.FIELD_SEPARATOR.join(
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


def assign_asset_types(
    policy: dict[str, Any],
    *,
    world_uid: str,
    structure_key_hex: str,
    assets: list[dict[str, Any]],
    nonempty_description_counts: Mapping[str, int],
    allow_infeasible: bool = False,
    maximum_nodes_override: int | None = None,
) -> tuple[list[dict[str, Any]] | None, int]:
    """Solve the exact fixed-capacity type CSP with deterministic propagation."""

    for asset in assets:
        if "asset_repeat_decision" not in asset:
            raise common.ContractError(
                "Identity repeat decisions must be materialized before type assignment"
            )
        decision = asset["asset_repeat_decision"]
        if asset.get("repeat_draw_name") is None:
            if decision is not None:
                raise common.ContractError("Non-repeat asset has a repeat decision")
        elif not isinstance(decision, bool):
            raise common.ContractError("Repeat-enabled asset lacks a Boolean decision")
    global_order = list(policy["identity_design"]["identity_types"])
    fixed_capacity = int(
        policy["identity_design"]["slot_feasibility"]["type_assignment"][
            "fixed_per_seller_type_capacity"
        ]
    )
    if set(nonempty_description_counts) != {
        seller
        for asset in assets
        for seller in asset["sellers"]
    }:
        raise common.ContractError("Observed description-count seller keyset drift")
    if any(
        int(count) < fixed_capacity
        for count in nonempty_description_counts.values()
    ):
        raise common.ContractError(
            "Seller lacks the registered minimum nonempty descriptions"
        )
    assigned: dict[str, str] = {}
    group_types: defaultdict[str, set[str]] = defaultdict(set)
    seller_type_demand: defaultdict[tuple[str, str], int] = defaultdict(int)
    fixed = [asset for asset in assets if asset["fixed_type"] is not None]
    for asset in fixed:
        identity_type = str(asset["fixed_type"])
        if identity_type not in asset["allowed_types"]:
            raise common.ContractError("Fixed identity type is outside its domain")
        if any(
            identity_type in group_types[group]
            for group in asset["distinct_groups"]
        ):
            raise common.ContractError("Fixed background type violates slot capacity")
        assigned[asset["identity_asset_uid"]] = identity_type
        for group in asset["distinct_groups"]:
            group_types[group].add(identity_type)
        for seller in asset["sellers"]:
            key = (seller, identity_type)
            seller_type_demand[key] += int(
                asset["occurrence_counts"][seller]
            )
            if seller_type_demand[key] > fixed_capacity:
                raise common.ContractError(
                    "Fixed identity type violates fixed seller/type capacity"
                )

    variable_assets = sorted(
        (
            asset
            for asset in assets
            if asset["fixed_type"] is None
        ),
        key=lambda asset: asset["identity_asset_uid"].encode("utf-8"),
    )
    variable = {
        str(asset["identity_asset_uid"]): asset for asset in variable_assets
    }
    if len(variable) != len(variable_assets):
        raise common.ContractError("Variable identity asset UID collision")
    maximum_nodes = (
        int(maximum_nodes_override)
        if maximum_nodes_override is not None
        else int(
            policy["identity_design"]["slot_feasibility"]["type_assignment"][
                "maximum_search_nodes"
            ]
        )
    )
    if maximum_nodes <= 0:
        raise common.ContractError("Identity type solver has no search-node budget")
    maximum_memoized_states = int(
        policy["identity_design"]["slot_feasibility"]["type_assignment"][
            "maximum_memoized_states"
        ]
    )
    type_bit = {
        identity_type: 1 << index
        for index, identity_type in enumerate(global_order)
    }
    sellers = common.utf8_sort(nonempty_description_counts)
    distinct_groups = common.utf8_sort(
        {
            group
            for asset in variable_assets
            for group in asset["distinct_groups"]
        }
    )
    node_count = 0
    failed_states: set[tuple[Any, ...]] = set()

    def can_assign(asset: Mapping[str, Any], identity_type: str) -> bool:
        if any(
            identity_type in group_types[group]
            for group in asset["distinct_groups"]
        ):
            return False
        return all(
            seller_type_demand[(seller, identity_type)]
            + int(asset["occurrence_counts"][seller])
            <= fixed_capacity
            for seller in asset["sellers"]
        )

    def apply(asset: Mapping[str, Any], identity_type: str, sign: int) -> None:
        for group in asset["distinct_groups"]:
            if sign > 0:
                group_types[group].add(identity_type)
            else:
                group_types[group].remove(identity_type)
        for seller in asset["sellers"]:
            seller_type_demand[(seller, identity_type)] += (
                sign * int(asset["occurrence_counts"][seller])
            )

    def options_for(asset_uid: str) -> list[str]:
        asset = variable[asset_uid]
        return [
            identity_type
            for identity_type in _rank_types(
                structure_key_hex,
                world_uid=world_uid,
                asset_uid=asset_uid,
                allowed_types=asset["allowed_types"],
                global_order=global_order,
            )
            if can_assign(asset, identity_type)
        ]

    def state_key(remaining_uids: frozenset[str]) -> tuple[Any, ...]:
        demand_state = tuple(
            seller_type_demand[(seller, identity_type)]
            for seller in sellers
            for identity_type in global_order
        )
        group_state = tuple(
            sum(type_bit[value] for value in group_types[group])
            for group in distinct_groups
        )
        return (
            tuple(sorted(remaining_uids, key=lambda value: value.encode("utf-8"))),
            demand_state,
            group_state,
        )

    def propagation_passes(
        remaining_uids: frozenset[str],
        option_map: Mapping[str, list[str]],
    ) -> bool:
        # Exact Hall checks for every registered all-different group (size <= 4).
        for group in distinct_groups:
            members = [
                asset_uid
                for asset_uid in remaining_uids
                if group in variable[asset_uid]["distinct_groups"]
            ]
            for subset_size in range(1, len(members) + 1):
                for subset in itertools.combinations(members, subset_size):
                    union_mask = 0
                    for asset_uid in subset:
                        for identity_type in option_map[asset_uid]:
                            union_mask |= type_bit[identity_type]
                    if union_mask.bit_count() < subset_size:
                        return False

        # Weighted seller/type Hall necessities over all 255 type subsets.
        for seller in sellers:
            incident = [
                asset_uid
                for asset_uid in remaining_uids
                if seller in variable[asset_uid]["occurrence_counts"]
            ]
            if not incident:
                continue
            domain_masks = {
                asset_uid: sum(
                    type_bit[identity_type]
                    for identity_type in option_map[asset_uid]
                )
                for asset_uid in incident
            }
            relevant_masks = {0}
            for domain_mask in domain_masks.values():
                relevant_masks.update(
                    mask | domain_mask for mask in tuple(relevant_masks)
                )
            for subset_mask in sorted(relevant_masks - {0}):
                residual_capacity = sum(
                    fixed_capacity
                    - seller_type_demand[(seller, identity_type)]
                    for identity_type in global_order
                    if subset_mask & type_bit[identity_type]
                )
                forced_demand = sum(
                    int(variable[asset_uid]["occurrence_counts"][seller])
                    for asset_uid in incident
                    if domain_masks[asset_uid] & ~subset_mask == 0
                )
                if forced_demand > residual_capacity:
                    return False
        return True

    def search(remaining_uids: frozenset[str]) -> bool:
        nonlocal node_count
        if not remaining_uids:
            return True
        key = state_key(remaining_uids)
        if key in failed_states:
            return False
        option_map = {
            asset_uid: options_for(asset_uid)
            for asset_uid in remaining_uids
        }
        if any(not options for options in option_map.values()):
            failed_states.add(key)
            return False
        if not propagation_passes(remaining_uids, option_map):
            failed_states.add(key)
            return False
        selected_uid = min(
            remaining_uids,
            key=lambda asset_uid: (
                len(option_map[asset_uid]),
                -max(
                    int(variable[asset_uid]["occurrence_counts"][seller])
                    for seller in variable[asset_uid]["sellers"]
                ),
                -len(variable[asset_uid]["sellers"]),
                asset_uid.encode("utf-8"),
            ),
        )
        asset = variable[selected_uid]
        next_remaining = remaining_uids - {selected_uid}
        for identity_type in option_map[selected_uid]:
            node_count += 1
            if node_count > maximum_nodes:
                raise common.ContractError("Identity type solver exhausted")
            assigned[selected_uid] = identity_type
            apply(asset, identity_type, 1)
            if search(next_remaining):
                return True
            apply(asset, identity_type, -1)
            del assigned[selected_uid]
        failed_states.add(key)
        if len(failed_states) > maximum_memoized_states:
            raise common.ContractError("Identity type solver memo budget exhausted")
        return False

    if not search(frozenset(variable)):
        if allow_infeasible:
            return None, node_count
        raise common.ContractError("Identity type solver found no feasible assignment")

    validation_group_types: defaultdict[str, set[str]] = defaultdict(set)
    validation_demand: defaultdict[tuple[str, str], int] = defaultdict(int)
    for asset in assets:
        identity_type = assigned[asset["identity_asset_uid"]]
        for group in asset["distinct_groups"]:
            if identity_type in validation_group_types[group]:
                raise common.ContractError("Type-solver distinct-group replay failed")
            validation_group_types[group].add(identity_type)
        for seller_uid in asset["sellers"]:
            key = (seller_uid, identity_type)
            validation_demand[key] += int(
                asset["occurrence_counts"][seller_uid]
            )
            if validation_demand[key] > fixed_capacity:
                raise common.ContractError("Type-solver fixed-capacity replay failed")
    output = [
        {**asset, "identity_type": assigned[asset["identity_asset_uid"]]}
        for asset in assets
    ]
    return output, node_count


class _Edge:
    __slots__ = ("to", "reverse", "capacity", "original_capacity", "kind", "key")

    def __init__(
        self,
        to: str,
        reverse: int,
        capacity: int,
        *,
        kind: str,
        key: tuple[str, str] | None,
    ) -> None:
        self.to = to
        self.reverse = reverse
        self.capacity = capacity
        self.original_capacity = capacity
        self.kind = kind
        self.key = key


def _add_edge(
    graph: dict[str, list[_Edge]],
    left: str,
    right: str,
    *,
    kind: str,
    key: tuple[str, str] | None = None,
) -> None:
    forward = _Edge(right, len(graph[right]), 1, kind=kind, key=key)
    reverse = _Edge(left, len(graph[left]), 0, kind="reverse", key=None)
    graph[left].append(forward)
    graph[right].append(reverse)


def _max_flow(
    graph: dict[str, list[_Edge]], source_uid: str, sink_uid: str
) -> int:
    flow = 0
    while True:
        level = {source_uid: 0}
        queue = deque([source_uid])
        while queue:
            node = queue.popleft()
            for edge in graph[node]:
                if edge.capacity and edge.to not in level:
                    level[edge.to] = level[node] + 1
                    queue.append(edge.to)
        if sink_uid not in level:
            return flow
        cursor = defaultdict(int)

        def dfs(node: str, amount: int) -> int:
            if node == sink_uid:
                return amount
            while cursor[node] < len(graph[node]):
                edge = graph[node][cursor[node]]
                if edge.capacity and level.get(edge.to) == level[node] + 1:
                    pushed = dfs(edge.to, min(amount, edge.capacity))
                    if pushed:
                        edge.capacity -= pushed
                        graph[edge.to][edge.reverse].capacity += pushed
                        return pushed
                cursor[node] += 1
            return 0

        while pushed := dfs(source_uid, 1):
            flow += pushed


def assign_occurrences_to_items(
    policy: dict[str, Any],
    *,
    mode: str,
    world_uid: str,
    structure_key_hex: str,
    assets: list[dict[str, Any]],
    items_by_seller: Mapping[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    occurrences: list[dict[str, Any]] = []
    for asset in assets:
        required_value_fields = {"identity_uid", "identity_value", "global_asset_index"}
        if not required_value_fields.issubset(asset):
            raise common.ContractError(
                "Identity values must be attached before occurrence-to-slot matching"
            )
        for seller_uid in asset["sellers"]:
            for occurrence_index in range(asset["occurrence_counts"][seller_uid]):
                occurrence_uid = "occ_" + common.canonical_sha256(
                    {
                        "identity_asset_uid": asset["identity_asset_uid"],
                        "seller_uid": seller_uid,
                        "occurrence_index": occurrence_index,
                    }
                )
                occurrences.append(
                    {
                        "occurrence_uid": occurrence_uid,
                        "identity_asset_uid": asset["identity_asset_uid"],
                        "seller_uid": seller_uid,
                        "identity_type": asset["identity_type"],
                        "identity_uid": asset["identity_uid"],
                        "identity_value": asset["identity_value"],
                        "global_asset_index": asset["global_asset_index"],
                        "bundle_uid": "bundle0_"
                        + common.canonical_sha256(
                            {
                                "world_uid": world_uid,
                                "seller_uid": seller_uid,
                                "identity_uid": asset["identity_uid"],
                            }
                        ),
                        "role": asset["role"],
                    }
                )
    source_uid = "src_" + common.sha256_bytes(
        world_uid.encode("utf-8")
        + common.FIELD_SEPARATOR
        + b"occurrence_slot_matching"
        + common.FIELD_SEPARATOR
        + b"source"
    )
    sink_uid = "snk_" + common.sha256_bytes(
        world_uid.encode("utf-8")
        + common.FIELD_SEPARATOR
        + b"occurrence_slot_matching"
        + common.FIELD_SEPARATOR
        + b"sink"
    )
    graph: dict[str, list[_Edge]] = defaultdict(list)
    edge_specs: list[tuple[int, bytes, bytes, str, str, str, tuple[str, str] | None]] = []
    kind_order = {
        "source_occurrence": 0,
        "occurrence_slot": 1,
        "slot_sink": 2,
    }
    slot_records: dict[str, tuple[str, str]] = {}

    for occurrence in occurrences:
        edge_specs.append(
            (
                kind_order["source_occurrence"],
                b"",
                occurrence["occurrence_uid"].encode("utf-8"),
                "source_occurrence",
                source_uid,
                occurrence["occurrence_uid"],
                None,
            )
        )
        for item in items_by_seller[occurrence["seller_uid"]]:
            if not item["description_nonempty"]:
                continue
            slot_uid = "isl_" + common.canonical_sha256(
                {
                    "item_uid": item["item_uid"],
                    "identity_type": occurrence["identity_type"],
                }
            )
            slot_records[slot_uid] = (item["item_uid"], occurrence["identity_type"])
            edge_specs.append(
                (
                    kind_order["occurrence_slot"],
                    b"",
                    (
                        occurrence["occurrence_uid"]
                        + structure.TUPLE_SEPARATOR
                        + slot_uid
                    ).encode("utf-8"),
                    "occurrence_slot",
                    occurrence["occurrence_uid"],
                    slot_uid,
                    (occurrence["occurrence_uid"], item["item_uid"]),
                )
            )
    for slot_uid in slot_records:
        edge_specs.append(
            (
                kind_order["slot_sink"],
                b"",
                slot_uid.encode("utf-8"),
                "slot_sink",
                slot_uid,
                sink_uid,
                None,
            )
        )
    ranked_specs = []
    for order, _empty, _legacy_serialization, kind, left, right, key in edge_specs:
        message = common.FIELD_SEPARATOR.join(
            (
                world_uid.encode("utf-8"),
                b"occurrence_slot_matching",
                kind.encode("ascii"),
                left.encode("utf-8"),
                right.encode("utf-8"),
            )
        )
        edge_serialization = (
            left.encode("utf-8")
            + common.FIELD_SEPARATOR
            + right.encode("utf-8")
        )
        ranked_specs.append(
            (
                order,
                hmac.new(
                    bytes.fromhex(structure_key_hex),
                    message,
                    hashlib.sha256,
                ).digest(),
                edge_serialization,
                kind,
                left,
                right,
                key,
            )
        )
    for _order, _digest, _serialization, kind, left, right, key in sorted(
        ranked_specs
    ):
        _add_edge(graph, left, right, kind=kind, key=key)
    flow = _max_flow(graph, source_uid, sink_uid)
    if flow != len(occurrences):
        raise common.ContractError(
            f"Occurrence-slot flow incomplete: {flow}/{len(occurrences)}"
        )
    item_by_uid = {
        item["item_uid"]: item
        for items in items_by_seller.values()
        for item in items
    }
    occurrence_by_uid = {
        occurrence["occurrence_uid"]: occurrence for occurrence in occurrences
    }
    assignments: list[dict[str, Any]] = []
    for edges in graph.values():
        for edge in edges:
            if (
                edge.kind == "occurrence_slot"
                and edge.original_capacity == 1
                and edge.capacity == 0
                and edge.key is not None
            ):
                occurrence_uid, item_uid = edge.key
                assignments.append(
                    {
                        **occurrence_by_uid[occurrence_uid],
                        "item_uid": item_uid,
                    }
                )
    if len(assignments) != len(occurrences):
        raise common.ContractError("Occurrence assignment extraction count drift")

    type_order = list(policy["identity_design"]["identity_types"])
    by_item: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in assignments:
        by_item[row["item_uid"]].append(row)
    id_key = policy["randomness"][mode]["id_key_hex"]
    slots: list[dict[str, Any]] = []
    for item_uid, rows in by_item.items():
        ordered = sorted(
            rows,
            key=lambda row: (
                type_order.index(row["identity_type"]),
                row["identity_asset_uid"].encode("utf-8"),
                row["occurrence_uid"].encode("utf-8"),
            ),
        )
        for ordinal, row in enumerate(ordered):
            slot_uid = structure.base_uid(
                key_hex=id_key,
                entity_kind="identity_slot",
                parent_uid_or_mode=item_uid,
                ordinal=ordinal,
            )
            slot = {**row, "slot_uid": slot_uid}
            slots.append(slot)
            item_by_uid[item_uid]["identity_slots"].append(slot)
    return slots, {
        "planned_occurrence_count": len(occurrences),
        "maximum_flow": flow,
        "slot_count": len(slots),
        "source_node_uid": source_uid,
        "sink_node_uid": sink_uid,
    }


def attach_identity_values(
    policy: dict[str, Any],
    *,
    mode: str,
    world_mode_global_ordinal: int,
    assets: list[dict[str, Any]],
    asset_ordinal_by_uid: Mapping[str, int],
) -> list[dict[str, Any]]:
    salt_map = policy["identity_design"]["identity_value_generation"][
        "salt_selection"
    ][f"{mode}_per_type_salt_counters"]
    if not isinstance(salt_map, dict):
        raise common.ContractError(f"Identity salt map is not frozen for {mode}")
    key_hex = policy["randomness"][mode]["identity_value_key_hex"]
    handle_encoding = str(
        policy["identity_design"]["identity_value_generation"][
            "handle_encoding_by_mode"
        ][mode]
    )
    pool_count = int(
        policy["identity_design"]["slot_feasibility"][
            "identity_asset_uid_pool"
        ]["count_per_world"]
    )
    output: list[dict[str, Any]] = []
    for asset in assets:
        asset_uid = asset["identity_asset_uid"]
        asset_ordinal = int(asset_ordinal_by_uid[asset_uid])
        global_index = world_mode_global_ordinal * pool_count + asset_ordinal
        identity_type = asset["identity_type"]
        value = identity_values.identity_value(
            key_hex=key_hex,
            identity_type=identity_type,
            salt=int(salt_map[identity_type]),
            global_asset_index=global_index,
            handle_encoding=handle_encoding,
        )
        identity_uid = "id_" + common.canonical_sha256(
            {
                "contact_type": identity_type.strip().lower(),
                "normalized_value": value.strip().lower(),
            }
        )
        output.append(
            {
                **asset,
                "identity_value": value,
                "identity_uid": identity_uid,
                "global_asset_index": global_index,
            }
        )
    return output


def solve_identity_plan(
    policy: dict[str, Any],
    *,
    mode: str,
    split: str,
    world_uid: str,
    world_mode_global_ordinal: int,
    structure_key_hex: str,
    membership: Mapping[str, Any],
    markets: Mapping[str, str],
    mechanisms: Mapping[str, Mapping[str, Any]],
    items_by_seller: Mapping[str, list[dict[str, Any]]],
    pre_slot_callback: Callable[[Sequence[Mapping[str, Any]]], Mapping[str, Any]],
) -> dict[str, Any]:
    """Run the only supported fail-closed identity-planning sequence."""

    graph_name = str(policy["identity_design"]["mechanism_by_split"][split])
    background_assets = build_background_assets(
        policy,
        world_uid=world_uid,
        structure_key_hex=structure_key_hex,
        seller_uids=membership["seller_uids"],
    )
    positive_assets, positive_targets = build_positive_assets(
        policy,
        world_uid=world_uid,
        graph_name=graph_name,
        structure_key_hex=structure_key_hex,
        membership=membership,
        markets=markets,
        mechanisms=mechanisms,
    )
    nonempty_description_counts = {
        seller_uid: sum(bool(item["description_nonempty"]) for item in items)
        for seller_uid, items in items_by_seller.items()
    }
    if set(nonempty_description_counts) != set(membership["seller_uids"]):
        raise common.ContractError("Item/seller keyset drift before identity solve")
    type_contract = policy["identity_design"]["slot_feasibility"][
        "type_assignment"
    ]
    maximum_type_nodes = int(type_contract["maximum_search_nodes"])
    maximum_membership_leaves = int(
        type_contract["maximum_membership_complete_assignments"]
    )
    total_type_solver_nodes = 0
    assets: list[dict[str, Any]] | None = None
    ordinal_by_uid: dict[str, int] | None = None
    negative_flags: list[dict[str, Any]] | None = None
    membership_audit: dict[str, Any] | None = None
    selected_membership_ordinal: int | None = None
    plans = _iter_hard_negative_plans(
        policy,
        world_uid=world_uid,
        graph_name=graph_name,
        structure_key_hex=structure_key_hex,
        membership=membership,
        mechanisms=mechanisms,
    )
    for complete_ordinal, plan in enumerate(
        itertools.islice(plans, maximum_membership_leaves)
    ):
        (
            negative_assets,
            candidate_negative_flags,
            candidate_membership_audit,
        ) = plan
        if (
            candidate_membership_audit[
                "selected_membership_complete_assignment_ordinal"
            ]
            != complete_ordinal
            or candidate_membership_audit[
                "membership_complete_assignments_examined"
            ]
            != complete_ordinal + 1
        ):
            raise common.ContractError(
                "Hard-negative membership iterator ordinal drift"
            )
        candidate_assets, candidate_ordinal_by_uid = assign_asset_uids(
            policy,
            mode=mode,
            world_uid=world_uid,
            graph_name=graph_name,
            structure_key_hex=structure_key_hex,
            assets_in_descriptor_order=[
                *background_assets,
                *positive_assets,
                *negative_assets,
            ],
        )
        candidate_assets = materialize_asset_repeat_decisions(
            policy,
            world_uid=world_uid,
            structure_key_hex=structure_key_hex,
            assets=candidate_assets,
        )
        remaining_type_nodes = maximum_type_nodes - total_type_solver_nodes
        if remaining_type_nodes <= 0:
            raise common.ContractError(
                "Identity type solver exhausted its cross-membership node budget"
            )
        typed_assets, leaf_type_nodes = assign_asset_types(
            policy,
            world_uid=world_uid,
            structure_key_hex=structure_key_hex,
            assets=candidate_assets,
            nonempty_description_counts=nonempty_description_counts,
            allow_infeasible=True,
            maximum_nodes_override=remaining_type_nodes,
        )
        if (
            isinstance(leaf_type_nodes, bool)
            or not isinstance(leaf_type_nodes, int)
            or leaf_type_nodes < 0
            or leaf_type_nodes > remaining_type_nodes
        ):
            raise common.ContractError(
                "Identity type solver returned an invalid cross-membership "
                "node count"
        )
        total_type_solver_nodes += leaf_type_nodes
        if typed_assets is None:
            continue
        assets = typed_assets
        ordinal_by_uid = candidate_ordinal_by_uid
        negative_flags = candidate_negative_flags
        membership_audit = candidate_membership_audit
        selected_membership_ordinal = complete_ordinal
        break
    if (
        assets is None
        or ordinal_by_uid is None
        or negative_flags is None
        or membership_audit is None
        or selected_membership_ordinal is None
    ):
        raise common.ContractError(
            "No hard-negative membership leaf has a feasible fixed-capacity type plan"
        )
    assets = attach_identity_values(
        policy,
        mode=mode,
        world_mode_global_ordinal=world_mode_global_ordinal,
        assets=assets,
        asset_ordinal_by_uid=ordinal_by_uid,
    )
    pre_slot_result = dict(pre_slot_callback(negative_flags))
    expected_pre_slot_schema = {
        "override_audit_count",
        "high_semantic_count",
        "exact_title_clone_count",
        "unique_override_seller_count",
        "unique_override_item_count",
        "noise_record_count",
        "override_audit_sha256",
        "noise_record_sha256",
    }
    if set(pre_slot_result) != expected_pre_slot_schema:
        raise common.ContractError("Pre-slot materializer audit schema drift")
    expected_counts = {
        "override_audit_count": 6,
        "high_semantic_count": 4,
        "exact_title_clone_count": 2,
        "unique_override_seller_count": 12,
        "unique_override_item_count": 12,
        "noise_record_count": 28,
    }
    if any(
        int(pre_slot_result[key]) != value
        for key, value in expected_counts.items()
    ):
        raise common.ContractError("Pre-slot materializer count gate failed")
    for key in ("override_audit_sha256", "noise_record_sha256"):
        value = str(pre_slot_result[key])
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise common.ContractError("Pre-slot materializer digest drift")
    slots, flow_audit = assign_occurrences_to_items(
        policy,
        mode=mode,
        world_uid=world_uid,
        structure_key_hex=structure_key_hex,
        assets=assets,
        items_by_seller=items_by_seller,
    )
    return {
        "graph_name": graph_name,
        "assets": assets,
        "slots": slots,
        "positive_targets": positive_targets,
        "negative_flags": negative_flags,
        "pre_slot_result": pre_slot_result,
        "solver_audit": {
            **membership_audit,
            "selected_membership_complete_assignment_ordinal": (
                selected_membership_ordinal
            ),
            "membership_complete_assignments_type_tested": (
                selected_membership_ordinal + 1
            ),
            "type_solver_node_count": total_type_solver_nodes,
            **flow_audit,
            "identity_asset_count": len(assets),
            "unused_identity_asset_uid_count": int(
                policy["identity_design"]["slot_feasibility"][
                    "identity_asset_uid_pool"
                ]["count_per_world"]
            )
            - len(assets),
        },
    }
