#!/usr/bin/env python3
"""Build five label-free, support-exact M1 identity33 derangements."""

from __future__ import annotations

import hashlib
import hmac
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import step28_v13_common as common


UNIVERSE_ORDER = ("primary_c40", "secondary_complement")


def _seed_id(seed: bytes) -> str:
    return "rws_" + hashlib.sha256(seed).hexdigest()


def _digest(seed: bytes, *parts: str) -> bytes:
    return hmac.new(
        seed,
        common.FIELD_SEPARATOR.join(
            part.encode("utf-8") for part in parts
        ),
        hashlib.sha256,
    ).digest()


def _validate_inputs(
    policy: Mapping[str, Any],
    *,
    mode: str,
    m2_identity33_all_pairs: Sequence[Mapping[str, Any]],
    candidate_pairs: Sequence[Mapping[str, Any]],
    complete_pair_endpoints: Sequence[Mapping[str, Any]],
) -> tuple[
    list[str],
    dict[tuple[str, str], dict[str, Any]],
    dict[tuple[str, str], tuple[str, str]],
    set[tuple[str, str]],
]:
    feature_names = [
        str(value) for value in policy["history_features"]["feature_names"]
    ]
    matrix_schema = ["canonical_pair_uid", "world_uid", *feature_names]
    candidate_schema = policy["candidate_design"][
        "public_safe_projection_columns"
    ]
    complete_schema = policy["relational_integrity"][
        "pair_projection_contract"
    ]["complete_model_pair_endpoints_schema"]
    if len(feature_names) != 33 or len(set(feature_names)) != 33:
        raise common.ContractError(
            "M1 derangement feature contract drift"
        )
    if any(list(row) != matrix_schema for row in m2_identity33_all_pairs):
        raise common.ContractError("M1 derangement M2 matrix schema drift")
    if any(list(row) != candidate_schema for row in candidate_pairs):
        raise common.ContractError(
            "M1 derangement C40 endpoint schema drift"
        )
    if any(list(row) != complete_schema for row in complete_pair_endpoints):
        raise common.ContractError(
            "M1 derangement complete-pair schema drift"
        )

    matrix_index: dict[tuple[str, str], dict[str, Any]] = {}
    for source_row in m2_identity33_all_pairs:
        row = dict(source_row)
        key = (str(row["world_uid"]), str(row["canonical_pair_uid"]))
        if key in matrix_index:
            raise common.ContractError(
                "M1 derangement M2 matrix key collision"
            )
        for name in feature_names:
            try:
                value = float(row[name])
            except (TypeError, ValueError) as exc:
                raise common.ContractError(
                    "M1 derangement M2 matrix value is nonnumeric"
                ) from exc
            if not (-float("inf") < value < float("inf")):
                raise common.ContractError(
                    "M1 derangement M2 matrix value is nonfinite"
                )
        matrix_index[key] = row

    endpoint_index: dict[tuple[str, str], tuple[str, str]] = {}
    for row in complete_pair_endpoints:
        world_uid = str(row["world_uid"])
        pair_uid = str(row["canonical_pair_uid"])
        left = str(row["seller_uid_left"])
        right = str(row["seller_uid_right"])
        key = (world_uid, pair_uid)
        if (
            key in endpoint_index
            or left == right
            or common.utf8_sort((left, right)) != [left, right]
            or pair_uid != common.canonical_pair_uid(left, right)
        ):
            raise common.ContractError(
                "M1 derangement complete-pair endpoint drift"
            )
        endpoint_index[key] = (left, right)
    c40_keys = {
        (str(row["world_uid"]), str(row["canonical_pair_uid"]))
        for row in candidate_pairs
    }
    if (
        len(c40_keys) != len(candidate_pairs)
        or set(matrix_index) != set(endpoint_index)
        or not c40_keys.issubset(endpoint_index)
    ):
        raise common.ContractError(
            "M1 derangement matrix/pair/C40 keyset drift"
        )
    matrix_counts = Counter(key[0] for key in matrix_index)
    c40_counts = Counter(key[0] for key in c40_keys)
    expected_world_count = int(
        policy["modes"][mode]["world_counts"]["train"]
    )
    if (
        len(matrix_counts) != expected_world_count
        or set(matrix_counts.values()) != {378}
        or matrix_counts.keys() != c40_counts.keys()
        or set(c40_counts.values()) != {40}
    ):
        raise common.ContractError(
            "M1 derangement train world cardinality drift"
        )
    return feature_names, matrix_index, endpoint_index, c40_keys


def _perfect_endpoint_disjoint_mapping(
    *,
    seed: bytes,
    world_uid: str,
    universe: str,
    pair_uids: Sequence[str],
    endpoints: Mapping[str, tuple[str, str]],
) -> dict[str, str]:
    """Map every destination pair to one endpoint-disjoint source pair."""

    destination_order = sorted(
        pair_uids,
        key=lambda pair_uid: (
            _digest(
                seed,
                world_uid,
                "m1_derangement_destination_order",
                universe,
                pair_uid,
            ),
            pair_uid.encode("utf-8"),
        ),
    )
    source_candidates: dict[str, list[str]] = {}
    for destination_uid in destination_order:
        destination_endpoints = set(endpoints[destination_uid])
        candidates = [
            source_uid
            for source_uid in pair_uids
            if not destination_endpoints.intersection(
                endpoints[source_uid]
            )
        ]
        source_candidates[destination_uid] = sorted(
            candidates,
            key=lambda source_uid: (
                _digest(
                    seed,
                    world_uid,
                    "m1_derangement_source_candidate",
                    universe,
                    destination_uid,
                    source_uid,
                ),
                source_uid.encode("utf-8"),
            ),
        )
        if not source_candidates[destination_uid]:
            raise common.ContractError(
                "M1 derangement destination has no disjoint source"
            )

    source_to_destination: dict[str, str] = {}

    def augment(destination_uid: str, seen_sources: set[str]) -> bool:
        for source_uid in source_candidates[destination_uid]:
            if source_uid in seen_sources:
                continue
            seen_sources.add(source_uid)
            previous = source_to_destination.get(source_uid)
            if previous is None or augment(previous, seen_sources):
                source_to_destination[source_uid] = destination_uid
                return True
        return False

    for destination_uid in destination_order:
        if not augment(destination_uid, set()):
            raise common.ContractError(
                "M1 endpoint-disjoint derangement has no perfect matching"
            )
    destination_to_source = {
        destination_uid: source_uid
        for source_uid, destination_uid in source_to_destination.items()
    }
    if (
        set(destination_to_source) != set(pair_uids)
        or set(destination_to_source.values()) != set(pair_uids)
        or any(
            destination_uid == source_uid
            or set(endpoints[destination_uid]).intersection(
                endpoints[source_uid]
            )
            for destination_uid, source_uid in destination_to_source.items()
        )
    ):
        raise common.ContractError(
            "M1 endpoint-disjoint matching replay failed"
        )
    return destination_to_source


def build_one_feature_derangement(
    policy: Mapping[str, Any],
    *,
    mode: str,
    split: str,
    seed_hex: str,
    m2_identity33_all_pairs: Sequence[Mapping[str, Any]],
    candidate_pairs: Sequence[Mapping[str, Any]],
    complete_pair_endpoints: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Create one M1 matrix without labels, oracle, or candidate evidence."""

    common.validate_policy(policy, mode=mode)
    if mode not in {"development_smoke", "training_ready"} or split != "train":
        raise common.ContractError(
            "M1 feature derangement is restricted to non-formal train data"
        )
    registered = list(policy["randomness"][mode]["rewire_key_hexes"])
    if seed_hex not in registered or registered.count(seed_hex) != 1:
        raise common.ContractError(
            "M1 feature derangement seed is not uniquely registered"
        )
    try:
        seed = bytes.fromhex(seed_hex)
    except ValueError as exc:
        raise common.ContractError(
            "M1 feature derangement seed is not hex"
        ) from exc
    if len(seed) != 32:
        raise common.ContractError(
            "M1 feature derangement seed must contain 32 bytes"
        )
    rewire_seed_id = _seed_id(seed)
    (
        feature_names,
        matrix_index,
        endpoint_index,
        c40_keys,
    ) = _validate_inputs(
        policy,
        mode=mode,
        m2_identity33_all_pairs=m2_identity33_all_pairs,
        candidate_pairs=candidate_pairs,
        complete_pair_endpoints=complete_pair_endpoints,
    )
    keys_by_world: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key in matrix_index:
        keys_by_world[key[0]].append(key)
    mapping_rows: list[dict[str, Any]] = []
    output_rows: list[dict[str, str]] = []
    for world_uid in common.utf8_sort(keys_by_world):
        world_keys = keys_by_world[world_uid]
        world_endpoint_index = {
            pair_uid: endpoint_index[(world_uid, pair_uid)]
            for _world_uid, pair_uid in world_keys
        }
        for universe in UNIVERSE_ORDER:
            universe_keys = [
                key
                for key in world_keys
                if ((key in c40_keys) == (universe == "primary_c40"))
            ]
            expected_count = 40 if universe == "primary_c40" else 338
            if len(universe_keys) != expected_count:
                raise common.ContractError(
                    "M1 derangement universe count drift"
                )
            pair_uids = common.utf8_sort(
                pair_uid for _world_uid, pair_uid in universe_keys
            )
            destination_to_source = _perfect_endpoint_disjoint_mapping(
                seed=seed,
                world_uid=world_uid,
                universe=universe,
                pair_uids=pair_uids,
                endpoints=world_endpoint_index,
            )
            original_vectors = Counter(
                tuple(
                    str(matrix_index[(world_uid, pair_uid)][name])
                    for name in feature_names
                )
                for pair_uid in pair_uids
            )
            deranged_vectors: Counter[tuple[str, ...]] = Counter()
            for destination_uid in pair_uids:
                source_uid = destination_to_source[destination_uid]
                source = matrix_index[(world_uid, source_uid)]
                vector = tuple(str(source[name]) for name in feature_names)
                deranged_vectors[vector] += 1
                output_rows.append(
                    {
                        "canonical_pair_uid": destination_uid,
                        "world_uid": world_uid,
                        **{
                            name: str(source[name])
                            for name in feature_names
                        },
                    }
                )
                mapping_rows.append(
                    {
                        "rewire_seed_id": rewire_seed_id,
                        "world_uid": world_uid,
                        "universe": universe,
                        "destination_pair_uid": destination_uid,
                        "source_pair_uid": source_uid,
                        "endpoint_disjoint_bool": True,
                        "feature_vector_sha256": common.canonical_sha256(
                            list(vector)
                        ),
                    }
                )
            if deranged_vectors != original_vectors:
                raise common.ContractError(
                    "M1 derangement changed a world/universe vector multiset"
                )
    output_rows.sort(
        key=lambda row: (
            row["world_uid"].encode("utf-8"),
            row["canonical_pair_uid"].encode("utf-8"),
        )
    )
    mapping_rows.sort(
        key=lambda row: (
            row["rewire_seed_id"].encode("utf-8"),
            row["world_uid"].encode("utf-8"),
            UNIVERSE_ORDER.index(str(row["universe"])),
            row["destination_pair_uid"].encode("utf-8"),
        )
    )
    expected_row_count = (
        int(policy["modes"][mode]["world_counts"][split]) * 378
    )
    if (
        len(output_rows) != expected_row_count
        or len(mapping_rows) != expected_row_count
        or len(
            {
                (row["world_uid"], row["canonical_pair_uid"])
                for row in output_rows
            }
        )
        != expected_row_count
    ):
        raise common.ContractError(
            "M1 derangement aggregate matrix cardinality drift"
        )
    result = {
        "rewire_seed_id": rewire_seed_id,
        "identity33_all_pairs": output_rows,
        "feature_derangement_mapping": mapping_rows,
        "joint_vector_multiset_exact_by_world_and_universe": True,
        "endpoint_disjoint_bijection_exact": True,
        "labels_or_controller_inputs_read": False,
        "candidate_trigger_or_audit_inputs_read": False,
    }
    result["canonical_self_hash"] = common.canonical_sha256(result)
    return result


def build_all_feature_derangements(
    policy: Mapping[str, Any],
    *,
    mode: str,
    split: str,
    m2_identity33_all_pairs: Sequence[Mapping[str, Any]],
    candidate_pairs: Sequence[Mapping[str, Any]],
    complete_pair_endpoints: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    seeds = list(policy["randomness"][mode]["rewire_key_hexes"])
    outputs = [
        build_one_feature_derangement(
            policy,
            mode=mode,
            split=split,
            seed_hex=seed_hex,
            m2_identity33_all_pairs=m2_identity33_all_pairs,
            candidate_pairs=candidate_pairs,
            complete_pair_endpoints=complete_pair_endpoints,
        )
        for seed_hex in seeds
    ]
    if (
        len(outputs) != int(policy["placebo"]["replicates"])
        or len({row["rewire_seed_id"] for row in outputs}) != 5
    ):
        raise common.ContractError(
            "M1 feature derangement replicate set drift"
        )
    return outputs
