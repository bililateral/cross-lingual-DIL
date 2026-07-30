#!/usr/bin/env python3
"""Deterministic label-free Step28-v13 identity-edge placebo rewiring."""

from __future__ import annotations

import hashlib
import hmac
from collections import Counter, defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import step28_v13_common as common
import step28_v13_production_chain as production


FIELD_ORDER = {"title": 0, "description": 1}
FLAG_NAMES = (
    "seller_facing_context",
    "product_data_risk_context",
    "direct_identity_eligible",
    "support_only",
)
SAFE_FLAG_NAMES = (
    "observed_seller_facing_context",
    "observed_product_data_risk_context",
    "observed_direct_identity_eligible",
    "observed_support_only",
)
EDGE_KIND_ORDER = {
    "source_identity": 0,
    "identity_identity_item": 1,
    "identity_item_slot": 2,
    "slot_sink": 3,
}


def _identity_uid(identity_type: str, normalized_value: str) -> str:
    return "id_" + common.canonical_sha256(
        {
            "contact_type": identity_type.strip().lower(),
            "normalized_value": normalized_value.strip().lower(),
        }
    )


def _original_bundle_uid(
    world_uid: str, seller_uid: str, identity_uid: str
) -> str:
    return "bundle0_" + common.canonical_sha256(
        {
            "world_uid": world_uid,
            "seller_uid": seller_uid,
            "identity_uid": identity_uid,
        }
    )


def _layer_uid(
    world_uid: str,
    identity_type: str,
    nuisance_class: str,
    edge_occurrence_count: int,
) -> str:
    return "layer_" + common.canonical_sha256(
        {
            "world_uid": world_uid,
            "identity_type": identity_type,
            "nuisance_class": nuisance_class,
            "edge_occurrence_count": edge_occurrence_count,
        }
    )


def _rewired_bundle_uid(
    rewire_seed_id: str,
    layer_uid: str,
    seller_uid: str,
    identity_uid: str,
) -> str:
    return "bundle_" + common.canonical_sha256(
        {
            "rewire_seed_id": rewire_seed_id,
            "layer_uid": layer_uid,
            "seller_uid": seller_uid,
            "identity_uid": identity_uid,
        }
    )


def _seed_hmac(seed: bytes, *parts: str) -> bytes:
    if not parts:
        raise common.ContractError("Rewire HMAC message is empty")
    return hmac.new(
        seed,
        common.FIELD_SEPARATOR.join(part.encode("utf-8") for part in parts),
        hashlib.sha256,
    ).digest()


def _edge_uid(edge: tuple[str, str]) -> str:
    return edge[0] + "\x1f" + edge[1]


def _canonical_edge_pair(
    left: tuple[str, str], right: tuple[str, str]
) -> str:
    edge_uids = sorted(
        (_edge_uid(left), _edge_uid(right)),
        key=lambda value: value.encode("utf-8"),
    )
    return edge_uids[0] + "\x1e" + edge_uids[1]


def _aggregate_nuisance(
    rows: Sequence[Mapping[str, Any]],
    *,
    seller_degree: int,
    direct_maximum: int,
) -> str:
    if not rows or seller_degree < 1:
        raise common.ContractError("Rewire nuisance input is empty")
    observed: list[tuple[int, int, int, int]] = []
    for row in rows:
        values = tuple(int(row[name]) for name in SAFE_FLAG_NAMES)
        if any(value not in {0, 1} for value in values):
            raise common.ContractError("Rewire safe-slot flags are not binary")
        seller_facing, risk, direct, support = values
        if risk and direct:
            raise common.ContractError("Risky rewire slot is direct eligible")
        if support and direct:
            raise common.ContractError(
                "Support rewire slot cannot be direct eligible"
            )
        if not risk and not support and (not direct or not seller_facing):
            raise common.ContractError("Direct rewire slot flags are inconsistent")
        observed.append((risk, support, direct, seller_facing))
    if any(row[0] for row in observed):
        return "risky_product"
    if any(row[1] for row in observed):
        return "public_support"
    if seller_degree > direct_maximum:
        return "high_frequency_direct"
    return "direct_or_private"


@dataclass
class _ValidatedInputs:
    seller_index: dict[str, dict[str, Any]]
    item_index: dict[str, dict[str, Any]]
    ast_index: dict[str, dict[str, Any]]
    safe_index: dict[str, dict[str, Any]]
    identity_rows: dict[str, list[dict[str, Any]]]
    identity_catalog: dict[str, dict[str, str]]
    edges: dict[tuple[str, str], list[dict[str, Any]]]
    edge_layer: dict[tuple[str, str], tuple[str, str, str, int]]


def _validate_inputs(
    policy: Mapping[str, Any],
    *,
    sellers: Sequence[Mapping[str, Any]],
    items: Sequence[Mapping[str, Any]],
    safe_slots: Sequence[Mapping[str, Any]],
    nuisance_ledger: Sequence[Mapping[str, Any]],
    render_asts: Sequence[Mapping[str, Any]],
) -> _ValidatedInputs:
    seller_schema = policy["relational_integrity"]["observed_core_schemas"][
        "sellers.csv"
    ]
    item_schema = policy["relational_integrity"]["observed_core_schemas"][
        "items.jsonl"
    ]
    ast_schema = policy["relational_integrity"]["observed_core_schemas"][
        "render_asts.jsonl_private"
    ]
    safe_schema = policy["placebo"]["rewire_safe_slot_schema"]
    seller_index: dict[str, dict[str, Any]] = {}
    for source_row in sellers:
        if list(source_row) != seller_schema:
            raise common.ContractError("Rewire seller schema/order drift")
        row = dict(source_row)
        seller_uid = str(row["seller_uid"])
        if (
            not seller_uid
            or seller_uid in seller_index
            or not str(row["world_uid"])
        ):
            raise common.ContractError("Rewire seller key drift")
        seller_index[seller_uid] = row
    item_index: dict[str, dict[str, Any]] = {}
    for source_row in items:
        if list(source_row) != item_schema:
            raise common.ContractError("Rewire item schema/order drift")
        row = dict(source_row)
        item_uid = str(row["item_uid"])
        seller = seller_index.get(str(row["seller_uid"]))
        if (
            not item_uid
            or item_uid in item_index
            or seller is None
            or str(seller["world_uid"]) != str(row["world_uid"])
        ):
            raise common.ContractError("Rewire item lineage drift")
        item_index[item_uid] = row
    ast_index: dict[str, dict[str, Any]] = {}
    ast_slot_uids: set[str] = set()
    for source_row in render_asts:
        if list(source_row) != ast_schema:
            raise common.ContractError("Rewire AST schema/order drift")
        row = dict(source_row)
        item_uid = str(row["item_uid"])
        item = item_index.get(item_uid)
        slot_uids = row["identity_slot_uids"]
        normalized_slot_uids = (
            [str(value) for value in slot_uids]
            if isinstance(slot_uids, list)
            else []
        )
        if (
            item is None
            or item_uid in ast_index
            or str(row["world_uid"]) != str(item["world_uid"])
            or str(row["seller_uid"]) != str(item["seller_uid"])
            or int(row["time_bucket"]) != int(item["time_bucket"])
            or not isinstance(slot_uids, list)
            or any(not value for value in normalized_slot_uids)
            or len(normalized_slot_uids) != len(set(normalized_slot_uids))
            or normalized_slot_uids
            != common.utf8_sort(normalized_slot_uids)
            or ast_slot_uids.intersection(normalized_slot_uids)
        ):
            raise common.ContractError("Rewire AST lineage drift")
        ast_index[item_uid] = row
        ast_slot_uids.update(normalized_slot_uids)
    if set(ast_index) != set(item_index):
        raise common.ContractError("Rewire AST/item keyset drift")

    safe_index: dict[str, dict[str, Any]] = {}
    identity_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    edges: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for source_row in safe_slots:
        if list(source_row) != safe_schema:
            raise common.ContractError("Rewire safe-slot schema/order drift")
        row = dict(source_row)
        slot_uid = str(row["slot_uid"])
        item = item_index.get(str(row["item_uid"]))
        identity_type = str(row["identity_type"]).strip().lower()
        normalized_value = str(
            row["downstream_canonical_value"]
        ).strip().lower()
        identity_uid = _identity_uid(identity_type, normalized_value)
        start, end = int(row["start"]), int(row["end"])
        if (
            not slot_uid
            or slot_uid in safe_index
            or item is None
            or str(row["world_uid"]) != str(item["world_uid"])
            or str(row["seller_uid"]) != str(item["seller_uid"])
            or str(row["field_name"]) != "description"
            or type(row["start"]) is not int
            or type(row["end"]) is not int
            or not 0 <= start < end <= len(str(item["description"]))
            or str(item["description"])[start:end] != str(row["raw_surface"])
            or str(row["identity_uid"]) != identity_uid
            or str(row["bundle_uid"])
            != _original_bundle_uid(
                str(row["world_uid"]),
                str(row["seller_uid"]),
                identity_uid,
            )
            or identity_type
            not in policy["identity_design"]["identity_types"]
        ):
            raise common.ContractError("Rewire safe-slot lineage/value drift")
        safe_index[slot_uid] = row
        identity_rows[identity_uid].append(row)
        edges[(str(row["seller_uid"]), identity_uid)].append(row)
    if set(safe_index) != ast_slot_uids:
        raise common.ContractError("Rewire safe-slot/AST universe drift")

    ledger_index: dict[str, str] = {}
    for source_row in nuisance_ledger:
        if list(source_row) != ["identity_uid", "nuisance_class"]:
            raise common.ContractError("Rewire nuisance-ledger schema/order drift")
        identity_uid = str(source_row["identity_uid"])
        if not identity_uid or identity_uid in ledger_index:
            raise common.ContractError("Rewire nuisance-ledger key drift")
        ledger_index[identity_uid] = str(source_row["nuisance_class"])
    if set(ledger_index) != set(identity_rows):
        raise common.ContractError("Rewire nuisance-ledger keyset drift")

    direct_maximum = int(
        policy["history_features"]["direct_token_seller_frequency_maximum"]
    )
    identity_catalog: dict[str, dict[str, str]] = {}
    for identity_uid, rows in identity_rows.items():
        worlds = {str(row["world_uid"]) for row in rows}
        types = {str(row["identity_type"]) for row in rows}
        normalized = {
            str(row["downstream_canonical_value"]).strip().lower()
            for row in rows
        }
        raw_surfaces = {str(row["raw_surface"]) for row in rows}
        seller_degree = len({str(row["seller_uid"]) for row in rows})
        if (
            len(worlds) != 1
            or len(types) != 1
            or len(normalized) != 1
            or len(raw_surfaces) != 1
        ):
            raise common.ContractError("Rewire identity catalog is ambiguous")
        nuisance = _aggregate_nuisance(
            rows,
            seller_degree=seller_degree,
            direct_maximum=direct_maximum,
        )
        if nuisance != ledger_index[identity_uid] or any(
            str(row["observed_nuisance_class"]) != nuisance for row in rows
        ):
            raise common.ContractError("Rewire nuisance class recomputation drift")
        identity_catalog[identity_uid] = {
            "world_uid": next(iter(worlds)),
            "identity_type": next(iter(types)),
            "normalized_value": next(iter(normalized)),
            "raw_surface": next(iter(raw_surfaces)),
            "nuisance_class": nuisance,
        }

    edge_layer: dict[tuple[str, str], tuple[str, str, str, int]] = {}
    counts_by_identity: dict[str, set[int]] = defaultdict(set)
    for edge, rows in edges.items():
        seller_uid, identity_uid = edge
        item_uids = {str(row["item_uid"]) for row in rows}
        if len(item_uids) != len(rows):
            raise common.ContractError(
                "One seller-identity edge repeats inside an item"
            )
        catalog = identity_catalog[identity_uid]
        seller = seller_index[seller_uid]
        count = len(rows)
        layer = (
            str(seller["world_uid"]),
            catalog["identity_type"],
            catalog["nuisance_class"],
            count,
        )
        if layer[0] != catalog["world_uid"]:
            raise common.ContractError("Rewire edge crosses worlds")
        edge_layer[edge] = layer
        counts_by_identity[identity_uid].add(count)
    if any(len(values) != 1 for values in counts_by_identity.values()):
        raise common.ContractError(
            "One identity spans multiple occurrence-count strata"
        )
    return _ValidatedInputs(
        seller_index=seller_index,
        item_index=item_index,
        ast_index=ast_index,
        safe_index=safe_index,
        identity_rows=dict(identity_rows),
        identity_catalog=identity_catalog,
        edges=dict(edges),
        edge_layer=edge_layer,
    )


def _stratum_sort_key(
    policy: Mapping[str, Any],
    layer: tuple[str, str, str, int],
) -> tuple[Any, ...]:
    type_order = {
        value: index
        for index, value in enumerate(policy["identity_design"]["identity_types"])
    }
    nuisance_order = {
        value: index
        for index, value in enumerate(policy["placebo"]["nuisance_priority"])
    }
    return (
        layer[0].encode("utf-8"),
        type_order[layer[1]],
        nuisance_order[layer[2]],
        layer[3],
    )


def _run_stratum_swaps(
    policy: Mapping[str, Any],
    *,
    seed: bytes,
    rewire_seed_id: str,
    layer: tuple[str, str, str, int],
    original_edges: set[tuple[str, str]],
    complete_type_edges: set[tuple[str, str]],
) -> tuple[set[tuple[str, str]], list[dict[str, Any]], dict[str, Any]]:
    world_uid, identity_type, nuisance_class, occurrence_count = layer
    fixed_rule = (
        "allow unchanged only for an unswappable non-direct_or_private "
        "observable nuisance stratum; an unswappable direct_or_private "
        "stratum fails"
    )
    if policy["placebo"]["structurally_fixed_stratum_rule"] != fixed_rule:
        raise common.ContractError(
            "Rewire structurally-fixed stratum rule drift"
        )
    layer_identifier = _layer_uid(*layer)
    current = set(original_edges)
    original_seller_degree = Counter(edge[0] for edge in original_edges)
    original_identity_degree = Counter(edge[1] for edge in original_edges)
    edge_count = len(original_edges)
    accepted = 0
    attempts = 0
    maximum_attempts = int(
        policy["placebo"]["maximum_attempt_multiplier_per_stratum_edge"]
    ) * max(edge_count, 1)
    required_accepted = int(
        policy["placebo"]["swap_acceptance_multiplier_per_stratum_edge"]
    ) * edge_count
    retention_maximum = float(
        policy["placebo"][
            "direct_or_private_original_edge_retention_rate_maximum"
        ]
    )
    manifest: list[dict[str, Any]] = []
    structurally_fixed = False
    while True:
        retained = len(current & original_edges)
        retention_rate = retained / edge_count if edge_count else 0.0
        accepted_gate = accepted >= required_accepted
        retention_gate = (
            nuisance_class != "direct_or_private"
            or retention_rate <= retention_maximum
        )
        if accepted_gate and retention_gate:
            break
        snapshot: list[
            tuple[bytes, bytes, tuple[str, str], tuple[str, str], str]
        ] = []
        ordered_edges = sorted(
            current, key=lambda edge: _edge_uid(edge).encode("utf-8")
        )
        for left_index, left in enumerate(ordered_edges):
            for right in ordered_edges[left_index + 1 :]:
                canonical_pair = _canonical_edge_pair(left, right)
                digest = _seed_hmac(
                    seed,
                    layer_identifier,
                    str(accepted),
                    canonical_pair,
                )
                snapshot.append(
                    (
                        digest,
                        canonical_pair.encode("utf-8"),
                        left,
                        right,
                        canonical_pair,
                    )
                )
        legal_found = False
        for digest, _pair_bytes, left, right, _canonical_pair in sorted(
            snapshot, key=lambda row: (row[0], row[1])
        ):
            if attempts >= maximum_attempts:
                raise common.ContractError(
                    "Rewire stratum reached its frozen attempt limit"
                )
            attempts += 1
            seller_left, identity_left = left
            seller_right, identity_right = right
            cross_left = (seller_left, identity_right)
            cross_right = (seller_right, identity_left)
            if (
                seller_left == seller_right
                or identity_left == identity_right
                or cross_left in complete_type_edges
                or cross_right in complete_type_edges
            ):
                continue
            legal_found = True
            complete_type_edges.remove(left)
            complete_type_edges.remove(right)
            current.remove(left)
            current.remove(right)
            current.add(cross_left)
            current.add(cross_right)
            complete_type_edges.add(cross_left)
            complete_type_edges.add(cross_right)
            ordered_current = sorted(
                (left, right), key=lambda edge: _edge_uid(edge).encode("utf-8")
            )
            first, second = ordered_current
            row = {
                "rewire_seed_id": rewire_seed_id,
                "layer_uid": layer_identifier,
                "iteration": accepted,
                "attempt_count_at_accept": attempts,
                "seller_uid_left": first[0],
                "identity_uid_left": first[1],
                "seller_uid_right": second[0],
                "identity_uid_right": second[1],
                "new_identity_uid_left": (
                    identity_right
                    if first == left
                    else identity_left
                ),
                "new_identity_uid_right": (
                    identity_left
                    if second == right
                    else identity_right
                ),
                "hmac_digest_hex": digest.hex(),
            }
            if list(row) != policy["placebo"]["rewire_manifest_schema"]:
                raise common.ContractError("Rewire manifest schema/order drift")
            manifest.append(row)
            accepted += 1
            break
        if legal_found:
            continue
        if (
            accepted == 0
            and nuisance_class != "direct_or_private"
        ):
            structurally_fixed = True
            break
        raise common.ContractError(
            "Rewire stratum has no legal candidate swap: "
            f"world={world_uid} type={identity_type} "
            f"nuisance={nuisance_class} count={occurrence_count} "
            f"E={edge_count} accepted={accepted} attempts={attempts}"
        )

    if (
        Counter(edge[0] for edge in current) != original_seller_degree
        or Counter(edge[1] for edge in current) != original_identity_degree
        or len(current) != edge_count
    ):
        raise common.ContractError("Rewire swap degree invariant failed")
    retained = len(current & original_edges)
    retention_rate = retained / edge_count if edge_count else 0.0
    audit = {
        "rewire_seed_id": rewire_seed_id,
        "layer_uid": layer_identifier,
        "world_uid": world_uid,
        "identity_type": identity_type,
        "nuisance_class": nuisance_class,
        "edge_occurrence_count": occurrence_count,
        "original_edge_count": edge_count,
        "attempt_count": attempts,
        "accepted_swap_count": accepted,
        "original_edge_retention_count": retained,
        "original_edge_retention_rate": f"{retention_rate:.12f}",
        "structurally_fixed_bool": (
            "true" if structurally_fixed else "false"
        ),
        "final_edge_multiset_sha256": common.canonical_sha256(
            sorted(
                (
                    {"seller_uid": seller_uid, "identity_uid": identity_uid}
                    for seller_uid, identity_uid in current
                ),
                key=lambda row: (
                    row["seller_uid"].encode("utf-8"),
                    row["identity_uid"].encode("utf-8"),
                ),
            )
        ),
    }
    if list(audit) != policy["placebo"]["rewire_stratum_audit_schema"]:
        raise common.ContractError("Rewire stratum-audit schema/order drift")
    return current, manifest, audit


@dataclass
class _FlowEdge:
    to: str
    reverse_index: int
    capacity: int


class _Dinic:
    def __init__(self) -> None:
        self.graph: dict[str, list[_FlowEdge]] = defaultdict(list)

    def add_edge(self, source: str, target: str, capacity: int) -> _FlowEdge:
        if capacity < 0:
            raise common.ContractError("Negative flow capacity")
        forward = _FlowEdge(target, len(self.graph[target]), capacity)
        reverse = _FlowEdge(source, len(self.graph[source]), 0)
        self.graph[source].append(forward)
        self.graph[target].append(reverse)
        return forward

    def max_flow(self, source: str, sink: str) -> int:
        total = 0
        while True:
            levels = {source: 0}
            queue: deque[str] = deque([source])
            while queue:
                node = queue.popleft()
                for edge in self.graph[node]:
                    if edge.capacity > 0 and edge.to not in levels:
                        levels[edge.to] = levels[node] + 1
                        queue.append(edge.to)
            if sink not in levels:
                return total
            cursors: dict[str, int] = defaultdict(int)

            def send(node: str, amount: int) -> int:
                if node == sink:
                    return amount
                while cursors[node] < len(self.graph[node]):
                    edge = self.graph[node][cursors[node]]
                    if (
                        edge.capacity > 0
                        and levels.get(edge.to) == levels[node] + 1
                    ):
                        pushed = send(edge.to, min(amount, edge.capacity))
                        if pushed:
                            edge.capacity -= pushed
                            reverse = self.graph[edge.to][edge.reverse_index]
                            reverse.capacity += pushed
                            return pushed
                    cursors[node] += 1
                return 0

            while True:
                pushed = send(source, 1 << 60)
                if not pushed:
                    break
                total += pushed


def _slotflow_assign(
    *,
    seed: bytes,
    layer_identifier: str,
    seller_uid: str,
    occurrence_count: int,
    identities: Sequence[str],
    slots: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    slotflow_uid = "sf_" + common.canonical_sha256(
        {"layer_uid": layer_identifier, "seller_uid": seller_uid}
    )
    source_uid = "src_" + common.sha256_bytes(
        (slotflow_uid + "\x1fsource").encode("utf-8")
    )
    sink_uid = "snk_" + common.sha256_bytes(
        (slotflow_uid + "\x1fsink").encode("utf-8")
    )
    slots_by_item: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for slot in slots:
        slots_by_item[str(slot["item_uid"])].append(slot)
    edge_specs: list[
        tuple[str, tuple[str, ...], int, str | None, str | None]
    ] = []
    for identity_uid in common.utf8_sort(identities):
        edge_specs.append(
            (
                "source_identity",
                (source_uid, identity_uid),
                occurrence_count,
                None,
                None,
            )
        )
        for item_uid in common.utf8_sort(slots_by_item):
            composite_uid = "ii_" + common.canonical_sha256(
                {
                    "layer_uid": layer_identifier,
                    "seller_uid": seller_uid,
                    "identity_uid": identity_uid,
                    "item_uid": item_uid,
                }
            )
            edge_specs.append(
                (
                    "identity_identity_item",
                    (identity_uid, composite_uid),
                    1,
                    None,
                    None,
                )
            )
            for slot in slots_by_item[item_uid]:
                slot_uid = str(slot["slot_uid"])
                edge_specs.append(
                    (
                        "identity_item_slot",
                        (composite_uid, slot_uid),
                        1,
                        identity_uid,
                        slot_uid,
                    )
                )
    for slot in slots:
        edge_specs.append(
            (
                "slot_sink",
                (str(slot["slot_uid"]), sink_uid),
                1,
                None,
                None,
            )
        )

    def sort_key(
        spec: tuple[str, tuple[str, ...], int, str | None, str | None]
    ) -> tuple[Any, ...]:
        kind, endpoints, _capacity, _identity_uid, _slot_uid = spec
        digest = _seed_hmac(
            seed, layer_identifier, "slotflow", kind, *endpoints
        )
        directed = "\x1f".join(endpoints).encode("utf-8")
        return EDGE_KIND_ORDER[kind], digest, directed

    solver = _Dinic()
    assignment_edges: list[tuple[str, str, _FlowEdge]] = []
    for kind, endpoints, capacity, identity_uid, slot_uid in sorted(
        edge_specs, key=sort_key
    ):
        forward = solver.add_edge(endpoints[0], endpoints[1], capacity)
        if kind == "identity_item_slot":
            if identity_uid is None or slot_uid is None:
                raise common.ContractError("Slotflow assignment metadata drift")
            assignment_edges.append((identity_uid, slot_uid, forward))
    required = len(slots)
    if solver.max_flow(source_uid, sink_uid) != required:
        raise common.ContractError("Rewire slot flow did not saturate")
    assignment: dict[str, str] = {}
    identity_counts: Counter[str] = Counter()
    for identity_uid, slot_uid, edge in assignment_edges:
        if edge.capacity == 0:
            if slot_uid in assignment:
                raise common.ContractError("Rewire slot received two identities")
            assignment[slot_uid] = identity_uid
            identity_counts[identity_uid] += 1
    if (
        len(assignment) != required
        or identity_counts
        != Counter({identity_uid: occurrence_count for identity_uid in identities})
    ):
        raise common.ContractError("Rewire slotflow assignment closure failed")
    return assignment


def _render_rewired_items(
    policy: Mapping[str, Any],
    *,
    rewire_seed_id: str,
    validated: _ValidatedInputs,
    assignment_identity_by_slot: Mapping[str, str],
    layer_uid_by_slot: Mapping[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    assignments: list[dict[str, Any]] = []
    rewired_items: list[dict[str, Any]] = []
    slots_by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for slot_uid, row in validated.safe_index.items():
        if slot_uid not in assignment_identity_by_slot:
            raise common.ContractError("Rewire assignment misses a safe slot")
        slots_by_item[str(row["item_uid"])].append(row)
    for item_uid in common.utf8_sort(validated.item_index):
        item = validated.item_index[item_uid]
        original_description = str(item["description"])
        cursor = 0
        chunks: list[str] = []
        output_length = 0
        for original in sorted(
            slots_by_item.get(item_uid, []),
            key=lambda row: (
                int(row["start"]),
                int(row["end"]),
                str(row["slot_uid"]).encode("utf-8"),
            ),
        ):
            start, end = int(original["start"]), int(original["end"])
            if start < cursor:
                raise common.ContractError("Rewire safe spans overlap")
            unchanged = original_description[cursor:start]
            chunks.append(unchanged)
            output_length += len(unchanged)
            new_identity_uid = assignment_identity_by_slot[
                str(original["slot_uid"])
            ]
            catalog = validated.identity_catalog[new_identity_uid]
            raw_surface = catalog["raw_surface"]
            new_start = output_length
            chunks.append(raw_surface)
            output_length += len(raw_surface)
            new_end = output_length
            layer_identifier = layer_uid_by_slot[str(original["slot_uid"])]
            assignment = {
                "rewire_seed_id": rewire_seed_id,
                "layer_uid": layer_identifier,
                "slot_uid": str(original["slot_uid"]),
                "original_bundle_uid": str(original["bundle_uid"]),
                "rewired_bundle_uid": _rewired_bundle_uid(
                    rewire_seed_id,
                    layer_identifier,
                    str(original["seller_uid"]),
                    new_identity_uid,
                ),
                "world_uid": str(original["world_uid"]),
                "item_uid": item_uid,
                "seller_uid": str(original["seller_uid"]),
                "field_name": str(original["field_name"]),
                "start": new_start,
                "end": new_end,
                "original_identity_uid": str(original["identity_uid"]),
                "rewired_identity_uid": new_identity_uid,
                "identity_type": catalog["identity_type"],
                "downstream_canonical_value": catalog["normalized_value"],
                "raw_surface": raw_surface,
                "time_bucket": int(original["time_bucket"]),
                "observed_seller_facing_context": int(
                    original["observed_seller_facing_context"]
                ),
                "observed_product_data_risk_context": int(
                    original["observed_product_data_risk_context"]
                ),
                "observed_direct_identity_eligible": int(
                    original["observed_direct_identity_eligible"]
                ),
                "observed_support_only": int(
                    original["observed_support_only"]
                ),
                "observed_nuisance_class": str(
                    original["observed_nuisance_class"]
                ),
            }
            if (
                list(assignment)
                != policy["placebo"]["rewired_slot_assignment_schema"]
            ):
                raise common.ContractError(
                    "Rewired slot-assignment schema/order drift"
                )
            assignments.append(assignment)
            cursor = end
        unchanged_tail = original_description[cursor:]
        chunks.append(unchanged_tail)
        rewired_description = "".join(chunks)
        row = {**item, "description": rewired_description}
        if (
            list(row)
            != policy["relational_integrity"]["observed_core_schemas"][
                "items.jsonl"
            ]
            or str(row["title"]) != str(item["title"])
        ):
            raise common.ContractError("Rewired observed-item schema/title drift")
        rewired_items.append(row)
    if set(assignment_identity_by_slot) != {
        str(row["slot_uid"]) for row in assignments
    }:
        raise common.ContractError("Rewired assignment keyset drift")
    assignments.sort(
        key=lambda row: (
            row["rewire_seed_id"].encode("utf-8"),
            row["world_uid"].encode("utf-8"),
            row["seller_uid"].encode("utf-8"),
            row["item_uid"].encode("utf-8"),
            int(row["start"]),
            row["slot_uid"].encode("utf-8"),
        )
    )
    rewired_items.sort(
        key=lambda row: (
            str(row["world_uid"]).encode("utf-8"),
            str(row["seller_uid"]).encode("utf-8"),
            str(row["item_uid"]).encode("utf-8"),
        )
    )
    return rewired_items, assignments


def _validate_rewired_parser(
    policy: Mapping[str, Any],
    *,
    mode: str,
    split: str,
    sellers: Sequence[Mapping[str, Any]],
    rewired_items: Sequence[Mapping[str, Any]],
    assignments: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sellers_by_world: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    items_by_world: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for seller in sellers:
        sellers_by_world[str(seller["world_uid"])].append(seller)
    for item in rewired_items:
        items_by_world[str(item["world_uid"])].append(item)
    parsed_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    for world_uid in common.utf8_sort(sellers_by_world):
        world_parsed = production.parse_observed_world(
            policy,
            mode=mode,
            split=split,
            sellers=sellers_by_world[world_uid],
            items=items_by_world[world_uid],
        )
        parsed_rows.extend(world_parsed)
        history_rows.extend(
            production.project_history_safe_occurrences(
                policy,
                mode=mode,
                split=split,
                sellers=sellers_by_world[world_uid],
                items=items_by_world[world_uid],
                parsed_rows=world_parsed,
            )
        )
    expected = {
        (
            str(row["item_uid"]),
            str(row["field_name"]),
            str(row["identity_type"]),
            str(row["downstream_canonical_value"]),
            int(row["observed_seller_facing_context"]),
            int(row["observed_product_data_risk_context"]),
            int(row["observed_direct_identity_eligible"]),
            int(row["observed_support_only"]),
        )
        for row in assignments
    }
    actual = {
        (
            str(row["item_uid"]),
            str(row["source_field"]),
            str(row["contact_type"]),
            str(row["normalized_value"]),
            int(row["seller_facing_context"]),
            int(row["product_data_risk_context"]),
            int(row["direct_identity_eligible"]),
            int(row["support_only"]),
        )
        for row in parsed_rows
    }
    if (
        len(expected) != len(assignments)
        or len(actual) != len(parsed_rows)
        or expected != actual
    ):
        raise common.ContractError(
            "Rewired parser rows/flags differ from slot-local expectation"
        )
    return parsed_rows, history_rows


def build_one_placebo(
    policy: Mapping[str, Any],
    *,
    mode: str,
    split: str,
    seed_hex: str,
    sellers: Sequence[Mapping[str, Any]],
    items: Sequence[Mapping[str, Any]],
    safe_slots: Sequence[Mapping[str, Any]],
    nuisance_ledger: Sequence[Mapping[str, Any]],
    render_asts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build one complete development-smoke train placebo."""

    if mode != "development_smoke" or split != "train":
        raise common.ContractError(
            "Placebo rewire implementation is development-smoke train only"
        )
    try:
        seed = bytes.fromhex(seed_hex)
    except ValueError as exc:
        raise common.ContractError("Rewire seed is not hex") from exc
    if len(seed) != 32:
        raise common.ContractError("Rewire seed must contain 32 raw bytes")
    registered = policy["randomness"][mode]["rewire_key_hexes"]
    if seed_hex not in registered or registered.count(seed_hex) != 1:
        raise common.ContractError("Rewire seed is not uniquely registered")
    rewire_seed_id = "rws_" + hashlib.sha256(seed).hexdigest()
    validated = _validate_inputs(
        policy,
        sellers=sellers,
        items=items,
        safe_slots=safe_slots,
        nuisance_ledger=nuisance_ledger,
        render_asts=render_asts,
    )

    original_edges_by_layer: dict[
        tuple[str, str, str, int], set[tuple[str, str]]
    ] = defaultdict(set)
    for edge, layer in validated.edge_layer.items():
        original_edges_by_layer[layer].add(edge)
    complete_edges_by_world_type: dict[
        tuple[str, str], set[tuple[str, str]]
    ] = defaultdict(set)
    for edge, layer in validated.edge_layer.items():
        complete_edges_by_world_type[(layer[0], layer[1])].add(edge)

    final_edges_by_layer: dict[
        tuple[str, str, str, int], set[tuple[str, str]]
    ] = {}
    manifest: list[dict[str, Any]] = []
    stratum_audits: list[dict[str, Any]] = []
    for layer in sorted(
        original_edges_by_layer,
        key=lambda value: _stratum_sort_key(policy, value),
    ):
        final_edges, layer_manifest, layer_audit = _run_stratum_swaps(
            policy,
            seed=seed,
            rewire_seed_id=rewire_seed_id,
            layer=layer,
            original_edges=set(original_edges_by_layer[layer]),
            complete_type_edges=complete_edges_by_world_type[
                (layer[0], layer[1])
            ],
        )
        final_edges_by_layer[layer] = final_edges
        manifest.extend(layer_manifest)
        stratum_audits.append(layer_audit)

    assignment_identity_by_slot: dict[str, str] = {}
    layer_uid_by_slot: dict[str, str] = {}
    for layer in sorted(
        final_edges_by_layer,
        key=lambda value: _stratum_sort_key(policy, value),
    ):
        layer_identifier = _layer_uid(*layer)
        occurrence_count = layer[3]
        original_layer_edges = original_edges_by_layer[layer]
        original_slots_by_seller: dict[
            str, list[dict[str, Any]]
        ] = defaultdict(list)
        for edge in original_layer_edges:
            original_slots_by_seller[edge[0]].extend(validated.edges[edge])
        final_identities_by_seller: dict[str, list[str]] = defaultdict(list)
        for seller_uid, identity_uid in final_edges_by_layer[layer]:
            final_identities_by_seller[seller_uid].append(identity_uid)
        if set(original_slots_by_seller) != set(final_identities_by_seller):
            raise common.ContractError("Rewire seller layer demand keyset drift")
        for seller_uid in common.utf8_sort(original_slots_by_seller):
            seller_assignment = _slotflow_assign(
                seed=seed,
                layer_identifier=layer_identifier,
                seller_uid=seller_uid,
                occurrence_count=occurrence_count,
                identities=final_identities_by_seller[seller_uid],
                slots=original_slots_by_seller[seller_uid],
            )
            for slot_uid, identity_uid in seller_assignment.items():
                if slot_uid in assignment_identity_by_slot:
                    raise common.ContractError(
                        "Rewire slot assigned in multiple layers"
                    )
                assignment_identity_by_slot[slot_uid] = identity_uid
                layer_uid_by_slot[slot_uid] = layer_identifier
    if set(assignment_identity_by_slot) != set(validated.safe_index):
        raise common.ContractError("Rewire did not assign every safe slot")

    rewired_items, assignments = _render_rewired_items(
        policy,
        rewire_seed_id=rewire_seed_id,
        validated=validated,
        assignment_identity_by_slot=assignment_identity_by_slot,
        layer_uid_by_slot=layer_uid_by_slot,
    )
    parsed_rows, history_rows = _validate_rewired_parser(
        policy,
        mode=mode,
        split=split,
        sellers=sellers,
        rewired_items=rewired_items,
        assignments=assignments,
    )
    rewired_asts = [
        {"rewire_seed_id": rewire_seed_id, **dict(row)}
        for row in sorted(
            render_asts,
            key=lambda row: (
                str(row["world_uid"]).encode("utf-8"),
                str(row["seller_uid"]).encode("utf-8"),
                str(row["item_uid"]).encode("utf-8"),
            ),
        )
    ]
    if any(
        list(row) != policy["placebo"]["rewired_ast_schema"]
        for row in rewired_asts
    ):
        raise common.ContractError("Rewired AST schema/order drift")
    manifest.sort(
        key=lambda row: (
            row["rewire_seed_id"].encode("utf-8"),
            row["layer_uid"].encode("utf-8"),
            int(row["iteration"]),
        )
    )
    if len({(row["layer_uid"], row["iteration"]) for row in manifest}) != len(
        manifest
    ):
        raise common.ContractError("Rewire manifest iteration collision")
    if len(parsed_rows) != len(safe_slots) or len(history_rows) != len(safe_slots):
        raise common.ContractError("Rewire occurrence-count preservation failed")
    return {
        "rewire_seed_id": rewire_seed_id,
        "rewire_manifest": manifest,
        "rewire_stratum_audit": stratum_audits,
        "rewired_slot_assignments": assignments,
        "rewired_asts": rewired_asts,
        "rewired_items": rewired_items,
        "rewired_parsed_identity_occurrences": parsed_rows,
        "rewired_history_safe_occurrences": history_rows,
        "label_or_controller_inputs_read": False,
        "canonical_self_hash": common.canonical_sha256(
            {
                "rewire_seed_id": rewire_seed_id,
                "rewire_manifest": manifest,
                "rewire_stratum_audit": stratum_audits,
                "rewired_slot_assignments": assignments,
                "rewired_asts": rewired_asts,
                "rewired_items": rewired_items,
                "rewired_parsed_identity_occurrences": parsed_rows,
                "rewired_history_safe_occurrences": history_rows,
            }
        ),
    }


def build_all_train_placebos(
    policy: Mapping[str, Any],
    *,
    mode: str,
    split: str,
    sellers: Sequence[Mapping[str, Any]],
    items: Sequence[Mapping[str, Any]],
    safe_slots: Sequence[Mapping[str, Any]],
    nuisance_ledger: Sequence[Mapping[str, Any]],
    render_asts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if mode != "development_smoke" or split != "train":
        raise common.ContractError(
            "All-placebo builder is development-smoke train only"
        )
    expected_world_count = int(
        policy["modes"][mode]["world_counts"]["train"]
    )
    seller_world_counts = Counter(
        str(row["world_uid"]) for row in sellers
    )
    seller_worlds = set(seller_world_counts)
    item_worlds = {str(row["world_uid"]) for row in items}
    slot_worlds = {str(row["world_uid"]) for row in safe_slots}
    ast_worlds = {str(row["world_uid"]) for row in render_asts}
    if (
        len(seller_worlds) != expected_world_count
        or set(seller_world_counts.values()) != {28}
        or item_worlds != seller_worlds
        or slot_worlds != seller_worlds
        or ast_worlds != seller_worlds
    ):
        raise common.ContractError(
            "All-placebo builder requires the complete train world set"
        )
    seeds = list(policy["randomness"][mode]["rewire_key_hexes"])
    if len(seeds) != int(policy["placebo"]["replicates"]) or len(set(seeds)) != 5:
        raise common.ContractError("Exactly five distinct rewire seeds are required")
    outputs = [
        build_one_placebo(
            policy,
            mode=mode,
            split=split,
            seed_hex=seed_hex,
            sellers=sellers,
            items=items,
            safe_slots=safe_slots,
            nuisance_ledger=nuisance_ledger,
            render_asts=render_asts,
        )
        for seed_hex in seeds
    ]
    if len({row["rewire_seed_id"] for row in outputs}) != 5:
        raise common.ContractError("Rewire seed-ID collision")
    return outputs
