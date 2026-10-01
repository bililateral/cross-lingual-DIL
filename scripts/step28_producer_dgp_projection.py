#!/usr/bin/env python3
"""Build the minimal producer-side Step 28-v13 typed DGP projection.

This module runs inside producer-private custody.  It is the only comparison
stage allowed to read the full producer world.  Its output excludes identity
values, rendered text, slot plans, parser rows, and non-comparison solver data.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any


TUPLE_SEPARATOR = "\x1e"
PROJECTION_VERSION = (
    "2026-07-28-step28-v13-producer-typed-dgp-projection-v1-draft"
)
PROJECTION_SCOPE = (
    "membership_market_style_mechanism_typed_identity_assets_"
    "positive_negative_targets_repeat_and_registered_overrides"
)
ASSET_DECISION_FIELDS = (
    "descriptor_kind",
    "descriptor_index",
    "descriptor_uid",
    "role",
    "sellers",
    "occurrence_counts",
    "allowed_types",
    "fixed_type",
    "distinct_groups",
    "repeat_draw_name",
    "repeat_probability",
    "identity_asset_uid",
    "asset_repeat_decision",
    "identity_type",
)
RAW_ASSET_FIELDS = set(ASSET_DECISION_FIELDS) | {
    "global_asset_index",
    "identity_uid",
    "identity_value",
}
RENDER_AST_FIELDS = {
    "world_uid",
    "seller_uid",
    "item_uid",
    "time_bucket",
    "category",
    "product",
    "attribute",
    "code",
    "delivery",
    "service",
    "title_skeleton_index",
    "description_skeleton_index",
    "title_nonempty",
    "description_nonempty",
    "effective_style_uid",
    "identity_slot_uids",
    "noise_slot_uid",
}
RAW_SOLVER_FIELDS = {
    "world_uid",
    "split",
    "graph_name",
    "market_proposal_counter",
    "membership_solver_node_count",
    "membership_complete_assignments_examined",
    "selected_membership_complete_assignment_ordinal",
    "membership_complete_assignments_type_tested",
    "type_solver_node_count",
    "identity_asset_count",
    "unused_identity_asset_uid_count",
    "risk_seller_uids",
    "zero_visible_seller_uids",
    "planned_occurrence_count",
    "slot_count",
    "maximum_flow",
    "source_node_uid",
    "sink_node_uid",
}
SOLVER_TRACE_FIELDS = (
    "world_uid",
    "split",
    "graph_name",
    "market_proposal_counter",
    "membership_solver_node_count",
    "membership_complete_assignments_examined",
    "selected_membership_complete_assignment_ordinal",
    "membership_complete_assignments_type_tested",
    "type_solver_node_count",
    "identity_asset_count",
    "unused_identity_asset_uid_count",
    "risk_seller_uids",
    "zero_visible_seller_uids",
)
PRIVATE_WORLD_FIELDS = {
    "controller_membership",
    "controller_style_groups",
    "identity_assets",
    "identity_slots_audit",
    "identity_slots_edit",
    "mechanism_assignments",
    "negative_flags",
    "noise_slots_audit",
    "override_audit",
    "positive_targets",
    "render_asts",
    "solver_audit",
}
PUBLIC_WORLD_FIELDS = {
    "world",
    "sellers",
    "items",
    "complete_model_pair_endpoints",
}


class ProducerProjectionError(ValueError):
    """Raised when producer-private inputs do not match the fixed schema."""


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


def _require_row_schema(
    rows: Sequence[Mapping[str, Any]],
    *,
    fields: set[str],
    label: str,
) -> None:
    if any(not isinstance(row, Mapping) or set(row) != fields for row in rows):
        raise ProducerProjectionError(
            f"PRODUCER_PROJECTION_ROW_SCHEMA_INVALID:{label}"
        )


def _repeat_decisions(
    *,
    mechanism_rows: Sequence[Mapping[str, Any]],
    asset_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for asset in asset_rows:
        draw_name = asset["repeat_draw_name"]
        if draw_name is None:
            if asset["asset_repeat_decision"] is not None:
                raise ProducerProjectionError(
                    "PRODUCER_PROJECTION_REPEAT_SCHEMA_INVALID"
                )
            continue
        decision = asset["asset_repeat_decision"]
        if not isinstance(decision, bool):
            raise ProducerProjectionError(
                "PRODUCER_PROJECTION_REPEAT_SCHEMA_INVALID"
            )
        output.append(
            {
                "decision_kind": str(draw_name),
                "subject_uid": str(asset["identity_asset_uid"]),
                "decision": decision,
            }
        )

    controllers = [
        str(row["controller_uid"])
        for row in mechanism_rows
        if row["mechanism"] == "single_hop_rotation"
    ]
    for controller_uid in controllers:
        prefix = f"{controller_uid}{TUPLE_SEPARATOR}"
        path_assets = [
            row
            for row in asset_rows
            if row["descriptor_kind"] == "single_hop_rotation"
            and str(row["descriptor_index"]).startswith(prefix)
        ]
        by_index: dict[int, Mapping[str, Any]] = {}
        for row in path_assets:
            suffix = str(row["descriptor_index"])[len(prefix) :]
            if not suffix.isdigit() or int(suffix) in by_index:
                raise ProducerProjectionError(
                    "PRODUCER_PROJECTION_SINGLE_HOP_DESCRIPTOR_INVALID"
                )
            by_index[int(suffix)] = row
        if set(by_index) != {0, 1}:
            raise ProducerProjectionError(
                "PRODUCER_PROJECTION_SINGLE_HOP_ASSET_COUNT_INVALID"
            )
        repeated_indices: list[int] = []
        for asset_index, row in by_index.items():
            counts = {
                int(value) for value in row["occurrence_counts"].values()
            }
            if counts == {2}:
                repeated_indices.append(asset_index)
            elif counts != {1}:
                raise ProducerProjectionError(
                    "PRODUCER_PROJECTION_SINGLE_HOP_REPEAT_INVALID"
                )
        if len(repeated_indices) > 1:
            raise ProducerProjectionError(
                "PRODUCER_PROJECTION_SINGLE_HOP_REPEAT_INVALID"
            )
        output.extend(
            (
                {
                    "decision_kind": "single_hop_path_repeat",
                    "subject_uid": controller_uid,
                    "decision": bool(repeated_indices),
                },
                {
                    "decision_kind": "single_hop_repeat_side",
                    "subject_uid": controller_uid,
                    "decision": (
                        "left_middle"
                        if repeated_indices == [0]
                        else (
                            "middle_right"
                            if repeated_indices == [1]
                            else ""
                        )
                    ),
                },
            )
        )
    output.sort(
        key=lambda row: (
            str(row["decision_kind"]).encode("utf-8"),
            str(row["subject_uid"]).encode("utf-8"),
        )
    )
    return output


def _override_decisions(
    *,
    override_rows: Sequence[Mapping[str, Any]],
    render_asts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    ast_by_item = {str(row["item_uid"]): row for row in render_asts}
    if len(ast_by_item) != len(render_asts):
        raise ProducerProjectionError(
            "PRODUCER_PROJECTION_RENDER_AST_ITEM_DUPLICATE"
        )
    output: list[dict[str, Any]] = []
    basic_fields = (
        "override_kind",
        "asset_index",
        "canonical_pair_uid",
        "seller_uid_left",
        "seller_uid_right",
        "item_uid_left",
        "item_uid_right",
    )
    for raw in override_rows:
        row = {field: raw[field] for field in basic_fields}
        kind = str(row["override_kind"])
        left = ast_by_item.get(str(row["item_uid_left"]))
        right = ast_by_item.get(str(row["item_uid_right"]))
        if left is None or right is None:
            raise ProducerProjectionError(
                "PRODUCER_PROJECTION_OVERRIDE_ITEM_AST_MISSING"
            )
        if (
            str(left["seller_uid"]) != str(row["seller_uid_left"])
            or str(right["seller_uid"]) != str(row["seller_uid_right"])
        ):
            raise ProducerProjectionError(
                "PRODUCER_PROJECTION_OVERRIDE_ITEM_SELLER_MISMATCH"
            )
        if kind == "high_semantic_similarity":
            if (
                left["category"] != right["category"]
                or left["product"] != right["product"]
                or left["attribute"] != right["attribute"]
                or left["title_skeleton_index"]
                == right["title_skeleton_index"]
            ):
                raise ProducerProjectionError(
                    "PRODUCER_PROJECTION_HIGH_SEMANTIC_AST_INVALID"
                )
            row.update(
                {
                    "category": left["category"],
                    "product": left["product"],
                    "attribute": left["attribute"],
                    "title_skeleton_index_left": left[
                        "title_skeleton_index"
                    ],
                    "title_skeleton_index_right": right[
                        "title_skeleton_index"
                    ],
                }
            )
        elif kind == "exact_title_clone":
            row.update(
                {
                    "category": None,
                    "product": None,
                    "attribute": None,
                    "title_skeleton_index_left": None,
                    "title_skeleton_index_right": None,
                }
            )
        else:
            raise ProducerProjectionError(
                "PRODUCER_PROJECTION_OVERRIDE_KIND_INVALID"
            )
        output.append(row)
    return output


def project_world(
    *,
    world: Mapping[str, Any],
    mode: str,
    split: str,
) -> dict[str, Any]:
    """Project one producer world to the exact comparator input envelope."""

    if mode not in {"development_smoke", "training_ready"}:
        raise ProducerProjectionError(
            "PRODUCER_PROJECTION_FORMAL_CAPABILITY_NOT_IMPLEMENTED"
        )
    if set(world) != {"public", "private"}:
        raise ProducerProjectionError(
            "PRODUCER_PROJECTION_WORLD_SCHEMA_INVALID"
        )
    public = world["public"]
    private = world["private"]
    if (
        not isinstance(public, Mapping)
        or set(public) != PUBLIC_WORLD_FIELDS
        or not isinstance(private, Mapping)
        or set(private) != PRIVATE_WORLD_FIELDS
    ):
        raise ProducerProjectionError(
            "PRODUCER_PROJECTION_WORLD_SCHEMA_INVALID"
        )
    if (
        not isinstance(public["world"], Mapping)
        or set(public["world"]) != {"world_uid"}
    ):
        raise ProducerProjectionError(
            "PRODUCER_PROJECTION_WORLD_UID_SCHEMA_INVALID"
        )
    world_uid = str(public["world"]["world_uid"])
    table_schemas = (
        (
            public["sellers"],
            {"world_uid", "seller_uid", "market"},
            "seller_markets",
        ),
        (
            private["controller_membership"],
            {"world_uid", "controller_uid", "seller_uid"},
            "controller_membership",
        ),
        (
            private["controller_style_groups"],
            {"world_uid", "controller_uid", "style_id"},
            "controller_style_groups",
        ),
        (
            private["mechanism_assignments"],
            {
                "world_uid",
                "controller_uid",
                "mechanism",
                "mechanism_slot_uid",
            },
            "mechanism_assignments",
        ),
        (
            private["identity_assets"],
            RAW_ASSET_FIELDS,
            "identity_assets",
        ),
        (
            private["positive_targets"],
            {
                "controller_uid",
                "mechanism",
                "mechanism_slot_uid",
                "seller_uid_left",
                "seller_uid_right",
                "canonical_pair_uid",
            },
            "positive_targets",
        ),
        (
            private["negative_flags"],
            {"canonical_pair_uid", "flag", "asset_index"},
            "negative_flags",
        ),
        (
            private["override_audit"],
            {
                "override_kind",
                "asset_index",
                "canonical_pair_uid",
                "seller_uid_left",
                "seller_uid_right",
                "item_uid_left",
                "item_uid_right",
            },
            "override_audit",
        ),
        (
            private["render_asts"],
            RENDER_AST_FIELDS,
            "render_asts",
        ),
    )
    for rows, fields, label in table_schemas:
        if not isinstance(rows, Sequence):
            raise ProducerProjectionError(
                f"PRODUCER_PROJECTION_TABLE_TYPE_INVALID:{label}"
            )
        _require_row_schema(rows, fields=fields, label=label)
    solver = private["solver_audit"]
    if not isinstance(solver, Mapping) or set(solver) != RAW_SOLVER_FIELDS:
        raise ProducerProjectionError(
            "PRODUCER_PROJECTION_SOLVER_SCHEMA_INVALID"
        )
    if (
        str(solver["world_uid"]) != world_uid
        or str(solver["split"]) != split
    ):
        raise ProducerProjectionError(
            "PRODUCER_PROJECTION_WORLD_OR_SPLIT_MISMATCH"
        )
    assets = [
        {field: row[field] for field in ASSET_DECISION_FIELDS}
        for row in private["identity_assets"]
    ]
    tables = {
        "controller_membership": [
            dict(row) for row in private["controller_membership"]
        ],
        "seller_markets": [dict(row) for row in public["sellers"]],
        "controller_style_groups": [
            dict(row) for row in private["controller_style_groups"]
        ],
        "mechanism_assignments": [
            dict(row) for row in private["mechanism_assignments"]
        ],
        "identity_asset_decisions": assets,
        "positive_targets": [
            dict(row) for row in private["positive_targets"]
        ],
        "negative_flags": [dict(row) for row in private["negative_flags"]],
        "registered_override_decisions": _override_decisions(
            override_rows=private["override_audit"],
            render_asts=private["render_asts"],
        ),
        "repeat_decisions": _repeat_decisions(
            mechanism_rows=private["mechanism_assignments"],
            asset_rows=private["identity_assets"],
        ),
        "solver_trace": {
            field: solver[field] for field in SOLVER_TRACE_FIELDS
        },
    }
    projection = {
        "version": PROJECTION_VERSION,
        "scope": PROJECTION_SCOPE,
        "mode": mode,
        "split": split,
        "world_uid": world_uid,
        "graph_name": str(solver["graph_name"]),
        "tables": tables,
        "typed_projection_sha256": _canonical_sha256(tables),
    }
    projection["canonical_self_hash"] = _canonical_sha256(projection)
    return projection
