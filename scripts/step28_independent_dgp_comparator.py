#!/usr/bin/env python3
"""No-secret comparator for two minimal Step 28-v13 typed DGP projections."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any


REPLAY_LEDGER_VERSION = (
    "2026-07-28-step28-v13-independent-typed-replay-v2-draft"
)
PRODUCER_PROJECTION_VERSION = (
    "2026-07-28-step28-v13-producer-typed-dgp-projection-v1-draft"
)
REPLAY_SCOPE = (
    "membership_market_style_mechanism_typed_identity_assets_"
    "positive_negative_targets_repeat_and_registered_overrides"
)
DEVELOPMENT_EVIDENCE_LEVEL = (
    "INDEPENDENT_TYPED_DGP_REPLAY_DEVELOPMENT_INTEGRATION_"
    "NOT_FORMAL_CUSTODY_SEAL"
)
TRAINING_READY_EVIDENCE_LEVEL = (
    "INDEPENDENT_TYPED_DGP_REPLAY_TRAINING_READY_"
    "MATHEMATICAL_INTEGRATION_NOT_FORMAL_CUSTODY_SEAL"
)
# Backward-compatible name used by the development-smoke receipt validator.
EVIDENCE_LEVEL = DEVELOPMENT_EVIDENCE_LEVEL
TABLE_ROW_FIELDS = {
    "controller_membership": {
        "world_uid",
        "controller_uid",
        "seller_uid",
    },
    "seller_markets": {"world_uid", "seller_uid", "market"},
    "controller_style_groups": {
        "world_uid",
        "controller_uid",
        "style_id",
    },
    "mechanism_assignments": {
        "world_uid",
        "controller_uid",
        "mechanism",
        "mechanism_slot_uid",
    },
    "identity_asset_decisions": {
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
    },
    "positive_targets": {
        "controller_uid",
        "mechanism",
        "mechanism_slot_uid",
        "seller_uid_left",
        "seller_uid_right",
        "canonical_pair_uid",
    },
    "negative_flags": {"canonical_pair_uid", "flag", "asset_index"},
    "repeat_decisions": {"decision_kind", "subject_uid", "decision"},
    "registered_override_decisions": {
        "override_kind",
        "asset_index",
        "canonical_pair_uid",
        "seller_uid_left",
        "seller_uid_right",
        "item_uid_left",
        "item_uid_right",
        "category",
        "product",
        "attribute",
        "title_skeleton_index_left",
        "title_skeleton_index_right",
    },
}
SOLVER_TRACE_FIELDS = {
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
}
TABLE_NAMES = set(TABLE_ROW_FIELDS) | {"solver_trace"}
OBSERVED_AUDIT_FIELDS = {
    "seller_uid_pool_sha256",
    "seller_count",
    "all_item_count",
    "nonempty_title_item_count",
    "nonempty_description_item_count",
    "all_item_uid_pool_sha256",
    "nonempty_title_item_uid_pool_sha256",
    "nonempty_description_item_uid_pool_sha256",
}


class IndependentDgpComparisonError(ValueError):
    """Raised when a sealed producer projection differs from replay."""


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


def _same(left: Any, right: Any) -> bool:
    return _canonical_json(left) == _canonical_json(right)


def build_observed_uid_pools(
    *,
    world_uid: str,
    sellers: Sequence[Mapping[str, Any]],
    items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Project public rows to UID-only pools; no raw text leaves this function."""

    seller_uids = sorted(
        (str(row["seller_uid"]) for row in sellers),
        key=lambda value: value.encode("utf-8"),
    )
    if (
        len(seller_uids) != 28
        or len(set(seller_uids)) != 28
        or any(
            set(row) != {"world_uid", "seller_uid", "market"}
            or str(row["world_uid"]) != world_uid
            for row in sellers
        )
    ):
        raise IndependentDgpComparisonError(
            "REPLAY_OBSERVED_SELLER_PROJECTION_INVALID"
        )
    seller_set = set(seller_uids)
    all_rows: list[dict[str, str]] = []
    title_rows: list[dict[str, str]] = []
    description_rows: list[dict[str, str]] = []
    required_item_fields = {
        "world_uid",
        "seller_uid",
        "item_uid",
        "time_bucket",
        "category",
        "title",
        "description",
    }
    for raw in items:
        if set(raw) != required_item_fields:
            raise IndependentDgpComparisonError(
                "REPLAY_OBSERVED_ITEM_SCHEMA_INVALID"
            )
        row = {
            "world_uid": str(raw["world_uid"]),
            "seller_uid": str(raw["seller_uid"]),
            "item_uid": str(raw["item_uid"]),
        }
        if (
            row["world_uid"] != world_uid
            or row["seller_uid"] not in seller_set
            or not row["item_uid"]
        ):
            raise IndependentDgpComparisonError(
                "REPLAY_OBSERVED_ITEM_FOREIGN_KEY_INVALID"
            )
        all_rows.append(row)
        if str(raw["title"]):
            title_rows.append(dict(row))
        if str(raw["description"]):
            description_rows.append(dict(row))
    sort_key = lambda row: (
        row["seller_uid"].encode("utf-8"),
        row["item_uid"].encode("utf-8"),
    )
    all_rows.sort(key=sort_key)
    title_rows.sort(key=sort_key)
    description_rows.sort(key=sort_key)
    if len({row["item_uid"] for row in all_rows}) != len(all_rows):
        raise IndependentDgpComparisonError(
            "REPLAY_OBSERVED_ITEM_UID_DUPLICATE"
        )
    return {
        "observed_seller_uids": seller_uids,
        "observed_all_item_uid_rows": all_rows,
        "observed_nonempty_title_item_uid_rows": title_rows,
        "observed_nonempty_description_item_uid_rows": description_rows,
    }


def _validate_tables(
    tables: Any,
    *,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(tables, Mapping) or set(tables) != TABLE_NAMES:
        raise IndependentDgpComparisonError(
            f"REPLAY_{label}_TABLE_SET_INVALID"
        )
    for table_name, fields in TABLE_ROW_FIELDS.items():
        rows = tables[table_name]
        if (
            not isinstance(rows, list)
            or any(
                not isinstance(row, Mapping) or set(row) != fields
                for row in rows
            )
        ):
            raise IndependentDgpComparisonError(
                f"REPLAY_{label}_{table_name.upper()}_SCHEMA_INVALID"
            )
    solver = tables["solver_trace"]
    if not isinstance(solver, Mapping) or set(solver) != SOLVER_TRACE_FIELDS:
        raise IndependentDgpComparisonError(
            f"REPLAY_{label}_SOLVER_TRACE_SCHEMA_INVALID"
        )
    return tables


def _validate_replay_envelope(
    expected_replay: Mapping[str, Any],
) -> Mapping[str, Any]:
    required = {
        "version",
        "scope",
        "mode",
        "world_uid",
        "split",
        "graph_name",
        "observed_uid_pool_audit",
        "tables",
        "typed_replay_sha256",
        "secret_serialized",
        "producer_private_input_used",
        "canonical_self_hash",
    }
    if set(expected_replay) != required:
        raise IndependentDgpComparisonError(
            "REPLAY_EXPECTED_LEDGER_SCHEMA_INVALID"
        )
    envelope = dict(expected_replay)
    claimed_self_hash = envelope.pop("canonical_self_hash")
    if (
        not isinstance(claimed_self_hash, str)
        or _canonical_sha256(envelope) != claimed_self_hash
    ):
        raise IndependentDgpComparisonError(
            "REPLAY_EXPECTED_LEDGER_SELF_HASH_MISMATCH"
        )
    if (
        expected_replay["version"] != REPLAY_LEDGER_VERSION
        or expected_replay["scope"] != REPLAY_SCOPE
    ):
        raise IndependentDgpComparisonError(
            "REPLAY_EXPECTED_LEDGER_IDENTITY_INVALID"
        )
    if (
        expected_replay["secret_serialized"] is not False
        or expected_replay["producer_private_input_used"] is not False
    ):
        raise IndependentDgpComparisonError(
            "REPLAY_EXPECTED_LEDGER_BOUNDARY_INVALID"
        )
    audit = expected_replay["observed_uid_pool_audit"]
    if not isinstance(audit, Mapping) or set(audit) != OBSERVED_AUDIT_FIELDS:
        raise IndependentDgpComparisonError(
            "REPLAY_EXPECTED_UID_AUDIT_SCHEMA_INVALID"
        )
    tables = _validate_tables(
        expected_replay["tables"],
        label="EXPECTED",
    )
    if (
        _canonical_sha256(tables)
        != expected_replay["typed_replay_sha256"]
    ):
        raise IndependentDgpComparisonError(
            "REPLAY_EXPECTED_LEDGER_HASH_MISMATCH"
        )
    return tables


def _validate_producer_envelope(
    producer_projection: Mapping[str, Any],
    *,
    expected_replay: Mapping[str, Any],
) -> Mapping[str, Any]:
    required = {
        "version",
        "scope",
        "mode",
        "split",
        "world_uid",
        "graph_name",
        "tables",
        "typed_projection_sha256",
        "canonical_self_hash",
    }
    if set(producer_projection) != required:
        raise IndependentDgpComparisonError(
            "REPLAY_PRODUCER_PROJECTION_SCHEMA_INVALID"
        )
    envelope = dict(producer_projection)
    claimed_self_hash = envelope.pop("canonical_self_hash")
    if (
        not isinstance(claimed_self_hash, str)
        or _canonical_sha256(envelope) != claimed_self_hash
    ):
        raise IndependentDgpComparisonError(
            "REPLAY_PRODUCER_PROJECTION_SELF_HASH_MISMATCH"
        )
    if (
        producer_projection["version"] != PRODUCER_PROJECTION_VERSION
        or producer_projection["scope"] != REPLAY_SCOPE
    ):
        raise IndependentDgpComparisonError(
            "REPLAY_PRODUCER_PROJECTION_IDENTITY_INVALID"
        )
    for field in ("mode", "split", "world_uid", "graph_name"):
        if producer_projection[field] != expected_replay[field]:
            raise IndependentDgpComparisonError(
                f"REPLAY_PRODUCER_PROJECTION_{field.upper()}_MISMATCH"
            )
    tables = _validate_tables(
        producer_projection["tables"],
        label="PRODUCER",
    )
    if (
        _canonical_sha256(tables)
        != producer_projection["typed_projection_sha256"]
    ):
        raise IndependentDgpComparisonError(
            "REPLAY_PRODUCER_PROJECTION_HASH_MISMATCH"
        )
    return tables


def compare_typed_dgp(
    *,
    expected_replay: Mapping[str, Any],
    producer_projection: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare two exact envelopes; this function has no key or oracle input."""

    expected_tables = _validate_replay_envelope(expected_replay)
    actual_tables = _validate_producer_envelope(
        producer_projection,
        expected_replay=expected_replay,
    )
    comparison_order = (
        (
            "controller_membership",
            "REPLAY_CONTROLLER_MEMBERSHIP_MISMATCH",
        ),
        ("seller_markets", "REPLAY_MARKET_ASSIGNMENT_MISMATCH"),
        (
            "controller_style_groups",
            "REPLAY_CONTROLLER_STYLE_GROUP_MISMATCH",
        ),
        (
            "mechanism_assignments",
            "REPLAY_MECHANISM_ASSIGNMENT_MISMATCH",
        ),
        ("positive_targets", "REPLAY_POSITIVE_TARGET_MISMATCH"),
        ("negative_flags", "REPLAY_NEGATIVE_TARGET_MISMATCH"),
        (
            "registered_override_decisions",
            "REPLAY_REGISTERED_OVERRIDE_DECISION_MISMATCH",
        ),
        ("repeat_decisions", "REPLAY_REPEAT_DECISION_MISMATCH"),
        (
            "identity_asset_decisions",
            "REPLAY_IDENTITY_ASSET_DECISION_MISMATCH",
        ),
    )
    component_receipts: dict[str, dict[str, Any]] = {}
    for table_name, error_code in comparison_order:
        expected = expected_tables[table_name]
        actual = actual_tables[table_name]
        if not _same(actual, expected):
            raise IndependentDgpComparisonError(error_code)
        component_receipts[table_name] = {
            "producer_row_count": len(actual),
            "replayer_row_count": len(expected),
            "producer_sha256": _canonical_sha256(actual),
            "replayer_sha256": _canonical_sha256(expected),
            "exact_equal": True,
        }

    expected_solver = expected_tables["solver_trace"]
    actual_solver = actual_tables["solver_trace"]
    if (
        actual_solver["market_proposal_counter"]
        != expected_solver["market_proposal_counter"]
    ):
        raise IndependentDgpComparisonError(
            "REPLAY_MARKET_PROPOSAL_COUNTER_MISMATCH"
        )
    if not _same(actual_solver, expected_solver):
        raise IndependentDgpComparisonError("REPLAY_SOLVER_TRACE_MISMATCH")
    component_receipts["solver_trace"] = {
        "producer_row_count": 1,
        "replayer_row_count": 1,
        "producer_sha256": _canonical_sha256(actual_solver),
        "replayer_sha256": _canonical_sha256(expected_solver),
        "exact_equal": True,
    }
    actual_sha256 = _canonical_sha256(actual_tables)
    expected_sha256 = _canonical_sha256(expected_tables)
    if actual_sha256 != expected_sha256:
        raise IndependentDgpComparisonError(
            "REPLAY_FULL_TYPED_PROJECTION_MISMATCH"
        )
    return {
        "version": (
            "2026-07-28-step28-v13-independent-comparison-v2-draft"
        ),
        "world_uid": str(expected_replay["world_uid"]),
        "mode": str(expected_replay["mode"]),
        "split": str(expected_replay["split"]),
        "scope": str(expected_replay["scope"]),
        "evidence_level": (
            TRAINING_READY_EVIDENCE_LEVEL
            if str(expected_replay["mode"]) == "training_ready"
            else DEVELOPMENT_EVIDENCE_LEVEL
        ),
        "independent_typed_dgp_replay_pass": True,
        "independent_decision_implementation": True,
        "formal_custody_seal": False,
        "producer_private_input_used_by_replayer": False,
        "structure_key_serialized": False,
        "full_typed_projection_exact": True,
        "producer_typed_projection_sha256": actual_sha256,
        "replayer_typed_projection_sha256": expected_sha256,
        "observed_uid_pool_audit": dict(
            expected_replay["observed_uid_pool_audit"]
        ),
        "component_receipts": component_receipts,
    }
