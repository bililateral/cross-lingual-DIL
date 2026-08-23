#!/usr/bin/env python3
"""V9.2 label-free matrix preparation for 21 descriptive and 7 hard views."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

import step28_v13_common as common
import step28_v13_v1_13_quality_gate_registry_v9_2 as gate_registry
import step28_v13_v1_13_quality_probe_preparer_v9 as v9


VERSION = "2026-08-23-step28-v13-v1-13-quality-probe-preparer-v9-2"
ORIGINAL_AUTHOR_SURFACES = v9.TEXT_SURFACES
COUNTERFACTUAL_HARD_SURFACE = "surface_style_deranged_full"
TEXT_SURFACES = (*ORIGINAL_AUTHOR_SURFACES, COUNTERFACTUAL_HARD_SURFACE)
ORIGINAL_AUTHOR_MATRIX_COUNT = 21
COUNTERFACTUAL_MATRIX_COUNT = 7
TOTAL_TEXT_MATRIX_COUNT = 28


class QualityProbePreparationV92Error(v9.QualityProbePreparationError):
    """Raised when the V9.2 28-matrix pre-truth bundle drifts."""


@dataclass(frozen=True)
class FrozenTextBundleV92:
    matrices: tuple[v9.FrozenFeatureMatrix, ...]
    actual_consumption_receipts: tuple[dict[str, Any], ...]
    commitment_sha256: str


def _build_views_with_independent_f_p_u_consumption(
    *,
    items: Sequence[Mapping[str, Any]],
    profiles: Sequence[Mapping[str, Any]],
    endpoints: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, np.ndarray], dict[str, tuple[str, ...]]]:
    """Build F, P and U separately, then compare with the frozen V9 composition."""

    reference_views, reference_names = v9.text_views.build_text_probe_views(
        items=items,
        profiles=profiles,
        endpoints=endpoints,
    )
    fixed, fixed_names = v9.text_views._build_fixed_support_views(
        items=items,
        endpoints=endpoints,
    )
    item_counts = Counter(str(row["seller_uid"]) for row in items)
    production, production_names, numeric, numeric_names = (
        v9.text_views._build_production_views(
            profiles=profiles,
            endpoints=endpoints,
            item_counts_by_seller=item_counts,
        )
    )
    joint_names = tuple(
        [f"p::{name}" for name in production_names["p_full"]]
        + [f"fs::{name}" for name in fixed_names["fs_full"]]
        + [f"numeric::{name}" for name in numeric_names]
    )
    views = {
        **fixed,
        **production,
        "u_joint_full": np.column_stack(
            (production["p_full"], fixed["fs_full"], numeric)
        ),
    }
    names = {**fixed_names, **production_names, "u_joint_full": joint_names}
    if tuple(views) != v9.text_views.VIEW_ORDER or tuple(names) != (
        v9.text_views.VIEW_ORDER
    ):
        raise QualityProbePreparationV92Error("F/P/U view order drift")
    for view in v9.text_views.VIEW_ORDER:
        if names[view] != reference_names[view] or not np.array_equal(
            views[view], reference_views[view]
        ):
            raise QualityProbePreparationV92Error(
                f"Independent F/P/U reconstruction drift: {view}"
            )
    return views, names


def _ordered_worlds_from_path(
    row_keys: Sequence[tuple[str, str]],
) -> tuple[str, ...]:
    worlds: list[str] = []
    closed: set[str] = set()
    active: str | None = None
    for world_uid, _pair_uid in row_keys:
        if world_uid != active:
            if active is not None:
                closed.add(active)
            if world_uid in closed:
                raise QualityProbePreparationV92Error(
                    "F/P/U actual path world rows are not contiguous"
                )
            worlds.append(world_uid)
            active = world_uid
    return tuple(worlds)


def _actual_consumption_receipt(
    *,
    surface: str,
    path: str,
    matrices: Sequence[v9.FrozenFeatureMatrix],
    items: Sequence[Mapping[str, Any]],
    profiles: Sequence[Mapping[str, Any]],
    eligibility: v9.FrozenTextEligibility,
) -> dict[str, Any]:
    selected = tuple(matrices)
    if not selected or any(value.row_keys != selected[0].row_keys for value in selected):
        raise QualityProbePreparationV92Error("F/P/U path row order drift")
    for value in selected:
        v9.verify_frozen_feature_matrix(value)
    v9.verify_frozen_text_eligibility(eligibility)
    row_keys = selected[0].row_keys
    if eligibility.row_keys != row_keys:
        raise QualityProbePreparationV92Error(
            "F/P/U path did not consume the frozen eligibility row order"
        )
    pair_order = [list(value) for value in row_keys]
    world_order = list(_ordered_worlds_from_path(row_keys))
    mask_rows = [
        [world_uid, pair_uid, bool(keep)]
        for (world_uid, pair_uid), keep in zip(
            row_keys, eligibility.values, strict=True
        )
    ]
    eligible_order = [
        [world_uid, pair_uid]
        for (world_uid, pair_uid), keep in zip(
            row_keys, eligibility.values, strict=True
        )
        if keep
    ]
    item_count_projection = sorted(
        Counter(str(row["seller_uid"]) for row in items).items(),
        key=lambda value: value[0].encode("utf-8"),
    )
    if path == "F":
        actual_sources = {
            "items_sha256": common.canonical_sha256(items),
            "profiles_sha256": None,
            "item_count_projection_sha256": common.canonical_sha256(
                item_count_projection
            ),
        }
    elif path == "P":
        actual_sources = {
            "items_sha256": None,
            "profiles_sha256": common.canonical_sha256(profiles),
            "item_count_projection_sha256": common.canonical_sha256(
                item_count_projection
            ),
        }
    elif path == "U":
        actual_sources = {
            "items_sha256": common.canonical_sha256(items),
            "profiles_sha256": common.canonical_sha256(profiles),
            "item_count_projection_sha256": common.canonical_sha256(
                item_count_projection
            ),
        }
    else:
        raise QualityProbePreparationV92Error("Unknown F/P/U path")
    receipt: dict[str, Any] = {
        "surface": surface,
        "path": path,
        "view_names": [value.view for value in selected],
        "actual_sources": actual_sources,
        "actual_source_bundle_sha256": common.canonical_sha256(actual_sources),
        "actual_matrix_bundle_sha256": common.canonical_sha256(
            [
                {
                    "view": value.view,
                    "commitment_sha256": value.commitment_sha256,
                }
                for value in selected
            ]
        ),
        "pair_order_sha256": common.canonical_sha256(pair_order),
        "world_order_sha256": common.canonical_sha256(world_order),
        "eligibility_mask_sha256": common.canonical_sha256(mask_rows),
        "eligible_pair_order_sha256": common.canonical_sha256(eligible_order),
        "full_pair_count": len(pair_order),
        "eligible_pair_count": len(eligible_order),
    }
    receipt["canonical_self_hash"] = common.canonical_sha256(receipt)
    return receipt


def prepare_text_surface_matrices(
    *,
    surface: str,
    items: Sequence[Mapping[str, Any]],
    profiles: Sequence[Mapping[str, Any]],
    endpoints: Sequence[Mapping[str, Any]],
    ordered_world_uids: Sequence[str],
    sources: Sequence[v9.SourceCommitment],
    expected_sellers_per_world: int = v9.EXPECTED_SELLERS_PER_WORLD,
    expected_pairs_per_world: int = v9.EXPECTED_PAIRS_PER_WORLD,
) -> tuple[v9.FrozenFeatureMatrix, ...]:
    if surface in ORIGINAL_AUTHOR_SURFACES:
        return v9.prepare_text_surface_matrices(
            surface=surface,
            items=items,
            profiles=profiles,
            endpoints=endpoints,
            ordered_world_uids=ordered_world_uids,
            sources=sources,
            expected_sellers_per_world=expected_sellers_per_world,
            expected_pairs_per_world=expected_pairs_per_world,
        )
    if surface != COUNTERFACTUAL_HARD_SURFACE:
        raise QualityProbePreparationV92Error("Unknown V9.2 text model surface")
    row_keys, worlds, sellers_by_world = v9._validate_endpoints(
        endpoints,
        ordered_world_uids=ordered_world_uids,
        expected_pairs_per_world=expected_pairs_per_world,
    )
    if any(
        len(sellers_by_world[world_uid]) != expected_sellers_per_world
        for world_uid in worlds
    ):
        raise QualityProbePreparationV92Error("Text seller count per world drift")
    matrices, names = _build_views_with_independent_f_p_u_consumption(
        items=items,
        profiles=profiles,
        endpoints=endpoints,
    )
    return tuple(
        v9._freeze_owned_feature_matrix(
            family="text",
            view=f"{surface}::{view}",
            values=matrices[view],
            row_keys=row_keys,
            column_names=names[view],
            sources=sources,
        )
        for view in v9.text_views.VIEW_ORDER
    )


def freeze_all_text_surfaces_before_truth(
    *,
    surface_rows: Mapping[
        str, tuple[Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]]
    ],
    endpoints: Sequence[Mapping[str, Any]],
    ordered_world_uids: Sequence[str],
    sources_by_surface: Mapping[str, Sequence[v9.SourceCommitment]],
    text_eligibility: v9.FrozenTextEligibility,
) -> FrozenTextBundleV92:
    if tuple(surface_rows) != TEXT_SURFACES or tuple(sources_by_surface) != TEXT_SURFACES:
        raise QualityProbePreparationV92Error(
            "V9.2 surfaces must be supplied in frozen four-surface order"
        )
    output: list[v9.FrozenFeatureMatrix] = []
    receipts: list[dict[str, Any]] = []
    for surface in TEXT_SURFACES:
        items, profiles = surface_rows[surface]
        surface_matrices = prepare_text_surface_matrices(
                surface=surface,
                items=items,
                profiles=profiles,
                endpoints=endpoints,
                ordered_world_uids=ordered_world_uids,
                sources=sources_by_surface[surface],
            )
        output.extend(surface_matrices)
        by_view = {value.view.split("::", 1)[1]: value for value in surface_matrices}
        for path, view_names in (
            ("F", ("fs_full", "fs_title", "fs_template_surface")),
            ("P", ("p_full", "p_topic", "p_template_surface")),
            ("U", ("u_joint_full",)),
        ):
            receipts.append(
                _actual_consumption_receipt(
                    surface=surface,
                    path=path,
                    matrices=tuple(by_view[name] for name in view_names),
                    items=items,
                    profiles=profiles,
                    eligibility=text_eligibility,
                )
            )
    if (
        len(output) != TOTAL_TEXT_MATRIX_COUNT
        or sum(value.view.startswith(COUNTERFACTUAL_HARD_SURFACE + "::") for value in output)
        != COUNTERFACTUAL_MATRIX_COUNT
    ):
        raise QualityProbePreparationV92Error("V9.2 text matrix count drift")
    if len(receipts) != len(TEXT_SURFACES) * 3:
        raise QualityProbePreparationV92Error("F/P/U receipt cardinality drift")
    bundle_payload = {
        "matrix_commitments": [value.commitment_sha256 for value in output],
        "actual_consumption_receipt_hashes": [
            value["canonical_self_hash"] for value in receipts
        ],
        "text_eligibility_commitment_sha256": text_eligibility.commitment_sha256,
    }
    return FrozenTextBundleV92(
        matrices=tuple(output),
        actual_consumption_receipts=tuple(receipts),
        commitment_sha256=common.canonical_sha256(bundle_payload),
    )


def split_text_matrix_roles(
    matrices: Sequence[v9.FrozenFeatureMatrix] | FrozenTextBundleV92,
) -> tuple[tuple[v9.FrozenFeatureMatrix, ...], tuple[v9.FrozenFeatureMatrix, ...]]:
    if isinstance(matrices, FrozenTextBundleV92):
        matrices = matrices.matrices
    if len(matrices) != TOTAL_TEXT_MATRIX_COUNT:
        raise QualityProbePreparationV92Error("V9.2 text matrix bundle is incomplete")
    descriptive = tuple(
        value
        for value in matrices
        if any(value.view.startswith(surface + "::") for surface in ORIGINAL_AUTHOR_SURFACES)
    )
    hard = tuple(
        value
        for value in matrices
        if value.view.startswith(COUNTERFACTUAL_HARD_SURFACE + "::")
    )
    if len(descriptive) != ORIGINAL_AUTHOR_MATRIX_COUNT or len(hard) != 7:
        raise QualityProbePreparationV92Error("V9.2 matrix role partition drift")
    return descriptive, hard


def validate_counterfactual_f_p_u_consumption(
    bundle: FrozenTextBundleV92,
) -> dict[str, Any]:
    """Compare independently recomputed F/P/U routing evidence for the hard surface."""

    if not isinstance(bundle, FrozenTextBundleV92):
        raise QualityProbePreparationV92Error("F/P/U validator requires a V9.2 bundle")
    receipts = [
        value
        for value in bundle.actual_consumption_receipts
        if value.get("surface") == COUNTERFACTUAL_HARD_SURFACE
    ]
    if [value.get("path") for value in receipts] != ["F", "P", "U"]:
        raise QualityProbePreparationV92Error(
            "Counterfactual F/P/U receipt order or cardinality drift"
        )
    for value in receipts:
        supplied_hash = value.get("canonical_self_hash")
        payload = dict(value)
        payload.pop("canonical_self_hash", None)
        if supplied_hash != common.canonical_sha256(payload):
            raise QualityProbePreparationV92Error("F/P/U receipt self-hash drift")
    by_path = {str(value["path"]): value for value in receipts}
    routing_fields = (
        "pair_order_sha256",
        "world_order_sha256",
        "eligibility_mask_sha256",
        "eligible_pair_order_sha256",
        "full_pair_count",
        "eligible_pair_count",
    )
    mismatch_count = sum(
        len({by_path[path][field] for path in ("F", "P", "U")}) != 1
        for field in routing_fields
    )
    f_sources = by_path["F"]["actual_sources"]
    p_sources = by_path["P"]["actual_sources"]
    u_sources = by_path["U"]["actual_sources"]
    if any(
        not isinstance(value, Mapping)
        or set(value)
        != {"items_sha256", "profiles_sha256", "item_count_projection_sha256"}
        for value in (f_sources, p_sources, u_sources)
    ):
        raise QualityProbePreparationV92Error("F/P/U source commitment schema drift")
    mismatch_count += int(
        f_sources["items_sha256"] is None
        or f_sources["profiles_sha256"] is not None
        or p_sources["items_sha256"] is not None
        or p_sources["profiles_sha256"] is None
        or u_sources["items_sha256"] is None
        or u_sources["profiles_sha256"] is None
    )
    mismatch_count += int(
        f_sources["items_sha256"] != u_sources["items_sha256"]
    )
    mismatch_count += int(
        p_sources["profiles_sha256"] != u_sources["profiles_sha256"]
    )
    mismatch_count += int(
        len(
            {
                value["item_count_projection_sha256"]
                for value in (f_sources, p_sources, u_sources)
            }
        )
        != 1
    )
    receipt: dict[str, Any] = {
        "version": VERSION,
        "surface": COUNTERFACTUAL_HARD_SURFACE,
        "gate_registry_sha256": gate_registry.GATE_REGISTRY_SHA256,
        "f_p_u_actual_consumption_mismatch_count": int(mismatch_count),
        "path_receipt_hashes": {
            path: by_path[path]["canonical_self_hash"] for path in ("F", "P", "U")
        },
        "pair_order_sha256": by_path["F"]["pair_order_sha256"],
        "world_order_sha256": by_path["F"]["world_order_sha256"],
        "eligibility_mask_sha256": by_path["F"]["eligibility_mask_sha256"],
        "eligible_pair_order_sha256": by_path["F"][
            "eligible_pair_order_sha256"
        ],
    }
    receipt["canonical_self_hash"] = common.canonical_sha256(receipt)
    return receipt


FrozenFeatureMatrix = v9.FrozenFeatureMatrix
FrozenTextEligibility = v9.FrozenTextEligibility
SourceCommitment = v9.SourceCommitment
verify_frozen_feature_matrix = v9.verify_frozen_feature_matrix
verify_frozen_text_eligibility = v9.verify_frozen_text_eligibility
current_feature_matrix_commitment_json = v9.current_feature_matrix_commitment_json
current_text_eligibility_commitment_json = v9.current_text_eligibility_commitment_json
