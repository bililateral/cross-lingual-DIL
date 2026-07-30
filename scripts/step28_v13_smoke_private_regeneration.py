#!/usr/bin/env python3
"""Development-only same-producer regeneration check for one Step 28-v13 world.

This module is deliberately outside the observed parser/redactor dependency
closure.  It reruns the producer to detect mutated in-memory artifacts, but it
is not the independent second implementation required for a formal release.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import step28_v13_common as common
import step28_v13_structure as structure
import step28_v13_world_builder as world_builder


EVIDENCE_LEVEL_BY_MODE = {
    "development_smoke": (
        "DEVELOPMENT_SMOKE_SAME_IMPLEMENTATION_NOT_FORMAL_SEAL"
    ),
    "training_ready": (
        "TRAINING_READY_SAME_IMPLEMENTATION_REGENERATION_"
        "NOT_INDEPENDENT_NOT_FORMAL_CUSTODY_SEAL"
    ),
}


def validate_producer_regeneration_match(
    policy: Mapping[str, Any],
    *,
    mode: str,
    split: str,
    template: Mapping[str, Any],
    fixture: Mapping[str, Any],
    style_profile: Mapping[str, Any],
    world: Mapping[str, Any],
) -> dict[str, Any]:
    """Regenerate one non-formal world and compare all supplied bytes."""

    if mode not in EVIDENCE_LEVEL_BY_MODE:
        raise common.ContractError(
            "Producer regeneration match is development-smoke only for "
            "formal-custody purposes"
        )
    scope = (
        "smoke producer regeneration"
        if mode == "development_smoke"
        else "training-ready producer regeneration"
    )
    if set(world) != {"public", "private"}:
        raise common.ContractError("Producer regeneration world schema drift")
    public = world["public"]
    if set(public) != {
        "world",
        "sellers",
        "items",
        "complete_model_pair_endpoints",
    }:
        raise common.ContractError("Producer regeneration public schema drift")
    world_row = public["world"]
    if (
        not isinstance(world_row, Mapping)
        or set(world_row) != {"world_uid"}
        or not isinstance(world_row["world_uid"], str)
    ):
        raise common.ContractError("Producer regeneration world UID drift")
    world_uid = world_row["world_uid"]
    matching_records = [
        row
        for row in structure.build_mode_world_pool(policy, mode=mode)
        if row["world_uid"] == world_uid and row["split"] == split
    ]
    if len(matching_records) != 1:
        raise common.ContractError(
            "Producer regeneration world is not one registered pool member"
        )
    expected_template, expected_fixture = (
        common.validate_policy_release_documents(policy, mode=mode)
    )
    if (
        common.canonical_json_bytes(template)
        != common.canonical_json_bytes(expected_template)
        or common.canonical_json_bytes(fixture)
        != common.canonical_json_bytes(expected_fixture)
    ):
        raise common.ContractError(
            "Producer regeneration received unregistered release documents"
        )
    expected = world_builder.build_world(
        policy=dict(policy),
        template=dict(template),
        fixture=dict(fixture),
        style_profile=dict(style_profile),
        mode=mode,
        world_record=matching_records[0],
        structure_key_hex=common.structure_key_for_split(
            policy,
            mode=mode,
            split=split,
        ),
    )
    comparisons: tuple[tuple[str, Any, Any], ...] = (
        ("private trace", world["private"], expected["private"]),
        ("public world", public["world"], expected["public"]["world"]),
        ("public sellers", public["sellers"], expected["public"]["sellers"]),
        ("public items", public["items"], expected["public"]["items"]),
        (
            "public complete pair endpoints",
            public["complete_model_pair_endpoints"],
            expected["public"]["complete_model_pair_endpoints"],
        ),
    )
    for label, observed, replayed in comparisons:
        if common.canonical_json_bytes(observed) != common.canonical_json_bytes(
            replayed
        ):
            raise common.ContractError(
                f"{label} differs from {scope}"
            )
    replay_sha256 = common.canonical_sha256(expected)
    return {
        "producer_regeneration_match_pass": True,
        "producer_regeneration_match_smoke_pass": (
            mode == "development_smoke"
        ),
        "producer_regeneration_match_sha256": replay_sha256,
        "producer_regeneration_independent_replay": False,
        "producer_regeneration_evidence_level": EVIDENCE_LEVEL_BY_MODE[mode],
    }
