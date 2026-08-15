#!/usr/bin/env python3
"""In-memory causal replay of the already exposed train ordinals 0..283.

This command writes no dataset, receipt, cache, seed, model, or metric.  It is
the only v9 execution beyond focused tests currently authorized by the repair
contract.  World metadata may be reconstructed from the frozen design pool,
but no development/audit world or train ordinal above 283 is built.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from typing import Any

import step28_v13_common as common
import step28_v13_v1_13_document_collision as collision
import step28_v13_v1_13_scientific_common_v9 as scientific
import step28_v13_v1_13_scientific_dataset_builder_v9 as dataset_builder
import step28_v13_v1_13_scientific_world_v9 as world_module


AUTHORIZED_SPLIT = "train"
AUTHORIZED_FINAL_ORDINAL = 283
AUTHORIZED_WORLD_COUNT = AUTHORIZED_FINAL_ORDINAL + 1


def run_replay() -> dict[str, Any]:
    policy = scientific.load_policy()
    context = scientific.build_execution_context(
        policy, execution_mode="design_preflight"
    )
    records = sorted(
        (
            row
            for row in context.world_records
            if row["split"] == AUTHORIZED_SPLIT
            and 0 <= int(row["split_ordinal"]) <= AUTHORIZED_FINAL_ORDINAL
        ),
        key=lambda row: int(row["split_ordinal"]),
    )
    if (
        len(records) != AUTHORIZED_WORLD_COUNT
        or [int(row["split_ordinal"]) for row in records]
        != list(range(AUTHORIZED_WORLD_COUNT))
        or any(row["split"] != AUTHORIZED_SPLIT for row in records)
    ):
        raise scientific.ScientificBuilderError(
            "Authorized causal-replay world boundary did not close"
        )
    template, fixture, style_profile = scientific.load_release_inputs(context)
    historical = collision.load_historical_exclusion_registries()
    current_item_hashes: set[str] = set()
    current_seller_hashes: set[str] = set()
    current_identity_hashes: set[str] = set()
    current_item_codes: set[str] = set()
    seen_uids: dict[str, set[str]] = {
        kind: set() for kind in dataset_builder.GLOBAL_UID_KINDS
    }
    candidate_histogram: Counter[int] = Counter()
    rejection_totals: Counter[str] = Counter()
    target: world_module.AcceptedScientificWorld | None = None
    for position, record in enumerate(records, start=1):
        accepted = world_module.build_scientific_world(
            policy=context.effective_policy,
            template=template,
            fixture=fixture,
            style_profile=style_profile,
            mode=context.base_mode,
            world_record=record,
            structure_key_hex=common.structure_key_for_split(
                context.effective_policy,
                mode=context.base_mode,
                split=AUTHORIZED_SPLIT,
            ),
            document_variation_key=context.document_variation_key,
            anonymous_handle_key=context.anonymous_handle_key,
            historical_item_hashes=historical.item_document_hashes,
            historical_seller_hashes=historical.seller_document_hashes,
            historical_identity_hashes=historical.identity_value_hashes,
            current_item_hashes=current_item_hashes,
            current_seller_hashes=current_seller_hashes,
            current_identity_hashes=current_identity_hashes,
            current_item_codes=current_item_codes,
            candidate_limit=32,
            identity_maximum_counter=128,
        )
        values = dataset_builder._world_uid_sets(accepted)
        dataset_builder._commit_uid_values(values, seen=seen_uids)
        candidate_histogram[accepted.candidate_index] += 1
        rejection_totals.update(accepted.rejection_counts)
        if int(record["split_ordinal"]) == AUTHORIZED_FINAL_ORDINAL:
            target = accepted
        if position == 1 or position % 10 == 0 or position == AUTHORIZED_WORLD_COUNT:
            print(
                json.dumps(
                    {
                        "status": "IN_MEMORY_CAUSAL_REPLAY_PROGRESS",
                        "completed_worlds": position,
                        "authorized_worlds": AUTHORIZED_WORLD_COUNT,
                        "latest_train_ordinal": int(record["split_ordinal"]),
                        "dataset_rows_written": 0,
                        "formal_seed_created": False,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )
    if target is None:
        raise scientific.ScientificBuilderError("Target ordinal 283 was not replayed")
    if len(current_item_codes) != len(current_item_hashes):
        raise scientific.ScientificBuilderError(
            "Replay code and item-document registries disagree"
        )
    return {
        "status": "PASS_IN_MEMORY_CAUSAL_REPLAY_THROUGH_TRAIN_ORDINAL_283",
        "claim_boundary": (
            "Implementation causal evidence only; no dataset quality, publication, "
            "formal generation, or training authorization"
        ),
        "processed_split": AUTHORIZED_SPLIT,
        "processed_train_ordinals": [0, AUTHORIZED_FINAL_ORDINAL],
        "processed_world_count": AUTHORIZED_WORLD_COUNT,
        "future_train_worlds_built": 0,
        "development_or_audit_worlds_built": 0,
        "dataset_rows_written": 0,
        "formal_seed_created": False,
        "model_or_metric_created": False,
        "candidate_histogram": {
            str(index): count for index, count in sorted(candidate_histogram.items())
        },
        "rejection_totals": {
            name: rejection_totals[name]
            for name in world_module.COLLISION_CATEGORIES
        },
        "item_document_registry_count": len(current_item_hashes),
        "seller_document_registry_count": len(current_seller_hashes),
        "identity_value_registry_count": len(current_identity_hashes),
        "item_code_registry_count": len(current_item_codes),
        "uid_registry_counts": {
            kind: len(seen_uids[kind]) for kind in dataset_builder.GLOBAL_UID_KINDS
        },
        "target_train_ordinal": AUTHORIZED_FINAL_ORDINAL,
        "target_world_uid": target.world_uid,
        "target_accepted_candidate_index": target.candidate_index,
        "target_candidates_examined": target.candidates_examined,
        "target_rejection_counts": target.rejection_counts,
        "target_structural_parent_sha256": target.structural_parent_sha256,
        "target_candidate_zero_lineage_reference_sha256": (
            target.candidate_zero_lineage_reference_sha256
        ),
        "target_identity33_sha256": target.identity33_sha256,
        "target_document_capacity_receipt_sha256": common.canonical_sha256(
            target.document_capacity_receipt
        ),
        "target_document_capacity_audit_sha256": common.canonical_sha256(
            target.document_capacity_audit
        ),
    }


def main() -> None:
    print(json.dumps(run_replay(), ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Step28-v13 v1.13 v9 causal replay failed: {exc}", file=sys.stderr)
        raise
