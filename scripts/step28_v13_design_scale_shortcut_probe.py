#!/usr/bin/env python3
"""Run a non-persistent scale probe for the Step28-v13 nuisance audit.

This utility deliberately uses a modified in-memory development policy.  It
does not publish a dataset, consume a formal namespace, or grant a formal
status.  Its only purpose is to distinguish small-smoke variance from a
systematic nuisance/label shortcut before the formal design is frozen.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import step28_v13_candidate_sampling as candidate_sampling
import step28_v13_common as common
import step28_v13_generate_dataset as generator
import step28_v13_history_features as history_features
import step28_v13_production_chain as production
import step28_v13_project_null_nuisance as nuisance_projector
import step28_v13_run_metadata_shortcut_audit as shortcut_audit
import step28_v13_seal_classification_labels as label_sealer
import step28_v13_structure as structure
import step28_v13_world_builder as world_builder


MODE = "development_smoke"
SPLIT = "train"


def run_probe(world_count: int, progress_every: int) -> dict[str, Any]:
    if world_count < 5:
        raise common.ContractError("Scale probe requires at least five worlds")
    base_policy = common.load_policy(mode=MODE)
    template, fixture, style_profile = generator._load_release_inputs(
        base_policy,
        mode=MODE,
    )
    policy = copy.deepcopy(base_policy)
    policy["modes"][MODE]["world_counts"][SPLIT] = world_count
    # The published smoke remains byte-replayable under its legacy base36
    # handles.  A scale probe must exercise the parser-safe handle alphabet
    # registered for the forthcoming formal release, otherwise values such
    # as a randomly generated ``...cvv...`` handle can change Step3 flags.
    policy["identity_design"]["identity_value_generation"][
        "handle_encoding_by_mode"
    ][MODE] = policy["identity_design"]["identity_value_generation"][
        "handle_encoding_by_mode"
    ]["formal"]
    records = [
        row
        for row in structure.build_mode_world_pool(policy, mode=MODE)
        if row["split"] == SPLIT
    ]
    if len(records) != world_count:
        raise common.ContractError("Scale-probe world count drift")
    structure_key = common.structure_key_for_split(
        policy,
        mode=MODE,
        split=SPLIT,
    )
    candidate_policy = candidate_sampling.build_public_candidate_policy(
        policy,
        mode=MODE,
        split=SPLIT,
    )
    candidate_key = policy["randomness"][MODE]["candidate_key_hex"]

    candidate_rows: list[dict[str, Any]] = []
    redacted_items: list[dict[str, Any]] = []
    history_item_rows: list[dict[str, Any]] = []
    membership_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for ordinal, record in enumerate(records, start=1):
        world = world_builder.build_world(
            policy=policy,
            template=template,
            fixture=fixture,
            style_profile=style_profile,
            mode=MODE,
            world_record=record,
            structure_key_hex=structure_key,
        )
        try:
            processed = production.process_world(
                policy,
                mode=MODE,
                split=SPLIT,
                template=template,
                world=world,
            )
        except Exception as error:
            raise common.ContractError(
                "Scale-probe production failure at "
                f"ordinal={ordinal} world_uid={record['world_uid']}"
            ) from error
        selected, _sampling_audit, _generation_audit = (
            candidate_sampling.build_world_c40(
                candidate_policy,
                candidate_key_hex=candidate_key,
                mode=MODE,
                split=SPLIT,
                sellers=world["public"]["sellers"],
                raw_observed_items=world["public"]["items"],
                complete_pair_endpoints=world["public"][
                    "complete_model_pair_endpoints"
                ],
            )
        )
        item_index = [
            {
                "world_uid": row["world_uid"],
                "seller_uid": row["seller_uid"],
                "item_uid": row["item_uid"],
                "time_bucket": row["time_bucket"],
            }
            for row in world["public"]["items"]
        ]
        attestation = production.build_history_projection_attestation(
            policy,
            mode=MODE,
            split=SPLIT,
            world_uid=record["world_uid"],
            sellers=world["public"]["sellers"],
            items=world["public"]["items"],
            history_safe_occurrences=processed["public"][
                "history_safe_occurrences"
            ],
            history_item_index=item_index,
            parsed_rows=processed["private"][
                "parsed_identity_occurrences"
            ],
            identity_slots_audit=world["private"][
                "identity_slots_audit"
            ],
            noise_slots_audit=world["private"]["noise_slots_audit"],
            render_asts=world["private"]["render_asts"],
        )
        # Exercise the same parser-to-identity33 chain even though the frozen
        # nuisance audit consumes only redacted items and the item index.
        history_features.build_identity33_all_pairs(
            policy,
            mode=MODE,
            split=SPLIT,
            history_safe_occurrences=processed["public"][
                "history_safe_occurrences"
            ],
            history_item_index=item_index,
            projection_attestations=[attestation],
            complete_model_pair_endpoints=world["public"][
                "complete_model_pair_endpoints"
            ],
        )
        candidate_rows.extend(selected)
        redacted_items.extend(processed["public"]["redacted_items"])
        history_item_rows.extend(item_index)
        membership_rows.extend(
            world["private"]["controller_membership"]
        )
        if ordinal % progress_every == 0 or ordinal == world_count:
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "worlds_complete": ordinal,
                        "worlds_total": world_count,
                        "elapsed_seconds": round(
                            time.perf_counter() - started,
                            3,
                        ),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    projection = nuisance_projector.build_projection(
        candidate_rows=candidate_rows,
        redacted_items=redacted_items,
        history_item_rows=history_item_rows,
        expected_world_count=world_count,
    )
    labels = label_sealer.build_labels(
        candidate_rows=candidate_rows,
        membership_rows=membership_rows,
        expected_world_count=world_count,
    )
    audit_report, _oof = shortcut_audit.run_audit(
        projection_rows=projection,
        label_rows=labels,
        split=SPLIT,
        expected_world_count=world_count,
        bootstrap_replicates=9999,
    )
    return {
        "version": (
            "2026-07-29-step28-v13-development-design-scale-"
            "shortcut-probe-v1"
        ),
        "status": "DEVELOPMENT_DESIGN_PROBE_NOT_FORMAL_EVIDENCE",
        "formal_namespace_consumed": False,
        "world_count": world_count,
        "item_count": len(redacted_items),
        "candidate_pair_count": len(candidate_rows),
        "elapsed_seconds": time.perf_counter() - started,
        "audit": audit_report,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-count", type=int, default=500)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_probe(args.world_count, args.progress_every)
    payload = (
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )
    if args.output is None:
        print(payload, end="")
        return
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite probe: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload, encoding="utf-8")
    print(f"Wrote scale probe: {output}", flush=True)


if __name__ == "__main__":
    main()
