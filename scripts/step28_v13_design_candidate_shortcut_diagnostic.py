#!/usr/bin/env python3
"""Diagnose and repair C40-induced nuisance shortcuts before formal freeze.

This script uses only a modified in-memory development policy and fresh design
worlds.  It never writes a dataset, consumes a formal structure key, or grants
formal status.  It compares the current text-triggered C40 with two fixed-HMAC
alternatives while holding every generated world and nuisance value fixed.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import step28_v13_candidate_sampling as candidate_sampling
import step28_v13_common as common
import step28_v13_generate_dataset as generator
import step28_v13_metadata_shortcut_common as shortcut_common
import step28_v13_project_null_nuisance as nuisance_projector
import step28_v13_run_metadata_shortcut_audit as shortcut_audit
import step28_v13_seal_classification_labels as label_sealer
import step28_v13_structure as structure
import step28_v13_world_builder as world_builder


MODE = "development_smoke"
SPLIT = "train"


def _safe_pair(row: Mapping[str, Any]) -> dict[str, str]:
    return {
        name: str(row[name])
        for name in shortcut_common.CANDIDATE_FIELDS
    }


def _rank_pairs(
    rows: Sequence[Mapping[str, Any]],
    *,
    key_hex: str,
    world_uid: str,
    domain: str,
) -> list[dict[str, str]]:
    return sorted(
        (_safe_pair(row) for row in rows),
        key=lambda row: (
            common.hmac_digest(
                key_hex,
                world_uid,
                domain,
                row["canonical_pair_uid"],
            ),
            row["canonical_pair_uid"].encode("utf-8"),
        ),
    )


def _alternative_c40(
    complete_pairs: Sequence[Mapping[str, Any]],
    memberships: Sequence[Mapping[str, Any]],
    *,
    key_hex: str,
    world_uid: str,
    positive_count: int | None,
) -> list[dict[str, str]]:
    controller = {
        str(row["seller_uid"]): str(row["controller_uid"])
        for row in memberships
    }
    if len(controller) != 28:
        raise common.ContractError("Diagnostic membership cardinality drift")
    ranked = _rank_pairs(
        complete_pairs,
        key_hex=key_hex,
        world_uid=world_uid,
        domain=(
            "diagnostic_uniform_c40"
            if positive_count is None
            else "diagnostic_label_stratified_c40"
        ),
    )
    if positive_count is None:
        selected = ranked[:40]
    else:
        positive = [
            row
            for row in ranked
            if controller[row["seller_uid_left"]]
            == controller[row["seller_uid_right"]]
        ]
        negative = [
            row
            for row in ranked
            if controller[row["seller_uid_left"]]
            != controller[row["seller_uid_right"]]
        ]
        if (
            not 1 <= positive_count < 40
            or len(positive) < positive_count
            or len(negative) < 40 - positive_count
        ):
            raise common.ContractError(
                "Diagnostic label-stratified C40 lacks class capacity"
            )
        selected = (
            positive[:positive_count]
            + negative[: 40 - positive_count]
        )
    return sorted(
        selected,
        key=lambda row: (
            row["world_uid"].encode("utf-8"),
            row["canonical_pair_uid"].encode("utf-8"),
        ),
    )


def _mechanism_stratified_c40(
    complete_pairs: Sequence[Mapping[str, Any]],
    memberships: Sequence[Mapping[str, Any]],
    positive_targets: Sequence[Mapping[str, Any]],
    negative_flags: Sequence[Mapping[str, Any]],
    *,
    key_hex: str,
    world_uid: str,
    positive_count: int,
) -> list[dict[str, str]]:
    """Cover every registered mechanism, then HMAC-fill within class."""

    controller = {
        str(row["seller_uid"]): str(row["controller_uid"])
        for row in memberships
    }
    ranked = _rank_pairs(
        complete_pairs,
        key_hex=key_hex,
        world_uid=world_uid,
        domain="diagnostic_mechanism_stratified_c40",
    )
    pair_by_uid = {
        row["canonical_pair_uid"]: row
        for row in ranked
    }
    rank_by_uid = {
        row["canonical_pair_uid"]: rank
        for rank, row in enumerate(ranked)
    }
    label_by_uid = {
        pair_uid: int(
            controller[row["seller_uid_left"]]
            == controller[row["seller_uid_right"]]
        )
        for pair_uid, row in pair_by_uid.items()
    }
    positive_groups: dict[str, set[str]] = defaultdict(set)
    negative_groups: dict[str, set[str]] = defaultdict(set)
    for row in positive_targets:
        if (
            "world_uid" in row
            and str(row["world_uid"]) != world_uid
        ):
            raise common.ContractError(
                "Positive mechanism target world drift"
            )
        pair_uid = str(row["canonical_pair_uid"])
        if pair_uid not in pair_by_uid or label_by_uid[pair_uid] != 1:
            raise common.ContractError(
                "Positive mechanism target is not a positive complete pair"
            )
        positive_groups[str(row["mechanism"])].add(pair_uid)
    for row in negative_flags:
        if (
            "world_uid" in row
            and str(row["world_uid"]) != world_uid
        ):
            raise common.ContractError(
                "Negative mechanism target world drift"
            )
        pair_uid = str(row["canonical_pair_uid"])
        if pair_uid not in pair_by_uid or label_by_uid[pair_uid] != 0:
            raise common.ContractError(
                "Negative mechanism target is not a negative complete pair"
            )
        negative_groups[str(row["flag"])].add(pair_uid)
    if not positive_groups or not negative_groups:
        raise common.ContractError(
            "Mechanism-stratified C40 lacks registered groups"
        )

    selected_positive = {
        min(pair_uids, key=rank_by_uid.__getitem__)
        for _mechanism, pair_uids in sorted(
            positive_groups.items(),
            key=lambda row: row[0].encode("utf-8"),
        )
    }
    selected_negative = {
        min(pair_uids, key=rank_by_uid.__getitem__)
        for _flag, pair_uids in sorted(
            negative_groups.items(),
            key=lambda row: row[0].encode("utf-8"),
        )
    }
    if (
        len(selected_positive) > positive_count
        or len(selected_negative) > 40 - positive_count
    ):
        raise common.ContractError(
            "Mechanism coverage exceeds the registered class budget"
        )
    for row in ranked:
        pair_uid = row["canonical_pair_uid"]
        if (
            label_by_uid[pair_uid] == 1
            and len(selected_positive) < positive_count
        ):
            selected_positive.add(pair_uid)
        elif (
            label_by_uid[pair_uid] == 0
            and len(selected_negative) < 40 - positive_count
        ):
            selected_negative.add(pair_uid)
        if (
            len(selected_positive) == positive_count
            and len(selected_negative) == 40 - positive_count
        ):
            break
    selected_uids = selected_positive | selected_negative
    if (
        len(selected_positive) != positive_count
        or len(selected_negative) != 40 - positive_count
        or len(selected_uids) != 40
        or any(
            not selected_uids.intersection(pair_uids)
            for pair_uids in positive_groups.values()
        )
        or any(
            not selected_uids.intersection(pair_uids)
            for pair_uids in negative_groups.values()
        )
    ):
        raise common.ContractError(
            "Mechanism-stratified C40 coverage/fill failed"
        )
    return sorted(
        (pair_by_uid[pair_uid] for pair_uid in selected_uids),
        key=lambda row: (
            row["world_uid"].encode("utf-8"),
            row["canonical_pair_uid"].encode("utf-8"),
        ),
    )


def _summary(
    projection: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    *,
    world_count: int,
) -> dict[str, Any]:
    label_by_pair = {
        str(row["canonical_pair_uid"]): int(row["label"])
        for row in labels
    }
    y = np.asarray(
        [label_by_pair[str(row["canonical_pair_uid"])] for row in projection],
        dtype=np.int64,
    )
    feature_rows: dict[str, Any] = {}
    for feature in shortcut_common.PAIR_FEATURES:
        values = np.asarray(
            [float(row[feature]) for row in projection],
            dtype=np.float64,
        )
        auc = float(roc_auc_score(y, values))
        feature_rows[feature] = {
            "positive_mean": float(np.mean(values[y == 1])),
            "negative_mean": float(np.mean(values[y == 0])),
            "roc_auc": auc,
            "roc_auc_symmetric": max(auc, 1.0 - auc),
        }
    report, _oof = shortcut_audit.run_audit(
        projection_rows=projection,
        label_rows=labels,
        split=SPLIT,
        expected_world_count=world_count,
        bootstrap_replicates=9999,
    )
    return {
        "class_counts": {
            "negative": int(np.sum(y == 0)),
            "positive": int(np.sum(y == 1)),
        },
        "univariate": feature_rows,
        "frozen14_audit": report,
    }


def run(
    world_count: int,
    progress_every: int,
    design_set: str,
) -> dict[str, Any]:
    base_policy = common.load_policy(mode=MODE)
    template, fixture, style_profile = generator._load_release_inputs(
        base_policy,
        mode=MODE,
    )
    policy = copy.deepcopy(base_policy)
    policy["modes"][MODE]["world_counts"][SPLIT] = world_count
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
        raise common.ContractError("Candidate diagnostic world-count drift")
    structure_key = common.structure_key_for_split(
        policy,
        mode=MODE,
        split=SPLIT,
    )
    if design_set == "all":
        public_candidate_policy = (
            candidate_sampling.build_public_candidate_policy(
                policy,
                mode=MODE,
                split=SPLIT,
            )
        )
    else:
        public_candidate_policy = None
    candidate_key = str(
        policy["randomness"][MODE]["candidate_key_hex"]
    )
    candidate_sets: dict[str, list[dict[str, str]]] = (
        {
            "current_text_triggered": [],
            "uniform_hmac": [],
            "label_stratified_16_24": [],
            "label_stratified_10_30": [],
        }
        if design_set == "all"
        else {
            "mechanism_stratified_16_24": [],
            "mechanism_stratified_10_30": [],
        }
    )
    selected_trigger_counts: dict[str, Counter[tuple[str, int]]] = {
        "current_text_triggered": Counter()
    }
    redacted_shape_items: list[dict[str, str]] = []
    history_items: list[dict[str, Any]] = []
    memberships: list[dict[str, Any]] = []
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
        public = world["public"]
        private = world["private"]
        if design_set == "all":
            selected, sampling_audit, _generation_audit = (
                candidate_sampling.build_world_c40(
                    public_candidate_policy,
                    candidate_key_hex=candidate_key,
                    mode=MODE,
                    split=SPLIT,
                    sellers=public["sellers"],
                    raw_observed_items=public["items"],
                    complete_pair_endpoints=public[
                        "complete_model_pair_endpoints"
                    ],
                )
            )
        label_by_pair = {}
        controller = {
            str(row["seller_uid"]): str(row["controller_uid"])
            for row in private["controller_membership"]
        }
        for row in public["complete_model_pair_endpoints"]:
            label_by_pair[str(row["canonical_pair_uid"])] = int(
                controller[str(row["seller_uid_left"])]
                == controller[str(row["seller_uid_right"])]
            )
        if design_set == "all":
            candidate_sets["current_text_triggered"].extend(selected)
            for row in sampling_audit:
                selected_value = row["selected_bool"]
                if selected_value is True or selected_value == "true":
                    pair_uid = str(row["canonical_pair_uid"])
                    selected_trigger_counts["current_text_triggered"][
                        (
                            str(row["primary_trigger"]),
                            label_by_pair[pair_uid],
                        )
                    ] += 1
            for name, positive_count in (
                ("uniform_hmac", None),
                ("label_stratified_16_24", 16),
                ("label_stratified_10_30", 10),
            ):
                candidate_sets[name].extend(
                    _alternative_c40(
                        public["complete_model_pair_endpoints"],
                        private["controller_membership"],
                        key_hex=candidate_key,
                        world_uid=str(record["world_uid"]),
                        positive_count=positive_count,
                    )
                )
        else:
            for name, positive_count in (
                ("mechanism_stratified_16_24", 16),
                ("mechanism_stratified_10_30", 10),
            ):
                candidate_sets[name].extend(
                    _mechanism_stratified_c40(
                        public["complete_model_pair_endpoints"],
                        private["controller_membership"],
                        private["positive_targets"],
                        private["negative_flags"],
                        key_hex=candidate_key,
                        world_uid=str(record["world_uid"]),
                        positive_count=positive_count,
                    )
                )
        redacted_shape_items.extend(
            {
                "world_uid": str(row["world_uid"]),
                "seller_uid": str(row["seller_uid"]),
                "item_uid": str(row["item_uid"]),
                "title": str(row["title"]),
                "description": str(row["description"]),
            }
            for row in public["items"]
        )
        history_items.extend(
            {
                "world_uid": str(row["world_uid"]),
                "seller_uid": str(row["seller_uid"]),
                "item_uid": str(row["item_uid"]),
                "time_bucket": int(row["time_bucket"]),
            }
            for row in public["items"]
        )
        memberships.extend(private["controller_membership"])
        if ordinal % progress_every == 0 or ordinal == world_count:
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "worlds_complete": ordinal,
                        "worlds_total": world_count,
                        "elapsed_seconds": round(
                            time.perf_counter() - started, 3
                        ),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    results: dict[str, Any] = {}
    for name, candidates in candidate_sets.items():
        projection = nuisance_projector.build_projection(
            candidate_rows=candidates,
            redacted_items=redacted_shape_items,
            history_item_rows=history_items,
            expected_world_count=world_count,
        )
        labels = label_sealer.build_labels(
            candidate_rows=candidates,
            membership_rows=memberships,
            expected_world_count=world_count,
        )
        results[name] = _summary(
            projection,
            labels,
            world_count=world_count,
        )
    trigger_rows = [
        {
            "primary_trigger": trigger,
            "label": label,
            "count": count,
        }
        for (trigger, label), count in sorted(
            selected_trigger_counts["current_text_triggered"].items()
        )
    ]
    return {
        "version": (
            "2026-07-29-step28-v13-candidate-shortcut-design-"
            "diagnostic-v1"
        ),
        "status": "DEVELOPMENT_DESIGN_DIAGNOSTIC_NOT_FORMAL_EVIDENCE",
        "formal_namespace_consumed": False,
        "world_count": world_count,
        "item_count": len(redacted_shape_items),
        "candidate_designs": results,
        "design_set": design_set,
        "current_selected_trigger_label_counts": trigger_rows,
        "elapsed_seconds": time.perf_counter() - started,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-count", type=int, default=500)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--design-set",
        choices=("all", "mechanism_only"),
        default="all",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.world_count < 5:
        raise common.ContractError("Diagnostic requires at least five worlds")
    result = run(
        args.world_count,
        args.progress_every,
        args.design_set,
    )
    payload = (
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n"
    )
    if args.output is None:
        print(payload, end="")
        return
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite candidate diagnostic: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload, encoding="utf-8")
    print(f"Wrote candidate diagnostic: {output}", flush=True)


if __name__ == "__main__":
    main()
