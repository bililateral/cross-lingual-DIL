#!/usr/bin/env python3
"""Run the design-only Step28-v13 v1.12 visible-text preceremony."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import step28_v13_production_chain as production
import step28_v13_profiles as profiles
import step28_v13_v1_12_formal_common as formal
import step28_v13_v1_12_preceremony as preceremony
import step28_v13_v1_12_text_shortcut_preflight as shortcut
import step28_v13_world_builder as world_builder


ROOT = Path(__file__).resolve().parents[1]
M0_FIELDS = (
    "category_concat_top",
    "signature_title_concat",
    "title_concat_top",
    "signature_description_concat",
    "description_concat_top",
)
INTERNAL_UID_RE = re.compile(
    r"(?:w|sel|itm|ctl|ias|id|qry)_[0-9a-f]{64}", re.IGNORECASE
)
FORBIDDEN_VISIBLE_TOKENS = (
    "same_controller",
    "different_controller",
    "controller_uid",
    "mechanism_slot_uid",
    "identity_asset_uid",
)
CLOSED_FORMAL_AUTHORIZATIONS = {
    "formal_seed_ceremony": False,
    "formal_train_generation": False,
    "formal_development_generation": False,
    "formal_audit_a_generation": False,
    "formal_audit_b_generation": False,
    "model_training": False,
    "audit_truth_unsealing": False,
}
FROZEN_VERSION_NON_REUSE_COMMITMENT = (
    "this frozen-version failure cannot be rescued by changing a key, "
    "domain, derangement, mask, threshold, or formal authorization"
)


class TextShortcutRunnerError(ValueError):
    """Raised when the design-only text preceremony cannot close."""


class TextShortcutStageError(TextShortcutRunnerError):
    """Wrap a fail-closed exception with its exact preceremony stage."""

    def __init__(self, stage: str, cause: Exception) -> None:
        super().__init__(f"stage={stage}: {type(cause).__name__}: {cause}")
        self.stage = stage
        self.cause_type = type(cause).__name__


@dataclass(frozen=True)
class WorldOriginalDiagnostic:
    world_uid: str
    pair_uids: tuple[str, ...]
    labels: np.ndarray
    feature_names: tuple[str, ...]
    scores: np.ndarray
    strata: tuple[str, ...]
    inventory_sets: dict[str, frozenset[str]]
    visible_leakage_count: int
    audit: dict[str, Any]


@dataclass(frozen=True)
class SplitOriginalDiagnostic:
    split: str
    world_uids: tuple[str, ...]
    pair_uids: tuple[str, ...]
    labels: np.ndarray
    feature_names: tuple[str, ...]
    scores: np.ndarray
    strata: tuple[str, ...]
    inventory_sets: dict[str, frozenset[str]]
    visible_leakage_count: int
    world_audits: tuple[dict[str, Any], ...]


def _project_original_visible_world(
    *,
    execution_policy: Mapping[str, Any],
    template: Mapping[str, Any],
    split: str,
    world: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the production parser/redactor/profile chain without a style intervention."""

    public = world["public"]
    private = world["private"]
    sellers = public["sellers"]
    items = public["items"]
    production._validate_observed_raw_against_private_ast(
        execution_policy,
        split=split,
        template=template,
        items=items,
        identity_slots_audit=private["identity_slots_audit"],
        noise_slots_audit=private["noise_slots_audit"],
        render_asts=private["render_asts"],
        override_audit=private["override_audit"],
    )
    parsed = production.parse_observed_world(
        execution_policy,
        mode="formal",
        split=split,
        sellers=sellers,
        items=items,
    )
    parser_audit = production.validate_parser_against_private_plan(
        execution_policy,
        mode="formal",
        split=split,
        sellers=sellers,
        items=items,
        parsed_rows=parsed,
        identity_slots_audit=private["identity_slots_audit"],
        noise_slots_audit=private["noise_slots_audit"],
        render_asts=private["render_asts"],
    )
    registry_profiles = production.registry_profiles_from_sellers(
        execution_policy, sellers=sellers
    )
    redaction = production.redact_observed_world(
        execution_policy,
        mode="formal",
        split=split,
        template=template,
        sellers=sellers,
        items=items,
        registry_profiles=registry_profiles,
        parsed_rows=parsed,
    )
    redaction_audit = production.validate_redaction_against_private_plan(
        execution_policy,
        mode="formal",
        split=split,
        template=template,
        sellers=sellers,
        items=items,
        redacted_items=redaction["redacted_items"],
        parsed_rows=parsed,
        identity_slots_audit=private["identity_slots_audit"],
        noise_slots_audit=private["noise_slots_audit"],
        render_asts=private["render_asts"],
        override_audit=private["override_audit"],
    )
    profile_safe_items = production.build_profile_safe_items(
        execution_policy,
        items=items,
        redacted_items=redaction["redacted_items"],
    )
    registered = preceremony.project_registered_visible_text(
        policy=execution_policy,
        template=template,
        sellers=sellers,
        items=items,
        parsed_rows=parsed,
    )
    if (
        registered["redacted_items"] != redaction["redacted_items"]
        or {
            str(row["item_uid"]): row for row in registered["profile_safe_items"]
        }
        != {str(row["item_uid"]): row for row in profile_safe_items}
    ):
        raise TextShortcutRunnerError(
            "Original registered projection disagrees with production redaction"
        )
    seller_profiles, profile_audit = profiles.build_world_profiles(
        execution_policy,
        mode="formal",
        split=split,
        sellers=sellers,
        items=profile_safe_items,
    )
    if (
        parser_audit.get("exact_rows_and_flags") is not True
        or int(redaction_audit["planned_identity_surface_residue_count"]) != 0
        or int(profile_audit["seller_count"]) != 28
        or len(seller_profiles) != 28
    ):
        raise TextShortcutRunnerError("Original production projection did not close")
    return {
        "seller_profiles": seller_profiles,
        "redacted_items": redaction["redacted_items"],
        "audit": {
            "parser": parser_audit,
            "redaction": redaction_audit,
            "profile": profile_audit,
            "parsed_identity_occurrence_count": len(parsed),
            "redacted_item_count": len(redaction["redacted_items"]),
            "seller_profile_count": len(seller_profiles),
        },
    }


def _full_pair_rows_and_labels(
    *,
    pair_rows: Sequence[Mapping[str, Any]],
    label_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], np.ndarray]:
    pair_index = {str(row["canonical_pair_uid"]): dict(row) for row in pair_rows}
    label_index = {str(row["canonical_pair_uid"]): int(row["label"]) for row in label_rows}
    if (
        len(pair_index) != 378
        or len(label_index) != 378
        or set(pair_index) != set(label_index)
    ):
        raise TextShortcutRunnerError("Original full-pair keyset drift")
    pair_uids = sorted(pair_index, key=lambda value: value.encode("utf-8"))
    ordered = [pair_index[pair_uid] for pair_uid in pair_uids]
    labels = np.asarray([label_index[pair_uid] for pair_uid in pair_uids], dtype=np.int8)
    if labels.shape != (378,) or int(np.sum(labels)) != 20:
        raise TextShortcutRunnerError("Original full-pair label closure failed")
    return ordered, labels


def _visible_leakage(values: Sequence[str]) -> int:
    return sum(
        INTERNAL_UID_RE.search(value) is not None
        or any(token in value.casefold() for token in FORBIDDEN_VISIBLE_TOKENS)
        for value in values
    )


def _build_original_world_diagnostic(
    *,
    policy: Mapping[str, Any],
    world: Mapping[str, Any],
    projection: Mapping[str, Any],
    label_rows: Sequence[Mapping[str, Any]],
) -> WorldOriginalDiagnostic:
    if tuple(policy["visible_attack"]["m0_fields_in_order"]) != M0_FIELDS:
        raise TextShortcutRunnerError("Original M0 field order drift")
    public = world["public"]
    private = world["private"]
    ordered_pairs, labels = _full_pair_rows_and_labels(
        pair_rows=public["complete_model_pair_endpoints"],
        label_rows=label_rows,
    )
    seller_profiles = projection["seller_profiles"]
    profile_index = {
        str(row["seller_uid"]): row for row in seller_profiles
    }
    if len(profile_index) != 28:
        raise TextShortcutRunnerError("Original seller-profile UID collision")
    seller_order = sorted(profile_index, key=lambda value: value.encode("utf-8"))
    seller_ordinal = {seller_uid: index for index, seller_uid in enumerate(seller_order)}
    left_indices = np.asarray(
        [seller_ordinal[str(row["seller_uid_left"])] for row in ordered_pairs],
        dtype=np.intp,
    )
    right_indices = np.asarray(
        [seller_ordinal[str(row["seller_uid_right"])] for row in ordered_pairs],
        dtype=np.intp,
    )
    documents = {
        field: [str(profile_index[seller_uid][field]) for seller_uid in seller_order]
        for field in M0_FIELDS
    }
    separator = bytes.fromhex(
        str(policy["visible_attack"]["combined_field_separator_utf8_hex"])
    ).decode("utf-8")
    documents["all_fields"] = [
        separator.join(str(profile_index[seller_uid][field]) for field in M0_FIELDS)
        for seller_uid in seller_order
    ]
    visible_values = [
        value for field_values in documents.values() for value in field_values
    ] + [
        str(row[name])
        for row in projection["redacted_items"]
        for name in ("title", "description")
    ]
    leakage_count = _visible_leakage(visible_values)
    if leakage_count:
        raise TextShortcutRunnerError("Original visible text contains forbidden residue")

    feature_names: list[str] = []
    feature_values: list[np.ndarray] = []
    for kind in ("char3", "word12"):
        vectorizer = shortcut._vectorizer(policy, kind=kind)
        for field in (*M0_FIELDS, "all_fields"):
            feature_names.append(f"{kind}_cosine__{field}")
            feature_values.append(
                shortcut._cosine_by_pair(
                    documents[field],
                    left_indices=left_indices,
                    right_indices=right_indices,
                    vectorizer=vectorizer,
                )
            )
    scores = np.column_stack(feature_values).astype(np.float64, copy=False)
    if scores.shape != (378, 12) or not np.all(np.isfinite(scores)):
        raise TextShortcutRunnerError("Original direct-score matrix drift")

    selected_rows = [
        row
        for row in private["negative_flags"]
        if str(row["flag"])
        in {"exact_title_clone_target", "high_semantic_similarity_target"}
    ]
    override_kind_by_flag = policy["mechanism_neutral_eligibility"][
        "override_kind_by_flag"
    ]
    selected_triples = {
        (
            str(override_kind_by_flag[str(row["flag"])]),
            int(row["asset_index"]),
            str(row["canonical_pair_uid"]),
        )
        for row in selected_rows
    }
    override_triples = {
        (
            str(row["override_kind"]),
            int(row["asset_index"]),
            str(row["canonical_pair_uid"]),
        )
        for row in private["override_audit"]
    }
    if (
        len(selected_triples) != 6
        or len(override_triples) != 6
        or selected_triples != override_triples
    ):
        raise TextShortcutRunnerError(
            "Original diagnostic mechanism/override lineage drift"
        )
    selected_flags = {
        str(row["canonical_pair_uid"]): str(row["flag"])
        for row in selected_rows
    }
    if Counter(selected_flags.values()) != Counter(
        {"exact_title_clone_target": 2, "high_semantic_similarity_target": 4}
    ):
        raise TextShortcutRunnerError("Original diagnostic mechanism strata drift")
    strata: list[str] = []
    for pair, label in zip(ordered_pairs, labels, strict=True):
        pair_uid = str(pair["canonical_pair_uid"])
        if int(label) == 1:
            stratum = "positive"
        elif pair_uid in selected_flags:
            stratum = selected_flags[pair_uid]
        else:
            stratum = "other_negative"
        strata.append(stratum)
    if Counter(strata) != Counter(
        {
            "positive": 20,
            "exact_title_clone_target": 2,
            "high_semantic_similarity_target": 4,
            "other_negative": 352,
        }
    ):
        raise TextShortcutRunnerError("Original diagnostic stratum counts drift")

    world_uid = str(public["world"]["world_uid"])
    seller_five_field_record_hashes = frozenset(
        preceremony.canonical_sha256(
            {field: str(profile_index[seller_uid][field]) for field in M0_FIELDS}
        )
        for seller_uid in seller_order
    )
    seller_document_hashes = frozenset(
        hashlib.sha256(
            "\n".join(
                str(profile_index[seller_uid][field]).strip()
                for field in M0_FIELDS
                if str(profile_index[seller_uid][field]).strip()
            ).encode("utf-8")
        ).hexdigest()
        for seller_uid in seller_order
    )
    item_document_hashes = frozenset(
        preceremony.canonical_sha256(
            {"title": str(row["title"]), "description": str(row["description"])}
        )
        for row in projection["redacted_items"]
    )
    inventory_sets = {
        "world_uid": frozenset({world_uid}),
        "seller_uid": frozenset(str(row["seller_uid"]) for row in public["sellers"]),
        "item_uid": frozenset(str(row["item_uid"]) for row in public["items"]),
        "canonical_pair_uid": frozenset(
            str(row["canonical_pair_uid"])
            for row in public["complete_model_pair_endpoints"]
        ),
        "controller_uid": frozenset(
            str(row["controller_uid"])
            for row in private["controller_membership"]
        ),
        "item_document_hash": item_document_hashes,
        "seller_document_hash": seller_document_hashes,
        "seller_five_field_record_hash": seller_five_field_record_hashes,
    }
    return WorldOriginalDiagnostic(
        world_uid=world_uid,
        pair_uids=tuple(str(row["canonical_pair_uid"]) for row in ordered_pairs),
        labels=labels,
        feature_names=tuple(feature_names),
        scores=scores,
        strata=tuple(strata),
        inventory_sets=inventory_sets,
        visible_leakage_count=leakage_count,
        audit=dict(projection["audit"]),
    )


def _aggregate_original_worlds(
    worlds: Sequence[WorldOriginalDiagnostic],
    *,
    split: str,
    expected_world_count: int,
) -> SplitOriginalDiagnostic:
    if (
        split not in {"train", "development"}
        or expected_world_count <= 0
        or len(worlds) != expected_world_count
    ):
        raise TextShortcutRunnerError("Original split aggregation boundary drift")
    ordered = sorted(worlds, key=lambda row: row.world_uid.encode("utf-8"))
    if len({row.world_uid for row in ordered}) != expected_world_count:
        raise TextShortcutRunnerError("Original split world UID collision")
    feature_names = ordered[0].feature_names
    if any(row.feature_names != feature_names for row in ordered):
        raise TextShortcutRunnerError("Original split feature-name drift")
    world_uids = tuple(
        row.world_uid for row in ordered for _pair_uid in row.pair_uids
    )
    pair_uids = tuple(pair_uid for row in ordered for pair_uid in row.pair_uids)
    labels = np.concatenate([row.labels for row in ordered]).astype(np.int8, copy=False)
    scores = np.vstack([row.scores for row in ordered]).astype(np.float64, copy=False)
    strata = tuple(stratum for row in ordered for stratum in row.strata)
    inventory_names = set(ordered[0].inventory_sets)
    if any(set(row.inventory_sets) != inventory_names for row in ordered):
        raise TextShortcutRunnerError("Original split inventory schema drift")
    inventory_sets = {
        name: frozenset().union(*(row.inventory_sets[name] for row in ordered))
        for name in inventory_names
    }
    expected_rows = expected_world_count * 378
    expected_strata = Counter(
        {
            "positive": expected_world_count * 20,
            "exact_title_clone_target": expected_world_count * 2,
            "high_semantic_similarity_target": expected_world_count * 4,
            "other_negative": expected_world_count * 352,
        }
    )
    expected_uid_counts = {
        "world_uid": expected_world_count,
        "seller_uid": expected_world_count * 28,
        "item_uid": sum(len(row.inventory_sets["item_uid"]) for row in ordered),
        "canonical_pair_uid": expected_rows,
        "controller_uid": expected_world_count * 12,
    }
    if (
        len(world_uids) != expected_rows
        or len(pair_uids) != expected_rows
        or len(set(pair_uids)) != expected_rows
        or labels.shape != (expected_rows,)
        or int(np.sum(labels)) != expected_world_count * 20
        or scores.shape != (expected_rows, 12)
        or not np.all(np.isfinite(scores))
        or Counter(strata) != expected_strata
        or any(
            len(inventory_sets[name]) != count
            for name, count in expected_uid_counts.items()
        )
        or sum(row.visible_leakage_count for row in ordered) != 0
    ):
        raise TextShortcutRunnerError("Original split aggregate closure failed")
    return SplitOriginalDiagnostic(
        split=split,
        world_uids=world_uids,
        pair_uids=pair_uids,
        labels=labels,
        feature_names=feature_names,
        scores=scores,
        strata=strata,
        inventory_sets=inventory_sets,
        visible_leakage_count=0,
        world_audits=tuple(
            {
                "world_uid": row.world_uid,
                "parser_exact_rows_and_flags": bool(
                    row.audit["parser"]["exact_rows_and_flags"]
                ),
                "planned_identity_surface_residue_count": int(
                    row.audit["redaction"][
                        "planned_identity_surface_residue_count"
                    ]
                ),
                "parsed_identity_occurrence_count": int(
                    row.audit["parsed_identity_occurrence_count"]
                ),
                "redacted_item_count": int(row.audit["redacted_item_count"]),
                "seller_profile_count": int(row.audit["seller_profile_count"]),
                "production_audit_sha256": preceremony.canonical_sha256(
                    {
                        "parser": row.audit["parser"],
                        "redaction": row.audit["redaction"],
                        "profile": row.audit["profile"],
                    }
                ),
            }
            for row in ordered
        ),
    )


def build_original_design_split(
    *,
    split: str,
    world_count: int,
    progress_every: int = 10,
) -> SplitOriginalDiagnostic:
    """Build nonpersistent original-text diagnostics under design capabilities."""

    if not 1 <= world_count <= 500 or progress_every < 0:
        raise TextShortcutRunnerError("Original design split scale drift")
    policy = shortcut.load_text_audit_policy()
    (
        execution_policy,
        template,
        fixture,
        style_profile,
        generator_capabilities,
        records,
        _validated,
    ) = shortcut._design_split_context(split=split)
    output: list[WorldOriginalDiagnostic] = []
    for index, record in enumerate(records[:world_count]):
        with formal.mounted_structure_capability(
            split=split,
            structure_key_hex=str(generator_capabilities["structure"]),
        ):
            world = world_builder.build_world(
                policy=execution_policy,
                template=template,
                fixture=fixture,
                style_profile=style_profile,
                mode="formal",
                world_record=record,
                structure_key_hex=str(generator_capabilities["structure"]),
            )
        projection = _project_original_visible_world(
            execution_policy=execution_policy,
            template=template,
            split=split,
            world=world,
        )
        labels = preceremony.validate_full_pair_labels(
            pair_rows=world["public"]["complete_model_pair_endpoints"],
            controller_membership=world["private"]["controller_membership"],
            expected_world_uid=str(record["world_uid"]),
        )
        output.append(
            _build_original_world_diagnostic(
                policy=policy,
                world=world,
                projection=projection,
                label_rows=labels,
            )
        )
        if progress_every and (index + 1) % progress_every == 0:
            print(
                f"ORIGINAL_TEXT_WORLD_PROGRESS {split} {index + 1}/{world_count}",
                flush=True,
            )
    return _aggregate_original_worlds(
        output, split=split, expected_world_count=world_count
    )


def _direct_metrics(
    *,
    policy: Mapping[str, Any],
    split: SplitOriginalDiagnostic,
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for index, name in enumerate(split.feature_names):
        values = split.scores[:, index]
        auc = shortcut._point_roc_auc(policy, split.labels, values)
        rows[name] = {
            "roc_auc": auc,
            "symmetric_roc_auc": max(auc, 1.0 - auc),
            "average_precision_forward": shortcut._point_average_precision(
                policy, split.labels, values
            ),
            "average_precision_reverse": shortcut._point_average_precision(
                policy, split.labels, -values
            ),
        }
    strata = np.asarray(split.strata, dtype=object)
    stratified: dict[str, Any] = {}
    for stratum in (
        "positive",
        "exact_title_clone_target",
        "high_semantic_similarity_target",
        "other_negative",
    ):
        selected = strata == stratum
        if not np.any(selected):
            raise TextShortcutRunnerError("Original descriptive stratum is empty")
        stratified[stratum] = {
            "row_count": int(np.sum(selected)),
            "feature_mean": {
                name: float(np.mean(split.scores[selected, index]))
                for index, name in enumerate(split.feature_names)
            },
            "feature_maximum": {
                name: float(np.max(split.scores[selected, index]))
                for index, name in enumerate(split.feature_names)
            },
        }
    return {
        "split": split.split,
        "status": "DESCRIPTIVE_ONLY_NO_FORMAL_GATE",
        "pair_count": len(split.pair_uids),
        "positive_count": int(np.sum(split.labels)),
        "random_average_precision_baseline": 10.0 / 189.0,
        "direct_character_and_word_metrics": rows,
        "registered_mechanism_strata": stratified,
        "threshold_or_model_fitted": False,
    }


def evaluate_original_descriptive_attacks(
    *,
    policy: Mapping[str, Any],
    train: SplitOriginalDiagnostic,
    development: SplitOriginalDiagnostic,
) -> dict[str, Any]:
    if (
        train.split != "train"
        or development.split != "development"
        or train.feature_names != development.feature_names
        or set(train.world_uids) & set(development.world_uids)
    ):
        raise TextShortcutRunnerError("Original descriptive split boundary drift")
    intersections = {
        name: len(train.inventory_sets[name] & development.inventory_sets[name])
        for name in sorted(train.inventory_sets)
    }
    if any(intersections.values()):
        nonzero = {name: count for name, count in intersections.items() if count}
        raise TextShortcutRunnerError(
            f"Original design cross-split intersection failed: {nonzero}"
        )
    if train.visible_leakage_count or development.visible_leakage_count:
        raise TextShortcutRunnerError(
            "Original design visible leakage failed: "
            f"train={train.visible_leakage_count} "
            f"development={development.visible_leakage_count}"
        )
    return {
        "status": "PASS_DESIGN_ORIGINAL_TEXT_ISOLATION_DESCRIPTIVE_ONLY",
        "train": _direct_metrics(policy=policy, split=train),
        "development": _direct_metrics(policy=policy, split=development),
        "cross_split_exact_intersection_counts": intersections,
        "visible_forbidden_residue_counts": {
            "train": train.visible_leakage_count,
            "development": development.visible_leakage_count,
        },
        "near_duplicate_features_included": True,
        "absolute_near_duplicate_threshold_applied": False,
        "formal_identity_hash_intersection_audit": {
            "status": "DEFERRED_UNTIL_FORMAL_IDENTITIES_EXIST",
            "reason": (
                "design capabilities are not formal identity capabilities; the "
                "privileged persisted-data audit must recompute this after generation"
            ),
        },
        "formal_gate": False,
    }


def validate_original_fast_full_parity() -> dict[str, Any]:
    """Prove that design fast projection equals full identity-remapped projection."""

    policy = shortcut.load_text_audit_policy()
    rows: list[dict[str, Any]] = []
    for split in ("train", "development"):
        (
            execution_policy,
            template,
            fixture,
            style_profile,
            generator_capabilities,
            records,
            validated,
        ) = shortcut._design_split_context(split=split)
        record = records[0]
        with formal.mounted_structure_capability(
            split=split,
            structure_key_hex=str(generator_capabilities["structure"]),
        ):
            world = world_builder.build_world(
                policy=execution_policy,
                template=template,
                fixture=fixture,
                style_profile=style_profile,
                mode="formal",
                world_record=record,
                structure_key_hex=str(generator_capabilities["structure"]),
            )
        fast_projection = _project_original_visible_world(
            execution_policy=execution_policy,
            template=template,
            split=split,
            world=world,
        )
        fast_labels = preceremony.validate_full_pair_labels(
            pair_rows=world["public"]["complete_model_pair_endpoints"],
            controller_membership=world["private"]["controller_membership"],
            expected_world_uid=str(record["world_uid"]),
        )
        fast = _build_original_world_diagnostic(
            policy=policy,
            world=world,
            projection=fast_projection,
            label_rows=fast_labels,
        )
        full_bundle = formal.materialize_world_bundle(
            execution_policy=execution_policy,
            template=template,
            fixture=fixture,
            style_profile=style_profile,
            split=split,
            world_record=record,
            generator_capabilities=generator_capabilities,
            historical_forbidden_hashes=validated["baseline"][
                "failed_identity_hashes"
            ],
            allocated_identity_hashes=set(),
            maximum_identity_counter=int(
                validated["draft"]["identity_collision_resolution"][
                    "maximum_counter"
                ]
            ),
        )
        full_world = {
            "public": {
                "world": {"world_uid": full_bundle["world_uid"]},
                "sellers": full_bundle["public"]["sellers"],
                "items": full_bundle["private"]["raw_identity_bearing_items"],
                "complete_model_pair_endpoints": full_bundle["public"][
                    "complete_model_pair_endpoints"
                ],
            },
            "private": {
                "negative_flags": full_bundle["private"]["negative_flags"],
                "override_audit": full_bundle["private"][
                    "registered_override_audit"
                ],
                "controller_membership": full_bundle["private"][
                    "controller_membership"
                ],
            },
        }
        full_projection = {
            "seller_profiles": full_bundle["public"]["seller_profiles"],
            "redacted_items": full_bundle["public"]["redacted_items"],
            "audit": fast_projection["audit"],
        }
        full = _build_original_world_diagnostic(
            policy=policy,
            world=full_world,
            projection=full_projection,
            label_rows=full_bundle["private"]["classification_labels"],
        )
        if (
            fast_projection["redacted_items"]
            != full_bundle["public"]["redacted_items"]
            or fast_projection["seller_profiles"]
            != full_bundle["public"]["seller_profiles"]
            or fast.pair_uids != full.pair_uids
            or not np.array_equal(fast.labels, full.labels)
            or fast.feature_names != full.feature_names
            or not np.array_equal(fast.scores, full.scores)
            or fast.strata != full.strata
            or fast.inventory_sets != full.inventory_sets
        ):
            raise TextShortcutRunnerError(
                f"Original fast/full projection parity failed: {split}"
            )
        rows.append(
            {
                "split": split,
                "world_uid": fast.world_uid,
                "pair_count": len(fast.pair_uids),
                "score_matrix_sha256": hashlib.sha256(
                    np.ascontiguousarray(
                        fast.scores.astype(">f8", copy=False)
                    ).tobytes()
                ).hexdigest(),
                "redacted_items_sha256": preceremony.canonical_sha256(
                    fast_projection["redacted_items"]
                ),
                "seller_profiles_sha256": preceremony.canonical_sha256(
                    fast_projection["seller_profiles"]
                ),
            }
        )
    return {
        "status": "PASS_ORIGINAL_FAST_FULL_REDACTED_PARITY",
        "design_only": True,
        "formal_seed_or_key_access": False,
        "world_count": 2,
        "rows": rows,
    }


def _assignment_split_summary(
    *,
    policy: Mapping[str, Any],
    split: shortcut.DesignSplitAttackData,
) -> dict[str, Any]:
    if (
        split.assignment_observed.shape != split.assignment_expected.shape
        or split.assignment_observed.shape[1]
        != len(split.assignment_relation_names)
        or split.assignment_world_uids != split.visible.world_uids
    ):
        raise TextShortcutRunnerError("Assignment descriptive split drift")
    labels = split.visible.labels
    relations: dict[str, Any] = {}
    for index, name in enumerate(split.assignment_relation_names):
        observed = split.assignment_observed[:, index]
        expected = split.assignment_expected[:, index]
        residual = observed - expected
        auc = shortcut._point_roc_auc(policy, labels, observed)
        relations[name] = {
            "roc_auc": auc,
            "symmetric_roc_auc": max(auc, 1.0 - auc),
            "observed_mean_positive": float(np.mean(observed[labels == 1])),
            "observed_mean_negative": float(np.mean(observed[labels == 0])),
            "exact_null_mean_positive": float(np.mean(expected[labels == 1])),
            "exact_null_mean_negative": float(np.mean(expected[labels == 0])),
            "residual_mean_positive": float(np.mean(residual[labels == 1])),
            "residual_mean_negative": float(np.mean(residual[labels == 0])),
        }
    return {
        "split": split.visible.split,
        "world_count": len(set(split.visible.world_uids)),
        "pair_count": len(split.visible.pair_uids),
        "relations": relations,
        "classifier_fitted": False,
    }


def _rowwise_audit_receipt(
    *,
    counterfactual_train_world_audits: Sequence[Mapping[str, Any]],
    counterfactual_development_world_audits: Sequence[Mapping[str, Any]],
    original_train: SplitOriginalDiagnostic,
    original_development: SplitOriginalDiagnostic,
) -> dict[str, Any]:
    blocks: dict[str, Any] = {}
    for split_name, counterfactual_audits, original in (
        ("train", counterfactual_train_world_audits, original_train),
        (
            "development",
            counterfactual_development_world_audits,
            original_development,
        ),
    ):
        if (
            len(counterfactual_audits) != 500
            or len(original.world_audits) != 500
            or any(
                row["parser_exact_rows_and_flags"] is not True
                or int(row["planned_identity_surface_residue_count"]) != 0
                or int(row["seller_profile_count"]) != 28
                or preceremony.HEX_SHA256_RE.fullmatch(
                    str(row.get("production_audit_sha256", ""))
                )
                is None
                for row in counterfactual_audits
            )
            or any(
                row["parser_exact_rows_and_flags"] is not True
                or int(row["planned_identity_surface_residue_count"]) != 0
                or not 56 <= int(row["redacted_item_count"]) <= 224
                or int(row["seller_profile_count"]) != 28
                or preceremony.HEX_SHA256_RE.fullmatch(
                    str(row.get("production_audit_sha256", ""))
                )
                is None
                for row in original.world_audits
            )
        ):
            raise TextShortcutRunnerError(
                f"Design rowwise production audit failed: {split_name}"
            )
        original_item_count = sum(
            int(row["redacted_item_count"]) for row in original.world_audits
        )
        style_change_counts = [
            int(row["source_style_changed_seller_count"])
            for row in counterfactual_audits
        ]
        title_change_counts = [
            int(row["raw_title_changed_item_count"])
            for row in counterfactual_audits
        ]
        description_change_counts = [
            int(row["raw_description_changed_item_count"])
            for row in counterfactual_audits
        ]
        blocks[split_name] = {
            "world_count": 500,
            "original_redacted_item_count": original_item_count,
            "original_seller_profile_count": 500 * 28,
            "original_pair_rows_recomputed": 500 * 378,
            "counterfactual_pair_rows_recomputed_after_neutral_mask": 500 * 372,
            "counterfactual_positive_rows": 500 * 20,
            "counterfactual_change_counts_descriptive_only": {
                "source_style_changed_seller_count_minimum": min(
                    style_change_counts
                ),
                "source_style_changed_seller_count_maximum": max(
                    style_change_counts
                ),
                "source_style_changed_seller_count_sum": sum(
                    style_change_counts
                ),
                "raw_title_changed_item_count_minimum": min(title_change_counts),
                "raw_title_changed_item_count_maximum": max(title_change_counts),
                "raw_title_changed_item_count_sum": sum(title_change_counts),
                "raw_description_changed_item_count_minimum": min(
                    description_change_counts
                ),
                "raw_description_changed_item_count_maximum": max(
                    description_change_counts
                ),
                "raw_description_changed_item_count_sum": sum(
                    description_change_counts
                ),
                "formal_gate": False,
            },
            "original_world_audit_sha256": preceremony.canonical_sha256(
                original.world_audits
            ),
            "counterfactual_world_audit_sha256": preceremony.canonical_sha256(
                counterfactual_audits
            ),
            "raw_rows_persisted": False,
        }
    return {
        "status": "PASS_DESIGN_WORLDS_ROW_BY_ROW_RECOMPUTED_IN_MEMORY",
        "splits": blocks,
        "formal_dataset_rows_audited": 0,
        "formal_dataset_audit_deferred_until_formal_dataset_exists": True,
    }


def _file_pin(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": preceremony.sha256_file(path),
    }


def _source_closure(policy: Mapping[str, Any]) -> dict[str, Any]:
    paths = {
        "runner": Path(__file__).resolve(),
        "preflight": Path(shortcut.__file__).resolve(),
        "assignment_null": Path(shortcut.assignment_null.__file__).resolve(),
        "counterfactual_text": Path(
            shortcut.counterfactual_text.__file__
        ).resolve(),
        "exact_logistic": Path(shortcut.exact_preflight.__file__).resolve(),
        "style_derangement": (
            ROOT / "scripts/step28_v13_v1_12_style_derangement.py"
        ),
        "formal_common": Path(formal.__file__).resolve(),
        "preceremony": Path(preceremony.__file__).resolve(),
        "production_chain": Path(production.__file__).resolve(),
        "profiles": Path(profiles.__file__).resolve(),
        "world_builder": Path(world_builder.__file__).resolve(),
        "step28_common": ROOT / "scripts/step28_common.py",
        "history_common": ROOT / "scripts/step28_history_common.py",
        "v13_common": ROOT / "scripts/step28_v13_common.py",
        "history_features": ROOT / "scripts/step28_v13_history_features.py",
        "identity_plan": ROOT / "scripts/step28_v13_identity_plan.py",
        "identity_values": ROOT / "scripts/step28_v13_identity_values.py",
        "nonidentity": ROOT / "scripts/step28_v13_nonidentity.py",
        "structure": ROOT / "scripts/step28_v13_structure.py",
        "text_renderer": ROOT / "scripts/step28_v13_text_renderer.py",
        "formal_build_draft": formal.DEFAULT_DRAFT_PATH,
        "policy": shortcut.DEFAULT_POLICY_PATH,
        "contract": ROOT / str(policy["contract"]["path"]),
    }
    if any(not path.is_file() for path in paths.values()):
        raise TextShortcutRunnerError("Design preflight source closure is incomplete")
    output = {name: _file_pin(path) for name, path in paths.items()}
    output["policy"]["canonical_self_hash"] = str(policy["canonical_self_hash"])
    output["formal_build_draft"]["canonical_self_hash"] = str(
        formal.load_and_validate_draft()["draft"]["canonical_self_hash"]
    )
    return output


def _minimal_failure_source_snapshot() -> dict[str, Any]:
    """Return a non-throwing source snapshot for pre-closure failures."""

    paths = {
        "runner": Path(__file__).resolve(),
        "preflight": Path(shortcut.__file__).resolve(),
        "formal_build_draft": formal.DEFAULT_DRAFT_PATH,
        "policy": shortcut.DEFAULT_POLICY_PATH,
        "contract": (
            ROOT
            / "docs/STEP28_V13_V1_12_TEXT_SHORTCUT_AUDIT_CONTRACT_20260808.zh.md"
        ),
    }
    output: dict[str, Any] = {}
    for name, path in paths.items():
        try:
            exists = path.is_file()
            record: dict[str, Any] = {
                "path": (
                    path.relative_to(ROOT).as_posix()
                    if ROOT in path.parents
                    else str(path)
                ),
                "exists": exists,
            }
            if exists:
                record.update(
                    {
                        "size_bytes": path.stat().st_size,
                        "sha256": preceremony.sha256_file(path),
                    }
                )
        except Exception as exc:  # pragma: no cover - OS-level failure
            record = {
                "path": str(path),
                "exists": "UNKNOWN_AFTER_METADATA_ERROR",
                "pin_error": f"{type(exc).__name__}: {exc}",
            }
        output[name] = record
    return output


def _require_stage_status(
    stage: str, result: Any, expected_status: str
) -> Mapping[str, Any]:
    if not isinstance(result, Mapping) or result.get("status") != expected_status:
        observed = result.get("status") if isinstance(result, Mapping) else None
        raise TextShortcutStageError(
            stage,
            TextShortcutRunnerError(
                f"Unexpected stage status: expected={expected_status} "
                f"observed={observed}"
            ),
        )
    return result


def _require_stage_mapping(stage: str, result: Any) -> Mapping[str, Any]:
    if not isinstance(result, Mapping):
        raise TextShortcutStageError(
            stage,
            TextShortcutRunnerError("Stage did not return a mapping"),
        )
    return result


def _run_stage(stage: str, function: Any, /, *args: Any, **kwargs: Any) -> Any:
    print(f"TEXT_PREFLIGHT_STAGE {stage}", flush=True)
    try:
        return function(*args, **kwargs)
    except Exception as exc:
        raise TextShortcutStageError(stage, exc) from exc


def _registered_gate_failure_receipt(
    *,
    run_id: str,
    policy: Mapping[str, Any],
    source_closure: Mapping[str, Any],
    failed_stage: str,
    gate_result: Mapping[str, Any],
    completed_stages: Mapping[str, Any],
    counterfactual_world_audit_sha256: Mapping[str, str],
) -> dict[str, Any]:
    if (
        set(counterfactual_world_audit_sha256) != {"train", "development"}
        or any(
            preceremony.HEX_SHA256_RE.fullmatch(str(value)) is None
            for value in counterfactual_world_audit_sha256.values()
        )
    ):
        raise TextShortcutRunnerError(
            "Registered-gate counterfactual audit hash closure failed"
        )
    receipt = {
        "version": "2026-08-08-step28-v13-v1-12-text-shortcut-preflight-failure-v1",
        "run_id": run_id,
        "status": "FAIL_DESIGN_TEXT_SHORTCUT_PREFLIGHT_REGISTERED_GATE",
        "design_only": True,
        "failure_stage": failed_stage,
        "failure_type": "registered_gate_failure",
        "failure_message": str(gate_result.get("status", "UNKNOWN_GATE_STATUS")),
        "gate_result": dict(gate_result),
        "completed_stages": dict(completed_stages),
        "counterfactual_world_audit_sha256": dict(
            counterfactual_world_audit_sha256
        ),
        "source_files": dict(source_closure),
        "formal_authorizations_after_failure": dict(policy["authorizations"]),
        "formal_seed_or_key_access": False,
        "formal_dataset_rows_produced": 0,
        "formal_dataset_rows_audited": 0,
        "model_training_authorized": False,
        "audit_truth_unsealed": False,
        "raw_design_worlds_or_matrices_persisted": False,
        "non_reuse_commitment": FROZEN_VERSION_NON_REUSE_COMMITMENT,
    }
    if set(receipt["formal_authorizations_after_failure"].values()) != {False}:
        raise TextShortcutRunnerError("Registered-gate failure authorization drift")
    return preceremony.with_canonical_self_hash(receipt)


def run_design_preflight(
    *, run_id: str, progress_every: int = 10
) -> dict[str, Any]:
    """Run the complete nonpersistent 500+500 design text preceremony."""

    policy = _run_stage("policy_closure", shortcut.load_text_audit_policy)
    source_closure = _run_stage("source_closure", _source_closure, policy)
    counterfactual_parity = _run_stage(
        "counterfactual_fast_full_parity",
        shortcut.validate_text_fast_full_parity,
    )
    _require_stage_status(
        "counterfactual_fast_full_parity",
        counterfactual_parity,
        "PASS_TEXT_FAST_FULL_REDACTED_PARITY",
    )
    original_parity = _run_stage(
        "original_fast_full_parity", validate_original_fast_full_parity
    )
    _require_stage_status(
        "original_fast_full_parity",
        original_parity,
        "PASS_ORIGINAL_FAST_FULL_REDACTED_PARITY",
    )
    counterfactual_train = _run_stage(
        "counterfactual_train_500_worlds",
        shortcut.build_design_split_attack_data,
        split="train", world_count=500, progress_every=progress_every
    )
    counterfactual_development = _run_stage(
        "counterfactual_development_500_worlds",
        shortcut.build_design_split_attack_data,
        split="development", world_count=500, progress_every=progress_every
    )
    counterfactual_world_audit_sha256 = {
        "train": preceremony.canonical_sha256(
            counterfactual_train.world_audits
        ),
        "development": preceremony.canonical_sha256(
            counterfactual_development.world_audits
        ),
    }
    assignment_train = _run_stage(
        "assignment_train_description",
        _assignment_split_summary,
        policy=policy, split=counterfactual_train
    )
    assignment_development = _run_stage(
        "assignment_development_description",
        _assignment_split_summary,
        policy=policy, split=counterfactual_development
    )
    assignment_gate = _run_stage(
        "assignment_development_hard_gate",
        shortcut.evaluate_assignment_null_gate,
        policy=policy, development=counterfactual_development
    )
    assignment_gate = _require_stage_mapping(
        "assignment_development_hard_gate", assignment_gate
    )
    if assignment_gate.get("status") != "PASS_ASSIGNMENT_NULL_GATES":
        return _registered_gate_failure_receipt(
            run_id=run_id,
            policy=policy,
            source_closure=source_closure,
            failed_stage="assignment_development_hard_gate",
            gate_result=assignment_gate,
            completed_stages={
                "counterfactual_fast_full_parity": counterfactual_parity,
                "original_fast_full_parity": original_parity,
                "assignment_train_description": assignment_train,
                "assignment_development_description": assignment_development,
            },
            counterfactual_world_audit_sha256=(
                counterfactual_world_audit_sha256
            ),
        )
    visible_gate = _run_stage(
        "visible_attack_models_and_hard_gates",
        shortcut.evaluate_visible_attack_family,
        policy=policy,
        train=counterfactual_train.visible,
        development=counterfactual_development.visible,
    )
    visible_gate = _require_stage_mapping(
        "visible_attack_models_and_hard_gates", visible_gate
    )
    if visible_gate.get("status") != "PASS_VISIBLE_TEXT_SHORTCUT_GATES":
        return _registered_gate_failure_receipt(
            run_id=run_id,
            policy=policy,
            source_closure=source_closure,
            failed_stage="visible_attack_models_and_hard_gates",
            gate_result=visible_gate,
            completed_stages={
                "counterfactual_fast_full_parity": counterfactual_parity,
                "original_fast_full_parity": original_parity,
                "assignment_train_description": assignment_train,
                "assignment_development_description": assignment_development,
                "assignment_development_hard_gate": assignment_gate,
            },
            counterfactual_world_audit_sha256=(
                counterfactual_world_audit_sha256
            ),
        )
    counterfactual_train_world_audits = counterfactual_train.world_audits
    counterfactual_development_world_audits = (
        counterfactual_development.world_audits
    )
    del counterfactual_train
    del counterfactual_development
    gc.collect()
    original_train = _run_stage(
        "original_train_500_worlds",
        build_original_design_split,
        split="train", world_count=500, progress_every=progress_every
    )
    original_development = _run_stage(
        "original_development_500_worlds",
        build_original_design_split,
        split="development", world_count=500, progress_every=progress_every
    )
    original_description = _run_stage(
        "original_description_and_cross_split_isolation",
        evaluate_original_descriptive_attacks,
        policy=policy,
        train=original_train,
        development=original_development,
    )
    _require_stage_status(
        "original_description_and_cross_split_isolation",
        original_description,
        "PASS_DESIGN_ORIGINAL_TEXT_ISOLATION_DESCRIPTIVE_ONLY",
    )
    rowwise = _run_stage(
        "rowwise_design_audit_receipt",
        _rowwise_audit_receipt,
        counterfactual_train_world_audits=counterfactual_train_world_audits,
        counterfactual_development_world_audits=(
            counterfactual_development_world_audits
        ),
        original_train=original_train,
        original_development=original_development,
    )
    _require_stage_status(
        "rowwise_design_audit_receipt",
        rowwise,
        "PASS_DESIGN_WORLDS_ROW_BY_ROW_RECOMPUTED_IN_MEMORY",
    )
    stage_statuses = {
        "counterfactual_fast_full_parity": counterfactual_parity["status"],
        "original_fast_full_parity": original_parity["status"],
        "assignment_null": assignment_gate["status"],
        "visible_text_shortcut": visible_gate["status"],
        "original_text_isolation": original_description["status"],
        "rowwise_design_audit": rowwise["status"],
    }
    passed = stage_statuses == {
        "counterfactual_fast_full_parity": "PASS_TEXT_FAST_FULL_REDACTED_PARITY",
        "original_fast_full_parity": "PASS_ORIGINAL_FAST_FULL_REDACTED_PARITY",
        "assignment_null": "PASS_ASSIGNMENT_NULL_GATES",
        "visible_text_shortcut": "PASS_VISIBLE_TEXT_SHORTCUT_GATES",
        "original_text_isolation": (
            "PASS_DESIGN_ORIGINAL_TEXT_ISOLATION_DESCRIPTIVE_ONLY"
        ),
        "rowwise_design_audit": (
            "PASS_DESIGN_WORLDS_ROW_BY_ROW_RECOMPUTED_IN_MEMORY"
        ),
    }
    receipt = {
        "version": "2026-08-08-step28-v13-v1-12-text-shortcut-preflight-v1",
        "run_id": run_id,
        "status": (
            "PASS_DESIGN_TEXT_SHORTCUT_PREFLIGHT_NO_FORMAL_AUTHORIZATION"
            if passed
            else "FAIL_DESIGN_TEXT_SHORTCUT_PREFLIGHT"
        ),
        "design_only": True,
        "stage_statuses": stage_statuses,
        "counterfactual_fast_full_parity": counterfactual_parity,
        "original_fast_full_parity": original_parity,
        "assignment_null": {
            "train_description": assignment_train,
            "development_description": assignment_development,
            "development_hard_gate": assignment_gate,
        },
        "counterfactual_visible_text_hard_gate": visible_gate,
        "original_visible_text_descriptive_only": original_description,
        "rowwise_design_audit": rowwise,
        "source_files": source_closure,
        "formal_authorizations_after_preflight": dict(policy["authorizations"]),
        "formal_seed_or_key_access": False,
        "formal_dataset_rows_produced": 0,
        "formal_dataset_rows_audited": 0,
        "model_training_authorized": False,
        "audit_truth_unsealed": False,
        "raw_design_worlds_or_matrices_persisted": False,
        "claim_boundary": (
            "a passing receipt validates the design audit mechanism only; actual "
            "formal train/development rows require a later privileged rowwise audit"
        ),
    }
    if set(receipt["formal_authorizations_after_preflight"].values()) != {False}:
        raise TextShortcutRunnerError("Design preflight changed an authorization")
    return preceremony.with_canonical_self_hash(receipt)


def _failure_receipt(*, run_id: str, error: Exception) -> dict[str, Any]:
    policy: Mapping[str, Any] | None = None
    policy_validation: dict[str, Any]
    try:
        policy = shortcut.load_text_audit_policy()
        observed_authorizations = dict(policy["authorizations"])
        policy_validation = {
            "status": "VALIDATED",
            "observed_authorizations": observed_authorizations,
        }
    except Exception as exc:
        observed_authorizations = dict(CLOSED_FORMAL_AUTHORIZATIONS)
        policy_validation = {
            "status": "FAILED_BEFORE_COMPLETE_POLICY_CLOSURE",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
    source_closure_status = "COMPLETE"
    source_closure_error: dict[str, str] | None = None
    source_files: Mapping[str, Any]
    try:
        if policy is None or getattr(error, "stage", None) == "source_closure":
            raise TextShortcutRunnerError(
                "validated source closure unavailable after early failure"
            )
        source_files = _source_closure(policy)
    except Exception as exc:
        source_closure_status = "FAILED_BEFORE_COMPLETE_SOURCE_CLOSURE"
        source_closure_error = {
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
        source_files = _minimal_failure_source_snapshot()
    receipt = {
        "version": "2026-08-08-step28-v13-v1-12-text-shortcut-preflight-failure-v1",
        "run_id": run_id,
        "status": "FAIL_DESIGN_TEXT_SHORTCUT_PREFLIGHT_EXCEPTION",
        "design_only": True,
        "failure_stage": getattr(
            error, "stage", "fail_closed_exception_before_complete_receipt"
        ),
        "error_type": type(error).__name__,
        "cause_type": getattr(error, "cause_type", type(error).__name__),
        "error_message": str(error),
        "policy_validation": policy_validation,
        "source_closure_status": source_closure_status,
        "source_closure_error": source_closure_error,
        "source_files": dict(source_files),
        "formal_authorizations_after_failure": dict(
            CLOSED_FORMAL_AUTHORIZATIONS
        ),
        "formal_seed_or_key_access": False,
        "formal_dataset_rows_produced": 0,
        "formal_dataset_rows_audited": 0,
        "model_training_authorized": False,
        "audit_truth_unsealed": False,
        "large_failed_payloads_or_matrices_persisted": False,
        "non_reuse_commitment": FROZEN_VERSION_NON_REUSE_COMMITMENT,
    }
    if any(observed_authorizations.values()):
        receipt["policy_validation"] = {
            **policy_validation,
            "status": "FAILED_AUTHORIZATION_CLOSURE",
        }
    return preceremony.with_canonical_self_hash(receipt)


def _resolve_output(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    reports = (ROOT / "reports").resolve()
    if reports not in path.parents or path.suffix.lower() != ".json":
        raise TextShortcutRunnerError("Receipt output must be a JSON file under reports/")
    return path


def _design_run_id() -> str:
    run_id = str(formal.load_and_validate_draft()["draft"]["run_id"])
    if re.fullmatch(r"[a-z0-9][a-z0-9_.-]{7,127}", run_id) is None:
        raise TextShortcutRunnerError("Design preflight run ID drift")
    return run_id


def _write_receipt_no_replace(path: Path, receipt: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(
            receipt,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    if preceremony.exists_long_path(path):
        raise TextShortcutRunnerError("Design preflight receipt already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    preceremony.write_bytes_no_replace_long_path(path, payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-design-preflight",
        action="store_true",
        help="run the complete 500+500 design-only text preceremony",
    )
    parser.add_argument(
        "--output",
        help="fresh JSON receipt path under reports/",
    )
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.run_design_preflight:
        policy = shortcut.load_text_audit_policy()
        print(
            json.dumps(
                {
                    "status": policy["status"],
                    "design_preflight_run": False,
                    "formal_authorizations": policy["authorizations"],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return
    if not args.output or args.progress_every <= 0:
        raise TextShortcutRunnerError(
            "--run-design-preflight requires --output and positive --progress-every"
        )
    output = _resolve_output(args.output)
    if preceremony.exists_long_path(output):
        raise TextShortcutRunnerError("Design preflight output must be fresh")
    run_id = "unresolved-v1.12-design-preflight"
    try:
        run_id = _run_stage("design_run_id", _design_run_id)
        receipt = run_design_preflight(
            run_id=run_id, progress_every=args.progress_every
        )
    except Exception as exc:
        failure = _failure_receipt(run_id=run_id, error=exc)
        _write_receipt_no_replace(output, failure)
        print(failure["status"], output, flush=True)
        raise
    _write_receipt_no_replace(output, receipt)
    print(receipt["status"], output, flush=True)
    if receipt["status"] != (
        "PASS_DESIGN_TEXT_SHORTCUT_PREFLIGHT_NO_FORMAL_AUTHORIZATION"
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
