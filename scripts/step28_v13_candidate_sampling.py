#!/usr/bin/env python3
"""Build the label-blind Step28-v13 C40 projection inside one world."""

from __future__ import annotations

import copy
import importlib
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import step28_v13_common as common


SPLITS = ("train", "development", "audit_a", "audit_b")
REACHABLE_TRIGGERS = (
    "shared_contact_exact",
    "shared_description_clone",
    "shared_title_clone",
    "profile_lexical_neighbor",
)
FALLBACK_TRIGGER = "fallback_hash"
AUDIT_FIELDS = (
    "canonical_pair_uid",
    "world_uid",
    "primary_trigger",
    "trigger_flags",
    "lexical_similarity",
    "structural_support_flag",
    "layer_size",
    "layer_quota",
    "hmac_digest_hex",
    "design_inclusion_probability",
    "selected_bool",
    "selected_rank",
)
SAFE_FIELDS = (
    "canonical_pair_uid",
    "world_uid",
    "seller_uid_left",
    "seller_uid_right",
)
CANDIDATE_POLICY_PROJECTION_VERSION = (
    "2026-07-28-step28-v13-public-candidate-policy-v1-draft"
)
CANDIDATE_STATIC_CONTRACT_SHA256 = (
    "56641833ffff904fce7fcc00c1cddcfbf6d4b244d37f249c4e"
    "539b1ea805061b"
)
CANDIDATE_POLICY_TOP_LEVEL_KEYS = {
    "version",
    "parent_policy_version",
    "mode",
    "split",
    "frozen_inputs",
    "observed_core_schemas",
    "complete_model_pair_endpoints_schema",
    "candidate_design",
}
CANDIDATE_FROZEN_INPUT_KEYS = {
    "step3_parser_profile_code",
    "step3_profile_schema",
    "step4_candidate_code",
    "step4_candidate_schema",
}
CANDIDATE_DESIGN_KEYS = {
    "pairs_per_world",
    "all_unordered_pairs_per_world",
    "statistics_fit_scope",
    "input_boundary",
    "step4_derived_config",
    "supported_exact_contact_triggers",
    "history_only_identity_types",
    "primary_trigger_priority",
    "structural_support_role",
    "zero_size_stratum",
    "allocation",
    "within_stratum_order",
    "selected_global_rank",
    "canonical_pair_uid",
    "public_safe_projection_columns",
    "sampling_audit_projection_columns",
    "sampling_audit_rows_per_world",
    "sampling_audit_row_order",
    "trigger_flags_exact_members_and_order",
    "sampling_audit_serialization",
    "step4_raw_evidence_must_not_be_persisted",
    "sampling_audit_forbidden_from_all_model_and_feature_workers",
    "candidate_may_read_oracle",
    "candidate_may_read_model_scores",
}
FORBIDDEN_CANDIDATE_POLICY_KEYS = {
    "randomness",
    "security",
    "identity_design",
    "supervision",
    "structure_key_hex",
    "id_namespace_key_hex",
    "id_key_hex",
    "identity_value_key_hex",
    "text_key_hex",
    "candidate_key_hex",
    "query_key_hex",
    "rewire_key_hexes",
    "controller_membership",
    "mechanism_assignments",
    "classification_labels",
    "retrieval_qrels",
    "solver_audit",
}


def _all_mapping_keys(value: Any) -> list[str]:
    keys: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            keys.append(str(key))
            keys.extend(_all_mapping_keys(nested))
    elif isinstance(value, list):
        for nested in value:
            keys.extend(_all_mapping_keys(nested))
    return keys


def validate_public_candidate_policy(
    candidate_policy: Mapping[str, Any],
    *,
    mode: str,
    split: str,
) -> None:
    """Validate the exact label/oracle/structure-secret-free projection."""

    if (
        not isinstance(candidate_policy, Mapping)
        or set(candidate_policy) != CANDIDATE_POLICY_TOP_LEVEL_KEYS
        or candidate_policy.get("version")
        != CANDIDATE_POLICY_PROJECTION_VERSION
        or candidate_policy.get("parent_policy_version")
        != common.POLICY_VERSION
        or candidate_policy.get("mode") != mode
        or candidate_policy.get("split") != split
        or mode != "development_smoke"
        or split not in SPLITS
    ):
        raise common.ContractError(
            "Public candidate policy projection envelope drift"
        )
    frozen_inputs = candidate_policy["frozen_inputs"]
    observed_schemas = candidate_policy["observed_core_schemas"]
    design = candidate_policy["candidate_design"]
    static_contract = {
        "frozen_inputs": frozen_inputs,
        "observed_core_schemas": observed_schemas,
        "complete_model_pair_endpoints_schema": candidate_policy[
            "complete_model_pair_endpoints_schema"
        ],
        "candidate_design": design,
    }
    if (
        not isinstance(frozen_inputs, Mapping)
        or set(frozen_inputs) != CANDIDATE_FROZEN_INPUT_KEYS
        or any(
            not isinstance(spec, Mapping)
            or set(spec) != {"path", "sha256"}
            for spec in frozen_inputs.values()
        )
        or not isinstance(observed_schemas, Mapping)
        or set(observed_schemas) != {"sellers.csv", "items.jsonl"}
        or observed_schemas["sellers.csv"]
        != ["world_uid", "seller_uid", "market"]
        or observed_schemas["items.jsonl"]
        != [
            "world_uid",
            "seller_uid",
            "item_uid",
            "time_bucket",
            "category",
            "title",
            "description",
        ]
        or candidate_policy["complete_model_pair_endpoints_schema"]
        != [
            "canonical_pair_uid",
            "world_uid",
            "seller_uid_left",
            "seller_uid_right",
        ]
        or not isinstance(design, Mapping)
        or set(design) != CANDIDATE_DESIGN_KEYS
        or FORBIDDEN_CANDIDATE_POLICY_KEYS.intersection(
            _all_mapping_keys(candidate_policy)
        )
        or common.canonical_sha256(static_contract)
        != CANDIDATE_STATIC_CONTRACT_SHA256
    ):
        raise common.ContractError(
            "Public candidate policy projection schema/secret drift"
        )


def build_public_candidate_policy(
    policy: Mapping[str, Any],
    *,
    mode: str,
    split: str,
) -> dict[str, Any]:
    """Project the full generator policy before entering candidate custody."""

    common.validate_policy(policy, mode=mode)
    if split not in SPLITS:
        raise common.ContractError("Unknown candidate policy split")
    frozen_names = (
        "step3_parser_profile_code",
        "step3_profile_schema",
        "step4_candidate_code",
        "step4_candidate_schema",
    )
    projection = {
        "version": CANDIDATE_POLICY_PROJECTION_VERSION,
        "parent_policy_version": str(policy["version"]),
        "mode": mode,
        "split": split,
        "frozen_inputs": {
            name: copy.deepcopy(policy["frozen_inputs"][name])
            for name in frozen_names
        },
        "observed_core_schemas": {
            name: copy.deepcopy(
                policy["relational_integrity"][
                    "observed_core_schemas"
                ][name]
            )
            for name in ("sellers.csv", "items.jsonl")
        },
        "complete_model_pair_endpoints_schema": copy.deepcopy(
            policy["relational_integrity"][
                "pair_projection_contract"
            ]["complete_model_pair_endpoints_schema"]
        ),
        "candidate_design": copy.deepcopy(policy["candidate_design"]),
    }
    validate_public_candidate_policy(
        projection, mode=mode, split=split
    )
    return projection


def _load_frozen_dependencies(
    candidate_policy: Mapping[str, Any],
) -> tuple[Any, Any, dict[str, Any]]:
    """Verify source bytes before importing the frozen Step3/Step4 wrappers."""

    step3_path = common.verify_file_pin(
        candidate_policy["frozen_inputs"]["step3_parser_profile_code"],
        label="Step3 seller-profile producer",
    )
    common.verify_file_pin(
        candidate_policy["frozen_inputs"]["step3_profile_schema"],
        label="Step3 seller-profile schema",
    )
    step4_path = common.verify_file_pin(
        candidate_policy["frozen_inputs"]["step4_candidate_code"],
        label="Step4 candidate producer",
    )
    step4_schema_path = common.verify_file_pin(
        candidate_policy["frozen_inputs"]["step4_candidate_schema"],
        label="Step4 candidate schema",
    )
    profiles_mod = importlib.import_module("step28_v13_profiles")
    step3_mod = importlib.import_module("step3_build_seller_profiles")
    step4_mod = importlib.import_module("step4_build_silver_candidates")
    if (
        common.sha256_file(step3_path)
        != common.sha256_file(Path(step3_mod.__file__).resolve())
        or common.sha256_file(step4_path)
        != common.sha256_file(Path(step4_mod.__file__).resolve())
    ):
        raise common.ContractError("Frozen candidate dependency import drift")
    return profiles_mod, step4_mod, common.load_json(step4_schema_path)


def _validate_world_inputs(
    candidate_policy: Mapping[str, Any],
    *,
    sellers: Sequence[Mapping[str, Any]],
    items: Sequence[Mapping[str, Any]],
    complete_pair_endpoints: Sequence[Mapping[str, Any]],
) -> tuple[str, dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    seller_schema = candidate_policy["observed_core_schemas"][
        "sellers.csv"
    ]
    item_schema = candidate_policy["observed_core_schemas"][
        "items.jsonl"
    ]
    pair_schema = candidate_policy[
        "complete_model_pair_endpoints_schema"
    ]
    if len(sellers) != 28 or len(complete_pair_endpoints) != 378:
        raise common.ContractError("C40 requires exactly 28 sellers and 378 pairs")

    seller_index: dict[str, dict[str, Any]] = {}
    world_uids: set[str] = set()
    for source_row in sellers:
        if list(source_row) != seller_schema:
            raise common.ContractError("C40 seller schema/order drift")
        row = dict(source_row)
        seller_uid = str(row["seller_uid"])
        if (
            not seller_uid
            or "|" in seller_uid
            or seller_uid in seller_index
            or not str(row["market"])
        ):
            raise common.ContractError("C40 seller key/value drift")
        seller_index[seller_uid] = row
        world_uids.add(str(row["world_uid"]))
    if len(world_uids) != 1:
        raise common.ContractError("C40 input must contain exactly one world")
    world_uid = next(iter(world_uids))

    item_index: dict[str, dict[str, Any]] = {}
    for source_row in items:
        if list(source_row) != item_schema:
            raise common.ContractError("C40 raw item schema/order drift")
        row = dict(source_row)
        item_uid = str(row["item_uid"])
        if (
            str(row["world_uid"]) != world_uid
            or str(row["seller_uid"]) not in seller_index
            or not item_uid
            or item_uid in item_index
        ):
            raise common.ContractError("C40 raw item lineage drift")
        item_index[item_uid] = row
    if not item_index or {
        str(row["seller_uid"]) for row in item_index.values()
    } != set(seller_index):
        raise common.ContractError("C40 raw items do not cover every seller")

    pair_index: dict[str, dict[str, Any]] = {}
    expected_pairs: set[str] = set()
    ordered_sellers = common.utf8_sort(seller_index)
    for left_index, left_uid in enumerate(ordered_sellers):
        for right_uid in ordered_sellers[left_index + 1 :]:
            expected_pairs.add(common.canonical_pair_uid(left_uid, right_uid))
    for source_row in complete_pair_endpoints:
        if list(source_row) != pair_schema:
            raise common.ContractError("C40 complete-pair schema/order drift")
        row = dict(source_row)
        left_uid = str(row["seller_uid_left"])
        right_uid = str(row["seller_uid_right"])
        pair_uid = str(row["canonical_pair_uid"])
        if (
            str(row["world_uid"]) != world_uid
            or left_uid not in seller_index
            or right_uid not in seller_index
            or pair_uid != common.canonical_pair_uid(left_uid, right_uid)
            or pair_uid in pair_index
        ):
            raise common.ContractError("C40 complete-pair lineage drift")
        pair_index[pair_uid] = row
    if set(pair_index) != expected_pairs:
        raise common.ContractError("C40 complete-pair universe is not exact")
    return world_uid, seller_index, pair_index


def _hamilton_quotas(
    layer_sizes: Mapping[str, int],
    *,
    total_slots: int,
    trigger_priority: Sequence[str],
) -> dict[str, int]:
    total_size = sum(int(layer_sizes.get(name, 0)) for name in trigger_priority)
    if total_size < total_slots:
        raise common.ContractError("Hamilton allocation has too few candidates")
    quotas: dict[str, int] = {}
    remainders: list[tuple[int, int, str]] = []
    allocated = 0
    for priority_index, name in enumerate(trigger_priority):
        size = int(layer_sizes.get(name, 0))
        if size < 0:
            raise common.ContractError("Negative C40 layer size")
        if size == 0:
            quotas[name] = 0
            continue
        numerator = total_slots * size
        quota, remainder = divmod(numerator, total_size)
        quotas[name] = quota
        allocated += quota
        remainders.append((-remainder, priority_index, name))
    remaining = total_slots - allocated
    if remaining < 0 or remaining > len(remainders):
        raise common.ContractError("Hamilton remainder allocation drift")
    for _negative_remainder, _priority_index, name in sorted(remainders)[
        :remaining
    ]:
        quotas[name] += 1
    if sum(quotas.values()) != total_slots:
        raise common.ContractError("Hamilton quota sum drift")
    return quotas


def _fixed_probability(quota: int, size: int) -> str:
    if size == 0:
        return ""
    return f"{quota / size:.12f}"


def build_world_c40(
    candidate_policy: Mapping[str, Any],
    *,
    candidate_key_hex: str,
    mode: str,
    split: str,
    sellers: Sequence[Mapping[str, Any]],
    raw_observed_items: Sequence[Mapping[str, Any]],
    complete_pair_endpoints: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, str]], list[dict[str, Any]], dict[str, Any]]:
    """Return safe C40 rows, all-378 audit rows, and a label-free audit."""

    if mode != "development_smoke":
        raise common.ContractError(
            "This candidate implementation is development-smoke only; "
            "formal C40 requires the split-private capability launcher"
        )
    validate_public_candidate_policy(
        candidate_policy, mode=mode, split=split
    )
    design = candidate_policy["candidate_design"]
    if (
        int(design["pairs_per_world"]) != 40
        or int(design["all_unordered_pairs_per_world"]) != 378
        or tuple(design["primary_trigger_priority"])
        != (*REACHABLE_TRIGGERS, FALLBACK_TRIGGER)
        or design["sampling_audit_projection_columns"] != list(AUDIT_FIELDS)
        or design["public_safe_projection_columns"] != list(SAFE_FIELDS)
        or design["sampling_audit_serialization"]["null_probability"] != ""
    ):
        raise common.ContractError("C40 policy contract drift")
    candidate_key_hex = str(candidate_key_hex)
    try:
        candidate_key = bytes.fromhex(candidate_key_hex)
    except ValueError as exc:
        raise common.ContractError("C40 candidate key is not hex") from exc
    if len(candidate_key) != 32:
        raise common.ContractError("C40 candidate key must be 32 bytes")

    world_uid, _seller_index, pair_index = _validate_world_inputs(
        candidate_policy,
        sellers=sellers,
        items=raw_observed_items,
        complete_pair_endpoints=complete_pair_endpoints,
    )
    profiles_mod, step4_mod, step4_schema = _load_frozen_dependencies(
        candidate_policy
    )
    profile_policy = {
        "modes": {mode: {}},
        "frozen_inputs": {
            "step3_profile_schema": candidate_policy["frozen_inputs"][
                "step3_profile_schema"
            ]
        },
        "relational_integrity": {
            "observed_core_schemas": candidate_policy[
                "observed_core_schemas"
            ]
        },
    }
    raw_profiles, raw_profile_audit = profiles_mod.build_world_profiles(
        profile_policy,
        mode=mode,
        split=split,
        sellers=sellers,
        items=raw_observed_items,
    )
    filtering = step4_schema["filtering_policy"]
    stopwords = {
        str(value).lower()
        for value in filtering["contact_noise_stopwords"]
    }
    step4_profiles = step4_mod.build_seller_profiles(
        rows=raw_profiles,
        data_bucket=f"step28_v13_{mode}_{split}",
        language="zh",
        stopwords=stopwords,
        min_config=filtering["content_minimums"],
        pgp_alias_map={},
    )
    derived_config = {
        key: value
        for key, value in design["step4_derived_config"].items()
        if key != "pgp_alias_map"
    }
    step4_mod.compute_retrieval_weights(step4_profiles, derived_config)
    step4_rows = step4_mod.build_candidates_for_pool(
        step4_profiles,
        derived_config,
        "zh",
        filtering["duplicate_cluster_limits"],
    )

    step4_index: dict[str, dict[str, Any]] = {}
    for source_row in step4_rows:
        row = dict(source_row)
        pair_uid = str(row.get("pair_uid", ""))
        if pair_uid not in pair_index or pair_uid in step4_index:
            raise common.ContractError("Frozen Step4 candidate pair drift")
        contact_types = set(filter(None, str(row["shared_contact_types"]).split("|")))
        if not contact_types.issubset(set(design["supported_exact_contact_triggers"])):
            raise common.ContractError("Step4 emitted an unsupported contact trigger")
        step4_index[pair_uid] = row

    projected: dict[str, dict[str, Any]] = {}
    layer_sizes: Counter[str] = Counter()
    for pair_uid in common.utf8_sort(pair_index):
        source = step4_index.get(pair_uid)
        source_flags = set()
        lexical_similarity = 0.0
        structural_support = False
        if source is not None:
            source_flags = set(
                filter(None, str(source["candidate_rule_hits"]).split("|"))
            )
            if not source_flags.issubset(
                {*REACHABLE_TRIGGERS, "structural_support"}
            ):
                raise common.ContractError(
                    "Step4 emitted an unreachable v13 candidate rule"
                )
            lexical_similarity = float(source["lexical_similarity"])
            structural_support = "structural_support" in source_flags
        flags = [name for name in REACHABLE_TRIGGERS if name in source_flags]
        primary_trigger = flags[0] if flags else FALLBACK_TRIGGER
        layer_sizes[primary_trigger] += 1
        digest = common.hmac_digest(
            candidate_key_hex, world_uid, pair_uid
        ).hex()
        projected[pair_uid] = {
            "flags": flags,
            "primary_trigger": primary_trigger,
            "lexical_similarity": lexical_similarity,
            "structural_support": structural_support,
            "digest": digest,
        }

    production_count = sum(layer_sizes[name] for name in REACHABLE_TRIGGERS)
    if production_count >= 40:
        quotas = _hamilton_quotas(
            layer_sizes,
            total_slots=40,
            trigger_priority=REACHABLE_TRIGGERS,
        )
        quotas[FALLBACK_TRIGGER] = 0
    else:
        quotas = {name: layer_sizes[name] for name in REACHABLE_TRIGGERS}
        quotas[FALLBACK_TRIGGER] = 40 - production_count
    selected: set[str] = set()
    for name in (*REACHABLE_TRIGGERS, FALLBACK_TRIGGER):
        rows = [
            pair_uid
            for pair_uid, row in projected.items()
            if row["primary_trigger"] == name
        ]
        rows.sort(
            key=lambda pair_uid: (
                bytes.fromhex(str(projected[pair_uid]["digest"])),
                pair_uid.encode("utf-8"),
            )
        )
        quota = int(quotas[name])
        if quota < 0 or quota > len(rows):
            raise common.ContractError("C40 layer quota exceeds its universe")
        selected.update(rows[:quota])
    if len(selected) != 40:
        raise common.ContractError("C40 selected pair count drift")

    global_order = sorted(
        selected,
        key=lambda pair_uid: (
            common.hmac_digest(
                candidate_key_hex,
                world_uid,
                "selected_global_rank",
                pair_uid,
            ),
            pair_uid.encode("utf-8"),
        ),
    )
    rank_by_pair = {
        pair_uid: rank for rank, pair_uid in enumerate(global_order, start=1)
    }
    audit_rows: list[dict[str, Any]] = []
    for pair_uid in common.utf8_sort(pair_index):
        row = projected[pair_uid]
        layer = str(row["primary_trigger"])
        audit_row = {
            "canonical_pair_uid": pair_uid,
            "world_uid": world_uid,
            "primary_trigger": layer,
            "trigger_flags": "|".join(row["flags"]),
            "lexical_similarity": f"{float(row['lexical_similarity']):.6f}",
            "structural_support_flag": (
                "true" if bool(row["structural_support"]) else "false"
            ),
            "layer_size": int(layer_sizes[layer]),
            "layer_quota": int(quotas[layer]),
            "hmac_digest_hex": str(row["digest"]),
            "design_inclusion_probability": _fixed_probability(
                int(quotas[layer]), int(layer_sizes[layer])
            ),
            "selected_bool": "true" if pair_uid in selected else "false",
            "selected_rank": (
                str(rank_by_pair[pair_uid]) if pair_uid in selected else ""
            ),
        }
        if list(audit_row) != list(AUDIT_FIELDS):
            raise common.ContractError("C40 audit schema/order drift")
        audit_rows.append(audit_row)

    safe_rows: list[dict[str, str]] = []
    for pair_uid in global_order:
        source = pair_index[pair_uid]
        safe_row = {name: str(source[name]) for name in SAFE_FIELDS}
        if list(safe_row) != list(SAFE_FIELDS):
            raise common.ContractError("C40 safe schema/order drift")
        safe_rows.append(safe_row)
    if (
        len(audit_rows) != 378
        or len(safe_rows) != 40
        or {row["canonical_pair_uid"] for row in safe_rows} != selected
        or sorted(
            int(row["selected_rank"])
            for row in audit_rows
            if row["selected_bool"] == "true"
        )
        != list(range(1, 41))
    ):
        raise common.ContractError("C40 output closure failed")

    label_free_audit = {
        "world_uid": world_uid,
        "complete_pair_count": len(pair_index),
        "production_triggered_pair_count": production_count,
        "selected_pair_count": len(safe_rows),
        "primary_layer_sizes": {
            name: int(layer_sizes[name])
            for name in (*REACHABLE_TRIGGERS, FALLBACK_TRIGGER)
        },
        "primary_layer_quotas": {
            name: int(quotas[name])
            for name in (*REACHABLE_TRIGGERS, FALLBACK_TRIGGER)
        },
        "raw_observed_item_input_sha256": common.canonical_sha256(
            list(raw_observed_items)
        ),
        "ephemeral_raw_profile_sha256": common.canonical_sha256(raw_profiles),
        "step3_profile_audit_sha256": common.canonical_sha256(
            raw_profile_audit
        ),
        "candidate_safe_projection_sha256": common.canonical_sha256(safe_rows),
        "candidate_sampling_audit_sha256": common.canonical_sha256(audit_rows),
        "labels_or_oracle_or_model_scores_read": False,
        "ephemeral_step4_raw_evidence_persisted": False,
    }
    return safe_rows, audit_rows, label_free_audit
