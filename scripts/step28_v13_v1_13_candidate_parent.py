#!/usr/bin/env python3
"""Design-only Step28-v13 v1.13 parent and trial-identity primitives.

This module deliberately has no formal-custody path and performs no writes.
It builds one development-smoke world in memory, freezes the structure that a
future document candidate may not change, independently records the sources of
the five Step3 fields used by M0, and allocates identity values exactly once on
a private copy of the already-committed identity-hash set.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import step28_v13_common as common
import step28_v13_history_features as history_features
import step28_v13_identity_values as identity_values
import step28_v13_nonidentity as nonidentity
import step28_v13_production_chain as production
import step28_v13_profiles as profiles_module
import step28_v13_structure as structure
import step28_v13_text_renderer as renderer
import step28_v13_v1_13_document_collision as collision
import step28_v13_v1_13_identity_remap as identity_remap
import step28_v13_world_builder as world_builder
import step3_build_seller_profiles as step3
import step7_v3_1_source_data as source


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_PATH = (
    ROOT / "schema" / "step28_v13_v1_13_candidate_parent_policy.json"
)
POLICY_VERSION = "2026-08-09-step28-v13-v1-13-candidate-parent-policy-v1"
POLICY_STATUS = (
    "DESIGN_ONLY_PARENT_AND_TRIAL_IDENTITY_IMPLEMENTATION_NO_FORMAL_"
    "AUTHORIZATION"
)
ALLOWED_MODE = "development_smoke"
SPLITS = ("train", "development", "audit_a", "audit_b")
HEX_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
FROZEN_INPUT_KEYS = frozenset(
    {
        "base_dataset_policy",
        "collision_policy",
        "collision_primitives",
        "common_contract",
        "identity_history",
        "history_common",
        "identity_remapper",
        "identity_values",
        "identity_plan",
        "legacy_step28_common",
        "nonidentity_builder",
        "production_chain",
        "profile_bridge",
        "redaction_common",
        "source_data",
        "step3_profile_code",
        "structure_builder",
        "text_renderer",
        "world_builder",
    }
)
FULL_RENDER_AST_FIELDS = (
    "world_uid",
    "seller_uid",
    "item_uid",
    "time_bucket",
    "category",
    "product",
    "attribute",
    "delivery",
    "service",
    "code",
    "title_skeleton_index",
    "description_skeleton_index",
    "effective_style_uid",
    "title_nonempty",
    "description_nonempty",
    "identity_slot_uids",
    "noise_slot_uid",
)
CANDIDATE_INVARIANT_RENDER_AST_FIELDS = (
    "world_uid",
    "seller_uid",
    "item_uid",
    "time_bucket",
    "code",
    "effective_style_uid",
    "title_nonempty",
    "description_nonempty",
    "identity_slot_uids",
    "noise_slot_uid",
)
HISTORICAL_IDENTITY_HASH_COUNT = 999_996
HISTORICAL_IDENTITY_HASHES_SHA256 = (
    "c82e86aa72c205d62ed4a03bd5166a32c8d98fde9bb5961ec1a0048b64ee1346"
)
DEVELOPMENT_SMOKE_WORLD_UID = (
    "w_003497845547650a980473b05e249937bf825ad0eaefa424baec74f2bd2210f3"
)
DEVELOPMENT_SMOKE_WORLD_RECORD_SHA256 = (
    "f147b016762dbe9aad452a58a8f937b104fe653379902cc461c1035c86f07210"
)
DEVELOPMENT_SMOKE_STRUCTURE_KEY_SHA256 = (
    "3088ac746944c07072b309277071356af0d9dddf71cae1a86aabf331312dd1e6"
)
DEVELOPMENT_SMOKE_IDENTITY_KEY_SHA256 = (
    "f5be119b0945e2c355beba49f983b4fa6dfa8fa02ebc53283335e89bcc1960e0"
)
FORMAL_AUTHORIZATION_KEYS = frozenset(
    {
        "candidate_generation",
        "formal_capability_derivation",
        "formal_dataset_generation",
        "formal_model_training",
        "formal_seed_ceremony",
    }
)
PROVENANCE_FIELDS = (
    "world_uid",
    "seller_uid",
    "output_field",
    "aggregation_role",
    "output_rank",
    "source_item_uids",
    "source_item_uids_sha256",
    "source_item_count",
    "first_seen_position",
    "item_uid",
    "extracted_segment_ordinal",
    "seller_df",
    "seller_df_seller_count",
    "seller_df_seller_uids_sha256",
)
ALLOCATION_RECEIPT_FIELDS = frozenset(
    {
        "world_uid",
        "identity_asset_count",
        "identity_slot_count",
        "changed_item_count",
        "maximum_counter",
        "maximum_selected_counter",
        "nonzero_counter_count",
        "forced_design_collision_count",
        "visible_text_candidate_rejection_count",
        "historical_intersection_count",
        "same_run_intersection_count",
        "selected_value_hashes_sha256",
        "allocation_audit_rows",
        "allocation_audit_rows_sha256",
    }
)
PROFILE_ROLES = (
    (
        "category_concat_top",
        "top_categories",
        "category_frequency_rank",
        "category",
    ),
    (
        "signature_title_concat",
        "signature_titles",
        "title_specificity_rank",
        "title",
    ),
    (
        "title_concat_top",
        "top_titles",
        "title_frequency_rank",
        "title",
    ),
    (
        "signature_description_concat",
        "signature_description_segments",
        "description_segment_specificity_rank",
        "description_segment",
    ),
    (
        "description_concat_top",
        "top_description_snippets",
        "description_snippet_frequency_rank",
        "description_snippet",
    ),
)


class CandidateParentError(common.ContractError):
    """Raised when a v1.13 parent or trial-identity invariant fails."""


def _canonical_bytes(value: Any) -> bytes:
    return common.canonical_json_bytes(value)


def _canonical_clone(value: Any) -> Any:
    return json.loads(_canonical_bytes(value).decode("utf-8"))


def _decode_canonical(payload: bytes, *, label: str) -> Any:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CandidateParentError(f"{label} canonical payload is invalid") from exc
    if _canonical_bytes(value) != payload:
        raise CandidateParentError(f"{label} payload is not canonical JSON")
    return value


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_exact_keys(
    value: Mapping[str, Any], expected: Collection[str], *, label: str
) -> None:
    if set(value) != set(expected):
        raise CandidateParentError(f"{label} keyset drift")


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or HEX_SHA256_RE.fullmatch(value) is None:
        raise CandidateParentError(f"{label} is not a lowercase SHA-256 value")
    return value


def _verify_self_hash(document: Mapping[str, Any], *, label: str) -> None:
    claimed = _require_sha256(
        document.get("canonical_self_hash"),
        label=f"{label}.canonical_self_hash",
    )
    payload = dict(document)
    payload.pop("canonical_self_hash")
    if common.canonical_sha256(payload) != claimed:
        raise CandidateParentError(f"{label} canonical self-hash drift")


def _validate_policy_document(policy: Mapping[str, Any]) -> None:
    """Validate exact policy semantics, including re-self-hashed tampering."""

    _require_exact_keys(
        policy,
        {
            "version",
            "status",
            "claim_boundary",
            "formal_authorizations",
            "allowed_mode",
            "required_split_values",
            "frozen_inputs",
            "candidate_independent_parent",
            "profile_contribution_provenance",
            "trial_identity_allocation",
            "development_smoke_parent",
            "canonical_self_hash",
        },
        label="candidate-parent policy",
    )
    _verify_self_hash(policy, label="candidate-parent policy")
    if (
        policy["version"] != POLICY_VERSION
        or policy["status"] != POLICY_STATUS
        or policy["allowed_mode"] != ALLOWED_MODE
        or tuple(policy["required_split_values"]) != SPLITS
    ):
        raise CandidateParentError("Candidate-parent policy version boundary drift")
    authorizations = policy["formal_authorizations"]
    if not isinstance(authorizations, dict):
        raise CandidateParentError("Formal authorization block is missing")
    _require_exact_keys(
        authorizations,
        FORMAL_AUTHORIZATION_KEYS,
        label="formal authorization block",
    )
    if any(value is not False for value in authorizations.values()):
        raise CandidateParentError("Candidate-parent policy cannot authorize work")
    frozen_inputs = policy["frozen_inputs"]
    if not isinstance(frozen_inputs, dict):
        raise CandidateParentError("Frozen candidate-parent inputs are missing")
    _require_exact_keys(
        frozen_inputs, FROZEN_INPUT_KEYS, label="frozen candidate-parent inputs"
    )
    for name, spec in frozen_inputs.items():
        if not isinstance(spec, dict):
            raise CandidateParentError(f"Frozen input is malformed: {name}")
        common.verify_file_pin(spec, label=f"v1.13 parent dependency {name}")
    parent = policy["candidate_independent_parent"]
    _require_exact_keys(
        parent,
        {
            "seller_count",
            "pair_count",
            "full_render_ast_schema",
            "candidate_invariant_render_ast_projection",
            "candidate_visible_fields_excluded_from_parent_fingerprint",
            "effective_style_factors_source",
            "registered_override_lineage_exact",
            "noise_target_lineage_exact",
            "identity_values_excluded_until_trial_allocation",
        },
        label="candidate-independent parent contract",
    )
    if (
        parent["seller_count"] != 28
        or parent["pair_count"] != 378
        or tuple(parent["full_render_ast_schema"]) != FULL_RENDER_AST_FIELDS
        or tuple(parent["candidate_invariant_render_ast_projection"])
        != CANDIDATE_INVARIANT_RENDER_AST_FIELDS
        or tuple(parent["candidate_visible_fields_excluded_from_parent_fingerprint"])
        != ("title", "description")
        or parent["effective_style_factors_source"]
        != "template.renderer_contract.style_factor_order"
        or parent["registered_override_lineage_exact"] is not True
        or parent["noise_target_lineage_exact"] is not True
        or parent["identity_values_excluded_until_trial_allocation"] is not True
    ):
        raise CandidateParentError("Candidate-independent parent semantics drift")
    provenance = policy["profile_contribution_provenance"]
    _require_exact_keys(
        provenance,
        {
            "public_profile_algorithm_unchanged",
            "public_profile_bytes_must_match_frozen_step3",
            "output_fields",
            "row_fields",
            "raw_contribution_values_persisted",
            "private_audit_only",
        },
        label="profile contribution contract",
    )
    if (
        tuple(provenance.get("row_fields", ())) != PROVENANCE_FIELDS
        or tuple(provenance.get("output_fields", ()))
        != tuple(row[0] for row in PROFILE_ROLES)
        or provenance.get("public_profile_algorithm_unchanged") is not True
        or provenance.get("public_profile_bytes_must_match_frozen_step3")
        is not True
        or provenance.get("raw_contribution_values_persisted") is not False
        or provenance.get("private_audit_only") is not True
    ):
        raise CandidateParentError("Profile contribution contract drift")
    trial = policy["trial_identity_allocation"]
    _require_exact_keys(
        trial,
        {
            "allocator_calls_per_world",
            "candidate_loop_receives_live_identity_set",
            "committed_hash_input_type",
            "trial_set_is_private_copy",
            "returned_state_serialization",
            "identity33_rows_per_world",
            "identity33_feature_count",
            "retry_after_allocator_entry_forbidden",
            "formal_custody_released",
            "historical_identity_hash_count",
            "historical_identity_hashes_sha256",
            "maximum_counter",
            "identity_candidate_domain",
            "identity_key_source",
            "identity_key_sha256",
            "normal_api_fault_injection",
            "allocation_audit_row_fields",
            "allocation_audit_contains_raw_identity_values",
        },
        label="trial identity allocation contract",
    )
    if (
        trial.get("allocator_calls_per_world") != 1
        or trial.get("candidate_loop_receives_live_identity_set") is not False
        or trial.get("committed_hash_input_type") != "frozenset"
        or trial.get("trial_set_is_private_copy") is not True
        or trial.get("identity33_rows_per_world") != 378
        or trial.get("identity33_feature_count") != 33
        or trial.get("returned_state_serialization") != "canonical-json-utf8-bytes"
        or trial.get("retry_after_allocator_entry_forbidden") is not True
        or trial.get("formal_custody_released") is not False
        or trial.get("historical_identity_hash_count")
        != HISTORICAL_IDENTITY_HASH_COUNT
        or trial.get("historical_identity_hashes_sha256")
        != HISTORICAL_IDENTITY_HASHES_SHA256
        or trial.get("maximum_counter") != 128
        or trial.get("identity_candidate_domain")
        != "step28-v13-v1.13-identity-value"
        or trial.get("identity_key_source")
        != "base_dataset_policy.randomness.development_smoke.identity_value_key_hex"
        or trial.get("identity_key_sha256")
        != DEVELOPMENT_SMOKE_IDENTITY_KEY_SHA256
        or trial.get("normal_api_fault_injection") is not False
        or tuple(trial.get("allocation_audit_row_fields", ()))
        != (
            "identity_asset_uid",
            "selected_counter",
            "visible_text_candidate_rejection_count",
            "selected_value_hash",
        )
        or trial.get("allocation_audit_contains_raw_identity_values") is not False
    ):
        raise CandidateParentError("Trial identity contract drift")
    smoke = policy.get("development_smoke_parent")
    _require_exact_keys(
        smoke,
        {
            "world_pool_index",
            "world_uid",
            "split",
            "world_record_canonical_sha256",
            "structure_key_sha256",
        },
        label="development-smoke parent contract",
    )
    if (
        smoke["world_pool_index"] != 0
        or smoke["world_uid"] != DEVELOPMENT_SMOKE_WORLD_UID
        or smoke["split"] != "audit_a"
        or smoke["world_record_canonical_sha256"]
        != DEVELOPMENT_SMOKE_WORLD_RECORD_SHA256
        or smoke["structure_key_sha256"]
        != DEVELOPMENT_SMOKE_STRUCTURE_KEY_SHA256
    ):
        raise CandidateParentError("Development-smoke parent selection drift")


def load_policy(path: Path = DEFAULT_POLICY_PATH) -> dict[str, Any]:
    """Load the sole design-only policy and revalidate every dependency pin."""

    if path.resolve() != DEFAULT_POLICY_PATH.resolve():
        raise CandidateParentError(
            "Only the canonical v1.13 candidate-parent policy may produce PASS"
        )
    try:
        policy = common.load_json(path)
    except common.ContractError as exc:
        raise CandidateParentError("Candidate-parent policy is invalid JSON") from exc
    _validate_policy_document(policy)
    return policy


@dataclass(frozen=True)
class CandidateIndependentParent:
    """Immutable byte envelope for a single candidate-independent world."""

    mode: str
    split: str
    world_uid: str
    bootstrap_world_bytes: bytes
    invariant_projection_bytes: bytes
    invariant_sha256: str
    profile_bytes: bytes
    profile_sha256: str
    profile_provenance_bytes: bytes
    profile_provenance_sha256: str

    def thaw_bootstrap_world(self) -> dict[str, Any]:
        value = _decode_canonical(
            self.bootstrap_world_bytes, label="candidate-independent world"
        )
        if not isinstance(value, dict):
            raise CandidateParentError("Candidate-independent world is not an object")
        return value

    def thaw_profiles(self) -> list[dict[str, Any]]:
        value = _decode_canonical(self.profile_bytes, label="parent profiles")
        if not isinstance(value, list):
            raise CandidateParentError("Parent profiles are not a list")
        return value

    def thaw_profile_provenance(self) -> dict[str, Any]:
        value = _decode_canonical(
            self.profile_provenance_bytes, label="profile provenance"
        )
        if not isinstance(value, dict):
            raise CandidateParentError("Profile provenance is not an object")
        return value


@dataclass(frozen=True)
class FrozenTrialIdentityParent:
    """Immutable result of the sole trial identity allocation for one world."""

    mode: str
    split: str
    world_uid: str
    world_bytes: bytes
    identity_parent_projection_bytes: bytes
    identity_parent_sha256: str
    identity33_bytes: bytes
    identity33_sha256: str
    allocation_receipt_bytes: bytes
    allocation_delta: tuple[str, ...]
    candidate_invariant_sha256: str
    profile_provenance_sha256: str
    profile_sha256: str

    def thaw_world(self) -> dict[str, Any]:
        value = _decode_canonical(self.world_bytes, label="trial identity world")
        if not isinstance(value, dict):
            raise CandidateParentError("Trial identity world is not an object")
        return value

    def thaw_identity33(self) -> list[dict[str, Any]]:
        value = _decode_canonical(self.identity33_bytes, label="identity33 parent")
        if not isinstance(value, list):
            raise CandidateParentError("Identity33 parent is not a list")
        return value


def _sorted_rows(
    rows: Sequence[Mapping[str, Any]], *fields: str
) -> list[dict[str, Any]]:
    return sorted(
        (dict(row) for row in rows),
        key=lambda row: tuple(str(row[field]).encode("utf-8") for field in fields),
    )


def _support_digest(values: Sequence[str]) -> str:
    ordered = common.utf8_sort(set(values))
    return common.canonical_sha256(ordered)


def build_profile_contribution_provenance(
    *,
    world_uid: str,
    profiles: Sequence[Mapping[str, Any]],
    profile_safe_items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Reconstruct private contributor lineage without changing Step3 output."""

    profile_by_seller: dict[str, dict[str, Any]] = {}
    for source_profile in profiles:
        profile = dict(source_profile)
        seller_uid = str(profile.get("seller_uid", ""))
        if not seller_uid or seller_uid in profile_by_seller:
            raise CandidateParentError("Profile seller keyset is invalid")
        profile_by_seller[seller_uid] = profile
    if len(profile_by_seller) != 28:
        raise CandidateParentError("Contribution provenance requires 28 profiles")

    ordered_items = sorted(
        (dict(row) for row in profile_safe_items),
        key=lambda row: (
            str(row["world_uid"]).encode("utf-8"),
            str(row["seller_uid"]).encode("utf-8"),
            str(row["item_uid"]).encode("utf-8"),
        ),
    )
    if (
        not ordered_items
        or len({str(row["item_uid"]) for row in ordered_items})
        != len(ordered_items)
        or any(str(row["world_uid"]) != world_uid for row in ordered_items)
        or {str(row["seller_uid"]) for row in ordered_items}
        != set(profile_by_seller)
    ):
        raise CandidateParentError("Contribution item lineage is invalid")

    occurrences: dict[
        tuple[str, str, str], list[tuple[int, str, int]]
    ] = defaultdict(list)
    title_norm_sellers: defaultdict[str, set[str]] = defaultdict(set)
    segment_norm_sellers: defaultdict[str, set[str]] = defaultdict(set)
    sequence_by_seller: Counter[str] = Counter()
    title_norms_by_seller: defaultdict[str, set[str]] = defaultdict(set)
    segment_norms_by_seller: defaultdict[str, set[str]] = defaultdict(set)
    for item in ordered_items:
        seller_uid = str(item["seller_uid"])
        item_uid = str(item["item_uid"])
        sequence_by_seller[seller_uid] += 1
        sequence = int(sequence_by_seller[seller_uid])
        ordinary = {
            "category": step3.clean_text(item["category"]),
            "title": step3.clean_text(item["title"]),
            "description_snippet": step3.description_snippet(
                item["description"]
            ),
        }
        for source_kind, value in ordinary.items():
            if value:
                occurrences[(seller_uid, source_kind, value)].append(
                    (sequence, item_uid, -1)
                )
        title_norm = step3.normalize_signature_text(ordinary["title"])
        if title_norm:
            title_norms_by_seller[seller_uid].add(title_norm)
        for segment_ordinal, segment in enumerate(
            step3.extract_description_segments(item["description"]), start=1
        ):
            occurrences[(seller_uid, "description_segment", segment)].append(
                (sequence * 100 + segment_ordinal, item_uid, segment_ordinal)
            )
            segment_norm = step3.normalize_signature_text(segment)
            if segment_norm:
                segment_norms_by_seller[seller_uid].add(segment_norm)
    for seller_uid, norms in title_norms_by_seller.items():
        for norm in norms:
            title_norm_sellers[norm].add(seller_uid)
    for seller_uid, norms in segment_norms_by_seller.items():
        for norm in norms:
            segment_norm_sellers[norm].add(seller_uid)

    rows: list[dict[str, Any]] = []
    for seller_uid in common.utf8_sort(profile_by_seller):
        profile = profile_by_seller[seller_uid]
        for output_field, list_field, role, source_kind in PROFILE_ROLES:
            selected = profile.get(list_field)
            if not isinstance(selected, list):
                raise CandidateParentError(
                    f"Frozen Step3 output is not a list: {list_field}"
                )
            if step3.concat_top(selected) != profile.get(output_field):
                raise CandidateParentError(
                    f"Frozen Step3 concat field drift: {output_field}"
                )
            for output_rank, selected_row in enumerate(selected, start=1):
                if not isinstance(selected_row, dict):
                    raise CandidateParentError("Selected Step3 contribution is invalid")
                value = selected_row.get("value")
                if not isinstance(value, str) or not value:
                    raise CandidateParentError("Selected Step3 value is empty")
                support = occurrences.get((seller_uid, source_kind, value), [])
                if not support:
                    raise CandidateParentError(
                        "Selected Step3 value lacks an observed contributor"
                    )
                support_item_uids = common.utf8_sort(
                    {item_uid for _position, item_uid, _ordinal in support}
                )
                if selected_row.get("count") != len(support):
                    raise CandidateParentError(
                        "Selected Step3 count disagrees with contributor replay"
                    )
                first_position, first_item_uid, first_segment_ordinal = min(
                    support, key=lambda row: (row[0], row[1].encode("utf-8"))
                )
                is_signature = source_kind in {"title", "description_segment"} and (
                    output_field.startswith("signature_")
                )
                if is_signature:
                    norm = step3.normalize_signature_text(value)
                    df_sellers = (
                        title_norm_sellers[norm]
                        if source_kind == "title"
                        else segment_norm_sellers[norm]
                    )
                    seller_df = len(df_sellers)
                    if selected_row.get("seller_df") != seller_df:
                        raise CandidateParentError(
                            "Signature seller_df disagrees with contributor replay"
                        )
                else:
                    df_sellers = set()
                    seller_df = 0
                row = {
                    "world_uid": world_uid,
                    "seller_uid": seller_uid,
                    "output_field": output_field,
                    "aggregation_role": role,
                    "output_rank": output_rank,
                    "source_item_uids": support_item_uids,
                    "source_item_uids_sha256": _support_digest(
                        support_item_uids
                    ),
                    "source_item_count": len(support_item_uids),
                    "first_seen_position": first_position,
                    "item_uid": (
                        first_item_uid
                        if source_kind == "description_segment"
                        else ""
                    ),
                    "extracted_segment_ordinal": (
                        first_segment_ordinal
                        if source_kind == "description_segment"
                        else -1
                    ),
                    "seller_df": seller_df,
                    "seller_df_seller_count": len(df_sellers),
                    "seller_df_seller_uids_sha256": _support_digest(
                        common.utf8_sort(df_sellers)
                    ),
                }
                _require_exact_keys(row, PROVENANCE_FIELDS, label="provenance row")
                rows.append(row)
    rows.sort(
        key=lambda row: (
            row["seller_uid"].encode("utf-8"),
            row["output_field"].encode("utf-8"),
            int(row["output_rank"]),
        )
    )
    if not rows:
        raise CandidateParentError("Contribution provenance is empty")
    return {
        "version": "2026-08-09-step28-v13-v1-13-profile-provenance-v1",
        "world_uid": world_uid,
        "seller_count": len(profile_by_seller),
        "profile_count": len(profiles),
        "contribution_row_count": len(rows),
        "raw_contribution_values_persisted": False,
        "private_audit_only": True,
        "rows": rows,
        "rows_sha256": common.canonical_sha256(rows),
    }


def _effective_style_rows(
    *,
    policy: Mapping[str, Any],
    template: Mapping[str, Any],
    mode: str,
    world: Mapping[str, Any],
) -> list[dict[str, Any]]:
    private = world["private"]
    membership = private["controller_membership"]
    style_groups = private["controller_style_groups"]
    style_prototypes = template["style_prototypes"]
    if (
        len(membership) != 28
        or len({str(row["seller_uid"]) for row in membership}) != len(membership)
        or len(style_groups) != 12
        or len({str(row["controller_uid"]) for row in style_groups})
        != len(style_groups)
        or len({str(row["style_id"]) for row in style_prototypes})
        != len(style_prototypes)
    ):
        raise CandidateParentError("Effective style lineage contains duplicate keys")
    controller_style_id = {
        str(row["controller_uid"]): str(row["style_id"])
        for row in style_groups
    }
    seller_controller = {
        str(row["seller_uid"]): str(row["controller_uid"])
        for row in membership
    }
    styles = {
        str(row["style_id"]): dict(row) for row in style_prototypes
    }
    if set(seller_controller.values()) != set(controller_style_id):
        raise CandidateParentError("Controller membership/style-group keysets drift")
    factors = [str(value) for value in template["renderer_contract"]["style_factor_order"]]
    if len(factors) != 6 or len(set(factors)) != 6:
        raise CandidateParentError("Frozen six-factor style contract drift")
    rows: list[dict[str, Any]] = []
    for seller_uid in common.utf8_sort(seller_controller):
        controller_uid = seller_controller[seller_uid]
        try:
            controller_style = styles[controller_style_id[controller_uid]]
        except KeyError as exc:
            raise CandidateParentError("Controller style lineage is incomplete") from exc
        effective = nonidentity.seller_effective_style(
            policy=dict(policy),
            template=dict(template),
            mode=mode,
            seller_uid=seller_uid,
            controller_style=controller_style,
        )
        rows.append(
            {
                "seller_uid": seller_uid,
                "controller_uid": controller_uid,
                "effective_style_uid": effective["effective_style_uid"],
                "style_factors": {name: effective[name] for name in factors},
            }
        )
    render_style_uids: defaultdict[str, set[str]] = defaultdict(set)
    for row in private["render_asts"]:
        render_style_uids[str(row["seller_uid"])].add(
            str(row["effective_style_uid"])
        )
    if (
        len(rows) != 28
        or set(render_style_uids) != set(seller_controller)
        or any(len(values) != 1 for values in render_style_uids.values())
        or any(
            next(iter(render_style_uids[row["seller_uid"]]))
            != row["effective_style_uid"]
            for row in rows
        )
    ):
        raise CandidateParentError("Effective style replay disagrees with render AST")
    return rows


def candidate_invariant_projection(
    *,
    policy: Mapping[str, Any],
    template: Mapping[str, Any],
    mode: str,
    split: str,
    world: Mapping[str, Any],
    profile_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Project only quantities a future natural-text candidate must preserve."""

    if mode != ALLOWED_MODE or split not in SPLITS:
        raise CandidateParentError("Candidate invariant projection mode/split drift")
    public = world.get("public")
    private = world.get("private")
    if not isinstance(public, dict) or not isinstance(private, dict):
        raise CandidateParentError("Candidate parent world boundary is malformed")
    render_asts = _sorted_rows(private["render_asts"], "item_uid")
    stage_policy = load_policy()
    full_ast_fields = tuple(
        stage_policy["candidate_independent_parent"]["full_render_ast_schema"]
    )
    invariant_ast_fields = tuple(
        stage_policy["candidate_independent_parent"][
            "candidate_invariant_render_ast_projection"
        ]
    )
    if any(set(row) != set(full_ast_fields) for row in render_asts):
        raise CandidateParentError("Render AST schema drift in parent projection")
    invariant_render_asts = [
        {field: row[field] for field in invariant_ast_fields} for row in render_asts
    ]
    item_structure = [
        {
            "world_uid": str(row["world_uid"]),
            "seller_uid": str(row["seller_uid"]),
            "item_uid": str(row["item_uid"]),
            "time_bucket": row["time_bucket"],
        }
        for row in _sorted_rows(public["items"], "item_uid")
    ]
    identity_assets = []
    for source_row in _sorted_rows(private["identity_assets"], "identity_asset_uid"):
        row = {
            key: value
            for key, value in source_row.items()
            if key not in {"identity_uid", "identity_value"}
        }
        identity_assets.append(row)
    identity_slots = []
    for source_row in _sorted_rows(private["identity_slots_audit"], "slot_uid"):
        identity_slots.append(
            {
                key: value
                for key, value in source_row.items()
                if key
                not in {
                    "start",
                    "end",
                    "identity_uid",
                    "raw_surface",
                    "downstream_canonical_value",
                }
            }
        )
    identity_slot_edits = []
    for source_row in _sorted_rows(private["identity_slots_edit"], "slot_uid"):
        identity_slot_edits.append(
            {
                key: value
                for key, value in source_row.items()
                if key
                not in {
                    "start",
                    "end",
                    "raw_surface",
                    "downstream_canonical_value",
                }
            }
        )
    noise_slots = []
    for source_row in _sorted_rows(private["noise_slots_audit"], "noise_slot_uid"):
        noise_slots.append(
            {
                key: value
                for key, value in source_row.items()
                if key not in {"start", "end", "raw_surface"}
            }
        )
    return {
        "version": "2026-08-09-step28-v13-v1-13-parent-projection-v1",
        "mode": mode,
        "split": split,
        "world": dict(public["world"]),
        "sellers": _sorted_rows(public["sellers"], "seller_uid"),
        "item_structure": item_structure,
        "complete_model_pair_endpoints": _sorted_rows(
            public["complete_model_pair_endpoints"], "canonical_pair_uid"
        ),
        "controller_membership": _sorted_rows(
            private["controller_membership"], "controller_uid", "seller_uid"
        ),
        "controller_style_groups": _sorted_rows(
            private["controller_style_groups"], "controller_uid"
        ),
        "effective_styles": _effective_style_rows(
            policy=policy,
            template=template,
            mode=mode,
            world=world,
        ),
        "mechanism_assignments": _sorted_rows(
            private["mechanism_assignments"], "controller_uid"
        ),
        "render_asts": invariant_render_asts,
        "identity_asset_structure": identity_assets,
        "identity_slot_structure": identity_slots,
        "identity_slot_edit_structure": identity_slot_edits,
        "noise_slot_targets": noise_slots,
        "positive_targets": _sorted_rows(
            private["positive_targets"], "controller_uid", "mechanism_slot_uid"
        ),
        "negative_flags": _sorted_rows(
            private["negative_flags"], "flag", "canonical_pair_uid"
        ),
        "registered_override_lineage": _sorted_rows(
            private["override_audit"], "asset_index", "override_kind"
        ),
        "solver_audit": dict(private["solver_audit"]),
        "profile_contribution_lineage_sha256": str(
            profile_provenance["rows_sha256"]
        ),
    }


def _build_profiles_and_provenance(
    *,
    base_policy: Mapping[str, Any],
    mode: str,
    split: str,
    world: Mapping[str, Any],
    template: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    processed = production.process_world(
        base_policy,
        mode=mode,
        split=split,
        template=template,
        world=world,
    )
    profiles, profile_audit = profiles_module.build_world_profiles(
        base_policy,
        mode=mode,
        split=split,
        sellers=world["public"]["sellers"],
        items=processed["public"]["profile_safe_items"],
    )
    provenance = build_profile_contribution_provenance(
        world_uid=str(world["public"]["world"]["world_uid"]),
        profiles=profiles,
        profile_safe_items=processed["public"]["profile_safe_items"],
    )
    return profiles, provenance, {"processed": processed, "audit": profile_audit}


def _load_validated_base_inputs() -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]
]:
    """Load each base input only through its pinned release reference."""

    base_policy = common.load_policy(
        common.repo_path("schema/step28_v13_synthetic_chinese_dataset_policy.json"),
        mode=ALLOWED_MODE,
    )
    template, fixture = common.validate_policy_release_documents(
        base_policy, mode=ALLOWED_MODE
    )
    style_spec = base_policy["style_reference_boundary"][
        "generator_release_inputs"
    ]["profile"]
    style_path = common.verify_file_pin(
        style_spec, label="candidate-parent style profile"
    )
    style_profile = common.load_json(style_path)
    common.validate_independent_replay_public_domains(
        base_policy,
        template=template,
        style_profile=style_profile,
    )
    return base_policy, template, fixture, style_profile


def build_candidate_independent_parent() -> CandidateIndependentParent:
    """Replay the sole pinned smoke parent without caller-selected inputs."""

    stage_policy = load_policy()
    mode = ALLOWED_MODE
    base_policy, template, fixture, style_profile = _load_validated_base_inputs()
    smoke = stage_policy["development_smoke_parent"]
    pool = structure.build_mode_world_pool(base_policy, mode=mode)
    world_record = pool[int(smoke["world_pool_index"])]
    split = str(world_record["split"])
    structure_key_hex = common.structure_key_for_split(
        base_policy, mode=mode, split=split
    )
    if (
        world_record["world_uid"] != smoke["world_uid"]
        or split != smoke["split"]
        or common.canonical_sha256(world_record)
        != smoke["world_record_canonical_sha256"]
        or hashlib.sha256(bytes.fromhex(structure_key_hex)).hexdigest()
        != smoke["structure_key_sha256"]
    ):
        raise CandidateParentError("Pinned development-smoke parent replay drift")
    world = world_builder.build_world(
        policy=_canonical_clone(base_policy),
        template=_canonical_clone(template),
        fixture=_canonical_clone(fixture),
        style_profile=_canonical_clone(style_profile),
        mode=mode,
        world_record=_canonical_clone(world_record),
        structure_key_hex=structure_key_hex,
    )
    profiles, provenance, profile_context = _build_profiles_and_provenance(
        base_policy=base_policy,
        mode=mode,
        split=split,
        world=world,
        template=template,
    )
    if (
        profile_context["audit"].get("labels_or_private_structure_read") is not False
        or int(profile_context["audit"].get("seller_count", -1)) != 28
    ):
        raise CandidateParentError("Frozen Step3 profile audit did not close")
    invariant = candidate_invariant_projection(
        policy=base_policy,
        template=template,
        mode=mode,
        split=split,
        world=world,
        profile_provenance=provenance,
    )
    bootstrap_bytes = _canonical_bytes(world)
    invariant_bytes = _canonical_bytes(invariant)
    profile_bytes = _canonical_bytes(profiles)
    provenance_bytes = _canonical_bytes(provenance)
    return CandidateIndependentParent(
        mode=mode,
        split=split,
        world_uid=str(world["public"]["world"]["world_uid"]),
        bootstrap_world_bytes=bootstrap_bytes,
        invariant_projection_bytes=invariant_bytes,
        invariant_sha256=_sha256_bytes(invariant_bytes),
        profile_bytes=profile_bytes,
        profile_sha256=_sha256_bytes(profile_bytes),
        profile_provenance_bytes=provenance_bytes,
        profile_provenance_sha256=_sha256_bytes(provenance_bytes),
    )


def _history_item_index(world: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {
            "world_uid": str(row["world_uid"]),
            "seller_uid": str(row["seller_uid"]),
            "item_uid": str(row["item_uid"]),
            "time_bucket": int(row["time_bucket"]),
        }
        for row in world["public"]["items"]
    ]
    return _sorted_rows(rows, "world_uid", "seller_uid", "item_uid")


def _identity_parent_projection(
    *,
    world: Mapping[str, Any],
    identity33: Sequence[Mapping[str, Any]],
    allocation_delta: Sequence[str],
) -> dict[str, Any]:
    private = world["private"]
    slots = []
    for source_row in _sorted_rows(private["identity_slots_audit"], "slot_uid"):
        slots.append(
            {
                key: value
                for key, value in source_row.items()
                if key not in {"start", "end"}
            }
        )
    slot_edits = []
    for source_row in _sorted_rows(private["identity_slots_edit"], "slot_uid"):
        slot_edits.append(
            {
                key: value
                for key, value in source_row.items()
                if key not in {"start", "end"}
            }
        )
    return {
        "version": "2026-08-09-step28-v13-v1-13-identity-parent-v1",
        "world_uid": str(world["public"]["world"]["world_uid"]),
        "identity_assets": _sorted_rows(
            private["identity_assets"], "identity_asset_uid"
        ),
        "identity_slots": slots,
        "identity_slot_edits": slot_edits,
        "identity33": [dict(row) for row in identity33],
        "allocation_delta": list(allocation_delta),
    }


def _replay_identity_allocation_audit(
    *,
    parent_world: Mapping[str, Any],
    allocated_world: Mapping[str, Any],
    template: Mapping[str, Any],
    key_hex: str,
    historical_forbidden: frozenset[str],
    maximum_counter: int,
) -> list[dict[str, Any]]:
    """Independently replay first-admissible counters without mutating state."""

    world_uid = str(parent_world["public"]["world"]["world_uid"])
    guards = renderer.context_guard_pool(template)
    visible_texts: list[str] = []
    visible_compacts: list[str] = []
    for item in parent_world["public"]["items"]:
        description = str(item["description"])
        boundaries = [
            position
            for guard in guards
            if (position := description.find(guard)) >= 0
        ]
        boundary = min(boundaries) if boundaries else None
        for text in (
            str(item["title"]),
            description if boundary is None else description[:boundary],
        ):
            normalized = source.normalize_redacted_text(text).casefold()
            visible_texts.append(normalized)
            visible_compacts.append(source.compact_identifier(normalized))
    original_assets = {
        str(row["identity_asset_uid"]): row
        for row in parent_world["private"]["identity_assets"]
    }
    allocated_assets = {
        str(row["identity_asset_uid"]): row
        for row in allocated_world["private"]["identity_assets"]
    }
    if (
        len(original_assets) != len(parent_world["private"]["identity_assets"])
        or len(allocated_assets) != len(allocated_world["private"]["identity_assets"])
        or set(original_assets) != set(allocated_assets)
    ):
        raise CandidateParentError("Identity allocation audit asset keyset drift")
    allocated_hashes: set[str] = set()
    rows: list[dict[str, Any]] = []
    for asset_uid in common.utf8_sort(original_assets):
        original = original_assets[asset_uid]
        allocated = allocated_assets[asset_uid]
        identity_type = str(original["identity_type"])
        selected_hash = identity_values.value_hash(str(allocated["identity_value"]))
        visible_rejections = 0
        selected_counter: int | None = None
        for counter in range(maximum_counter + 1):
            candidate_value = identity_remap._candidate_identity_value(
                key_hex=key_hex,
                world_uid=world_uid,
                identity_asset_uid=asset_uid,
                identity_type=identity_type,
                counter=counter,
            )
            candidate_hash = identity_values.value_hash(candidate_value)
            if identity_remap._identity_value_collides_with_visible_text(
                candidate_value,
                visible_texts=visible_texts,
                visible_compacts=visible_compacts,
            ):
                visible_rejections += 1
                continue
            if candidate_hash in historical_forbidden or candidate_hash in allocated_hashes:
                continue
            if candidate_hash != selected_hash:
                raise CandidateParentError(
                    "Allocated identity is not the first admissible candidate"
                )
            allocated_hashes.add(candidate_hash)
            selected_counter = counter
            break
        if selected_counter is None:
            raise CandidateParentError("Allocated identity audit exhausted its domain")
        rows.append(
            {
                "identity_asset_uid": asset_uid,
                "selected_counter": selected_counter,
                "visible_text_candidate_rejection_count": visible_rejections,
                "selected_value_hash": selected_hash,
            }
        )
    return rows


def _validate_allocation_receipt(
    receipt: Mapping[str, Any],
    *,
    world_uid: str,
    identity_asset_count: int,
    identity_slot_count: int,
    changed_item_count: int,
    maximum_counter: int,
    allocation_delta: Sequence[str],
    allocated_asset_hashes: Mapping[str, str],
    expected_allocation_audit_rows: Sequence[Mapping[str, Any]],
) -> None:
    _require_exact_keys(
        receipt, ALLOCATION_RECEIPT_FIELDS, label="trial identity receipt"
    )
    integer_fields = ALLOCATION_RECEIPT_FIELDS - {
        "world_uid",
        "selected_value_hashes_sha256",
        "allocation_audit_rows",
        "allocation_audit_rows_sha256",
    }
    if any(
        isinstance(receipt[name], bool) or not isinstance(receipt[name], int)
        for name in integer_fields
    ) or any(receipt[name] < 0 for name in integer_fields):
        raise CandidateParentError("Trial identity receipt integer domain drift")
    audit_rows = receipt["allocation_audit_rows"]
    if not isinstance(audit_rows, list) or any(
        not isinstance(row, dict)
        or set(row)
        != {
            "identity_asset_uid",
            "selected_counter",
            "visible_text_candidate_rejection_count",
            "selected_value_hash",
        }
        for row in audit_rows
    ):
        raise CandidateParentError("Trial identity allocation audit schema drift")
    if (
        audit_rows != list(expected_allocation_audit_rows)
        or
        receipt["allocation_audit_rows_sha256"]
        != common.canonical_sha256(audit_rows)
        or [row["identity_asset_uid"] for row in audit_rows]
        != common.utf8_sort(allocated_asset_hashes)
        or any(
            not isinstance(row["identity_asset_uid"], str)
            or isinstance(row["selected_counter"], bool)
            or not isinstance(row["selected_counter"], int)
            or not 0 <= row["selected_counter"] <= maximum_counter
            or isinstance(row["visible_text_candidate_rejection_count"], bool)
            or not isinstance(row["visible_text_candidate_rejection_count"], int)
            or not 0 <= row["visible_text_candidate_rejection_count"]
            <= maximum_counter + 1
            or not isinstance(row["selected_value_hash"], str)
            or HEX_SHA256_RE.fullmatch(row["selected_value_hash"]) is None
            or allocated_asset_hashes.get(row["identity_asset_uid"])
            != row["selected_value_hash"]
            for row in audit_rows
        )
    ):
        raise CandidateParentError("Trial identity per-asset audit did not close")
    selected_counters = [row["selected_counter"] for row in audit_rows]
    visible_rejections = sum(
        row["visible_text_candidate_rejection_count"] for row in audit_rows
    )
    if (
        receipt["world_uid"] != world_uid
        or receipt["identity_asset_count"] != identity_asset_count
        or len(audit_rows) != identity_asset_count
        or receipt["identity_slot_count"] != identity_slot_count
        or receipt["maximum_counter"] != maximum_counter
        or receipt["maximum_selected_counter"] != max(selected_counters)
        or receipt["nonzero_counter_count"]
        != sum(value > 0 for value in selected_counters)
        or receipt["changed_item_count"] != changed_item_count
        or receipt["forced_design_collision_count"] != 0
        or receipt["visible_text_candidate_rejection_count"] != visible_rejections
        or receipt["historical_intersection_count"] != 0
        or receipt["same_run_intersection_count"] != 0
        or receipt["selected_value_hashes_sha256"]
        != common.canonical_sha256(list(allocation_delta))
    ):
        raise CandidateParentError("Trial identity receipt did not independently close")


class OneTimeTrialIdentityAllocator:
    """One-use guard around identity allocation on a private set copy."""

    def __init__(self) -> None:
        self._entered = False

    def allocate(
        self,
        *,
        parent: CandidateIndependentParent,
    ) -> FrozenTrialIdentityParent:
        """Allocate once; entry is consumed even if validation later fails."""

        if self._entered:
            raise CandidateParentError(
                "Trial identity allocator may be entered only once per world"
            )
        self._entered = True
        stage_policy = load_policy()
        expected_parent = build_candidate_independent_parent()
        if parent != expected_parent:
            raise CandidateParentError("Trial allocator received an unpinned parent")
        committed_identity_hashes: frozenset[str] = frozenset()
        registries = collision.load_historical_exclusion_registries()
        historical_forbidden_hashes = registries.identity_value_hashes
        trial_contract = stage_policy["trial_identity_allocation"]
        if (
            len(historical_forbidden_hashes)
            != trial_contract["historical_identity_hash_count"]
            or common.canonical_sha256(common.utf8_sort(historical_forbidden_hashes))
            != trial_contract["historical_identity_hashes_sha256"]
        ):
            raise CandidateParentError("Authoritative historical identity registry drift")
        if historical_forbidden_hashes & committed_identity_hashes:
            raise CandidateParentError("Committed identities intersect history")
        base_policy, template, _fixture, _style_profile = _load_validated_base_inputs()
        identity_remap_key_hex = str(
            base_policy["randomness"][ALLOWED_MODE]["identity_value_key_hex"]
        )
        if (
            hashlib.sha256(bytes.fromhex(identity_remap_key_hex)).hexdigest()
            != trial_contract["identity_key_sha256"]
        ):
            raise CandidateParentError("Pinned development-smoke identity key drift")
        maximum_counter = int(trial_contract["maximum_counter"])
        trial_allocated = set(committed_identity_hashes)
        world, receipt = identity_remap.remap_world_identity_values(
            parent.thaw_bootstrap_world(),
            template=template,
            key_hex=identity_remap_key_hex,
            historical_forbidden=historical_forbidden_hashes,
            allocated_in_trial=trial_allocated,
            maximum_counter=maximum_counter,
        )
        delta = tuple(common.utf8_sort(trial_allocated - committed_identity_hashes))
        asset_hashes = tuple(
            common.utf8_sort(
                identity_values.value_hash(str(row["identity_value"]))
                for row in world["private"]["identity_assets"]
            )
        )
        if (
            not delta
            or delta != asset_hashes
            or len(delta) != len(set(delta))
            or set(delta) & historical_forbidden_hashes
            or set(delta) & committed_identity_hashes
        ):
            raise CandidateParentError("Trial identity allocation delta did not close")
        expected_allocation_audit_rows = _replay_identity_allocation_audit(
            parent_world=parent.thaw_bootstrap_world(),
            allocated_world=world,
            template=template,
            key_hex=identity_remap_key_hex,
            historical_forbidden=historical_forbidden_hashes,
            maximum_counter=maximum_counter,
        )
        _validate_allocation_receipt(
            receipt,
            world_uid=parent.world_uid,
            identity_asset_count=len(world["private"]["identity_assets"]),
            identity_slot_count=len(world["private"]["identity_slots_audit"]),
            changed_item_count=len(
                {
                    str(row["item_uid"])
                    for row in world["private"]["identity_slots_audit"]
                }
            ),
            maximum_counter=maximum_counter,
            allocation_delta=delta,
            allocated_asset_hashes={
                str(row["identity_asset_uid"]): identity_values.value_hash(
                    str(row["identity_value"])
                )
                for row in world["private"]["identity_assets"]
            },
            expected_allocation_audit_rows=expected_allocation_audit_rows,
        )

        profiles, provenance, context = _build_profiles_and_provenance(
            base_policy=base_policy,
            mode=parent.mode,
            split=parent.split,
            world=world,
            template=template,
        )
        profile_bytes = _canonical_bytes(profiles)
        provenance_bytes = _canonical_bytes(provenance)
        if (
            profile_bytes != parent.profile_bytes
            or provenance_bytes != parent.profile_provenance_bytes
        ):
            raise CandidateParentError(
                "Identity allocation changed visible Step3 profile lineage"
            )
        invariant = candidate_invariant_projection(
            policy=base_policy,
            template=template,
            mode=parent.mode,
            split=parent.split,
            world=world,
            profile_provenance=provenance,
        )
        invariant_bytes = _canonical_bytes(invariant)
        if invariant_bytes != parent.invariant_projection_bytes:
            raise CandidateParentError(
                "Identity allocation changed candidate-independent structure"
            )

        processed = context["processed"]
        item_index = _history_item_index(world)
        parsed = processed["private"]["parsed_identity_occurrences"]
        history_rows = processed["public"]["history_safe_occurrences"]
        attestation = production.build_history_projection_attestation(
            base_policy,
            mode=parent.mode,
            split=parent.split,
            world_uid=parent.world_uid,
            sellers=world["public"]["sellers"],
            items=world["public"]["items"],
            history_safe_occurrences=history_rows,
            history_item_index=item_index,
            parsed_rows=parsed,
            identity_slots_audit=world["private"]["identity_slots_audit"],
            noise_slots_audit=world["private"]["noise_slots_audit"],
            render_asts=world["private"]["render_asts"],
        )
        pair_schema = [
            str(value)
            for value in base_policy["relational_integrity"][
                "pair_projection_contract"
            ]["complete_model_pair_endpoints_schema"]
        ]
        ordered_pair_endpoints = [
            {name: row[name] for name in pair_schema}
            for row in world["public"]["complete_model_pair_endpoints"]
        ]
        identity33, identity33_audit = history_features.build_identity33_all_pairs(
            base_policy,
            mode=parent.mode,
            split=parent.split,
            history_safe_occurrences=history_rows,
            history_item_index=item_index,
            projection_attestations=[attestation],
            complete_model_pair_endpoints=ordered_pair_endpoints,
        )
        identity33_bytes = _canonical_bytes(identity33)
        if (
            len(identity33) != 378
            or identity33_audit.get("feature_count") != 33
            or identity33_audit.get("identity33_sha256")
            != common.canonical_sha256(identity33)
        ):
            raise CandidateParentError("Trial identity33 parent did not close")
        identity_parent = _identity_parent_projection(
            world=world,
            identity33=identity33,
            allocation_delta=delta,
        )
        identity_parent_bytes = _canonical_bytes(identity_parent)
        return FrozenTrialIdentityParent(
            mode=parent.mode,
            split=parent.split,
            world_uid=parent.world_uid,
            world_bytes=_canonical_bytes(world),
            identity_parent_projection_bytes=identity_parent_bytes,
            identity_parent_sha256=_sha256_bytes(identity_parent_bytes),
            identity33_bytes=identity33_bytes,
            identity33_sha256=_sha256_bytes(identity33_bytes),
            allocation_receipt_bytes=_canonical_bytes(receipt),
            allocation_delta=delta,
            candidate_invariant_sha256=parent.invariant_sha256,
            profile_provenance_sha256=parent.profile_provenance_sha256,
            profile_sha256=parent.profile_sha256,
        )


def main() -> None:
    policy = load_policy()
    print(
        json.dumps(
            {
                "status": "PASS_DESIGN_ONLY_POLICY_AND_DEPENDENCY_PREFLIGHT",
                "version": policy["version"],
                "formal_authorizations": policy["formal_authorizations"],
                "formal_rows_generated": 0,
                "formal_seeds_generated": 0,
                "candidate_text_generated": 0,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
