#!/usr/bin/env python3
"""Restricted, label-blind natural-expression variation for Step28-v13 v1.13.

This module is deliberately limited to one development-smoke world.  The pure
variation function receives only an anonymous, canonical safe view and one
already-derived candidate key.  A separate trusted adapter owns the ephemeral
handle-to-UID binding, restores the frozen identity clauses, and replays the
production parser, redactor, Step3 profiles, contributor provenance, and
identity33 closure.  Nothing in this module writes files or derives formal
capabilities.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import step28_v13_common as common
import step28_v13_history_features as history_features
import step28_v13_text_renderer as renderer
import step28_v13_v1_13_candidate_parent as candidate_parent
import step28_v13_v1_13_document_collision as collision
import step28_v13_v1_13_pure_natural_renderer as pure_renderer
import step28_v13_world_builder as world_builder


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_PATH = (
    ROOT / "schema" / "step28_v13_v1_13_natural_variation_policy.json"
)
POLICY_VERSION = "2026-08-10-step28-v13-v1-13-natural-variation-policy-v1"
POLICY_STATUS = "DESIGN_ONLY_RESTRICTED_NATURAL_VARIATION_NO_FORMAL_AUTHORIZATION"
CLAIM_BOUNDARY = (
    "Restricted anonymous candidate view, independently keyed natural-expression "
    "variation, trusted identity reassembly, and full in-memory development-smoke "
    "closure only. No formal seed, capability, row, dataset, model, metric, "
    "collision retry, transaction, or release may be created."
)
FORMAL_AUTHORIZATION_KEYS = frozenset(
    {
        "audit_truth_access",
        "formal_capability_derivation",
        "formal_candidate_generation",
        "formal_dataset_generation",
        "formal_model_training",
        "formal_seed_ceremony",
    }
)
VIEW_VERSION = pure_renderer.VIEW_VERSION
OUTPUT_VERSION = pure_renderer.OUTPUT_VERSION
BINDING_VERSION = "2026-08-10-step28-v13-v1-13-private-handle-binding-v1"
ALLOWED_MODE = "development_smoke"
HANDLE_DOMAIN = b"step28-v13-v1.13-development-smoke-anonymous-handle"
ITEM_VIEW_FIELDS = pure_renderer.ITEM_VIEW_FIELDS
STYLE_FIELDS = pure_renderer.STYLE_FIELDS
CANDIDATE_ITEM_FIELDS = pure_renderer.CANDIDATE_ITEM_FIELDS
SAFE_VIEW_FIELDS = pure_renderer.SAFE_VIEW_FIELDS
SAFE_LIBRARY_FIELDS = pure_renderer.SAFE_LIBRARY_FIELDS
RestrictedCandidateView = pure_renderer.RestrictedCandidateView
NaturalExpressionCandidate = pure_renderer.NaturalExpressionCandidate
NaturalVariationError = pure_renderer.PureNaturalVariationError
render_candidate_natural_expressions = (
    pure_renderer.render_candidate_natural_expressions
)


FROZEN_INPUT_KEYS = frozenset(
    {
        "collision_contract",
        "collision_policy",
        "candidate_parent_policy",
        "base_dataset_policy",
        "collision_primitives",
        "candidate_parent",
        "natural_variation",
        "natural_variation_tests",
        "pure_natural_renderer",
        "common_contract",
        "identity_history",
        "production_chain",
        "text_renderer",
        "world_builder",
    }
)


def _canonical_bytes(value: Any) -> bytes:
    return common.canonical_json_bytes(value)


def _canonical_clone(value: Any) -> Any:
    return json.loads(_canonical_bytes(value).decode("utf-8"))


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _decode_canonical(payload: bytes, *, label: str) -> Any:
    if not isinstance(payload, bytes):
        raise NaturalVariationError(f"{label} must be canonical bytes")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NaturalVariationError(f"{label} is not canonical UTF-8 JSON") from exc
    if _canonical_bytes(value) != payload:
        raise NaturalVariationError(f"{label} is not in canonical byte form")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], fields: Sequence[str], *, label: str
) -> None:
    if set(value) != set(fields):
        raise NaturalVariationError(f"{label} keyset drift")


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise NaturalVariationError(f"{label} is not a lowercase SHA-256")
    return value


def _verify_self_hash(document: Mapping[str, Any], *, label: str) -> None:
    expected = _require_sha256(document.get("canonical_self_hash"), label=label)
    unsigned = {key: value for key, value in document.items() if key != "canonical_self_hash"}
    if common.canonical_sha256(unsigned) != expected:
        raise NaturalVariationError(f"{label} canonical self-hash drift")


def _verify_pin(spec: Mapping[str, Any], *, label: str) -> Path:
    _require_exact_keys(spec, ("path", "sha256", "size_bytes"), label=label)
    path = common.repo_path(str(spec["path"]))
    if not path.is_file():
        raise NaturalVariationError(f"{label} path is missing")
    if path.stat().st_size != int(spec["size_bytes"]):
        raise NaturalVariationError(f"{label} size drift")
    if common.sha256_file(path) != _require_sha256(spec["sha256"], label=label):
        raise NaturalVariationError(f"{label} hash drift")
    return path


def _validate_policy(policy: Mapping[str, Any]) -> None:
    required = {
        "version",
        "status",
        "claim_boundary",
        "formal_authorizations",
        "allowed_mode",
        "frozen_inputs",
        "development_smoke_keys",
        "restricted_candidate_view",
        "candidate_output",
        "expected_smoke_world",
        "canonical_self_hash",
    }
    if set(policy) != required:
        raise NaturalVariationError("Natural-variation policy keyset drift")
    _verify_self_hash(policy, label="natural-variation policy")
    if (
        policy["version"] != POLICY_VERSION
        or policy["status"] != POLICY_STATUS
        or policy["claim_boundary"] != CLAIM_BOUNDARY
        or policy["allowed_mode"] != ALLOWED_MODE
    ):
        raise NaturalVariationError("Natural-variation policy identity drift")
    authorizations = policy["formal_authorizations"]
    if (
        not isinstance(authorizations, dict)
        or set(authorizations) != FORMAL_AUTHORIZATION_KEYS
        or any(value is not False for value in authorizations.values())
    ):
        raise NaturalVariationError("Formal authorization must remain entirely false")
    if set(policy["frozen_inputs"]) != FROZEN_INPUT_KEYS:
        raise NaturalVariationError("Natural-variation frozen-input closure drift")
    for key, spec in policy["frozen_inputs"].items():
        if not isinstance(spec, dict):
            raise NaturalVariationError(f"Frozen input is malformed: {key}")
        _verify_pin(spec, label=f"natural-variation frozen input {key}")
    keys = policy["development_smoke_keys"]
    _require_exact_keys(
        keys,
        (
            "document_variation_key_hex",
            "document_variation_key_sha256",
            "anonymous_handle_key_hex",
            "anonymous_handle_key_sha256",
            "keys_are_public_test_vectors",
            "formal_reuse_forbidden",
        ),
        label="development-smoke keys",
    )
    for name in ("document_variation", "anonymous_handle"):
        raw = keys[f"{name}_key_hex"]
        if (
            not isinstance(raw, str)
            or len(raw) != 64
            or any(character not in "0123456789abcdef" for character in raw)
        ):
            raise NaturalVariationError(f"{name} test key is malformed")
        if hashlib.sha256(bytes.fromhex(raw)).hexdigest() != keys[f"{name}_key_sha256"]:
            raise NaturalVariationError(f"{name} test-key commitment drift")
    if keys["keys_are_public_test_vectors"] is not True or keys["formal_reuse_forbidden"] is not True:
        raise NaturalVariationError("Development keys are not explicitly non-formal")
    view = policy["restricted_candidate_view"]
    _require_exact_keys(
        view,
        (
            "view_version",
            "item_fields",
            "style_fields",
            "safe_library_fields",
            "safe_library_sha256",
            "contains_raw_uids",
            "contains_labels",
            "contains_controllers",
            "contains_identity_values_or_features",
            "contains_market_or_query_fields",
            "contains_registered_override_mechanisms",
            "pure_renderer_loads_policy",
            "ephemeral_binding_passed_to_variation_function",
        ),
        label="restricted candidate-view policy",
    )
    if (
        view.get("view_version") != VIEW_VERSION
        or tuple(view.get("item_fields", [])) != ITEM_VIEW_FIELDS
        or tuple(view.get("style_fields", [])) != STYLE_FIELDS
        or tuple(view.get("safe_library_fields", [])) != SAFE_LIBRARY_FIELDS
        or view.get("contains_raw_uids") is not False
        or view.get("contains_labels") is not False
        or view.get("contains_controllers") is not False
        or view.get("contains_identity_values_or_features") is not False
        or view.get("contains_market_or_query_fields") is not False
        or view.get("contains_registered_override_mechanisms") is not False
        or view.get("pure_renderer_loads_policy") is not False
        or view.get("ephemeral_binding_passed_to_variation_function") is not False
    ):
        raise NaturalVariationError("Restricted candidate-view contract drift")
    _require_sha256(view.get("safe_library_sha256"), label="safe-library hash")
    output = policy["candidate_output"]
    _require_exact_keys(
        output,
        (
            "version",
            "item_fields",
            "identity_text_assembled_outside_variation_function",
            "natural_output_hash_covers_trusted_registered_overrides",
            "full_production_closure_required",
            "candidate_text_persisted",
        ),
        label="candidate output policy",
    )
    if (
        output.get("version") != OUTPUT_VERSION
        or tuple(output.get("item_fields", [])) != CANDIDATE_ITEM_FIELDS
        or output.get("identity_text_assembled_outside_variation_function") is not True
        or output.get("natural_output_hash_covers_trusted_registered_overrides")
        is not True
        or output.get("full_production_closure_required") is not True
        or output.get("candidate_text_persisted") is not False
    ):
        raise NaturalVariationError("Candidate output contract drift")
    smoke = policy["expected_smoke_world"]
    _require_exact_keys(
        smoke,
        (
            "world_uid",
            "split",
            "item_count",
            "seller_count",
            "pair_count",
            "identity_asset_count",
            "noise_target_count",
            "registered_override_count",
            "restricted_view_sha256",
            "candidate0_natural_output_sha256",
            "candidate0_world_sha256",
            "natural_output_hashes_sha256",
            "world_hashes_sha256",
            "candidate_invariant_sha256",
            "candidate_parent_full_state_sha256",
            "identity_parent_sha256",
            "frozen_trial_identity_full_state_sha256",
            "identity33_sha256",
            "profile_provenance_sha256",
            "unique_natural_output_count",
            "unique_world_count",
            "changed_item_count_vs_candidate0_min",
            "changed_item_count_vs_candidate0_max",
        ),
        label="expected smoke-world contract",
    )
    if (
        smoke.get("world_uid")
        != "w_003497845547650a980473b05e249937bf825ad0eaefa424baec74f2bd2210f3"
        or smoke.get("split") != "audit_a"
        or smoke.get("item_count") != 105
        or smoke.get("seller_count") != 28
        or smoke.get("pair_count") != 378
        or smoke.get("identity_asset_count") != 84
        or smoke.get("noise_target_count") != 28
        or smoke.get("registered_override_count") != 6
        or smoke.get("unique_natural_output_count") != 32
        or smoke.get("unique_world_count") != 32
        or smoke.get("changed_item_count_vs_candidate0_min") != 51
        or smoke.get("changed_item_count_vs_candidate0_max") != 105
    ):
        raise NaturalVariationError("Expected smoke-world cardinality drift")
    for field in (
        "restricted_view_sha256",
        "candidate0_natural_output_sha256",
        "candidate0_world_sha256",
        "natural_output_hashes_sha256",
        "world_hashes_sha256",
        "candidate_invariant_sha256",
        "candidate_parent_full_state_sha256",
        "identity_parent_sha256",
        "frozen_trial_identity_full_state_sha256",
        "identity33_sha256",
        "profile_provenance_sha256",
    ):
        _require_sha256(smoke.get(field), label=f"expected smoke-world {field}")


def load_policy(path: Path = DEFAULT_POLICY_PATH) -> dict[str, Any]:
    if path.resolve() != DEFAULT_POLICY_PATH.resolve():
        raise NaturalVariationError("Only the canonical natural-variation policy may pass")
    try:
        value = common.load_json(path)
    except common.ContractError as exc:
        raise NaturalVariationError("Natural-variation policy is invalid JSON") from exc
    _validate_policy(value)
    return value


@dataclass(frozen=True)
class _PrivateHandleBinding:
    """Trusted-only ephemeral binding; never passed to the variation function."""

    binding_bytes: bytes
    binding_sha256: str
    view_sha256: str

    def thaw(self) -> dict[str, Any]:
        value = _decode_canonical(self.binding_bytes, label="private handle binding")
        if not isinstance(value, dict):
            raise NaturalVariationError("Private handle binding is not an object")
        if _sha256_bytes(self.binding_bytes) != self.binding_sha256:
            raise NaturalVariationError("Private handle-binding hash drift")
        if value.get("view_sha256") != self.view_sha256:
            raise NaturalVariationError("Private handle binding targets another view")
        return value


@dataclass(frozen=True)
class AssembledDevelopmentCandidate:
    """In-memory, fully revalidated development candidate."""

    candidate_index: int
    world_bytes: bytes
    world_sha256: str
    profiles_bytes: bytes
    profiles_sha256: str
    profile_provenance_bytes: bytes
    profile_provenance_sha256: str
    identity33_bytes: bytes
    identity33_sha256: str
    natural_output_sha256: str
    candidate_invariant_sha256: str
    identity_parent_sha256: str

    def thaw_world(self) -> dict[str, Any]:
        value = _decode_canonical(self.world_bytes, label="assembled candidate world")
        if not isinstance(value, dict):
            raise NaturalVariationError("Assembled candidate world is not an object")
        if _sha256_bytes(self.world_bytes) != self.world_sha256:
            raise NaturalVariationError("Assembled candidate-world hash drift")
        return value


def _anonymous_handle(*, key: bytes, kind: str, value: str) -> str:
    if len(key) != 32 or not kind or not value:
        raise NaturalVariationError("Anonymous handle input is malformed")
    digest = hmac.new(
        key,
        common.FIELD_SEPARATOR.join((HANDLE_DOMAIN, kind.encode("ascii"), value.encode("utf-8"))),
        hashlib.sha256,
    ).hexdigest()
    return f"h_{kind}_{digest[:32]}"


def _safe_library(
    *,
    base_policy: Mapping[str, Any],
    template: Mapping[str, Any],
    fixture: Mapping[str, Any],
    split: str,
) -> dict[str, Any]:
    """Project the exact label-free textual whitelist needed by the pure renderer."""

    lexicon = template["generic_lexicon"]
    categories = [str(value) for value in lexicon["categories"]]
    library = template["split_libraries"][split]
    category_classes: defaultdict[tuple[int, int, tuple[int, ...]], list[str]] = defaultdict(list)
    for category in categories:
        products = lexicon["category_products"][category]
        category_classes[
            (len(category), len(products), tuple(len(value) for value in products))
        ].append(category)
    styles = renderer.reachable_effective_styles(template)
    title_classes: defaultdict[tuple[int, ...], list[int]] = defaultdict(list)
    for index, skeleton in enumerate(library["title_skeletons"]):
        signature = tuple(
            len(
                renderer.render_base_title(
                    skeleton=str(skeleton),
                    product="商品词词词词",
                    attribute="属性词词",
                    code="QABCDEFGHIJ",
                    style=style,
                    template=template,
                )
            )
            for style in styles
        )
        title_classes[signature].append(index)
    description_classes: defaultdict[tuple[tuple[int, ...], ...], list[int]] = defaultdict(list)
    for index, skeleton in enumerate(library["description_skeletons"]):
        signature = tuple(
            tuple(
                len(segment)
                for segment in candidate_parent.step3.extract_description_segments(
                    renderer.render_base_description(
                        skeleton=str(skeleton),
                        product="商品词词词词",
                        attribute="属性词词",
                        code="QABCDEFGHIJ",
                        delivery="交付词" * 10,
                        service="服务词" * 12,
                        style=style,
                        template=template,
                    )
                )
            )
            for style in styles
        )
        description_classes[signature].append(index)
    def length_classes(values: Sequence[str]) -> list[list[str]]:
        grouped: defaultdict[int, list[str]] = defaultdict(list)
        for value in values:
            grouped[len(value)].append(str(value))
        return [group for _length, group in sorted(grouped.items())]

    def surface_structure_classes(values: Sequence[str]) -> list[list[str]]:
        grouped: defaultdict[tuple[int, tuple[tuple[int, str], ...]], list[str]] = (
            defaultdict(list)
        )
        for value in values:
            text = str(value)
            signature = (
                len(text),
                tuple(
                    (index, character)
                    for index, character in enumerate(text)
                    if not character.isalnum()
                ),
            )
            grouped[signature].append(text)
        return [group for _signature, group in sorted(grouped.items())]

    noise_template_classes: defaultdict[int, list[int]] = defaultdict(list)
    for index, value in enumerate(template["identity_clause_templates"]["must_ignore"]):
        noise_template_classes[len(str(value).format(value=""))].append(index)
    output = {
        "categories": categories,
        "category_products": {
            category: [str(value) for value in lexicon["category_products"][category]]
            for category in categories
        },
        "attributes": [str(value) for value in lexicon["attributes"]],
        "delivery": [str(value) for value in lexicon["delivery"]],
        "service": [str(value) for value in lexicon["service"]],
        "title_modifiers": [str(value) for value in lexicon["title_modifiers"]],
        "title_skeletons": [str(value) for value in library["title_skeletons"]],
        "description_skeletons": [str(value) for value in library["description_skeletons"]],
        "traditional_substitutions": {
            str(key): str(value)
            for key, value in template["renderer_contract"]["traditional_substitutions"].items()
        },
        "must_ignore_templates": [
            str(value) for value in template["identity_clause_templates"]["must_ignore"]
        ],
        "must_ignore_values": [
            str(value) for value in fixture["must_ignore_adversarial_values"]
        ],
        "category_permutation_classes": [
            values for _signature, values in sorted(category_classes.items())
        ],
        "attribute_permutation_classes": [
            [str(value)] for value in lexicon["attributes"]
        ],
        "delivery_permutation_classes": surface_structure_classes(
            lexicon["delivery"]
        ),
        "service_permutation_classes": surface_structure_classes(
            lexicon["service"]
        ),
        "title_skeleton_permutation_classes": [
            values
            for _signature, values in sorted(
                title_classes.items(), key=lambda pair: pair[0]
            )
        ],
        "description_skeleton_permutation_classes": [
            values
            for _signature, values in sorted(
                description_classes.items(), key=lambda pair: pair[0]
            )
        ],
        "noise_template_permutation_classes": [
            values for _length, values in sorted(noise_template_classes.items())
        ],
        "noise_value_permutation_classes": length_classes(
            fixture["must_ignore_adversarial_values"]
        ),
    }
    _require_exact_keys(output, SAFE_LIBRARY_FIELDS, label="safe library")
    _validate_safe_library(output, base_policy=base_policy)
    return output


def _validate_safe_library(
    library: Mapping[str, Any], *, base_policy: Mapping[str, Any] | None = None
) -> None:
    pure_renderer.validate_safe_library(library)
    _require_exact_keys(library, SAFE_LIBRARY_FIELDS, label="safe library")
    categories = library["categories"]
    products = library["category_products"]
    if (
        not isinstance(categories, list)
        or not categories
        or len(categories) != len(set(categories))
        or not isinstance(products, dict)
        or set(products) != set(categories)
    ):
        raise NaturalVariationError("Safe category library is malformed")
    text_lists = (
        "attributes",
        "delivery",
        "service",
        "title_modifiers",
        "title_skeletons",
        "description_skeletons",
        "must_ignore_templates",
        "must_ignore_values",
    )
    for name in text_lists:
        values = library[name]
        if (
            not isinstance(values, list)
            or not values
            or len(values) != len(set(values))
            or any(
                not isinstance(value, str)
                or not value
                or value != unicodedata.normalize("NFC", value)
                for value in values
            )
        ):
            raise NaturalVariationError(f"Safe whitelist is malformed: {name}")
    for category in categories:
        values = products[category]
        if (
            not isinstance(values, list)
            or not values
            or len(values) != len(set(values))
            or any(not isinstance(value, str) or not value for value in values)
        ):
            raise NaturalVariationError("Safe category/product whitelist is malformed")
    substitutions = library["traditional_substitutions"]
    if (
        not isinstance(substitutions, dict)
        or not substitutions
        or any(len(key) != 1 or len(value) != 1 for key, value in substitutions.items())
    ):
        raise NaturalVariationError("Safe traditional-substitution table is malformed")
    if any(not value.endswith(renderer.DESCRIPTION_SUFFIX) for value in library["description_skeletons"]):
        raise NaturalVariationError("Safe description skeleton suffix drift")
    category_classes = library["category_permutation_classes"]
    if (
        not isinstance(category_classes, list)
        or not category_classes
        or len([value for group in category_classes for value in group])
        != len(categories)
        or set(value for group in category_classes for value in group)
        != set(categories)
        or any(
            not isinstance(group, list)
            or not group
            or len(
                {
                    (
                        len(category),
                        len(library["category_products"][category]),
                        tuple(
                            len(value)
                            for value in library["category_products"][category]
                        ),
                    )
                    for category in group
                }
            )
            != 1
            for group in category_classes
        )
    ):
        raise NaturalVariationError("Safe category permutation classes drift")
    for field, values_field in (
        ("attribute_permutation_classes", "attributes"),
        ("delivery_permutation_classes", "delivery"),
        ("service_permutation_classes", "service"),
        ("noise_value_permutation_classes", "must_ignore_values"),
    ):
        classes = library[field]
        flattened = [str(value) for group in classes for value in group]
        if (
            not isinstance(classes, list)
            or len(flattened) != len(library[values_field])
            or set(flattened) != set(library[values_field])
            or any(len({len(str(value)) for value in group}) != 1 for group in classes)
        ):
            raise NaturalVariationError(f"Safe length permutation classes drift: {field}")
        if field in {"delivery_permutation_classes", "service_permutation_classes"}:
            for group in classes:
                signatures = {
                    (
                        len(str(value)),
                        tuple(
                            (index, character)
                            for index, character in enumerate(str(value))
                            if not character.isalnum()
                        ),
                    )
                    for value in group
                }
                if len(signatures) != 1:
                    raise NaturalVariationError(
                        f"Safe punctuation permutation classes drift: {field}"
                    )
    title_classes = library["title_skeleton_permutation_classes"]
    title_indices = [int(value) for group in title_classes for value in group]
    if (
        not isinstance(title_classes, list)
        or sorted(title_indices) != list(range(len(library["title_skeletons"])))
        or len(title_indices) != len(set(title_indices))
    ):
        raise NaturalVariationError("Safe title-skeleton classes drift")
    description_classes = library["description_skeleton_permutation_classes"]
    description_indices = [
        int(value) for group in description_classes for value in group
    ]
    if (
        not isinstance(description_classes, list)
        or not description_classes
        or sorted(description_indices)
        != list(range(len(library["description_skeletons"])))
        or len(description_indices) != len(set(description_indices))
    ):
        raise NaturalVariationError("Safe description-skeleton classes drift")
    noise_classes = library["noise_template_permutation_classes"]
    noise_indices = [int(value) for group in noise_classes for value in group]
    if (
        not isinstance(noise_classes, list)
        or sorted(noise_indices) != list(range(len(library["must_ignore_templates"])))
        or len(noise_indices) != len(set(noise_indices))
    ):
        raise NaturalVariationError("Safe noise-template classes drift")
    if base_policy is not None:
        expected_template = common.load_json(
            common.verify_file_pin(
                base_policy["template_library"],
                label="natural-variation template library",
            )
        )
        if categories != list(expected_template["generic_lexicon"]["categories"]):
            raise NaturalVariationError("Safe category order disagrees with pinned template")


def _scan_for_forbidden_view_content(value: Any) -> None:
    forbidden_key_atoms = (
        "world",
        "seller",
        "item_uid",
        "pair",
        "query",
        "controller",
        "market",
        "label",
        "identity",
        "candidate_index",
        "retry",
        "override",
        "clone",
        "semantic",
    )
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).casefold()
            if any(atom in lowered for atom in forbidden_key_atoms):
                raise NaturalVariationError(f"Restricted view exposes forbidden key: {key}")
            _scan_for_forbidden_view_content(child)
    elif isinstance(value, list):
        for child in value:
            _scan_for_forbidden_view_content(child)


def _iter_string_values(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _iter_string_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_string_values(child)


def _iter_lower_hex_32_values(value: Any):
    if (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _iter_lower_hex_32_values(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _iter_lower_hex_32_values(child)


def _build_restricted_view_and_binding(
    *,
    frozen: candidate_parent.FrozenTrialIdentityParent,
    expected_parent: candidate_parent.CandidateIndependentParent,
) -> tuple[RestrictedCandidateView, _PrivateHandleBinding]:
    stage_policy = load_policy()
    base_policy, template, fixture, _style_profile = (
        candidate_parent._load_validated_base_inputs()
    )
    development_keys = stage_policy["development_smoke_keys"]
    public_test_authorities = {
        str(development_keys["document_variation_key_hex"]),
        str(development_keys["anonymous_handle_key_hex"]),
        str(development_keys["document_variation_key_sha256"]),
        str(development_keys["anonymous_handle_key_sha256"]),
    }
    base_randomness_values = set(
        _iter_lower_hex_32_values(base_policy["randomness"])
    )
    if (
        len(public_test_authorities) != 4
        or public_test_authorities & base_randomness_values
    ):
        raise NaturalVariationError(
            "Development-only variation keys are not isolated from base randomness"
        )
    if (
        frozen.mode != ALLOWED_MODE
        or frozen.world_uid != stage_policy["expected_smoke_world"]["world_uid"]
        or frozen.split != stage_policy["expected_smoke_world"]["split"]
        or frozen.candidate_invariant_sha256 != expected_parent.invariant_sha256
        or frozen.profile_provenance_sha256 != expected_parent.profile_provenance_sha256
    ):
        raise NaturalVariationError("Frozen identity parent is outside the sole smoke boundary")
    world = frozen.thaw_world()
    item_rows = sorted(
        (dict(row) for row in world["public"]["items"]),
        key=lambda row: str(row["item_uid"]).encode("utf-8"),
    )
    ast_by_item = {
        str(row["item_uid"]): dict(row) for row in world["private"]["render_asts"]
    }
    if len(ast_by_item) != len(world["private"]["render_asts"]):
        raise NaturalVariationError("Render AST contains duplicate item UIDs")
    handle_key = bytes.fromhex(
        stage_policy["development_smoke_keys"]["anonymous_handle_key_hex"]
    )
    item_uid_to_handle = {
        str(row["item_uid"]): _anonymous_handle(
            key=handle_key, kind="item", value=str(row["item_uid"])
        )
        for row in item_rows
    }
    if len(set(item_uid_to_handle.values())) != len(item_uid_to_handle):
        raise NaturalVariationError("Anonymous item-handle collision")
    effective_rows = candidate_parent._effective_style_rows(
        policy=base_policy,
        template=template,
        mode=ALLOWED_MODE,
        world=world,
    )
    style_by_seller = {
        str(row["seller_uid"]): {
            **dict(row["style_factors"]),
        }
        for row in effective_rows
    }
    safe_items: list[dict[str, Any]] = []
    for item in item_rows:
        item_uid = str(item["item_uid"])
        ast = ast_by_item.get(item_uid)
        if ast is None:
            raise NaturalVariationError("Public item lacks a render AST")
        style = style_by_seller.get(str(item["seller_uid"]))
        if style is None or set(style) != set(STYLE_FIELDS):
            raise NaturalVariationError("Item safe view lacks its fixed style")
        safe_row = {
            "item_handle": item_uid_to_handle[item_uid],
            "code": str(ast["code"]),
            "effective_style": style,
            "title_nonempty": bool(ast["title_nonempty"]),
            "description_nonempty": bool(ast["description_nonempty"]),
            "baseline_category": str(ast["category"]),
            "baseline_product": str(ast["product"]),
            "baseline_attribute": str(ast["attribute"]),
            "baseline_delivery": str(ast["delivery"]),
            "baseline_service": str(ast["service"]),
            "baseline_title_skeleton_index": int(ast["title_skeleton_index"]),
            "baseline_description_skeleton_index": int(
                ast["description_skeleton_index"]
            ),
        }
        _require_exact_keys(safe_row, ITEM_VIEW_FIELDS, label="safe item row")
        safe_items.append(safe_row)
    safe_items.sort(key=lambda row: row["item_handle"].encode("utf-8"))

    registered_overrides: list[dict[str, Any]] = []
    for row in sorted(
        world["private"]["override_audit"],
        key=lambda value: (
            str(value["override_kind"]).encode("utf-8"),
            int(value["asset_index"]),
        ),
    ):
        if row["override_kind"] not in {
            "high_semantic_similarity",
            "exact_title_clone",
        }:
            raise NaturalVariationError("Unknown registered override in safe projection")
        registered_overrides.append(
            {
                "override_kind": str(row["override_kind"]),
                "asset_index": int(row["asset_index"]),
                "canonical_pair_uid": str(row["canonical_pair_uid"]),
                "item_uid_left": str(row["item_uid_left"]),
                "item_uid_right": str(row["item_uid_right"]),
            }
        )

    safe_library = _safe_library(
        base_policy=base_policy,
        template=template,
        fixture=fixture,
        split=frozen.split,
    )
    safe_library_sha256 = common.canonical_sha256(safe_library)
    if safe_library_sha256 != stage_policy["restricted_candidate_view"]["safe_library_sha256"]:
        raise NaturalVariationError("Pinned safe-library projection drift")

    noise_targets: list[dict[str, str]] = []
    noise_handle_to_slot_uid: dict[str, str] = {}
    noise_audit_by_slot = {
        str(row["noise_slot_uid"]): dict(row)
        for row in world["private"]["noise_slots_audit"]
    }
    if len(noise_audit_by_slot) != len(world["private"]["noise_slots_audit"]):
        raise NaturalVariationError("Noise audit contains duplicate slot UIDs")
    for ast in sorted(ast_by_item.values(), key=lambda row: str(row["item_uid"]).encode("utf-8")):
        noise_slot_uid = str(ast["noise_slot_uid"])
        if not noise_slot_uid:
            continue
        noise_audit = noise_audit_by_slot.get(noise_slot_uid)
        if noise_audit is None:
            raise NaturalVariationError("Noise AST target lacks its fixed audit row")
        raw_surface = str(noise_audit["raw_surface"])
        matching_noise_choices = [
            (template_index, value_index)
            for template_index in range(
                len(safe_library["must_ignore_templates"])
            )
            for value_index in range(len(safe_library["must_ignore_values"]))
            if renderer.must_ignore_clause(
                template_index=template_index,
                value=str(safe_library["must_ignore_values"][value_index]),
                template=template,
            )
            == raw_surface
        ]
        if len(matching_noise_choices) != 1:
            raise NaturalVariationError("Fixed noise expression is not uniquely reversible")
        baseline_template_index, baseline_value_index = matching_noise_choices[0]
        noise_handle = _anonymous_handle(
            key=handle_key, kind="noise", value=noise_slot_uid
        )
        if noise_handle in noise_handle_to_slot_uid:
            raise NaturalVariationError("Anonymous noise-handle collision")
        noise_handle_to_slot_uid[noise_handle] = noise_slot_uid
        noise_targets.append(
            {
                "noise_handle": noise_handle,
                "item_handle": item_uid_to_handle[str(ast["item_uid"])],
                "baseline_template_index": baseline_template_index,
                "baseline_value_index": baseline_value_index,
            }
        )
    noise_targets.sort(key=lambda row: row["noise_handle"].encode("utf-8"))

    view_value = {
        "version": VIEW_VERSION,
        "item_count": len(safe_items),
        "items": safe_items,
        "noise_targets": noise_targets,
        "safe_library": safe_library,
        "safe_library_sha256": safe_library_sha256,
    }
    _require_exact_keys(view_value, SAFE_VIEW_FIELDS, label="restricted candidate view")
    _validate_restricted_view(view_value, policy=stage_policy)
    forbidden_private_literals = {
        str(world["public"]["world"]["world_uid"]),
        *(str(row["seller_uid"]) for row in world["public"]["sellers"]),
        *(str(row["item_uid"]) for row in world["public"]["items"]),
        *(
            str(row["canonical_pair_uid"])
            for row in world["public"]["complete_model_pair_endpoints"]
        ),
        *(
            str(row["controller_uid"])
            for row in world["private"]["controller_membership"]
        ),
        *(
            str(row[field])
            for row in world["private"]["identity_assets"]
            for field in ("identity_asset_uid", "identity_uid", "identity_value")
            if field in row
        ),
        *(
            str(row[field])
            for row in world["private"]["identity_slots_audit"]
            for field in (
                "slot_uid",
                "bundle_uid",
                "identity_uid",
                "raw_surface",
                "downstream_canonical_value",
            )
            if field in row
        ),
        *(
            str(row["noise_slot_uid"])
            for row in world["private"]["noise_slots_audit"]
        ),
    }
    safe_strings = tuple(_iter_string_values(view_value))
    if any(
        len(literal) >= 5 and literal in safe_value
        for literal in forbidden_private_literals
        for safe_value in safe_strings
    ):
        raise NaturalVariationError(
            "Restricted candidate view contains a private literal"
        )
    view_bytes = _canonical_bytes(view_value)
    view_sha256 = _sha256_bytes(view_bytes)
    view = RestrictedCandidateView(view_bytes=view_bytes, view_sha256=view_sha256)

    binding_value = {
        "version": BINDING_VERSION,
        "view_sha256": view_sha256,
        "candidate_invariant_sha256": expected_parent.invariant_sha256,
        "identity_parent_sha256": frozen.identity_parent_sha256,
        "item_handle_to_item_uid": {
            handle: uid
            for uid, handle in sorted(
                item_uid_to_handle.items(), key=lambda pair: pair[1].encode("utf-8")
            )
        },
        "noise_handle_to_noise_slot_uid": {
            key: noise_handle_to_slot_uid[key]
            for key in sorted(noise_handle_to_slot_uid, key=lambda value: value.encode("utf-8"))
        },
        "registered_overrides": registered_overrides,
    }
    binding_bytes = _canonical_bytes(binding_value)
    binding = _PrivateHandleBinding(
        binding_bytes=binding_bytes,
        binding_sha256=_sha256_bytes(binding_bytes),
        view_sha256=view_sha256,
    )
    smoke = stage_policy["expected_smoke_world"]
    if (
        len(safe_items) != smoke["item_count"]
        or len(registered_overrides) != smoke["registered_override_count"]
        or len(noise_targets) != smoke["noise_target_count"]
    ):
        raise NaturalVariationError("Restricted view cardinality drift")
    return view, binding


def _validate_restricted_view(value: Mapping[str, Any], *, policy: Mapping[str, Any]) -> None:
    pure_renderer.validate_restricted_view(value)
    _scan_for_forbidden_view_content(value)
    library = value["safe_library"]
    library_sha256 = common.canonical_sha256(library)
    if (
        value["safe_library_sha256"] != library_sha256
        or library_sha256
        != policy["restricted_candidate_view"]["safe_library_sha256"]
    ):
        raise NaturalVariationError("Restricted safe-library hash drift")
    items = value["items"]
    if (
        not isinstance(items, list)
        or len(items) != value["item_count"]
        or len(items) != policy["expected_smoke_world"]["item_count"]
    ):
        raise NaturalVariationError("Restricted item cardinality drift")


render_candidate_natural_expressions = (
    pure_renderer.render_candidate_natural_expressions
)


def _bytes_commitment(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, bytes):
        raise NaturalVariationError(f"{label} must be immutable bytes")
    return {
        "size_bytes": len(value),
        "sha256": _sha256_bytes(value),
    }


def _candidate_parent_full_state_sha256(
    parent: candidate_parent.CandidateIndependentParent,
) -> str:
    if not isinstance(parent, candidate_parent.CandidateIndependentParent):
        raise NaturalVariationError("Candidate parent type drift")
    state = {
        "mode": parent.mode,
        "split": parent.split,
        "world_uid": parent.world_uid,
        "bootstrap_world_bytes": _bytes_commitment(
            parent.bootstrap_world_bytes, label="candidate parent bootstrap world"
        ),
        "invariant_projection_bytes": _bytes_commitment(
            parent.invariant_projection_bytes,
            label="candidate parent invariant projection",
        ),
        "invariant_sha256": parent.invariant_sha256,
        "profile_bytes": _bytes_commitment(
            parent.profile_bytes, label="candidate parent profiles"
        ),
        "profile_sha256": parent.profile_sha256,
        "profile_provenance_bytes": _bytes_commitment(
            parent.profile_provenance_bytes,
            label="candidate parent profile provenance",
        ),
        "profile_provenance_sha256": parent.profile_provenance_sha256,
    }
    if (
        _sha256_bytes(parent.invariant_projection_bytes) != parent.invariant_sha256
        or _sha256_bytes(parent.profile_bytes) != parent.profile_sha256
        or _sha256_bytes(parent.profile_provenance_bytes)
        != parent.profile_provenance_sha256
    ):
        raise NaturalVariationError("Candidate parent internal commitment drift")
    return common.canonical_sha256(state)


def _frozen_identity_full_state_sha256(
    frozen: candidate_parent.FrozenTrialIdentityParent,
    *,
    parent: candidate_parent.CandidateIndependentParent,
) -> str:
    if not isinstance(frozen, candidate_parent.FrozenTrialIdentityParent):
        raise NaturalVariationError("Frozen trial identity parent type drift")
    if not isinstance(frozen.allocation_delta, tuple) or any(
        not isinstance(value, str) for value in frozen.allocation_delta
    ):
        raise NaturalVariationError("Frozen trial allocation delta type drift")
    state = {
        "mode": frozen.mode,
        "split": frozen.split,
        "world_uid": frozen.world_uid,
        "world_bytes": _bytes_commitment(
            frozen.world_bytes, label="frozen trial world"
        ),
        "identity_parent_projection_bytes": _bytes_commitment(
            frozen.identity_parent_projection_bytes,
            label="frozen identity-parent projection",
        ),
        "identity_parent_sha256": frozen.identity_parent_sha256,
        "identity33_bytes": _bytes_commitment(
            frozen.identity33_bytes, label="frozen identity33"
        ),
        "identity33_sha256": frozen.identity33_sha256,
        "allocation_receipt_bytes": _bytes_commitment(
            frozen.allocation_receipt_bytes,
            label="frozen allocation receipt",
        ),
        "allocation_delta": list(frozen.allocation_delta),
        "candidate_invariant_sha256": frozen.candidate_invariant_sha256,
        "profile_provenance_sha256": frozen.profile_provenance_sha256,
        "profile_sha256": frozen.profile_sha256,
    }
    if (
        _sha256_bytes(frozen.identity_parent_projection_bytes)
        != frozen.identity_parent_sha256
        or _sha256_bytes(frozen.identity33_bytes) != frozen.identity33_sha256
        or frozen.mode != parent.mode
        or frozen.split != parent.split
        or frozen.world_uid != parent.world_uid
        or frozen.candidate_invariant_sha256 != parent.invariant_sha256
        or frozen.profile_provenance_sha256 != parent.profile_provenance_sha256
        or frozen.profile_sha256 != parent.profile_sha256
    ):
        raise NaturalVariationError("Frozen trial identity parent commitment drift")
    return common.canonical_sha256(state)


def _assemble_and_validate(
    *,
    candidate_index: int,
    expected_parent: candidate_parent.CandidateIndependentParent,
    frozen: candidate_parent.FrozenTrialIdentityParent,
    natural: NaturalExpressionCandidate,
) -> AssembledDevelopmentCandidate:
    if type(candidate_index) is not int or not 0 <= candidate_index <= 31:
        raise NaturalVariationError("Trusted candidate index must be an integer from 0 through 31")
    stage_policy = load_policy()
    smoke_contract = stage_policy["expected_smoke_world"]
    if (
        _candidate_parent_full_state_sha256(expected_parent)
        != smoke_contract["candidate_parent_full_state_sha256"]
        or _frozen_identity_full_state_sha256(frozen, parent=expected_parent)
        != smoke_contract["frozen_trial_identity_full_state_sha256"]
    ):
        raise NaturalVariationError("Trusted parent or frozen identity state drift")
    expected_view, expected_binding = _build_restricted_view_and_binding(
        frozen=frozen,
        expected_parent=expected_parent,
    )
    document_variation_key = bytes.fromhex(
        stage_policy["development_smoke_keys"]["document_variation_key_hex"]
    )
    expected_candidate_key = collision.derive_candidate_key(
        document_variation_key=document_variation_key,
        split=frozen.split,
        world_uid=frozen.world_uid,
        candidate_index=candidate_index,
    )
    expected_natural = render_candidate_natural_expressions(
        restricted_view=expected_view,
        candidate_key=expected_candidate_key,
    )
    if not isinstance(natural, NaturalExpressionCandidate) or natural != expected_natural:
        raise NaturalVariationError(
            "Natural candidate is not the exact trusted redraw for its candidate index"
        )
    view = expected_view
    binding = expected_binding
    if natural.view_sha256 != view.view_sha256 or binding.view_sha256 != view.view_sha256:
        raise NaturalVariationError("Candidate/view/binding commitment mismatch")
    binding_value = binding.thaw()
    _require_exact_keys(
        binding_value,
        (
            "version",
            "view_sha256",
            "candidate_invariant_sha256",
            "identity_parent_sha256",
            "item_handle_to_item_uid",
            "noise_handle_to_noise_slot_uid",
            "registered_overrides",
        ),
        label="private handle binding",
    )
    if (
        binding_value["version"] != BINDING_VERSION
        or binding_value["view_sha256"] != view.view_sha256
        or binding_value["candidate_invariant_sha256"] != expected_parent.invariant_sha256
        or binding_value["identity_parent_sha256"] != frozen.identity_parent_sha256
    ):
        raise NaturalVariationError("Private handle binding authority drift")
    natural_value = natural.thaw()
    if (
        natural_value.get("version") != OUTPUT_VERSION
        or natural_value.get("view_sha256") != view.view_sha256
        or natural_value.get("item_count") != len(natural_value.get("items", []))
    ):
        raise NaturalVariationError("Natural candidate envelope drift")
    item_handle_to_uid = binding_value["item_handle_to_item_uid"]
    if set(item_handle_to_uid) != {
        str(row["item_handle"]) for row in natural_value["items"]
    }:
        raise NaturalVariationError("Natural candidate item-handle keyset drift")
    candidate_rows = {
        str(row["item_handle"]): dict(row) for row in natural_value["items"]
    }
    if len(candidate_rows) != len(natural_value["items"]):
        raise NaturalVariationError("Natural candidate contains duplicate item handles")

    uid_to_item_handle = {
        str(uid): str(handle) for handle, uid in item_handle_to_uid.items()
    }
    if len(uid_to_item_handle) != len(item_handle_to_uid):
        raise NaturalVariationError("Private item binding is not one-to-one")
    registered_overrides = binding_value["registered_overrides"]
    if not isinstance(registered_overrides, list):
        raise NaturalVariationError("Private registered overrides are malformed")
    seen_override_assets: set[tuple[str, int]] = set()
    used_override_items: set[str] = set()
    for override in registered_overrides:
        if not isinstance(override, dict):
            raise NaturalVariationError("Private registered override is not an object")
        _require_exact_keys(
            override,
            (
                "override_kind",
                "asset_index",
                "canonical_pair_uid",
                "item_uid_left",
                "item_uid_right",
            ),
            label="private registered override",
        )
        kind = override["override_kind"]
        asset_index = override["asset_index"]
        left_uid = override["item_uid_left"]
        right_uid = override["item_uid_right"]
        override_key = (str(kind), int(asset_index)) if type(asset_index) is int else None
        if (
            kind not in {"high_semantic_similarity", "exact_title_clone"}
            or type(asset_index) is not int
            or asset_index < 0
            or not isinstance(override["canonical_pair_uid"], str)
            or not isinstance(left_uid, str)
            or not isinstance(right_uid, str)
            or left_uid == right_uid
            or left_uid not in uid_to_item_handle
            or right_uid not in uid_to_item_handle
            or override_key in seen_override_assets
            or left_uid in used_override_items
            or right_uid in used_override_items
        ):
            raise NaturalVariationError("Private registered override lineage drift")
        seen_override_assets.add(override_key)
        used_override_items.update((left_uid, right_uid))
        left = candidate_rows[uid_to_item_handle[left_uid]]
        right = candidate_rows[uid_to_item_handle[right_uid]]
        if kind == "high_semantic_similarity":
            if (
                left["category"] != right["category"]
                or left["product"] != right["product"]
                or left["attribute"] != right["attribute"]
                or left["title_skeleton_index"] == right["title_skeleton_index"]
            ):
                raise NaturalVariationError(
                    "Bijection changed a registered high-semantic relation"
                )
        else:
            if not left["title"] or not right["title"]:
                raise NaturalVariationError("Exact-title clone endpoint lacks a title")
            right["title"] = left["title"]

    trusted_natural_value = {
        "version": OUTPUT_VERSION,
        "view_sha256": view.view_sha256,
        "item_count": len(candidate_rows),
        "items": [
            candidate_rows[handle]
            for handle in sorted(candidate_rows, key=lambda value: value.encode("utf-8"))
        ],
    }
    trusted_natural_bytes = _canonical_bytes(trusted_natural_value)
    trusted_natural_sha256 = _sha256_bytes(trusted_natural_bytes)

    base_policy, template, fixture, _style_profile = candidate_parent._load_validated_base_inputs()
    world = frozen.thaw_world()
    public_item_by_uid = {
        str(row["item_uid"]): row for row in world["public"]["items"]
    }
    ast_by_uid = {
        str(row["item_uid"]): row for row in world["private"]["render_asts"]
    }
    identity_by_item: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in world["private"]["identity_slots_audit"]:
        identity_by_item[str(row["item_uid"])].append(dict(row))
    noise_handle_to_slot = binding_value["noise_handle_to_noise_slot_uid"]
    noise_slot_to_handle = {value: key for key, value in noise_handle_to_slot.items()}
    if len(noise_slot_to_handle) != len(noise_handle_to_slot):
        raise NaturalVariationError("Private noise binding is not one-to-one")
    view_value = view.thaw()
    noise_item_to_handle = {
        str(row["item_handle"]): str(row["noise_handle"])
        for row in view_value["noise_targets"]
    }
    role_to_family = base_policy["identity_design"]["role_to_template_family"]
    items_by_seller: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    noise_records_by_item: dict[str, dict[str, Any]] = {}
    all_uid_literals = {
        frozen.world_uid,
        *public_item_by_uid,
        *(str(row["seller_uid"]) for row in world["public"]["sellers"]),
        *(str(row["controller_uid"]) for row in world["private"]["controller_membership"]),
        *(str(row["canonical_pair_uid"]) for row in world["public"]["complete_model_pair_endpoints"]),
    }
    for handle, item_uid in item_handle_to_uid.items():
        candidate_row = candidate_rows[handle]
        public_item = public_item_by_uid.get(str(item_uid))
        ast = ast_by_uid.get(str(item_uid))
        if public_item is None or ast is None:
            raise NaturalVariationError("Private item binding targets an unknown item")
        for field in (
            "category",
            "product",
            "attribute",
            "delivery",
            "service",
            "title_skeleton_index",
            "description_skeleton_index",
        ):
            ast[field] = candidate_row[field]
        public_item["category"] = candidate_row["category"]
        public_item["title"] = candidate_row["title"]
        slots = []
        for source_slot in sorted(
            identity_by_item[str(item_uid)],
            key=lambda row: str(row["slot_uid"]).encode("utf-8"),
        ):
            family = role_to_family[str(source_slot["planned_role"])]
            expected_clause = renderer.identity_clause(
                template_family=str(family),
                identity_type=str(source_slot["identity_type"]),
                normalized_value=str(source_slot["raw_surface"]),
                template=template,
            )
            if expected_clause.count(str(source_slot["raw_surface"])) != 1:
                raise NaturalVariationError("Frozen identity clause no longer round-trips")
            slots.append(
                {
                    "slot_uid": str(source_slot["slot_uid"]),
                    "bundle_uid": str(source_slot["bundle_uid"]),
                    "identity_uid": str(source_slot["identity_uid"]),
                    "role": str(source_slot["planned_role"]),
                    "identity_type": str(source_slot["identity_type"]),
                    "identity_value": str(source_slot["raw_surface"]),
                }
            )
        item_state = {
            "world_uid": str(public_item["world_uid"]),
            "seller_uid": str(public_item["seller_uid"]),
            "item_uid": str(item_uid),
            "time_bucket": int(public_item["time_bucket"]),
            "title_nonempty": bool(ast["title_nonempty"]),
            "description_nonempty": bool(ast["description_nonempty"]),
            "base_description": str(candidate_row["base_description"]),
            "noise_clause": str(candidate_row["noise_clause"]),
            "identity_slots": slots,
        }
        items_by_seller[str(public_item["seller_uid"])].append(item_state)
        noise_handle = noise_item_to_handle.get(handle)
        if noise_handle is not None:
            noise_slot_uid = str(noise_handle_to_slot[noise_handle])
            if noise_slot_uid != str(ast["noise_slot_uid"]):
                raise NaturalVariationError("Noise target changed during candidate assembly")
            noise_records_by_item[str(item_uid)] = {
                "noise_slot_uid": noise_slot_uid,
                "raw_surface": str(candidate_row["noise_clause"]),
            }
        elif candidate_row["noise_clause"] or ast["noise_slot_uid"]:
            raise NaturalVariationError("Candidate noise lineage is incomplete")
        visible_without_identity = (
            str(candidate_row["title"])
            + str(candidate_row["base_description"])
            + str(candidate_row["noise_clause"])
        )
        if any(uid and uid in visible_without_identity for uid in all_uid_literals):
            raise NaturalVariationError("Private UID leaked into candidate-visible text")

    new_identity_audit, new_identity_edit, new_noise_audit = world_builder._render_identity_slots(
        policy=base_policy,
        template=template,
        fixture=fixture,
        items_by_seller=items_by_seller,
        noise_records_by_item=noise_records_by_item,
    )
    rendered_description_by_uid = {
        str(item["item_uid"]): str(item["description"])
        for rows in items_by_seller.values()
        for item in rows
    }
    if set(rendered_description_by_uid) != set(public_item_by_uid):
        raise NaturalVariationError("Candidate descriptions did not cover every item")
    for item_uid, description in rendered_description_by_uid.items():
        public_item_by_uid[item_uid]["description"] = description
    world["private"]["identity_slots_audit"] = new_identity_audit
    world["private"]["identity_slots_edit"] = new_identity_edit
    world["private"]["noise_slots_audit"] = new_noise_audit

    profiles, provenance, context = candidate_parent._build_profiles_and_provenance(
        base_policy=base_policy,
        mode=frozen.mode,
        split=frozen.split,
        world=world,
        template=template,
    )
    provenance_bytes = _canonical_bytes(provenance)
    if provenance_bytes != expected_parent.profile_provenance_bytes:
        expected_provenance = expected_parent.thaw_profile_provenance()
        expected_rows = expected_provenance.get("rows", [])
        observed_rows = provenance.get("rows", [])
        key_fields = ("seller_uid", "output_field", "output_rank")
        expected_index = {
            tuple(row[field] for field in key_fields): row for row in expected_rows
        }
        observed_index = {
            tuple(row[field] for field in key_fields): row for row in observed_rows
        }
        mismatch_fields: Counter[str] = Counter()
        mismatch_output_fields: Counter[str] = Counter()
        mismatch_expected_description_skeletons: Counter[int] = Counter()
        mismatch_observed_description_skeletons: Counter[int] = Counter()
        mismatch_expected_noise_targets = 0
        mismatch_observed_noise_targets = 0
        for key in set(expected_index) & set(observed_index):
            expected_row = expected_index[key]
            observed_row = observed_index[key]
            row_mismatch = False
            for field in set(expected_row) | set(observed_row):
                if expected_row.get(field) != observed_row.get(field):
                    mismatch_fields[str(field)] += 1
                    row_mismatch = True
            if row_mismatch:
                mismatch_output_fields[str(expected_row["output_field"])] += 1
                for source_uid in expected_row.get("source_item_uids", []):
                    mismatch_expected_description_skeletons[
                        int(ast_by_uid[str(source_uid)]["description_skeleton_index"])
                    ] += 1
                    mismatch_expected_noise_targets += int(
                        bool(ast_by_uid[str(source_uid)]["noise_slot_uid"])
                    )
                for source_uid in observed_row.get("source_item_uids", []):
                    mismatch_observed_description_skeletons[
                        int(ast_by_uid[str(source_uid)]["description_skeleton_index"])
                    ] += 1
                    mismatch_observed_noise_targets += int(
                        bool(ast_by_uid[str(source_uid)]["noise_slot_uid"])
                    )
        missing_keys = set(expected_index) - set(observed_index)
        extra_keys = set(observed_index) - set(expected_index)
        summary = {
            "expected_row_count": len(expected_rows),
            "observed_row_count": len(observed_rows),
            "missing_key_count": len(missing_keys),
            "extra_key_count": len(extra_keys),
            "missing_output_fields": dict(
                sorted(Counter(str(key[1]) for key in missing_keys).items())
            ),
            "extra_output_fields": dict(
                sorted(Counter(str(key[1]) for key in extra_keys).items())
            ),
            "mismatch_fields": dict(sorted(mismatch_fields.items())),
            "mismatch_output_fields": dict(sorted(mismatch_output_fields.items())),
            "mismatch_expected_description_skeletons": dict(
                sorted(mismatch_expected_description_skeletons.items())
            ),
            "mismatch_observed_description_skeletons": dict(
                sorted(mismatch_observed_description_skeletons.items())
            ),
            "mismatch_expected_noise_target_count": mismatch_expected_noise_targets,
            "mismatch_observed_noise_target_count": mismatch_observed_noise_targets,
        }
        raise NaturalVariationError(
            "Candidate reselected Step3 contribution lineage: "
            + json.dumps(summary, sort_keys=True)
        )
    invariant = candidate_parent.candidate_invariant_projection(
        policy=base_policy,
        template=template,
        mode=frozen.mode,
        split=frozen.split,
        world=world,
        profile_provenance=provenance,
    )
    invariant_bytes = _canonical_bytes(invariant)
    if invariant_bytes != expected_parent.invariant_projection_bytes:
        raise NaturalVariationError("Candidate changed its candidate-independent parent")

    processed = context["processed"]
    item_index = candidate_parent._history_item_index(world)
    history_rows = processed["public"]["history_safe_occurrences"]
    parsed = processed["private"]["parsed_identity_occurrences"]
    attestation = candidate_parent.production.build_history_projection_attestation(
        base_policy,
        mode=frozen.mode,
        split=frozen.split,
        world_uid=frozen.world_uid,
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
        for value in base_policy["relational_integrity"]["pair_projection_contract"][
            "complete_model_pair_endpoints_schema"
        ]
    ]
    endpoints = [
        {field: row[field] for field in pair_schema}
        for row in world["public"]["complete_model_pair_endpoints"]
    ]
    identity33, identity33_audit = history_features.build_identity33_all_pairs(
        base_policy,
        mode=frozen.mode,
        split=frozen.split,
        history_safe_occurrences=history_rows,
        history_item_index=item_index,
        projection_attestations=[attestation],
        complete_model_pair_endpoints=endpoints,
    )
    identity33_bytes = _canonical_bytes(identity33)
    if (
        identity33_bytes != frozen.identity33_bytes
        or identity33_audit.get("feature_count") != 33
        or identity33_audit.get("identity33_sha256") != common.canonical_sha256(identity33)
    ):
        raise NaturalVariationError("Candidate changed the frozen identity33 parent")
    identity_parent = candidate_parent._identity_parent_projection(
        world=world,
        identity33=identity33,
        allocation_delta=frozen.allocation_delta,
    )
    identity_parent_bytes = _canonical_bytes(identity_parent)
    if identity_parent_bytes != frozen.identity_parent_projection_bytes:
        raise NaturalVariationError("Candidate changed frozen identity assets or slots")
    world_bytes = _canonical_bytes(world)
    profiles_bytes = _canonical_bytes(profiles)
    return AssembledDevelopmentCandidate(
        candidate_index=candidate_index,
        world_bytes=world_bytes,
        world_sha256=_sha256_bytes(world_bytes),
        profiles_bytes=profiles_bytes,
        profiles_sha256=_sha256_bytes(profiles_bytes),
        profile_provenance_bytes=provenance_bytes,
        profile_provenance_sha256=_sha256_bytes(provenance_bytes),
        identity33_bytes=identity33_bytes,
        identity33_sha256=_sha256_bytes(identity33_bytes),
        natural_output_sha256=trusted_natural_sha256,
        candidate_invariant_sha256=_sha256_bytes(invariant_bytes),
        identity_parent_sha256=_sha256_bytes(identity_parent_bytes),
    )


class DevelopmentSmokeVariationSession:
    """One identity allocation followed by any number of in-memory candidates."""

    def __init__(self) -> None:
        self._policy = load_policy()
        self._parent = candidate_parent.build_candidate_independent_parent()
        allocator = candidate_parent.OneTimeTrialIdentityAllocator()
        self._frozen = allocator.allocate(parent=self._parent)
        self._view, _binding = _build_restricted_view_and_binding(
            frozen=self._frozen,
            expected_parent=self._parent,
        )
        self._rendered_indices: set[int] = set()
        self._failed = False

    @property
    def restricted_view_sha256(self) -> str:
        return self._view.view_sha256

    @property
    def identity_parent_sha256(self) -> str:
        return self._frozen.identity_parent_sha256

    def render(self, candidate_index: int) -> AssembledDevelopmentCandidate:
        if self._failed:
            raise NaturalVariationError(
                "Development candidate session is permanently failed"
            )
        if type(candidate_index) is not int or not 0 <= candidate_index <= 31:
            self._failed = True
            raise NaturalVariationError("Candidate index must be an integer from 0 through 31")
        if candidate_index != len(self._rendered_indices):
            self._failed = True
            raise NaturalVariationError(
                "Development candidates must be rendered once in order"
            )
        self._rendered_indices.add(candidate_index)
        completed = False
        try:
            document_variation_key = bytes.fromhex(
                self._policy["development_smoke_keys"]["document_variation_key_hex"]
            )
            candidate_key = collision.derive_candidate_key(
                document_variation_key=document_variation_key,
                split=self._frozen.split,
                world_uid=self._frozen.world_uid,
                candidate_index=candidate_index,
            )
            natural = render_candidate_natural_expressions(
                restricted_view=self._view,
                candidate_key=candidate_key,
            )
            result = _assemble_and_validate(
                candidate_index=candidate_index,
                expected_parent=self._parent,
                frozen=self._frozen,
                natural=natural,
            )
            completed = True
            return result
        finally:
            if not completed:
                self._failed = True


def main() -> None:
    session = DevelopmentSmokeVariationSession()
    result = session.render(0)
    policy = load_policy()
    print(
        json.dumps(
            {
                "status": "PASS_DEVELOPMENT_SMOKE_NATURAL_VARIATION_ONLY",
                "version": policy["version"],
                "formal_authorizations": policy["formal_authorizations"],
                "formal_seeds_generated": 0,
                "formal_rows_generated": 0,
                "formal_models_trained": 0,
                "in_memory_candidate_count": 1,
                "candidate_index": result.candidate_index,
                "restricted_view_sha256": session.restricted_view_sha256,
                "identity_parent_sha256": session.identity_parent_sha256,
                "natural_output_sha256": result.natural_output_sha256,
                "world_sha256": result.world_sha256,
                "profile_provenance_sha256": result.profile_provenance_sha256,
                "identity33_sha256": result.identity33_sha256,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
