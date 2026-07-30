#!/usr/bin/env python3
"""Execute the exhaustive Step 28-v13 parser/template/redactor contract.

This is a pre-generation fixture.  It produces no labels, model scores, or
scientific metrics.  Every expectation is derived from the frozen fixture
specification before the production parser is called.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import step28_history_common as history
import step28_v13_common as common
import step28_v13_text_renderer as renderer
import step3_build_seller_profiles as step3
import step4_build_silver_candidates as step4
import step7_v3_1_source_data as source
import step7_v4_common as redactor


DEFAULT_FIXTURE_PATH = (
    common.ROOT / "schema" / "step28_v13_parser_template_fixture.json"
)
SPLITS = ("train", "development", "audit_a", "audit_b")
ROLE_ORDER = (
    "direct_or_private",
    "high_frequency_direct",
    "public_support",
    "risky_product",
)
FLAG_ORDER = (
    "seller_facing_context",
    "product_data_risk_context",
    "direct_identity_eligible",
    "support_only",
)


def _pin_or_hash(spec: Mapping[str, Any], *, label: str) -> tuple[Path, str]:
    path = common.repo_path(str(spec["path"]))
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    observed = common.sha256_file(path)
    expected = spec.get("sha256")
    if expected is not None and observed != str(expected).lower():
        raise common.ContractError(
            f"{label} drift: expected={expected} observed={observed}"
        )
    return path, observed


def _validate_dependencies(fixture: Mapping[str, Any]) -> dict[str, str]:
    pins: dict[str, str] = {}
    for key in (
        "parser",
        "step3_profile_contact_extractor",
        "history_token_key",
        "step4_contact_normalizer",
        "step7_v4_redactor",
    ):
        _, pins[key] = _pin_or_hash(fixture[key], label=key)
    dependency = fixture["step7_v4_redactor"]["pinned_lazy_dependency"]
    _, pins["step7_v4_redactor_dependency"] = _pin_or_hash(
        dependency, label="step7_v4_redactor dependency"
    )
    template_path, pins["template_library"] = _pin_or_hash(
        fixture["template_library"], label="template library"
    )
    runner_spec = fixture["full_render_context_contract"]["runner"]
    runner_path, pins["runner"] = _pin_or_hash(
        runner_spec, label="fixture runner"
    )
    if runner_path.resolve() != Path(__file__).resolve():
        raise common.ContractError("Fixture runner path does not identify this script")
    pins["fixture"] = common.sha256_file(DEFAULT_FIXTURE_PATH)

    if len(step4.EN_STOPWORDS) != int(
        fixture["step4_contact_normalizer"]["stopwords_count"]
    ):
        raise common.ContractError("Step4 stopword count drift")
    stopwords_hash = common.canonical_sha256(
        common.utf8_sort(str(value) for value in step4.EN_STOPWORDS)
    )
    if (
        stopwords_hash
        != fixture["step4_contact_normalizer"][
            "sorted_stopwords_canonical_json_sha256"
        ]
    ):
        raise common.ContractError("Step4 stopword hash drift")
    pins["step4_stopwords"] = stopwords_hash
    pins["template_path"] = template_path.relative_to(common.ROOT).as_posix()
    return pins


def _role_type_cases(
    fixture: Mapping[str, Any], identity_types: Sequence[str]
) -> list[tuple[str, str]]:
    output: list[tuple[str, str]] = []
    for role in ROLE_ORDER:
        allowed = set(fixture["dgp_allowed_role_types"][role])
        output.extend((role, identity_type) for identity_type in identity_types if identity_type in allowed)
    if len(output) != 25 or len(set(output)) != 25:
        raise common.ContractError("Role/type fixture must contain exactly 25 unique cases")
    return output


def _noise_choices(
    fixture: Mapping[str, Any], template: Mapping[str, Any]
) -> list[str]:
    output = [""]
    for template_index in range(
        len(template["identity_clause_templates"]["must_ignore"])
    ):
        for value in fixture["must_ignore_adversarial_values"]:
            output.append(
                renderer.must_ignore_clause(
                    template_index=template_index,
                    value=value,
                    template=template,
                )
            )
    if len(output) != 10:
        raise common.ContractError("Noise fixture must contain empty plus nine cases")
    return output


def _case_uid_payload(
    *,
    case_kind: str,
    split: str,
    title_index: int,
    description_index: int | None,
    style_uid: str,
    product_index: int,
    attribute_index: int,
    code_index: int,
    delivery_index: int | None,
    service_index: int | None,
    noise_index: int | None,
    seller_index: int,
    slots: Sequence[tuple[str, str, int]],
) -> dict[str, Any]:
    """Canonical case object; absent title-only fields use explicit JSON null."""

    return {
        "case_kind": case_kind,
        "split": split,
        "title_skeleton_index": title_index,
        "description_skeleton_index": description_index,
        "effective_style_uid": style_uid,
        "product_index": product_index,
        "attribute_index": attribute_index,
        "code_index": code_index,
        "delivery_index": delivery_index,
        "service_index": service_index,
        "noise_index": noise_index,
        "seller_index": seller_index,
        "ordered_role_type_cases": [
            {
                "role": role,
                "identity_type": identity_type,
                "sample_variant_index": variant,
            }
            for role, identity_type, variant in slots
        ],
    }


def _case(
    *,
    case_kind: str,
    split: str,
    title_index: int,
    description_index: int | None,
    style_index: int,
    styles: Sequence[Mapping[str, Any]],
    product_index: int,
    attribute_index: int,
    code_index: int,
    delivery_index: int | None,
    service_index: int | None,
    noise_index: int | None,
    slots: Sequence[tuple[str, str, int]],
    family_ordinal: int,
) -> dict[str, Any]:
    style_uid = str(styles[style_index]["effective_style_uid"])
    payload = _case_uid_payload(
        case_kind=case_kind,
        split=split,
        title_index=title_index,
        description_index=description_index,
        style_uid=style_uid,
        product_index=product_index,
        attribute_index=attribute_index,
        code_index=code_index,
        delivery_index=delivery_index,
        service_index=service_index,
        noise_index=noise_index,
        seller_index=family_ordinal % 28,
        slots=slots,
    )
    return {
        **payload,
        "case_uid": "case_" + common.canonical_sha256(payload),
        "effective_style_index": style_index,
        "family_ordinal": family_ordinal,
    }


def iter_cases(
    *,
    fixture: Mapping[str, Any],
    styles: Sequence[Mapping[str, Any]],
    identity_types: Sequence[str],
) -> Iterable[dict[str, Any]]:
    role_cases = _role_type_cases(fixture, identity_types)
    code_count = len(
        fixture["full_render_context_contract"]["fixture_code_values"]
    )
    if code_count != 16:
        raise common.ContractError(
            "Title-modifier fixture must register exactly 16 code values"
        )
    ordinal = 0
    for split in SPLITS:
        for description_index in range(8):
            for style_index in range(len(styles)):
                for role_index, (role, identity_type) in enumerate(role_cases):
                    local = description_index * len(role_cases) + role_index
                    yield _case(
                        case_kind="single_role_full_render",
                        split=split,
                        title_index=(ordinal // 360) % 8,
                        description_index=description_index,
                        style_index=style_index,
                        styles=styles,
                        product_index=local % 12,
                        attribute_index=local % 10,
                        code_index=(ordinal // 120) % code_count,
                        delivery_index=ordinal % 6,
                        service_index=(ordinal // 6) % 6,
                        noise_index=(ordinal // 36) % 10,
                        slots=[(role, identity_type, 0)],
                        family_ordinal=ordinal,
                    )
                    ordinal += 1

    ordinal = 0
    for left_role, left_type in role_cases:
        for right_role, right_type in role_cases:
            yield _case(
                case_kind="adjacent_ordered_roles",
                split=SPLITS[ordinal % 4],
                title_index=(ordinal // 360) % 8,
                description_index=(ordinal // 4) % 8,
                style_index=ordinal % len(styles),
                styles=styles,
                product_index=ordinal % 12,
                attribute_index=(ordinal // 12) % 10,
                code_index=(ordinal // 120) % code_count,
                delivery_index=ordinal % 6,
                service_index=(ordinal // 6) % 6,
                noise_index=(ordinal // 36) % 10,
                slots=[(left_role, left_type, 0), (right_role, right_type, 1)],
                family_ordinal=ordinal,
            )
            ordinal += 1

    ordinal = 0
    maximum_roles = fixture["full_render_context_contract"][
        "maximum_multislot_roles"
    ]
    maximum_slots = [
        (maximum_roles[identity_type], identity_type, 0)
        for identity_type in identity_types
    ]
    for split in SPLITS:
        for style_index in range(len(styles)):
            yield _case(
                case_kind="maximum_eight_slots",
                split=split,
                title_index=(ordinal // 8) % 8,
                description_index=ordinal % 8,
                style_index=style_index,
                styles=styles,
                product_index=ordinal % 12,
                attribute_index=(ordinal // 12) % 10,
                code_index=(ordinal // 120) % code_count,
                delivery_index=ordinal % 6,
                service_index=(ordinal // 6) % 6,
                noise_index=(ordinal // 36) % 10,
                slots=maximum_slots,
                family_ordinal=ordinal,
            )
            ordinal += 1

    ordinal = 0
    for split in SPLITS:
        for title_index in range(8):
            for style_index in range(len(styles)):
                for code_index in range(code_count):
                    yield _case(
                        case_kind="title_only",
                        split=split,
                        title_index=title_index,
                        description_index=None,
                        style_index=style_index,
                        styles=styles,
                        product_index=ordinal % 12,
                        attribute_index=(ordinal // 12) % 10,
                        code_index=code_index,
                        delivery_index=None,
                        service_index=None,
                        noise_index=None,
                        slots=[],
                        family_ordinal=ordinal,
                    )
                    ordinal += 1


def _fixture_registry(
    fixture: Mapping[str, Any], identity_types: Sequence[str]
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, str]]:
    profiles: list[dict[str, Any]] = []
    literals: dict[str, list[str]] = {}
    variants = fixture["sample_value_variants"]
    for ordinal in range(28):
        seller_uid = f"fixture_seller_{ordinal:02d}"
        profile = {
            "seller_uid": seller_uid,
            "source_seller_raw": seller_uid,
            "alias_normalized": seller_uid,
        }
        profiles.append(profile)
        assigned = (
            identity_types[ordinal % len(identity_types)],
            identity_types[(ordinal + 3) % len(identity_types)],
        )
        surfaces: set[str] = set()
        for identity_type in assigned:
            raw = str(variants[identity_type][0])
            normalized = raw.casefold()
            for value in (raw, normalized):
                admitted = source.safe_signal_literal(identity_type, value)
                if admitted is None:
                    raise common.ContractError(
                        f"Registry fixture value rejected: {identity_type}"
                    )
                surfaces.add(admitted)
        literals[seller_uid] = common.utf8_sort(surfaces)

    global_tokens = source.global_identity_tokens(literals, profiles)
    contextual_aliases = source.contextual_global_alias_tokens(profiles, literals)
    contextual_deletions = redactor.v4_contextual_alias_deletion_tokens(
        contextual_aliases
    )
    seller_literals = {
        profile["seller_uid"]: source.seller_identity_literals(profile)
        for profile in profiles
    }
    seller_phrases = {
        profile["seller_uid"]: source.seller_identity_phrase_tokens(profile)
        for profile in profiles
    }
    registry: dict[str, Any] = {
        "global_identity_tokens": global_tokens,
        "contextual_global_alias_tokens": contextual_aliases,
        "contextual_alias_deletion_tokens": contextual_deletions,
        "seller_identity_literals": seller_literals,
        "seller_identity_phrase_tokens": seller_phrases,
        "seller_contextual_collision_tokens": {
            profile["seller_uid"]: set() for profile in profiles
        },
        "audited_global_identity_phrase_tokens": set(
            source.AUDITED_GLOBAL_IDENTITY_PHRASE_TOKENS
        ),
    }

    def serializable(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                key: serializable(value[key])
                for key in common.utf8_sort(str(item) for item in value)
            }
        if isinstance(value, (set, frozenset)):
            return common.utf8_sort(str(item) for item in value)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [serializable(item) for item in value]
        return value

    hashes = {
        name: common.canonical_sha256(serializable(value))
        for name, value in registry.items()
    }
    expected_hashes = fixture["production_redaction_registry_fixture"].get(
        "expected_registry_hashes"
    )
    if expected_hashes is not None and hashes != expected_hashes:
        raise common.ContractError("Production redaction registry hash drift")
    return profiles, registry, hashes


def _redact(
    value: str,
    *,
    seller_uid: str,
    registry: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    return redactor.redact_raw_field(
        value,
        seller_uid=seller_uid,
        seller_literals=registry["seller_identity_literals"][seller_uid],
        seller_phrase_tokens=registry["seller_identity_phrase_tokens"][seller_uid],
        global_tokens=registry["global_identity_tokens"],
        contextual_aliases=registry["contextual_global_alias_tokens"],
        contextual_alias_deletions=registry[
            "contextual_alias_deletion_tokens"
        ],
        seller_contextual_collision_tokens=registry[
            "seller_contextual_collision_tokens"
        ][seller_uid],
        audited_global_phrases=registry[
            "audited_global_identity_phrase_tokens"
        ],
    )


def _expected_row(
    *,
    fixture: Mapping[str, Any],
    role: str,
    identity_type: str,
    raw_value: str,
) -> tuple[Any, ...]:
    family = fixture["role_to_template_family"][role]
    flags = fixture["expected_role_flags"][family][identity_type]
    return (
        "description",
        identity_type,
        raw_value.casefold(),
        *[int(value) for value in flags],
    )


def _actual_row(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["source_field"]),
        str(row["contact_type"]).casefold(),
        str(row["normalized_value"]).strip().casefold(),
        *[int(row[name]) for name in FLAG_ORDER],
    )


def _render_case(
    case: Mapping[str, Any],
    *,
    template: Mapping[str, Any],
    fixture: Mapping[str, Any],
    styles: Sequence[Mapping[str, Any]],
    noise_choices: Sequence[str],
) -> tuple[str, str, str, list[tuple[Any, ...]], list[str]]:
    lexicon = template["generic_lexicon"]
    style = styles[int(case["effective_style_index"])]
    split_library = template["split_libraries"][case["split"]]
    product = lexicon["products"][int(case["product_index"])]
    attribute = lexicon["attributes"][int(case["attribute_index"])]
    code = fixture["full_render_context_contract"]["fixture_code_values"][
        int(case["code_index"])
    ]
    title = renderer.render_base_title(
        skeleton=split_library["title_skeletons"][int(case["title_skeleton_index"])],
        product=product,
        attribute=attribute,
        code=code,
        style=style,
        template=template,
    )
    if case["case_kind"] == "title_only":
        return title, "", "", [], []

    base_description = renderer.render_base_description(
        skeleton=split_library["description_skeletons"][
            int(case["description_skeleton_index"])
        ],
        product=product,
        attribute=attribute,
        code=code,
        delivery=lexicon["delivery"][int(case["delivery_index"])],
        service=lexicon["service"][int(case["service_index"])],
        style=style,
        template=template,
    )
    identity_clauses: list[str] = []
    expected_rows: list[tuple[Any, ...]] = []
    surfaces: list[str] = []
    for slot in case["ordered_role_type_cases"]:
        role = str(slot["role"])
        identity_type = str(slot["identity_type"])
        variant = int(slot["sample_variant_index"])
        raw_value = str(fixture["sample_value_variants"][identity_type][variant])
        family = fixture["role_to_template_family"][role]
        identity_clauses.append(
            renderer.identity_clause(
                template_family=family,
                identity_type=identity_type,
                normalized_value=raw_value,
                template=template,
            )
        )
        expected_rows.append(
            _expected_row(
                fixture=fixture,
                role=role,
                identity_type=identity_type,
                raw_value=raw_value,
            )
        )
        surfaces.append(raw_value)
    noise = noise_choices[int(case["noise_index"])]
    description = renderer.render_description(
        base_description=base_description,
        noise_clause=noise,
        identity_clauses=identity_clauses,
        selector_uid=str(case["case_uid"]),
        template=template,
    )
    return title, description, base_description, expected_rows, surfaces


def _validate_profile_chain(
    fixture: Mapping[str, Any], template: Mapping[str, Any]
) -> dict[str, Any]:
    projected: dict[str, list[str]] = {}
    normalized_projection: dict[str, list[str]] = {}
    for identity_type, value in fixture["sample_values"].items():
        family = (
            "direct"
            if identity_type != "external_url"
            else "support"
        )
        description = renderer.identity_clause(
            template_family=family,
            identity_type=identity_type,
            normalized_value=value,
            template=template,
        )
        extracted = step3.extract_contacts("", description, "")
        values = list(extracted.get(identity_type, []))
        if identity_type in fixture["profile_contact_absent_types"]:
            if values:
                raise common.ContractError(
                    f"Profile extractor unexpectedly projected {identity_type}"
                )
            continue
        expected = fixture["expected_step3_profile_contact_projection"][
            identity_type
        ]
        if values != [expected[1]]:
            raise common.ContractError(
                f"Step3 profile contact projection drift for {identity_type}"
            )
        normalized = step4.normalize_contact_value(
            identity_type, values[0], set(step4.EN_STOPWORDS)
        )
        expected_step4 = fixture[
            "expected_step4_contact_keys_after_profile_projection"
        ][identity_type]
        if [identity_type, normalized] != expected_step4:
            raise common.ContractError(
                f"Step4 contact normalization drift for {identity_type}"
            )
        token = history.token_key(
            {"contact_type": identity_type, "normalized_value": normalized}
        )
        if list(token or ()) != fixture["expected_history_token_keys"][identity_type]:
            raise common.ContractError(
                f"History token key drift for {identity_type}"
            )
        projected[identity_type] = values
        normalized_projection[identity_type] = [normalized]
    return {
        "step3_profile_projection": projected,
        "step4_normalized_projection": normalized_projection,
        "absent_types": list(fixture["profile_contact_absent_types"]),
    }


def _validate_parser_only_stress(
    *,
    fixture: Mapping[str, Any],
    template: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove why the separately listed numeric stress value cannot enter the DGP."""

    checked = 0
    for template_index in range(
        len(template["identity_clause_templates"]["must_ignore"])
    ):
        for value in fixture["parser_only_redactor_incompatible_stress_values"]:
            text = renderer.must_ignore_clause(
                template_index=template_index,
                value=str(value),
                template=template,
            )
            meta = {
                "data_bucket": "step28_v13_fixture",
                "source_dataset": "step28_v13_fixture",
                "source_row_number": f"parser_only_{checked}",
                "seller_uid": "fixture_seller_00",
                "source_market_raw": "fixture_market",
                "source_seller_raw": "fixture_seller_00",
                "source_seller_id_raw": "fixture_seller_00",
                "alias_normalized": "fixture_seller_00",
            }
            rows = step3.extract_item_identity_signals(
                meta,
                title_raw="",
                description_raw=text,
                structured_snapshot="",
                extra_fields=None,
            )
            if rows:
                raise common.ContractError(
                    "Parser-only must-ignore stress unexpectedly produced a parser row"
                )
            clean, _ = _redact(
                text,
                seller_uid="fixture_seller_00",
                registry=registry,
            )
            if clean == source.normalize_redacted_text(text) or str(value) in clean:
                raise common.ContractError(
                    "Parser-only stress no longer demonstrates redactor incompatibility"
                )
            checked += 1
    return {
        "case_count": checked,
        "step3_parser_row_count": 0,
        "step7_v4_redaction_change_count": checked,
        "main_dgp_use_forbidden": True,
    }


def run_fixture(
    *,
    policy_path: Path,
    fixture_path: Path,
    output_path: Path | None,
    maximum_cases: int | None,
) -> dict[str, Any]:
    policy = common.load_policy(policy_path, mode="development_smoke")
    if fixture_path.resolve() != DEFAULT_FIXTURE_PATH.resolve():
        raise common.ContractError("Only the policy-pinned fixture path may be executed")
    fixture = common.load_json(fixture_path)
    dependency_hashes = _validate_dependencies(fixture)
    template_path = common.repo_path(fixture["template_library"]["path"])
    template = common.load_json(template_path)
    styles = renderer.reachable_effective_styles(template)
    expected_styles = int(
        fixture["full_render_context_contract"][
            "expected_reachable_effective_style_count"
        ]
    )
    if len(styles) != expected_styles:
        raise common.ContractError(
            f"Effective-style count drift: {len(styles)} != {expected_styles}"
        )
    style_manifest_hash = common.canonical_sha256(styles)
    expected_style_hash = fixture["full_render_context_contract"].get(
        "expected_effective_style_manifest_sha256"
    )
    if (
        expected_style_hash is not None
        and style_manifest_hash != str(expected_style_hash).lower()
    ):
        raise common.ContractError("Effective-style manifest hash drift")
    identity_types = list(policy["identity_design"]["identity_types"])
    profiles, registry, registry_hashes = _fixture_registry(
        fixture, identity_types
    )
    noise_choices = _noise_choices(fixture, template)
    profile_chain = _validate_profile_chain(fixture, template)
    parser_only_stress = _validate_parser_only_stress(
        fixture=fixture,
        template=template,
        registry=registry,
    )

    family_counts: Counter[str] = Counter()
    parser_row_counts: Counter[str] = Counter()
    redaction_diagnostics: Counter[str] = Counter()
    coverage: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(
        lambda: {"products": set(), "attributes": set()}
    )
    case_records: list[tuple[str, str]] = []
    case_uids: set[str] = set()
    title_modifier_coverage: set[str] = set()
    guards = renderer.context_guard_pool(template)
    context_contract = fixture["full_render_context_contract"]
    fixture_codes = list(context_contract["fixture_code_values"])
    if (
        len(fixture_codes) != 16
        or len(set(fixture_codes)) != 16
        or {
            renderer.title_modifier(code, template)
            for code in fixture_codes
        }
        != set(template["generic_lexicon"]["title_modifiers"])
    ):
        raise common.ContractError("Title-modifier fixture coverage drift")
    if (
        int(context_contract["context_radius_code_points"]) != 90
        or int(context_contract["context_guard_pool_count"]) != len(guards)
        or int(context_contract["minimum_context_guard_code_points"])
        != min(len(value) for value in guards)
        or int(context_contract["maximum_context_guard_code_points"])
        != max(len(value) for value in guards)
        or str(context_contract["context_guard_pool_sha256"]).lower()
        != common.canonical_sha256(list(guards))
    ):
        raise common.ContractError("Context guard fixture contract drift")
    expected_total = int(
        fixture["full_render_context_contract"]["expected_case_count"]
    )

    for total_index, case in enumerate(
        iter_cases(
            fixture=fixture,
            styles=styles,
            identity_types=identity_types,
        )
    ):
        if maximum_cases is not None and total_index >= maximum_cases:
            break
        uid = str(case["case_uid"])
        if uid in case_uids:
            raise common.ContractError(f"Duplicate fixture case UID: {uid}")
        case_uids.add(uid)
        family = str(case["case_kind"])
        family_counts[family] += 1
        if family == "title_only":
            title_modifier_coverage.add(
                renderer.title_modifier(
                    fixture_codes[int(case["code_index"])],
                    template,
                )
            )
        title, description, base_description, expected_rows, surfaces = _render_case(
            case,
            template=template,
            fixture=fixture,
            styles=styles,
            noise_choices=noise_choices,
        )

        meta = {
            "data_bucket": "step28_v13_fixture",
            "source_dataset": "step28_v13_fixture",
            "source_row_number": uid,
            "seller_uid": f"fixture_seller_{int(case['seller_index']):02d}",
            "source_market_raw": "fixture_market",
            "source_seller_raw": f"fixture_seller_{int(case['seller_index']):02d}",
            "source_seller_id_raw": f"fixture_seller_{int(case['seller_index']):02d}",
            "alias_normalized": f"fixture_seller_{int(case['seller_index']):02d}",
        }
        parsed = step3.extract_item_identity_signals(
            meta,
            title_raw=title,
            description_raw=description,
            structured_snapshot="",
            extra_fields=None,
        )
        observed_rows = Counter(_actual_row(row) for row in parsed)
        wanted_rows = Counter(expected_rows)
        if observed_rows != wanted_rows:
            raise common.ContractError(
                f"Parser contract drift in {uid}: "
                f"missing={list((wanted_rows-observed_rows).elements())[:3]} "
                f"extra={list((observed_rows-wanted_rows).elements())[:3]}"
            )
        if any(row["source_field"] == "title" for row in parsed):
            raise common.ContractError(f"Title parser false positive in {uid}")
        for row in parsed:
            token = history.token_key(row)
            expected_token = (
                str(row["contact_type"]).casefold(),
                str(row["normalized_value"]).strip().casefold(),
            )
            if token != expected_token:
                raise common.ContractError(f"History token drift in {uid}")
            parser_row_counts[str(row["contact_type"])] += 1

        seller_uid = meta["seller_uid"]
        clean_title, title_diagnostics = _redact(
            title, seller_uid=seller_uid, registry=registry
        )
        clean_description, description_diagnostics = _redact(
            description, seller_uid=seller_uid, registry=registry
        )
        expected_clean_title = source.normalize_redacted_text(title)
        if clean_title != expected_clean_title:
            raise common.ContractError(f"Title redaction changed base text in {uid}")
        for value in surfaces:
            if value.casefold() in clean_description.casefold():
                raise common.ContractError(f"Identity surface survived redaction in {uid}")
        normalized_base = source.normalize_redacted_text(base_description)
        if normalized_base and normalized_base not in clean_description:
            raise common.ContractError(f"Base description changed during redaction in {uid}")
        noise_index = case["noise_index"]
        noise = "" if noise_index is None else noise_choices[int(noise_index)]
        normalized_noise = source.normalize_redacted_text(noise)
        if normalized_noise and normalized_noise not in clean_description:
            raise common.ContractError(f"Must-ignore text changed during redaction in {uid}")
        selected_guards = (
            renderer.context_guard_sequence(
                selector_uid=uid,
                count=len(expected_rows) + 1,
                template=template,
            )
            if expected_rows
            else ()
        )
        if any(clean_description.count(guard) != 1 for guard in selected_guards):
            raise common.ContractError(f"Context guard changed during redaction in {uid}")
        if sum(clean_description.count(guard) for guard in guards) != len(
            selected_guards
        ):
            raise common.ContractError(
                f"Unexpected context guard multiplicity in {uid}"
            )
        for diagnostics in (title_diagnostics, description_diagnostics):
            for key in (
                "generic_identifier_match_count",
                "seller_local_alias_match_count",
                "seller_local_alias_phrase_match_count",
                "audited_global_identity_phrase_match_count",
                "global_identifier_token_match_count",
                "contextual_alias_match_count",
            ):
                redaction_diagnostics[key] += int(diagnostics.get(key, 0))

        if family == "single_role_full_render":
            style_uid = str(case["effective_style_uid"])
            group = coverage[(str(case["split"]), style_uid)]
            group["products"].add(
                template["generic_lexicon"]["products"][int(case["product_index"])]
            )
            group["attributes"].add(
                template["generic_lexicon"]["attributes"][
                    int(case["attribute_index"])
                ]
            )

        outcome = {
            "case_uid": uid,
            "case_kind": family,
            "title_sha256": common.sha256_bytes(title.encode("utf-8")),
            "description_sha256": common.sha256_bytes(description.encode("utf-8")),
            "parser_rows_sha256": common.canonical_sha256(
                sorted(observed_rows.elements())
            ),
            "redacted_title_sha256": common.sha256_bytes(
                clean_title.encode("utf-8")
            ),
            "redacted_description_sha256": common.sha256_bytes(
                clean_description.encode("utf-8")
            ),
        }
        case_records.append((uid, common.canonical_sha256(outcome)))

    limited = maximum_cases is not None
    observed_total = sum(family_counts.values())
    if not limited and observed_total != expected_total:
        raise common.ContractError(
            f"Fixture case-count drift: {observed_total} != {expected_total}"
        )
    if not limited:
        expected_products = set(template["generic_lexicon"]["products"])
        expected_attributes = set(template["generic_lexicon"]["attributes"])
        expected_groups = len(SPLITS) * len(styles)
        if len(coverage) != expected_groups:
            raise common.ContractError("Split/style coverage group count drift")
        failures = [
            (split, style_uid)
            for (split, style_uid), values in coverage.items()
            if values["products"] != expected_products
            or values["attributes"] != expected_attributes
        ]
        if failures:
            raise common.ContractError(
                f"Product/attribute coverage failed for {len(failures)} groups"
            )
        if title_modifier_coverage != set(
            template["generic_lexicon"]["title_modifiers"]
        ):
            raise common.ContractError("Title-modifier render coverage failed")

    manifest_hasher = hashlib.sha256()
    for uid, outcome_hash in sorted(
        case_records, key=lambda value: value[0].encode("utf-8")
    ):
        manifest_hasher.update(
            common.canonical_json_bytes(
                {"case_uid": uid, "outcome_sha256": outcome_hash}
            )
            + b"\n"
        )
    outcome_manifest_hash = manifest_hasher.hexdigest()
    expected_outcome_hash = fixture["full_render_context_contract"][
        "output_manifest"
    ].get("expected_case_outcome_manifest_sha256")
    if (
        not limited
        and expected_outcome_hash is not None
        and outcome_manifest_hash != str(expected_outcome_hash).lower()
    ):
        raise common.ContractError("Case-outcome manifest hash drift")
    result = {
        "version": "2026-07-29-step28-v13-parser-template-fixture-result-v3",
        "status": "LIMITED_DEBUG_ONLY" if limited else "PASS",
        "scientific_metrics_produced": False,
        "dependency_hashes": dependency_hashes,
        "runner_sha256": common.sha256_file(Path(__file__)),
        "effective_style_count": len(styles),
        "effective_style_manifest_sha256": style_manifest_hash,
        "registry_hashes": registry_hashes,
        "fixture_seller_count": len(profiles),
        "case_count": observed_total,
        "expected_full_case_count": expected_total,
        "family_counts": dict(sorted(family_counts.items())),
        "parser_row_counts": dict(sorted(parser_row_counts.items())),
        "redaction_diagnostics": dict(sorted(redaction_diagnostics.items())),
        "case_outcome_manifest_sha256": outcome_manifest_hash,
        "profile_contact_chain": profile_chain,
        "parser_only_redactor_incompatible_stress": parser_only_stress,
        "gates": {
            "exact_parser_rows_and_flags": True,
            "zero_unexpected_parser_rows": True,
            "title_only_zero_parser_rows": True,
            "production_redactor_identity_removal": True,
            "must_ignore_preservation": True,
            "base_and_guard_preservation": True,
            "all_split_style_product_attribute_coverage": not limited,
            "all_title_modifiers_covered": (
                not limited
                and title_modifier_coverage
                == set(template["generic_lexicon"]["title_modifiers"])
            ),
        },
    }
    if output_path is not None:
        common.write_json(output_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_policy_argument(parser)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE_PATH)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--maximum-cases",
        type=int,
        help="Debug only; a limited run can never satisfy the full fixture gate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.maximum_cases is not None and args.maximum_cases <= 0:
        raise common.ContractError("--maximum-cases must be positive")
    output = args.output
    if output is None:
        fixture = common.load_json(args.fixture)
        output = common.repo_path(
            fixture["full_render_context_contract"]["output_manifest"][
                "path"
            ]
        )
    result = run_fixture(
        policy_path=args.policy,
        fixture_path=args.fixture,
        output_path=output,
        maximum_cases=args.maximum_cases,
    )
    print(
        "Step28-v13 parser/template fixture "
        f"{result['status']}: cases={result['case_count']} "
        f"styles={result['effective_style_count']} "
        f"manifest={result['case_outcome_manifest_sha256']}"
    )


if __name__ == "__main__":
    main()
