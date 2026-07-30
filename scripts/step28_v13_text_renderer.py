#!/usr/bin/env python3
"""Exact Step 28-v13 synthetic text renderer shared by fixtures and generators."""

from __future__ import annotations

import hashlib
import itertools
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

import step28_v13_common as common


DESCRIPTION_SUFFIX = "{noise_clause}{context_guard}{identity_clause}"


def effective_style_uid(style: Mapping[str, Any]) -> str:
    return "estyle_" + hashlib.sha256(common.canonical_json_bytes(dict(style))).hexdigest()


def reachable_effective_styles(template: Mapping[str, Any]) -> list[dict[str, Any]]:
    renderer = template["renderer_contract"]
    factor_order = list(renderer["style_factor_order"])
    domains = renderer["style_factor_domains"]
    output: dict[str, dict[str, Any]] = {}
    for prototype in template["style_prototypes"]:
        if set(prototype) != {"style_id", *factor_order}:
            raise common.ContractError("Style prototype schema drift")
        for selected in itertools.combinations(factor_order, 2):
            style = {name: prototype[name] for name in factor_order}
            for factor in selected:
                domain = list(domains[factor])
                try:
                    position = domain.index(style[factor])
                except ValueError as exc:
                    raise common.ContractError(
                        f"Style value is outside its frozen domain: {factor}"
                    ) from exc
                style[factor] = domain[(position + 1) % len(domain)]
            uid = effective_style_uid(style)
            prior = output.setdefault(uid, style)
            if prior != style:
                raise common.ContractError("Effective style UID collision")
    return [
        {"effective_style_uid": uid, **output[uid]}
        for uid in common.utf8_sort(output)
    ]


def _translation_table(template: Mapping[str, Any]) -> dict[int, str]:
    values = template["renderer_contract"]["traditional_substitutions"]
    if any(len(source) != 1 or len(target) != 1 for source, target in values.items()):
        raise common.ContractError("Traditional substitution must map one code point to one")
    return str.maketrans(values)


def context_guard_pool(template: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the registered, parser-isolating natural guard pool."""

    raw = template["identity_clause_templates"].get("context_guards")
    minimum = int(
        template["renderer_contract"]["identity_context_isolation"][
            "minimum_code_points_on_each_populated_identity_side"
        ]
    )
    if (
        not isinstance(raw, list)
        or len(raw) < 9
        or any(not isinstance(value, str) for value in raw)
    ):
        raise common.ContractError("Context guard pool is malformed")
    guards = tuple(unicodedata.normalize("NFC", value) for value in raw)
    if (
        len(guards) != len(set(guards))
        or any(not value or len(value) < minimum for value in guards)
        or any(value != raw[index] for index, value in enumerate(guards))
    ):
        raise common.ContractError(
            "Context guards must be unique, NFC and longer than the parser radius"
        )
    return guards


def context_guard_sequence(
    *,
    selector_uid: str,
    count: int,
    template: Mapping[str, Any],
) -> tuple[str, ...]:
    """Select distinct guards without consulting labels or structure secrets."""

    if not isinstance(selector_uid, str) or not selector_uid:
        raise common.ContractError("Context guard selector UID is empty")
    guards = context_guard_pool(template)
    if type(count) is not int or not 0 <= count <= len(guards):
        raise common.ContractError("Context guard count exceeds the frozen pool")
    selector = selector_uid.encode("utf-8")
    ranked = sorted(
        guards,
        key=lambda guard: (
            hashlib.sha256(
                selector
                + common.FIELD_SEPARATOR
                + guard.encode("utf-8")
            ).digest(),
            guard.encode("utf-8"),
        ),
    )
    return tuple(ranked[:count])


def _code_symbol_indexes(code: str) -> tuple[int, ...]:
    if (
        not isinstance(code, str)
        or len(code) != 11
        or not code.startswith("Q")
        or any(value not in "ABCDEFGHIJKLMNOP" for value in code[1:])
    ):
        raise common.ContractError("Parser-safe item code is malformed")
    return tuple(ord(value) - ord("A") for value in code[1:])


def title_modifier(code: str, template: Mapping[str, Any]) -> str:
    """Map one label-blind code symbol to a shared natural title modifier."""

    indexes = _code_symbol_indexes(code)
    raw = template["generic_lexicon"].get("title_modifiers")
    if (
        not isinstance(raw, list)
        or len(raw) != 16
        or any(not isinstance(value, str) or not value for value in raw)
        or len(set(raw)) != len(raw)
    ):
        raise common.ContractError("Title-modifier domain must contain 16 unique strings")
    return unicodedata.normalize("NFC", raw[indexes[-2]])


def english_tag_visible(code: str) -> bool:
    """Expose a seller tag on a label-blind 3/16 subset of item titles."""

    return _code_symbol_indexes(code)[-1] < 3


def _style_values(style: Mapping[str, Any]) -> tuple[str, str]:
    separator = str(style["separator"])
    line_mode = str(style["line_mode"])
    if line_mode == "single":
        effective_separator = separator
    elif line_mode == "double":
        effective_separator = separator + "\n"
    elif line_mode == "bullet":
        effective_separator = "\n• "
    else:
        raise common.ContractError(f"Unknown line mode: {line_mode}")
    ending = str(style["ending"])
    if bool(style["repeat_punctuation"]):
        ending += ending
    return effective_separator, ending


def _transform_base(
    value: str,
    *,
    style: Mapping[str, Any],
    template: Mapping[str, Any],
    description: bool,
) -> str:
    output = value
    if description and style["line_mode"] == "bullet":
        output = "• " + output
    if bool(style["traditional_variant"]):
        output = output.translate(_translation_table(template))
    return unicodedata.normalize("NFC", output)


def render_base_title(
    *,
    skeleton: str,
    product: str,
    attribute: str,
    code: str,
    style: Mapping[str, Any],
    template: Mapping[str, Any],
) -> str:
    unknown = set(style) - {
        "effective_style_uid",
        "separator",
        "ending",
        "line_mode",
        "english_tag",
        "traditional_variant",
        "repeat_punctuation",
    }
    if unknown:
        raise common.ContractError(f"Unknown effective-style fields: {sorted(unknown)}")
    output = skeleton.format(
        product=product,
        attribute=attribute,
        title_modifier=title_modifier(code, template),
        code=code,
    )
    output = _transform_base(
        output,
        style=style,
        template=template,
        description=False,
    )
    tag = str(style["english_tag"])
    if tag and english_tag_visible(code):
        output += " " + tag
    return unicodedata.normalize("NFC", output)


def render_base_description(
    *,
    skeleton: str,
    product: str,
    attribute: str,
    code: str,
    delivery: str,
    service: str,
    style: Mapping[str, Any],
    template: Mapping[str, Any],
) -> str:
    if not skeleton.endswith(DESCRIPTION_SUFFIX):
        raise common.ContractError("Description skeleton does not have the frozen suffix")
    separator, ending = _style_values(style)
    output = skeleton.format(
        product=product,
        attribute=attribute,
        code=code,
        delivery=delivery,
        service=service,
        separator=separator,
        ending=ending,
        noise_clause="",
        context_guard="",
        identity_clause="",
    )
    return _transform_base(
        output,
        style=style,
        template=template,
        description=True,
    )


def identity_clause(
    *,
    template_family: str,
    identity_type: str,
    normalized_value: str,
    template: Mapping[str, Any],
) -> str:
    try:
        skeleton = template["identity_clause_templates"][template_family][identity_type]
    except KeyError as exc:
        raise common.ContractError(
            f"Missing identity clause for {template_family}/{identity_type}"
        ) from exc
    return unicodedata.normalize("NFC", skeleton.format(value=normalized_value))


def must_ignore_clause(
    *,
    template_index: int,
    value: str,
    template: Mapping[str, Any],
) -> str:
    values = template["identity_clause_templates"]["must_ignore"]
    return unicodedata.normalize("NFC", values[template_index].format(value=value))


def render_description(
    *,
    base_description: str,
    noise_clause: str,
    identity_clauses: Sequence[str],
    selector_uid: str,
    template: Mapping[str, Any],
) -> str:
    if not base_description:
        if noise_clause or identity_clauses:
            raise common.ContractError("Empty descriptions cannot receive noise or identity slots")
        return ""
    output = base_description + noise_clause
    if not identity_clauses:
        return unicodedata.normalize("NFC", output)
    guards = context_guard_sequence(
        selector_uid=selector_uid,
        count=len(identity_clauses) + 1,
        template=template,
    )
    output += guards[0]
    for clause, guard in zip(identity_clauses, guards[1:], strict=True):
        output += str(clause) + guard
    return unicodedata.normalize("NFC", output)


def nibble_code(digest: bytes) -> str:
    """Map the first 10 hex nibbles to parser-safe A..P and prefix Q."""

    hex_value = digest.hex()[:10]
    return "Q" + "".join(chr(ord("A") + int(value, 16)) for value in hex_value)
