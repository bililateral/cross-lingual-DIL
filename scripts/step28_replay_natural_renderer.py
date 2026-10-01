#!/usr/bin/env python3
"""V9 capability-minimal natural-expression renderer for Step28-v13 v1.13.

This module is intentionally self-contained.  It performs no filesystem or
policy access and imports no project module.  Its only runtime authorities are
the canonical restricted-view bytes and one already-derived 32-byte candidate
key supplied by the trusted host.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


FIELD_SEPARATOR = b"\x1f"
RNG_DOMAIN = "step28-v13-v1.13-natural-variation"
VIEW_VERSION = "2026-08-14-step28-v13-v1-13-restricted-candidate-view-v9"
OUTPUT_VERSION = (
    "2026-08-14-step28-v13-v1-13-natural-candidate-v9-document-capacity-v1"
)
DESCRIPTION_SUFFIX = "{noise_clause}{context_guard}{identity_clause}"
BASE_SKELETON_COUNT = 8
TITLE_CODE_TWIN_SUFFIX = " 编号{code}"
DESCRIPTION_CODE_TWIN_INSERTION = "{separator}编号{code}{ending}"
ATTRIBUTE_ROTATION_DOMAIN = (
    "step28-v13-v1.13-v8.attribute.semantic-orbit.keyed-rotation-v2"
)
# These are admissible alternative states of one content slot, not synonym sets.
ATTRIBUTE_SEMANTIC_ORBITS = (
    ("标准版", "组合版", "多规格"),
    ("轻量版", "更新版", "通用版"),
    ("可选配色",),
    ("分批交付",),
    ("附使用说明",),
    ("支持自选参数",),
    ("含基础售后",),
)

ITEM_VIEW_FIELDS = (
    "item_handle",
    "code",
    "effective_style",
    "title_nonempty",
    "description_nonempty",
    "baseline_category",
    "baseline_product",
    "baseline_attribute",
    "baseline_delivery",
    "baseline_service",
    "baseline_title_skeleton_index",
    "baseline_description_skeleton_index",
)
STYLE_FIELDS = (
    "separator",
    "ending",
    "line_mode",
    "english_tag",
    "traditional_variant",
    "repeat_punctuation",
)
CANDIDATE_ITEM_FIELDS = (
    "item_handle",
    "category",
    "product",
    "attribute",
    "delivery",
    "service",
    "title_skeleton_index",
    "description_skeleton_index",
    "title",
    "base_description",
    "noise_clause",
)
SAFE_VIEW_FIELDS = (
    "version",
    "item_count",
    "items",
    "noise_targets",
    "safe_library",
    "safe_library_sha256",
)
SAFE_LIBRARY_FIELDS = (
    "categories",
    "category_products",
    "attributes",
    "delivery",
    "service",
    "title_modifiers",
    "title_skeletons",
    "description_skeletons",
    "traditional_substitutions",
    "must_ignore_templates",
    "must_ignore_values",
    "category_permutation_classes",
    "attribute_permutation_classes",
    "delivery_permutation_classes",
    "service_permutation_classes",
    "title_skeleton_permutation_classes",
    "description_skeleton_permutation_classes",
    "noise_template_permutation_classes",
    "noise_value_permutation_classes",
)


class PureNaturalVariationError(ValueError):
    """Fail-closed error raised by the capability-minimal renderer."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _decode_canonical(payload: bytes, *, label: str) -> Any:
    if not isinstance(payload, bytes):
        raise PureNaturalVariationError(f"{label} must be canonical bytes")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PureNaturalVariationError(f"{label} is not canonical UTF-8 JSON") from exc
    if canonical_bytes(value) != payload:
        raise PureNaturalVariationError(f"{label} is not in canonical byte form")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], fields: Sequence[str], *, label: str
) -> None:
    if set(value) != set(fields):
        raise PureNaturalVariationError(f"{label} keyset drift")


@dataclass(frozen=True)
class RestrictedCandidateView:
    """Canonical anonymous bytes accepted by the pure renderer."""

    view_bytes: bytes
    view_sha256: str

    def thaw(self) -> dict[str, Any]:
        value = _decode_canonical(self.view_bytes, label="restricted candidate view")
        if not isinstance(value, dict):
            raise PureNaturalVariationError("Restricted candidate view is not an object")
        if sha256_bytes(self.view_bytes) != self.view_sha256:
            raise PureNaturalVariationError("Restricted candidate-view hash drift")
        return value


@dataclass(frozen=True)
class NaturalExpressionCandidate:
    """Anonymous pure output carrying only its input commitments."""

    output_bytes: bytes
    output_sha256: str
    view_sha256: str
    candidate_key_sha256: str

    def thaw(self) -> dict[str, Any]:
        value = _decode_canonical(self.output_bytes, label="natural-expression candidate")
        if not isinstance(value, dict):
            raise PureNaturalVariationError("Natural-expression candidate is not an object")
        if sha256_bytes(self.output_bytes) != self.output_sha256:
            raise PureNaturalVariationError("Natural-expression candidate hash drift")
        return value


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
                raise PureNaturalVariationError(
                    f"Restricted view exposes forbidden key: {key}"
                )
            _scan_for_forbidden_view_content(child)
    elif isinstance(value, list):
        for child in value:
            _scan_for_forbidden_view_content(child)


def _capacity_twin(skeleton: str, *, description: bool) -> str:
    if "{code}" in skeleton:
        raise PureNaturalVariationError("A code-bearing skeleton has no capacity twin")
    if description:
        if not skeleton.endswith(DESCRIPTION_SUFFIX):
            raise PureNaturalVariationError("Description skeleton suffix drift")
        return (
            skeleton[: -len(DESCRIPTION_SUFFIX)]
            + DESCRIPTION_CODE_TWIN_INSERTION
            + DESCRIPTION_SUFFIX
        )
    return skeleton + TITLE_CODE_TWIN_SUFFIX


def _capacity_index_map(
    skeletons: Sequence[str], *, description: bool
) -> dict[int, int]:
    values = list(skeletons)
    if len(values) < BASE_SKELETON_COUNT:
        raise PureNaturalVariationError("Base skeleton domain is incomplete")
    base = values[:BASE_SKELETON_COUNT]
    expected_extras = [
        _capacity_twin(value, description=description)
        for value in base
        if "{code}" not in value
    ]
    if values[BASE_SKELETON_COUNT:] != expected_extras:
        raise PureNaturalVariationError("Capacity-twin skeleton extension drift")
    mapping: dict[int, int] = {}
    extra_index = BASE_SKELETON_COUNT
    for base_index, skeleton in enumerate(base):
        if "{code}" in skeleton:
            mapping[base_index] = base_index
        else:
            mapping[base_index] = extra_index
            extra_index += 1
    if (
        len(mapping) != BASE_SKELETON_COUNT
        or len(set(mapping.values())) != BASE_SKELETON_COUNT
        or any("{code}" not in values[target] for target in mapping.values())
    ):
        raise PureNaturalVariationError("Capacity-twin mapping is not injective")
    return mapping


def validate_safe_library(library: Mapping[str, Any]) -> None:
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
        raise PureNaturalVariationError("Safe category library is malformed")
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
            raise PureNaturalVariationError(f"Safe whitelist is malformed: {name}")
    attribute_values = [
        value for orbit in ATTRIBUTE_SEMANTIC_ORBITS for value in orbit
    ]
    if (
        len(attribute_values) != len(set(attribute_values))
        or set(attribute_values) != set(library["attributes"])
    ):
        raise PureNaturalVariationError(
            "Frozen attribute semantic orbits do not cover the safe library"
        )
    if len(library["title_modifiers"]) != 16:
        raise PureNaturalVariationError("Title-modifier domain must contain 16 values")
    for category in categories:
        values = products[category]
        if (
            not isinstance(values, list)
            or not values
            or len(values) != len(set(values))
            or any(not isinstance(value, str) or not value for value in values)
        ):
            raise PureNaturalVariationError(
                "Safe category/product whitelist is malformed"
            )
    substitutions = library["traditional_substitutions"]
    if (
        not isinstance(substitutions, dict)
        or not substitutions
        or any(
            not isinstance(key, str)
            or not isinstance(value, str)
            or len(key) != 1
            or len(value) != 1
            for key, value in substitutions.items()
        )
    ):
        raise PureNaturalVariationError(
            "Safe traditional-substitution table is malformed"
        )
    if any(
        not value.endswith(DESCRIPTION_SUFFIX)
        for value in library["description_skeletons"]
    ):
        raise PureNaturalVariationError("Safe description skeleton suffix drift")
    _capacity_index_map(library["title_skeletons"], description=False)
    _capacity_index_map(library["description_skeletons"], description=True)

    category_classes = library["category_permutation_classes"]
    flattened_categories = [value for group in category_classes for value in group]
    if (
        not isinstance(category_classes, list)
        or not category_classes
        or len(flattened_categories) != len(categories)
        or set(flattened_categories) != set(categories)
        or any(
            not isinstance(group, list)
            or not group
            or len(
                {
                    (
                        len(category),
                        len(products[category]),
                        tuple(len(value) for value in products[category]),
                    )
                    for category in group
                }
            )
            != 1
            for group in category_classes
        )
    ):
        raise PureNaturalVariationError("Safe category permutation classes drift")

    for field, values_field in (
        ("attribute_permutation_classes", "attributes"),
        ("delivery_permutation_classes", "delivery"),
        ("service_permutation_classes", "service"),
        ("noise_value_permutation_classes", "must_ignore_values"),
    ):
        classes = library[field]
        if not isinstance(classes, list) or any(
            not isinstance(group, list) or not group for group in classes
        ):
            raise PureNaturalVariationError(f"Safe permutation classes drift: {field}")
        flattened = [str(value) for group in classes for value in group]
        if (
            len(flattened) != len(library[values_field])
            or len(flattened) != len(set(flattened))
            or set(flattened) != set(library[values_field])
            or any(len({len(str(value)) for value in group}) != 1 for group in classes)
        ):
            raise PureNaturalVariationError(
                f"Safe length permutation classes drift: {field}"
            )
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
                    raise PureNaturalVariationError(
                        f"Safe punctuation permutation classes drift: {field}"
                    )

    for field, values_field in (
        ("title_skeleton_permutation_classes", "title_skeletons"),
        ("description_skeleton_permutation_classes", "description_skeletons"),
        ("noise_template_permutation_classes", "must_ignore_templates"),
    ):
        classes = library[field]
        if not isinstance(classes, list) or any(
            not isinstance(group, list) or not group for group in classes
        ):
            raise PureNaturalVariationError(f"Safe index classes drift: {field}")
        indices = [value for group in classes for value in group]
        if (
            any(type(value) is not int for value in indices)
            or sorted(indices) != list(range(len(library[values_field])))
            or len(indices) != len(set(indices))
        ):
            raise PureNaturalVariationError(f"Safe index classes drift: {field}")
    for group in library["noise_template_permutation_classes"]:
        if len(
            {
                len(library["must_ignore_templates"][index].format(value=""))
                for index in group
            }
        ) != 1:
            raise PureNaturalVariationError("Noise-template structure class drift")


def validate_restricted_view(value: Mapping[str, Any]) -> None:
    _require_exact_keys(value, SAFE_VIEW_FIELDS, label="restricted candidate view")
    if value["version"] != VIEW_VERSION:
        raise PureNaturalVariationError("Restricted candidate-view version drift")
    _scan_for_forbidden_view_content(value)
    library = value["safe_library"]
    if not isinstance(library, dict):
        raise PureNaturalVariationError("Restricted safe library is not an object")
    validate_safe_library(library)
    library_sha256 = sha256_bytes(canonical_bytes(library))
    if value["safe_library_sha256"] != library_sha256:
        raise PureNaturalVariationError("Restricted safe-library hash drift")
    items = value["items"]
    if (
        type(value["item_count"]) is not int
        or value["item_count"] <= 0
        or not isinstance(items, list)
        or len(items) != value["item_count"]
    ):
        raise PureNaturalVariationError("Restricted item cardinality drift")
    handles: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            raise PureNaturalVariationError("Restricted item is not an object")
        _require_exact_keys(item, ITEM_VIEW_FIELDS, label="restricted item")
        handle = item["item_handle"]
        style = item["effective_style"]
        if (
            not isinstance(handle, str)
            or not handle.startswith("h_item_")
            or handle in handles
            or not isinstance(item["code"], str)
            or len(item["code"]) != 11
            or not item["code"].startswith("Q")
            or any(value not in "ABCDEFGHIJKLMNOP" for value in item["code"][1:])
            or type(item["title_nonempty"]) is not bool
            or type(item["description_nonempty"]) is not bool
            or not isinstance(style, dict)
            or set(style) != set(STYLE_FIELDS)
            or not isinstance(style["separator"], str)
            or not isinstance(style["ending"], str)
            or style["line_mode"] not in {"single", "double", "bullet"}
            or not isinstance(style["english_tag"], str)
            or type(style["traditional_variant"]) is not bool
            or type(style["repeat_punctuation"]) is not bool
            or any(value != unicodedata.normalize("NFC", value) for value in style.values() if isinstance(value, str))
            or item["baseline_category"] not in library["categories"]
            or item["baseline_product"]
            not in library["category_products"][item["baseline_category"]]
            or item["baseline_attribute"] not in library["attributes"]
            or item["baseline_delivery"] not in library["delivery"]
            or item["baseline_service"] not in library["service"]
            or type(item["baseline_title_skeleton_index"]) is not int
            or not 0
            <= item["baseline_title_skeleton_index"]
            < BASE_SKELETON_COUNT
            or type(item["baseline_description_skeleton_index"]) is not int
            or not 0
            <= item["baseline_description_skeleton_index"]
            < BASE_SKELETON_COUNT
        ):
            raise PureNaturalVariationError("Restricted item content drift")
        handles.add(handle)
    noise_items: set[str] = set()
    noise_handles: set[str] = set()
    for row in value["noise_targets"]:
        if not isinstance(row, dict):
            raise PureNaturalVariationError("Restricted noise target is not an object")
        _require_exact_keys(
            row,
            (
                "noise_handle",
                "item_handle",
                "baseline_template_index",
                "baseline_value_index",
            ),
            label="noise target",
        )
        if (
            not isinstance(row["noise_handle"], str)
            or not row["noise_handle"].startswith("h_noise_")
            or row["noise_handle"] in noise_handles
            or row["item_handle"] not in handles
            or row["item_handle"] in noise_items
            or type(row["baseline_template_index"]) is not int
            or not 0
            <= row["baseline_template_index"]
            < len(library["must_ignore_templates"])
            or type(row["baseline_value_index"]) is not int
            or not 0
            <= row["baseline_value_index"]
            < len(library["must_ignore_values"])
        ):
            raise PureNaturalVariationError("Restricted noise-target lineage drift")
        noise_handles.add(row["noise_handle"])
        noise_items.add(row["item_handle"])


def _permutation_map(
    *, candidate_key: bytes, selector: str, values: Sequence[Any]
) -> dict[Any, Any]:
    source = list(values)
    if not source or len(source) != len(set(source)):
        raise PureNaturalVariationError("Candidate permutation domain is invalid")
    ranked = sorted(
        source,
        key=lambda value: (
            hmac.new(
                candidate_key,
                FIELD_SEPARATOR.join(
                    (
                        RNG_DOMAIN.encode("ascii"),
                        selector.encode("utf-8"),
                        str(value).encode("utf-8"),
                    )
                ),
                hashlib.sha256,
            ).digest(),
            str(value).encode("utf-8"),
        ),
    )
    return dict(zip(source, ranked, strict=True))


def _placeholder_signature(value: str) -> tuple[str, ...]:
    names = (
        "product",
        "attribute",
        "title_modifier",
        "code",
        "delivery",
        "service",
        "separator",
        "ending",
        "noise_clause",
        "context_guard",
        "identity_clause",
        "value",
    )
    return tuple(name for name in names if "{" + name + "}" in value)


def _traditional_response(value: str, table: Mapping[int, str]) -> bool:
    """Return whether the reachable traditional-style transform changes text."""

    return value.translate(table) != value


def _visible_shape(value: str) -> tuple[tuple[str, int], ...]:
    output: list[tuple[str, int]] = []
    for character in unicodedata.normalize("NFC", value):
        if character.isspace():
            kind = "space"
        elif character.isdigit():
            kind = "digit"
        elif character.isalpha():
            kind = "alpha"
        else:
            kind = "punctuation"
        output.append((kind, ord(character) if kind in {"space", "punctuation"} else 0))
    return tuple(output)


def _validate_attribute_map(
    mapping: Mapping[str, str], *, library: Mapping[str, Any]
) -> None:
    """Prove semantic, visible-shape, and cross-style equality closure."""

    values = tuple(str(value) for value in library["attributes"])
    if set(mapping) != set(values) or set(mapping.values()) != set(values):
        raise PureNaturalVariationError("Attribute map is not a full bijection")
    orbit_by_value = {
        value: orbit_index
        for orbit_index, orbit in enumerate(ATTRIBUTE_SEMANTIC_ORBITS)
        for value in orbit
    }
    substitutions = str.maketrans(library["traditional_substitutions"])
    visible = {
        value: (
            unicodedata.normalize("NFC", value),
            unicodedata.normalize("NFC", value.translate(substitutions)),
        )
        for value in values
    }
    for source, target in mapping.items():
        if orbit_by_value[source] != orbit_by_value[target]:
            raise PureNaturalVariationError("Attribute crossed its frozen semantic orbit")
        if tuple(_visible_shape(value) for value in visible[source]) != tuple(
            _visible_shape(value) for value in visible[target]
        ):
            raise PureNaturalVariationError(
                "Attribute changed its reachable visible structure"
            )
    states = tuple((value, style) for value in values for style in (0, 1))
    for left_value, left_style in states:
        for right_value, right_style in states:
            before_equal = visible[left_value][left_style] == visible[right_value][right_style]
            after_equal = (
                visible[mapping[left_value]][left_style]
                == visible[mapping[right_value]][right_style]
            )
            if before_equal != after_equal:
                raise PureNaturalVariationError(
                    "Attribute changed cross-style visible equality"
                )


def _attribute_rotation_map(
    *, candidate_key: bytes, library: Mapping[str, Any]
) -> dict[str, str]:
    """Key each semantic-orbit rotation on a separate, registry-blind domain."""

    output: dict[str, str] = {}
    for orbit_index, orbit in enumerate(ATTRIBUTE_SEMANTIC_ORBITS):
        values = tuple(orbit)
        if len(values) == 1:
            output[values[0]] = values[0]
            continue
        digest = hmac.new(
            candidate_key,
            FIELD_SEPARATOR.join(
                (
                    RNG_DOMAIN.encode("ascii"),
                    ATTRIBUTE_ROTATION_DOMAIN.encode("ascii"),
                    str(orbit_index).encode("ascii"),
                )
            ),
            hashlib.sha256,
        ).digest()
        # Identity is deliberately one legal orbit state.  Excluding it would
        # make every two-value orbit a constant swap for all candidate keys,
        # eliminating the candidate-to-candidate variation this repair needs.
        shift = int.from_bytes(digest[:8], "big") % len(values)
        output.update(
            {
                source: values[(source_index + shift) % len(values)]
                for source_index, source in enumerate(values)
            }
        )
    _validate_attribute_map(output, library=library)
    return output


def _refined_permutation_map(
    *,
    candidate_key: bytes,
    selector: str,
    groups: Sequence[Sequence[Any]],
    signature,
) -> dict[Any, Any]:
    """Permute only inside classes with identical rendering responses.

    The upstream safe library groups values by length and punctuation shape.
    V8 additionally freezes placeholder dependencies and the response to the
    reachable simplified-to-traditional transform.  Otherwise two baseline
    strings that are equal under different style settings can be split by an
    apparently bijective lexical permutation, changing cross-seller document
    frequency and therefore frozen profile-contribution lineage.
    """

    output: dict[Any, Any] = {}
    for outer_index, group in enumerate(groups):
        buckets: dict[tuple[Any, ...], list[Any]] = {}
        for value in group:
            key = tuple(signature(value))
            buckets.setdefault(key, []).append(value)
        for key in sorted(buckets, key=canonical_bytes):
            values = buckets[key]
            mapped = _permutation_map(
                candidate_key=candidate_key,
                selector=(
                    f"{selector}-outer-{outer_index}-response-"
                    + sha256_bytes(canonical_bytes(key))
                ),
                values=values,
            )
            if set(output) & set(mapped):
                raise PureNaturalVariationError(
                    "Refined candidate permutation classes overlap"
                )
            output.update(mapped)
    return output


def _build_v9_permutation_maps(
    *, candidate_key: bytes, library: Mapping[str, Any]
) -> dict[str, dict[Any, Any]]:
    """Build deterministic label-free maps that preserve rendering responses."""

    substitutions = str.maketrans(library["traditional_substitutions"])

    def text_signature(value: Any) -> tuple[Any, ...]:
        text = str(value)
        return (_traditional_response(text, substitutions),)

    def skeleton_signature(values_field: str):
        def signature(index: Any) -> tuple[Any, ...]:
            text = str(library[values_field][int(index)])
            return (
                _placeholder_signature(text),
                _traditional_response(text, substitutions),
            )

        return signature

    category_groups = library["category_permutation_classes"]

    def category_signature(value: Any) -> tuple[Any, ...]:
        category = str(value)
        products = library["category_products"][category]
        return (
            _traditional_response(category, substitutions),
            tuple(_traditional_response(str(product), substitutions) for product in products),
        )

    category_map = _refined_permutation_map(
        candidate_key=candidate_key,
        selector="category-classes-v8",
        groups=category_groups,
        signature=category_signature,
    )

    text_maps = {
        name: _refined_permutation_map(
            candidate_key=candidate_key,
            selector=name + "-v8",
            groups=library[name],
            signature=text_signature,
        )
        for name in (
            "attribute_permutation_classes",
            "delivery_permutation_classes",
            "service_permutation_classes",
            "noise_value_permutation_classes",
        )
    }
    index_maps = {
        name: _refined_permutation_map(
            candidate_key=candidate_key,
            selector=name + "-v8",
            groups=library[name],
            signature=skeleton_signature(values_field),
        )
        for name, values_field in (
            ("title_skeleton_permutation_classes", "title_skeletons"),
            (
                "description_skeleton_permutation_classes",
                "description_skeletons",
            ),
            ("noise_template_permutation_classes", "must_ignore_templates"),
        )
    }

    categories_by_size: dict[int, list[str]] = {}
    for category, products in library["category_products"].items():
        categories_by_size.setdefault(len(products), []).append(str(category))
    product_index_maps: dict[int, dict[int, int]] = {}
    for size, categories in sorted(categories_by_size.items()):
        ordered_categories = sorted(categories, key=lambda value: value.encode("utf-8"))
        product_index_maps[size] = _refined_permutation_map(
            candidate_key=candidate_key,
            selector=f"product-index-domain-{size}-v8",
            groups=[list(range(size))],
            signature=lambda index, ordered_categories=ordered_categories: (
                tuple(
                    _traditional_response(
                        str(library["category_products"][category][int(index)]),
                        substitutions,
                    )
                    for category in ordered_categories
                ),
            ),
        )

    return {
        "category": category_map,
        "attribute": _attribute_rotation_map(
            candidate_key=candidate_key, library=library
        ),
        "delivery": text_maps["delivery_permutation_classes"],
        "service": text_maps["service_permutation_classes"],
        "title_skeleton": index_maps["title_skeleton_permutation_classes"],
        "description_skeleton": index_maps[
            "description_skeleton_permutation_classes"
        ],
        "noise_template": index_maps["noise_template_permutation_classes"],
        "noise_value": text_maps["noise_value_permutation_classes"],
        "product_index": product_index_maps,
    }


def _code_symbol_indexes(code: str) -> tuple[int, ...]:
    if (
        not isinstance(code, str)
        or len(code) != 11
        or not code.startswith("Q")
        or any(value not in "ABCDEFGHIJKLMNOP" for value in code[1:])
    ):
        raise PureNaturalVariationError("Parser-safe item code is malformed")
    return tuple(ord(value) - ord("A") for value in code[1:])


def _title_modifier(code: str, library: Mapping[str, Any]) -> str:
    return unicodedata.normalize(
        "NFC", str(library["title_modifiers"][_code_symbol_indexes(code)[-2]])
    )


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
        raise PureNaturalVariationError(f"Unknown line mode: {line_mode}")
    ending = str(style["ending"])
    if bool(style["repeat_punctuation"]):
        ending += ending
    return effective_separator, ending


def _transform_base(
    value: str,
    *,
    style: Mapping[str, Any],
    library: Mapping[str, Any],
    description: bool,
) -> str:
    output = value
    if description and style["line_mode"] == "bullet":
        output = "• " + output
    if bool(style["traditional_variant"]):
        output = output.translate(str.maketrans(library["traditional_substitutions"]))
    return unicodedata.normalize("NFC", output)


def _render_base_title(
    *,
    skeleton: str,
    product: str,
    attribute: str,
    code: str,
    style: Mapping[str, Any],
    library: Mapping[str, Any],
) -> str:
    output = skeleton.format(
        product=product,
        attribute=attribute,
        title_modifier=_title_modifier(code, library),
        code=code,
    )
    output = _transform_base(
        output, style=style, library=library, description=False
    )
    tag = str(style["english_tag"])
    if tag and _code_symbol_indexes(code)[-1] < 3:
        output += " " + tag
    return unicodedata.normalize("NFC", output)


def _render_base_description(
    *,
    skeleton: str,
    product: str,
    attribute: str,
    code: str,
    delivery: str,
    service: str,
    style: Mapping[str, Any],
    library: Mapping[str, Any],
) -> str:
    if not skeleton.endswith(DESCRIPTION_SUFFIX):
        raise PureNaturalVariationError(
            "Description skeleton does not have the frozen suffix"
        )
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
        output, style=style, library=library, description=True
    )


def render_candidate_natural_expressions(
    *, restricted_view: RestrictedCandidateView, candidate_key: bytes
) -> NaturalExpressionCandidate:
    """Render one anonymous candidate using exactly two runtime authorities."""

    if not isinstance(restricted_view, RestrictedCandidateView):
        raise PureNaturalVariationError("Restricted candidate-view type drift")
    if not isinstance(candidate_key, bytes) or len(candidate_key) != 32:
        raise PureNaturalVariationError("Candidate key must be exactly 32 bytes")
    view = restricted_view.thaw()
    validate_restricted_view(view)
    library = view["safe_library"]

    maps = _build_v9_permutation_maps(
        candidate_key=candidate_key, library=library
    )
    category_map = maps["category"]
    attribute_map = maps["attribute"]
    delivery_map = maps["delivery"]
    service_map = maps["service"]
    title_skeleton_map = maps["title_skeleton"]
    description_skeleton_map = maps["description_skeleton"]
    product_index_maps = maps["product_index"]

    rows: dict[str, dict[str, Any]] = {}
    for item in view["items"]:
        handle = str(item["item_handle"])
        baseline_category = str(item["baseline_category"])
        category = str(category_map[baseline_category])
        baseline_products = library["category_products"][baseline_category]
        product_index = baseline_products.index(item["baseline_product"])
        target_products = library["category_products"][category]
        mapped_index = int(product_index_maps[len(baseline_products)][product_index])
        rows[handle] = {
            "item_handle": handle,
            "category": category,
            "product": str(target_products[mapped_index]),
            "attribute": str(attribute_map[item["baseline_attribute"]]),
            "delivery": str(delivery_map[item["baseline_delivery"]]),
            "service": str(service_map[item["baseline_service"]]),
            "title_skeleton_index": int(
                title_skeleton_map[item["baseline_title_skeleton_index"]]
            ),
            "description_skeleton_index": int(
                description_skeleton_map[
                    item["baseline_description_skeleton_index"]
                ]
            ),
            "title": "",
            "base_description": "",
            "noise_clause": "",
        }

    title_capacity_map = _capacity_index_map(
        library["title_skeletons"], description=False
    )
    description_capacity_map = _capacity_index_map(
        library["description_skeletons"], description=True
    )
    view_items = {str(row["item_handle"]): row for row in view["items"]}
    for handle, row in rows.items():
        item = view_items[handle]
        style = dict(item["effective_style"])
        if item["description_nonempty"]:
            row["description_skeleton_index"] = description_capacity_map[
                row["description_skeleton_index"]
            ]
        elif item["title_nonempty"]:
            row["title_skeleton_index"] = title_capacity_map[
                row["title_skeleton_index"]
            ]
        else:
            raise PureNaturalVariationError(
                "V9 capacity parent contains a joint-empty item"
            )
        row["title"] = (
            _render_base_title(
                skeleton=library["title_skeletons"][row["title_skeleton_index"]],
                product=str(row["product"]),
                attribute=str(row["attribute"]),
                code=str(item["code"]),
                style=style,
                library=library,
            )
            if item["title_nonempty"]
            else ""
        )
        row["base_description"] = (
            _render_base_description(
                skeleton=library["description_skeletons"][
                    row["description_skeleton_index"]
                ],
                product=str(row["product"]),
                attribute=str(row["attribute"]),
                code=str(item["code"]),
                delivery=str(row["delivery"]),
                service=str(row["service"]),
                style=style,
                library=library,
            )
            if item["description_nonempty"]
            else ""
        )
        visible_carrier = str(row["title"]) + str(row["base_description"])
        if visible_carrier.count(str(item["code"])) < 1:
            raise PureNaturalVariationError("Item code did not survive its carrier")

    noise_template_map = {
        int(source): int(target)
        for source, target in maps["noise_template"].items()
    }
    noise_value_text_map = maps["noise_value"]
    noise_by_item: dict[str, str] = {}
    for target in view["noise_targets"]:
        item_handle = str(target["item_handle"])
        template_index = noise_template_map[int(target["baseline_template_index"])]
        baseline_value = library["must_ignore_values"][
            int(target["baseline_value_index"])
        ]
        clause = unicodedata.normalize(
            "NFC",
            library["must_ignore_templates"][template_index].format(
                value=str(noise_value_text_map[baseline_value])
            ),
        )
        if item_handle in noise_by_item:
            raise PureNaturalVariationError(
                "Candidate assigned two noise clauses to one item"
            )
        noise_by_item[item_handle] = clause
    for item_handle, clause in noise_by_item.items():
        rows[item_handle]["noise_clause"] = clause

    output_rows = [
        rows[handle]
        for handle in sorted(rows, key=lambda value: value.encode("utf-8"))
    ]
    for row in output_rows:
        _require_exact_keys(row, CANDIDATE_ITEM_FIELDS, label="candidate item output")
    visible_values = [
        str(row[field])
        for row in output_rows
        for field in (
            "category",
            "product",
            "attribute",
            "delivery",
            "service",
            "title",
            "base_description",
            "noise_clause",
        )
    ]
    forbidden_literals = set(rows) | {
        str(row["noise_handle"]) for row in view["noise_targets"]
    }
    forbidden_literals.add(candidate_key.hex())
    if any(
        literal and literal in visible
        for literal in forbidden_literals
        for visible in visible_values
    ):
        raise PureNaturalVariationError(
            "Anonymous selector or candidate key leaked into visible text"
        )
    output = {
        "version": OUTPUT_VERSION,
        "view_sha256": restricted_view.view_sha256,
        "item_count": len(output_rows),
        "items": output_rows,
    }
    output_bytes = canonical_bytes(output)
    return NaturalExpressionCandidate(
        output_bytes=output_bytes,
        output_sha256=sha256_bytes(output_bytes),
        view_sha256=restricted_view.view_sha256,
        candidate_key_sha256=hashlib.sha256(candidate_key).hexdigest(),
    )
