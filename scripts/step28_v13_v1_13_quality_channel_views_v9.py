#!/usr/bin/env python3
"""Label-free Step28-v13 v1.13 v9 channel views and probe features.

This implementation-only module neither opens a dataset nor fits a model.  The
caller must construct the whitelisted projections before pair truth is opened.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Callable, Iterable, Mapping, Sequence, cast


VERSION = "2026-08-14-step28-v13-v1-13-quality-channel-views-v9"
CODE_PATTERN = r"Q[A-P]{10}"
CODE_RE = re.compile(rf"^{CODE_PATTERN}$")
CODE_TOKEN_RE = re.compile(rf"(?<![A-Z]){CODE_PATTERN}(?![A-Z])")
RAW_CODE_RE = re.compile(CODE_PATTERN)
MASK_TOKEN = "QXXXXXXXXXX"
CODE_PAYLOAD_LENGTH = 10
CODE_ALPHABET = tuple("ABCDEFGHIJKLMNOP")
SELLER_SLOT_COUNT = 28
ITEM_SLOT_COUNT = 8
PROFILE_FIELDS = (
    "category_concat_top",
    "signature_title_concat",
    "title_concat_top",
    "signature_description_concat",
    "description_concat_top",
)
PROFILE_CODE_QUANTITIES = (
    "occurrence_count",
    "distinct_count",
    "own_count",
    "foreign_count",
)
NUMERIC_DELTA_FIELDS = (
    "title_length_median",
    "description_length_median",
    "digit_ratio_mean",
    "punct_ratio_mean",
    "repeated_title_share",
    "repeated_description_share",
)
PUBLIC_SELLER_QUANTITIES = (
    "owned_code_count",
    "visible_code_occurrence_count",
    "title_code_occurrence_count",
    "description_code_occurrence_count",
    "foreign_distinct_code_count",
)
ITEM_FIELDS = ("title", "description")
ALL_VISIBLE_FIELDS = (*ITEM_FIELDS, *PROFILE_FIELDS)
PUBLIC_FEATURE_WIDTH = 2992


class QualityChannelViewError(ValueError):
    """Raised when a label-free channel-view contract is violated."""


@dataclass(frozen=True)
class RegisteredCodeSpan:
    """One AST-registered visible code span in a rendered field."""

    start: int
    end: int
    code: str
    ast_node_id: str


@dataclass(frozen=True)
class CodeOccurrence:
    """One public, model-visible code occurrence without generator causes."""

    code: str
    field: str
    is_own: bool


@dataclass(frozen=True)
class SellerCodeView:
    """Whitelisted seller-level input for the public code-channel probe."""

    owned_codes: tuple[str, ...]
    visible_occurrences: tuple[CodeOccurrence, ...]
    profile_occurrences: Mapping[str, tuple[CodeOccurrence, ...]]
    numeric_profile_deltas: Mapping[str, float]


def public_feature_names() -> tuple[str, ...]:
    names = [f"absdiff__{name}" for name in PUBLIC_SELLER_QUANTITIES]
    names.extend(f"sum__{name}" for name in PUBLIC_SELLER_QUANTITIES)
    names.extend(
        (
            "intersection_count__owned_codes",
            "jaccard__owned_codes",
            "intersection_count__visible_codes",
            "jaccard__visible_codes",
            "maximum__cross_code_position_match_ratio",
            "mean__cross_code_position_match_ratio",
            "maximum__cross_code_common_prefix_ratio",
            "mean__cross_code_common_prefix_ratio",
            "maximum__cross_code_common_suffix_ratio",
            "mean__cross_code_common_suffix_ratio",
        )
    )
    for position in range(CODE_PAYLOAD_LENGTH):
        names.extend(
            (
                f"total_variation__owned__payload_position_{position:02d}",
                f"dot_product__owned__payload_position_{position:02d}",
            )
        )
    for position in range(CODE_PAYLOAD_LENGTH):
        for symbol in CODE_ALPHABET:
            names.extend(
                (
                    f"absdiff__owned__payload_position_{position:02d}__symbol_{symbol}",
                    f"sum__owned__payload_position_{position:02d}__symbol_{symbol}",
                )
            )
    for position in range(CODE_PAYLOAD_LENGTH):
        for symbol in CODE_ALPHABET:
            names.extend(
                (
                    f"absdiff__visible_all_fields__payload_position_{position:02d}__symbol_{symbol}",
                    f"sum__visible_all_fields__payload_position_{position:02d}__symbol_{symbol}",
                )
            )
    for field in ALL_VISIBLE_FIELDS:
        for position in range(CODE_PAYLOAD_LENGTH):
            for symbol in CODE_ALPHABET:
                names.extend(
                    (
                        f"absdiff__visible_field__{field}__payload_position_{position:02d}__symbol_{symbol}",
                        f"sum__visible_field__{field}__payload_position_{position:02d}__symbol_{symbol}",
                    )
                )
    for field in PROFILE_FIELDS:
        for quantity in PROFILE_CODE_QUANTITIES:
            names.extend(
                (
                    f"absdiff__profile__{field}__{quantity}",
                    f"sum__profile__{field}__{quantity}",
                )
            )
    for field in PROFILE_FIELDS:
        names.extend(
            (
                f"intersection_count__profile__{field}__visible_codes",
                f"jaccard__profile__{field}__visible_codes",
            )
        )
    for field in PROFILE_FIELDS:
        names.extend(
            (
                f"absdiff__profile__{field}__own_code_survival_rate",
                f"sum__profile__{field}__own_code_survival_rate",
            )
        )
    for field in NUMERIC_DELTA_FIELDS:
        names.extend(
            (
                f"absdiff__full_minus_neutral__{field}",
                f"sum__full_minus_neutral__{field}",
            )
        )
    if len(names) != PUBLIC_FEATURE_WIDTH or len(set(names)) != PUBLIC_FEATURE_WIDTH:
        raise AssertionError(
            f"Public code feature schema is not exactly {PUBLIC_FEATURE_WIDTH}-wide"
        )
    return tuple(names)


def decoded_feature_names() -> tuple[str, ...]:
    names: list[str] = []
    for left in range(SELLER_SLOT_COUNT):
        for right in range(left + 1, SELLER_SLOT_COUNT):
            names.append(f"seller_slot_pair__{left:02d}__{right:02d}")
    names.extend(
        (
            "minimum__seller_slot",
            "maximum__seller_slot",
            "absdiff__seller_slot",
            "sum__seller_slot",
            "absdiff__item_slot_count",
            "sum__item_slot_count",
            "intersection_count__item_slot_mask",
            "union_count__item_slot_mask",
            "jaccard__item_slot_mask",
            "hamming_distance__item_slot_mask",
        )
    )
    if len(names) != 388 or len(set(names)) != 388:
        raise AssertionError("Decoded slot feature schema is not exactly 388-wide")
    return tuple(names)


PUBLIC_FEATURE_NAMES = public_feature_names()
DECODED_FEATURE_NAMES = decoded_feature_names()


def _validate_code(code: object) -> str:
    if not isinstance(code, str) or CODE_RE.fullmatch(code) is None:
        raise QualityChannelViewError("Malformed registered item code")
    return code


def mask_registered_code_spans(
    text: str,
    *,
    registered_codes: Iterable[str],
    registered_spans: Sequence[RegisteredCodeSpan],
) -> tuple[str, Mapping[str, int]]:
    """Mask only AST-registered spans and reject every unregistered code token."""

    if not isinstance(text, str):
        raise QualityChannelViewError("Visible text must be a string")
    if MASK_TOKEN in text:
        raise QualityChannelViewError("Mask token already occurs in full text")
    registered_sequence = tuple(_validate_code(code) for code in registered_codes)
    registered = set(registered_sequence)
    if not registered:
        raise QualityChannelViewError("Registered code inventory must be nonempty")
    if len(registered) != len(registered_sequence):
        raise QualityChannelViewError("Registered code inventory contains duplicates")
    unsorted_spans: list[RegisteredCodeSpan] = []
    for span in registered_spans:
        if (
            not isinstance(span, RegisteredCodeSpan)
            or isinstance(span.start, bool)
            or isinstance(span.end, bool)
            or not isinstance(span.start, int)
            or not isinstance(span.end, int)
            or not isinstance(span.ast_node_id, str)
        ):
            raise QualityChannelViewError("Registered code span schema is invalid")
        unsorted_spans.append(span)
    normalized: list[RegisteredCodeSpan] = []
    seen_node_ids: set[str] = set()
    previous_end = -1
    for span in sorted(unsorted_spans, key=lambda value: (value.start, value.end)):
        code = _validate_code(span.code)
        if code not in registered:
            raise QualityChannelViewError("AST span refers to an unregistered code")
        if (
            not 0 <= span.start < span.end <= len(text)
            or span.start < previous_end
        ):
            raise QualityChannelViewError("Registered code spans overlap or are invalid")
        if not span.ast_node_id or span.ast_node_id in seen_node_ids:
            raise QualityChannelViewError("AST code node IDs must be nonempty and unique")
        if text[span.start : span.end] != code:
            raise QualityChannelViewError("AST span bytes disagree with the registered code")
        previous_end = span.end
        seen_node_ids.add(span.ast_node_id)
        normalized.append(span)
    observed = [
        (match.start(), match.end(), match.group(0))
        for match in RAW_CODE_RE.finditer(text)
    ]
    expected = [(span.start, span.end, span.code) for span in normalized]
    if observed != expected:
        raise QualityChannelViewError("Visible code tokens and AST-registered spans differ")
    pieces: list[str] = []
    cursor = 0
    counts = {code: 0 for code in sorted(registered)}
    for span in normalized:
        pieces.extend((text[cursor : span.start], MASK_TOKEN))
        counts[span.code] += 1
        cursor = span.end
    pieces.append(text[cursor:])
    masked = "".join(pieces)
    if RAW_CODE_RE.search(masked) is not None or len(masked) != len(text):
        raise QualityChannelViewError("Literal code masking did not close")
    return masked, counts


def _validate_occurrences(
    occurrences: Sequence[CodeOccurrence],
    *,
    allowed_fields: set[str],
    owned: set[str],
) -> tuple[str, ...]:
    codes: list[str] = []
    for occurrence in occurrences:
        code = _validate_code(occurrence.code)
        if occurrence.field not in allowed_fields:
            raise QualityChannelViewError("Code occurrence has an invalid mounted field")
        if not isinstance(occurrence.is_own, bool) or occurrence.is_own != (code in owned):
            raise QualityChannelViewError("Own-code occurrence disagrees with inventory")
        codes.append(code)
    return tuple(codes)


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QualityChannelViewError(f"{name} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise QualityChannelViewError(f"{name} must be finite")
    return normalized


def _validate_seller_view(view: SellerCodeView) -> dict[str, object]:
    owned = tuple(_validate_code(code) for code in view.owned_codes)
    if not owned or len(set(owned)) != len(owned):
        raise QualityChannelViewError("Owned code inventory must be nonempty and unique")
    owned_set = set(owned)
    visible_codes = _validate_occurrences(
        view.visible_occurrences,
        allowed_fields={"title", "description"},
        owned=owned_set,
    )
    if not owned_set <= set(visible_codes):
        raise QualityChannelViewError("At least one owned code is not model-visible")
    if set(view.profile_occurrences) != set(PROFILE_FIELDS):
        raise QualityChannelViewError("Mounted profile field schema drift")
    profile_codes: dict[str, tuple[str, ...]] = {}
    profile_quantities: dict[str, dict[str, float]] = {}
    profile_survival: dict[str, float] = {}
    for field in PROFILE_FIELDS:
        codes = _validate_occurrences(
            view.profile_occurrences[field], allowed_fields={field}, owned=owned_set
        )
        code_set = set(codes)
        own_set = code_set & owned_set
        profile_codes[field] = codes
        profile_quantities[field] = {
            "occurrence_count": float(len(codes)),
            "distinct_count": float(len(code_set)),
            "own_count": float(sum(code in owned_set for code in codes)),
            "foreign_count": float(sum(code not in owned_set for code in codes)),
        }
        profile_survival[field] = len(own_set) / len(owned_set)
    if set(view.numeric_profile_deltas) != set(NUMERIC_DELTA_FIELDS):
        raise QualityChannelViewError("Numeric profile delta schema drift")
    numeric_deltas = {
        name: _finite_float(view.numeric_profile_deltas[name], name=name)
        for name in NUMERIC_DELTA_FIELDS
    }
    title_count = sum(o.field == "title" for o in view.visible_occurrences)
    description_count = sum(o.field == "description" for o in view.visible_occurrences)
    item_field_codes = {
        field: tuple(
            occurrence.code
            for occurrence in view.visible_occurrences
            if occurrence.field == field
        )
        for field in ITEM_FIELDS
    }
    field_occurrence_codes = {**item_field_codes, **profile_codes}
    mounted_occurrence_codes = tuple(
        code
        for field in ALL_VISIBLE_FIELDS
        for code in field_occurrence_codes[field]
    )
    if not mounted_occurrence_codes:
        raise QualityChannelViewError("Mounted code occurrence inventory is empty")
    return {
        "owned": owned_set,
        "visible": set(visible_codes),
        "quantities": {
            "owned_code_count": float(len(owned_set)),
            "visible_code_occurrence_count": float(len(visible_codes)),
            "title_code_occurrence_count": float(title_count),
            "description_code_occurrence_count": float(description_count),
            "foreign_distinct_code_count": float(len(set(visible_codes) - owned_set)),
        },
        "profile_codes": profile_codes,
        "profile_quantities": profile_quantities,
        "profile_survival": profile_survival,
        "numeric_deltas": numeric_deltas,
        "field_occurrence_codes": field_occurrence_codes,
        "mounted_occurrence_codes": mounted_occurrence_codes,
    }


def _jaccard(left: set[str] | set[int], right: set[str] | set[int]) -> float:
    union = left | right
    return 0.0 if not union else len(left & right) / len(union)


def _common_prefix_length(left: str, right: str) -> int:
    count = 0
    for left_symbol, right_symbol in zip(left, right):
        if left_symbol != right_symbol:
            break
        count += 1
    return count


def _cross_code_summaries(left: set[str], right: set[str]) -> tuple[float, ...]:
    if not left or not right:
        raise QualityChannelViewError("Cross-code summaries require two nonempty sets")
    position_matches: list[float] = []
    prefix_matches: list[float] = []
    suffix_matches: list[float] = []
    for left_code in sorted(left):
        for right_code in sorted(right):
            left_payload = left_code[1:]
            right_payload = right_code[1:]
            position_matches.append(
                sum(a == b for a, b in zip(left_payload, right_payload))
                / CODE_PAYLOAD_LENGTH
            )
            prefix_matches.append(
                _common_prefix_length(left_payload, right_payload) / CODE_PAYLOAD_LENGTH
            )
            suffix_matches.append(
                _common_prefix_length(left_payload[::-1], right_payload[::-1])
                / CODE_PAYLOAD_LENGTH
            )
    return (
        max(position_matches),
        sum(position_matches) / len(position_matches),
        max(prefix_matches),
        sum(prefix_matches) / len(prefix_matches),
        max(suffix_matches),
        sum(suffix_matches) / len(suffix_matches),
    )


def _position_histogram(
    codes: Iterable[str], position: int, *, allow_empty: bool = False
) -> tuple[float, ...]:
    normalized = tuple(codes)
    if not normalized and not allow_empty:
        raise QualityChannelViewError("Code histogram requires a nonempty set")
    counts = [0] * len(CODE_ALPHABET)
    for code in normalized:
        counts[ord(code[position + 1]) - ord("A")] += 1
    if not normalized:
        return tuple(0.0 for _symbol in CODE_ALPHABET)
    return tuple(count / len(normalized) for count in counts)


def _append_absolute_composition(
    values: list[float],
    left_codes: Iterable[str],
    right_codes: Iterable[str],
    *,
    allow_empty: bool,
) -> None:
    for position in range(CODE_PAYLOAD_LENGTH):
        left_histogram = _position_histogram(
            left_codes, position, allow_empty=allow_empty
        )
        right_histogram = _position_histogram(
            right_codes, position, allow_empty=allow_empty
        )
        for left_value, right_value in zip(left_histogram, right_histogram):
            values.extend((abs(left_value - right_value), left_value + right_value))


def build_public_code_pair_features(
    left: SellerCodeView, right: SellerCodeView
) -> tuple[float, ...]:
    """Build the frozen exchange-symmetric public code-channel view."""

    left_view = _validate_seller_view(left)
    right_view = _validate_seller_view(right)
    left_quantities = cast(dict[str, float], left_view["quantities"])
    right_quantities = cast(dict[str, float], right_view["quantities"])
    values: list[float] = []
    values.extend(
        abs(left_quantities[name] - right_quantities[name])
        for name in PUBLIC_SELLER_QUANTITIES
    )
    values.extend(
        left_quantities[name] + right_quantities[name]
        for name in PUBLIC_SELLER_QUANTITIES
    )
    left_owned = cast(set[str], left_view["owned"])
    right_owned = cast(set[str], right_view["owned"])
    left_visible = cast(set[str], left_view["visible"])
    right_visible = cast(set[str], right_view["visible"])
    values.extend(
        (
            float(len(left_owned & right_owned)),
            _jaccard(left_owned, right_owned),
            float(len(left_visible & right_visible)),
            _jaccard(left_visible, right_visible),
        )
    )
    values.extend(_cross_code_summaries(left_visible, right_visible))
    left_histograms: list[tuple[float, ...]] = []
    right_histograms: list[tuple[float, ...]] = []
    for position in range(CODE_PAYLOAD_LENGTH):
        left_histogram = _position_histogram(left_owned, position)
        right_histogram = _position_histogram(right_owned, position)
        left_histograms.append(left_histogram)
        right_histograms.append(right_histogram)
        values.extend(
            (
                0.5 * sum(abs(a - b) for a, b in zip(left_histogram, right_histogram)),
                sum(a * b for a, b in zip(left_histogram, right_histogram)),
            )
        )
    _append_absolute_composition(
        values, left_owned, right_owned, allow_empty=False
    )
    left_mounted = cast(tuple[str, ...], left_view["mounted_occurrence_codes"])
    right_mounted = cast(tuple[str, ...], right_view["mounted_occurrence_codes"])
    _append_absolute_composition(
        values, left_mounted, right_mounted, allow_empty=False
    )
    left_field_codes = cast(
        dict[str, tuple[str, ...]], left_view["field_occurrence_codes"]
    )
    right_field_codes = cast(
        dict[str, tuple[str, ...]], right_view["field_occurrence_codes"]
    )
    for field in ALL_VISIBLE_FIELDS:
        _append_absolute_composition(
            values,
            left_field_codes[field],
            right_field_codes[field],
            allow_empty=True,
        )
    left_profile_quantities = cast(
        dict[str, dict[str, float]], left_view["profile_quantities"]
    )
    right_profile_quantities = cast(
        dict[str, dict[str, float]], right_view["profile_quantities"]
    )
    for field in PROFILE_FIELDS:
        for quantity in PROFILE_CODE_QUANTITIES:
            left_value = left_profile_quantities[field][quantity]
            right_value = right_profile_quantities[field][quantity]
            values.extend((abs(left_value - right_value), left_value + right_value))
    left_profile_codes = cast(dict[str, tuple[str, ...]], left_view["profile_codes"])
    right_profile_codes = cast(dict[str, tuple[str, ...]], right_view["profile_codes"])
    for field in PROFILE_FIELDS:
        left_codes = set(left_profile_codes[field])
        right_codes = set(right_profile_codes[field])
        values.extend((float(len(left_codes & right_codes)), _jaccard(left_codes, right_codes)))
    left_survival = cast(dict[str, float], left_view["profile_survival"])
    right_survival = cast(dict[str, float], right_view["profile_survival"])
    for field in PROFILE_FIELDS:
        values.extend(
            (
                abs(left_survival[field] - right_survival[field]),
                left_survival[field] + right_survival[field],
            )
        )
    left_deltas = cast(dict[str, float], left_view["numeric_deltas"])
    right_deltas = cast(dict[str, float], right_view["numeric_deltas"])
    for field in NUMERIC_DELTA_FIELDS:
        values.extend(
            (
                abs(left_deltas[field] - right_deltas[field]),
                left_deltas[field] + right_deltas[field],
            )
        )
    if len(values) != len(PUBLIC_FEATURE_NAMES) or not all(
        math.isfinite(value) for value in values
    ):
        raise QualityChannelViewError("Public code feature vector failed closure")
    return tuple(values)


def _validate_slot(value: object, *, upper_bound: int, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value < upper_bound
    ):
        raise QualityChannelViewError(f"{name} is outside its frozen range")
    return value


def _validate_item_slots(values: Sequence[int], *, name: str) -> tuple[int, ...]:
    slots = tuple(
        _validate_slot(value, upper_bound=ITEM_SLOT_COUNT, name=name)
        for value in values
    )
    if not slots or len(set(slots)) != len(slots):
        raise QualityChannelViewError(f"{name} must be nonempty and unique")
    if tuple(sorted(slots)) != tuple(range(len(slots))):
        raise QualityChannelViewError(f"{name} must be consecutive from zero")
    return slots


def build_decoded_slot_pair_features(
    *,
    left_seller_slot: int,
    right_seller_slot: int,
    left_item_slots: Sequence[int],
    right_item_slots: Sequence[int],
) -> tuple[float, ...]:
    """Build the frozen 388-wide private decoded-slot-only view."""

    left_slot = _validate_slot(
        left_seller_slot, upper_bound=SELLER_SLOT_COUNT, name="left_seller_slot"
    )
    right_slot = _validate_slot(
        right_seller_slot, upper_bound=SELLER_SLOT_COUNT, name="right_seller_slot"
    )
    if left_slot == right_slot:
        raise QualityChannelViewError("A seller pair must use two different slots")
    left_items = _validate_item_slots(left_item_slots, name="left_item_slots")
    right_items = _validate_item_slots(right_item_slots, name="right_item_slots")
    minimum = min(left_slot, right_slot)
    maximum = max(left_slot, right_slot)
    pair_name = f"seller_slot_pair__{minimum:02d}__{maximum:02d}"
    values = [0.0] * (SELLER_SLOT_COUNT * (SELLER_SLOT_COUNT - 1) // 2)
    values[DECODED_FEATURE_NAMES.index(pair_name)] = 1.0
    left_mask = set(left_items)
    right_mask = set(right_items)
    intersection = left_mask & right_mask
    union = left_mask | right_mask
    values.extend(
        (
            float(minimum),
            float(maximum),
            float(maximum - minimum),
            float(maximum + minimum),
            float(abs(len(left_items) - len(right_items))),
            float(len(left_items) + len(right_items)),
            float(len(intersection)),
            float(len(union)),
            len(intersection) / len(union),
            float(len(left_mask ^ right_mask)),
        )
    )
    if len(values) != len(DECODED_FEATURE_NAMES) or not all(
        math.isfinite(value) for value in values
    ):
        raise QualityChannelViewError("Decoded slot feature vector failed closure")
    return tuple(values)


def assert_neutralized_code_permutation_invariant(
    *,
    codes: Sequence[str],
    build_model_mount: Callable[[Mapping[str, str]], bytes],
) -> None:
    """Require a neutralized model mount to ignore a deterministic code rotation."""

    normalized = tuple(_validate_code(code) for code in codes)
    if not normalized or len(set(normalized)) != len(normalized):
        raise QualityChannelViewError("Permutation audit codes must be unique")
    baseline = {code: code for code in normalized}
    rotated_values = normalized[1:] + normalized[:1]
    permuted = dict(zip(normalized, rotated_values))
    baseline_bytes = build_model_mount(baseline)
    permuted_bytes = build_model_mount(permuted)
    if not isinstance(baseline_bytes, bytes) or not isinstance(permuted_bytes, bytes):
        raise QualityChannelViewError("Model-mount audit callback must return bytes")
    if baseline_bytes != permuted_bytes:
        raise QualityChannelViewError(
            "Neutralized model mount changes under a legal code permutation"
        )


def assert_all_derived_symbol_states_collapse(
    *, build_model_mount: Callable[[str], bytes]
) -> None:
    """Exhaust all 16x16 code-derived modifier/tag states."""

    observed: list[bytes] = []
    for second_last in CODE_ALPHABET:
        for last in CODE_ALPHABET:
            code = "Q" + ("A" * 8) + second_last + last
            _validate_code(code)
            value = build_model_mount(code)
            if not isinstance(value, bytes):
                raise QualityChannelViewError(
                    "Derived-symbol audit callback must return bytes"
                )
            observed.append(value)
    if len(observed) != 256 or len(set(observed)) != 1:
        raise QualityChannelViewError(
            "Neutralized model mount depends on a derived-symbol state"
        )
