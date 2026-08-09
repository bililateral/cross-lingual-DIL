#!/usr/bin/env python3
"""Controller-blind deterministic uniform seller derangement for Step28 v1.12."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence


DOMAIN = b"step28-v13-v1.12-visible-style-counterfactual-derangement-v1"
FIELD_SEPARATOR = b"\x1f"
SELLER_COUNT = 28
MAXIMUM_ATTEMPTS = 1024
VALID_SPLITS = frozenset({"train", "development", "audit_a", "audit_b"})
UINT256_SIZE = 1 << 256


class StyleDerangementError(ValueError):
    """Raised when the frozen public derangement contract cannot close."""


@dataclass(frozen=True)
class StyleSourceDerangement:
    """One replayable 28-to-28 mapping in canonical target order."""

    split: str
    world_uid: str
    attempt: int
    seller_set_sha256: str
    mapping_sha256: str
    target_source_pairs: tuple[tuple[str, str], ...]

    def as_mapping(self) -> dict[str, str]:
        return dict(self.target_source_pairs)

    def as_rows(self) -> list[dict[str, str]]:
        return [
            {
                "split": self.split,
                "world_uid": self.world_uid,
                "target_seller_uid": target,
                "source_seller_uid": source,
            }
            for target, source in self.target_source_pairs
        ]


def _length_prefixed_utf8(value: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise StyleDerangementError("Derangement identifiers must be nonempty strings")
    encoded = value.encode("utf-8")
    if len(encoded) >= 1 << 32:
        raise StyleDerangementError("Derangement identifier exceeds uint32 length")
    return len(encoded).to_bytes(4, "big", signed=False) + encoded


def _seller_set_blob(ordered_sellers: Sequence[str]) -> bytes:
    return b"".join(_length_prefixed_utf8(value) for value in ordered_sellers)


class _Sha256CounterStream:
    """One attempt-local SHA-256 counter stream with unbiased randbelow."""

    def __init__(self, attempt_seed: bytes) -> None:
        if not isinstance(attempt_seed, bytes) or len(attempt_seed) != 32:
            raise StyleDerangementError("Attempt seed must be a raw SHA-256 digest")
        self._attempt_seed = attempt_seed
        self._counter = 0

    def _next_uint256(self) -> int:
        if self._counter >= 1 << 64:
            raise StyleDerangementError("SHA-256 block counter exhausted")
        block = hashlib.sha256(
            self._attempt_seed
            + self._counter.to_bytes(8, "big", signed=False)
        ).digest()
        self._counter += 1
        return int.from_bytes(block, "big", signed=False)

    def randbelow(self, upper: int) -> int:
        if type(upper) is not int or upper <= 0:
            raise StyleDerangementError("randbelow upper bound must be positive int")
        limit = UINT256_SIZE - (UINT256_SIZE % upper)
        while True:
            value = self._next_uint256()
            if value < limit:
                return value % upper


def _attempt_seed(
    *, split: str, world_uid: str, seller_blob: bytes, attempt: int
) -> bytes:
    if not 0 <= attempt < MAXIMUM_ATTEMPTS:
        raise StyleDerangementError("Derangement attempt is outside frozen range")
    return hashlib.sha256(
        FIELD_SEPARATOR.join(
            (
                DOMAIN,
                split.encode("ascii"),
                world_uid.encode("utf-8"),
                seller_blob,
                attempt.to_bytes(4, "big", signed=False),
            )
        )
    ).digest()


def _candidate_permutation(
    ordered_sellers: tuple[str, ...], *, attempt_seed: bytes
) -> tuple[str, ...]:
    values = list(ordered_sellers)
    stream = _Sha256CounterStream(attempt_seed)
    for index in range(len(values) - 1, 0, -1):
        selected = stream.randbelow(index + 1)
        values[index], values[selected] = values[selected], values[index]
    return tuple(values)


def _mapping_digest(pairs: Sequence[tuple[str, str]]) -> str:
    payload = b"".join(
        _length_prefixed_utf8(target) + _length_prefixed_utf8(source)
        for target, source in pairs
    )
    return hashlib.sha256(payload).hexdigest()


def build_style_source_derangement(
    *, split: str, world_uid: str, seller_uids: Sequence[str]
) -> StyleSourceDerangement:
    """Draw one uniform derangement without accepting any private input."""

    if split not in VALID_SPLITS:
        raise StyleDerangementError("Unknown split for style-source derangement")
    if not isinstance(world_uid, str) or not world_uid:
        raise StyleDerangementError("Derangement world_uid must be nonempty")
    if isinstance(seller_uids, (str, bytes)):
        raise StyleDerangementError("seller_uids must be a sequence of identifiers")
    supplied = tuple(seller_uids)
    if (
        len(supplied) != SELLER_COUNT
        or any(not isinstance(value, str) or not value for value in supplied)
        or len(set(supplied)) != SELLER_COUNT
    ):
        raise StyleDerangementError("Derangement requires 28 unique seller identifiers")
    ordered = tuple(sorted(supplied, key=lambda value: value.encode("utf-8")))
    seller_blob = _seller_set_blob(ordered)
    seller_set_sha256 = hashlib.sha256(seller_blob).hexdigest()
    for attempt in range(MAXIMUM_ATTEMPTS):
        candidate = _candidate_permutation(
            ordered,
            attempt_seed=_attempt_seed(
                split=split,
                world_uid=world_uid,
                seller_blob=seller_blob,
                attempt=attempt,
            ),
        )
        if any(target == source for target, source in zip(ordered, candidate)):
            continue
        pairs = tuple(zip(ordered, candidate))
        if (
            len(pairs) != SELLER_COUNT
            or {target for target, _source in pairs} != set(ordered)
            or {source for _target, source in pairs} != set(ordered)
        ):
            raise StyleDerangementError("Derangement bijection closure failed")
        return StyleSourceDerangement(
            split=split,
            world_uid=world_uid,
            attempt=attempt,
            seller_set_sha256=seller_set_sha256,
            mapping_sha256=_mapping_digest(pairs),
            target_source_pairs=pairs,
        )
    raise StyleDerangementError("No fixed-point-free mapping in 1024 attempts")
