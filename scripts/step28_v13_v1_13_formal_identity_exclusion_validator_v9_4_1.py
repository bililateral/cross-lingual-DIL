#!/usr/bin/env python3
"""Privileged hash-only exclusion for sealed V9.4 method-root identities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


VERSION = "2026-08-29-step28-v13-v1-13-v9-4-1-formal-identity-exclusion-v1"
SPLITS = ("train", "development", "audit_a", "audit_b")
ROW_FIELDS = {
    "world_uid", "asset_uid", "identity_type", "value_sha256", "role",
    "mechanism", "seller_occurrences",
}
HEX64 = frozenset("0123456789abcdef")
_ISSUER = object()


class SealedIdentityExclusionError(ValueError):
    """Raised without returning any sealed identity row or hash."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(HEX64)
    )


class SealedMethodIdentityExclusion:
    """Keep method-root identity commitments sealed behind a disjointness check."""

    __slots__ = ("__value_hashes", "__asset_uids", "__audit", "__issuer")

    def __init__(
        self,
        *,
        value_hashes: frozenset[str],
        asset_uids: frozenset[str],
        audit: Mapping[str, Any],
        issuer: object,
    ) -> None:
        if issuer is not _ISSUER:
            raise SealedIdentityExclusionError("Direct capability construction forbidden")
        self.__value_hashes = value_hashes
        self.__asset_uids = asset_uids
        self.__audit = dict(audit)
        self.__issuer = issuer

    def require_disjoint(
        self,
        *,
        value_hashes: Iterable[str],
        asset_uids: Iterable[str],
    ) -> None:
        values = tuple(value_hashes)
        assets = tuple(asset_uids)
        if any(not _is_sha256(value) for value in values):
            raise SealedIdentityExclusionError("Formal identity hash schema drift")
        if any(not isinstance(value, str) or not value for value in assets):
            raise SealedIdentityExclusionError("Formal identity asset UID schema drift")
        if (
            set(values).intersection(self.__value_hashes)
            or set(assets).intersection(self.__asset_uids)
        ):
            raise SealedIdentityExclusionError(
                "Formal identity collides with the sealed method root"
            )

    def public_audit(self) -> dict[str, Any]:
        return dict(self.__audit)


def open_validator(
    *, method_private_root: Path, root_manifest: Mapping[str, Any],
) -> SealedMethodIdentityExclusion:
    commitments = {
        str(row["path"]): row
        for row in root_manifest.get("private_file_commitments", ())
    }
    expected_paths = {
        f"{split}/identity_plan.jsonl" for split in SPLITS
    }
    if not expected_paths.issubset(commitments):
        raise SealedIdentityExclusionError(
            "Method-root identity-plan commitments are incomplete"
        )
    value_hashes: set[str] = set()
    asset_uids: set[str] = set()
    split_audit: dict[str, Any] = {}
    for split in SPLITS:
        relative = f"{split}/identity_plan.jsonl"
        spec = commitments[relative]
        path = method_private_root / relative
        if (
            not path.is_file()
            or path.stat().st_size != spec["size_bytes"]
            or sha256_file(path) != spec["sha256"]
        ):
            raise SealedIdentityExclusionError(
                f"Sealed method identity-plan bytes drift: {split}"
            )
        rows = 0
        split_values: set[str] = set()
        split_assets: set[str] = set()
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                value = json.loads(line)
                if not isinstance(value, dict) or set(value) != ROW_FIELDS:
                    raise SealedIdentityExclusionError(
                        f"Sealed identity row schema drift: {split}:{line_number}"
                    )
                value_hash = value["value_sha256"]
                asset_uid = value["asset_uid"]
                if not _is_sha256(value_hash) or not isinstance(asset_uid, str):
                    raise SealedIdentityExclusionError(
                        f"Sealed identity projection drift: {split}:{line_number}"
                    )
                split_values.add(value_hash)
                split_assets.add(asset_uid)
                rows += 1
        if rows != len(split_values) or rows != len(split_assets):
            raise SealedIdentityExclusionError(
                f"Sealed identity multiplicity drift: {split}"
            )
        if value_hashes.intersection(split_values) or asset_uids.intersection(
            split_assets
        ):
            raise SealedIdentityExclusionError(
                "Method-root identity commitment crosses splits"
            )
        value_hashes.update(split_values)
        asset_uids.update(split_assets)
        split_audit[split] = {
            "source_path": relative,
            "source_sha256": spec["sha256"],
            "source_size_bytes": spec["size_bytes"],
            "row_count": rows,
            "semantic_projection_read_count": 1,
        }
    audit = {
        "version": VERSION,
        "status": "SEALED_METHOD_IDENTITY_EXCLUSION_READY",
        "projected_fields": ["asset_uid", "value_sha256"],
        "private_values_returned": 0,
        "private_rows_returned": 0,
        "split_audit": split_audit,
    }
    return SealedMethodIdentityExclusion(
        value_hashes=frozenset(value_hashes),
        asset_uids=frozenset(asset_uids),
        audit=audit,
        issuer=_ISSUER,
    )
