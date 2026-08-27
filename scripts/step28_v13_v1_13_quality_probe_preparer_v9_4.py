#!/usr/bin/env python3
"""Freeze V9.4 proxy matrices before any label connector is available."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Any

import numpy as np

import step28_v13_v1_13_model_visible_matrix_v9_4 as matrix_v94
import step28_v13_v1_13_model_visible_prebuild_source_v9_4 as source_v94
import step28_v13_v1_13_quality_probe_core_v9_4 as core_v94


VERSION = "2026-08-27-step28-v13-v1-13-quality-probe-preparer-v9-4"
FORMAL_WORLD_COUNT = 500
_PREPARER_ISSUER = object()


class QualityProbePreparerV94Error(core_v94.QualityProbeCoreV94Error):
    """Raised when label-free V9.4 matrix preparation drifts."""


@dataclass(frozen=True)
class PreparedSplit:
    split: str
    matrix: core_v94.FrozenMatrix
    world_commitments: tuple[tuple[str, str], ...]
    commitment: Mapping[str, Any]
    _issuer: object


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _prepare_split(
    *,
    worlds: Sequence[Mapping[str, Any]],
    noise_signatures: Sequence[Mapping[str, Any]],
    time_key_hex: str,
    expected_world_count: int,
    split_schedule_commitment_sha256: str,
    schedule_pair_audit_commitment_sha256: str,
    noise_signature_set_commitment_sha256: str,
) -> PreparedSplit:
    if type(expected_world_count) is not int or expected_world_count <= 0:
        raise QualityProbePreparerV94Error("Expected world count drift")
    for value in (
        split_schedule_commitment_sha256,
        schedule_pair_audit_commitment_sha256,
        noise_signature_set_commitment_sha256,
    ):
        if (
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise QualityProbePreparerV94Error("Upstream commitment drift")
    if len(worlds) != expected_world_count:
        raise QualityProbePreparerV94Error("Prepared split world count drift")
    if any(type(row) is not dict for row in worlds):
        raise QualityProbePreparerV94Error("Prepared world row type drift")
    if any(type(row) is not dict for row in noise_signatures):
        raise QualityProbePreparerV94Error("Prepared signature row type drift")
    try:
        time_key_bytes = bytes.fromhex(time_key_hex)
    except (TypeError, ValueError) as error:
        raise QualityProbePreparerV94Error("Prepared time key drift") from error
    if (
        not isinstance(time_key_hex, str)
        or len(time_key_hex) != 64
        or len(time_key_bytes) != 32
        or time_key_hex != time_key_hex.lower()
    ):
        raise QualityProbePreparerV94Error("Prepared time key drift")
    splits = {row.get("split") for row in worlds}
    if len(splits) != 1 or next(iter(splits)) not in source_v94.ALLOWED_SPLITS:
        raise QualityProbePreparerV94Error("Prepared split domain drift")
    split = next(iter(splits))
    if [row.get("world_ordinal") for row in worlds] != list(
        range(expected_world_count)
    ):
        raise QualityProbePreparerV94Error("Prepared world ordinal drift")
    world_uids = [row.get("world_uid") for row in worlds]
    if (
        any(not isinstance(value, str) or not value for value in world_uids)
        or len(world_uids) != len(set(world_uids))
        or world_uids
        != sorted(world_uids, key=lambda value: value.encode("utf-8"))
    ):
        raise QualityProbePreparerV94Error("Prepared world UID drift")
    all_seller_uids = [
        seller_uid
        for world in worlds
        for seller_uid in world.get("seller_uids", [])
    ]
    if len(all_seller_uids) != expected_world_count * 28 or len(
        all_seller_uids
    ) != len(set(all_seller_uids)):
        raise QualityProbePreparerV94Error("Prepared seller UID split collision")
    row_keys: list[tuple[str, str]] = []
    matrices: list[np.ndarray] = []
    world_commitments: list[tuple[str, str]] = []
    for world in worlds:
        rows = source_v94.build_truth_free_world_projection(
            world=world,
            noise_signatures=noise_signatures,
            time_key_hex=time_key_hex,
        )
        frozen = matrix_v94.freeze_matrix(rows, expected_row_count=378)
        if {key[0] for key in frozen.row_keys} != {world["world_uid"]}:
            raise QualityProbePreparerV94Error("Prepared world matrix key drift")
        row_keys.extend(frozen.row_keys)
        matrices.append(frozen.values)
        world_commitments.append(
            (world["world_uid"], frozen.joint_commitment_sha256)
        )
    values = np.ascontiguousarray(np.vstack(matrices), dtype=np.dtype("<f8"))
    matrix = core_v94.freeze_matrix(
        view="model_visible_14",
        values=values,
        row_keys=tuple(row_keys),
        column_names=matrix_v94.PAIR_FEATURES,
        take_ownership=True,
    )
    commitment_payload = {
        "version": VERSION,
        "split": split,
        "world_count": expected_world_count,
        "row_count": expected_world_count * 378,
        "world_source_sha256": _canonical_sha256(list(worlds)),
        "noise_signatures_sha256": _canonical_sha256(list(noise_signatures)),
        "time_key_commitment_sha256": hashlib.sha256(time_key_bytes).hexdigest(),
        "split_schedule_commitment_sha256": (
            split_schedule_commitment_sha256
        ),
        "schedule_pair_audit_commitment_sha256": (
            schedule_pair_audit_commitment_sha256
        ),
        "noise_signature_set_commitment_sha256": (
            noise_signature_set_commitment_sha256
        ),
        "world_commitments": tuple(world_commitments),
        "matrix_raw_f8_c_sha256": matrix.commitment["matrix_raw_f8_c_sha256"],
        "matrix_row_keys_sha256": matrix.commitment["row_keys_sha256"],
        "matrix_column_names_sha256": matrix.commitment["column_names_sha256"],
    }
    commitment_payload["prepared_commitment_sha256"] = _canonical_sha256(
        commitment_payload
    )
    prepared = PreparedSplit(
        split=split,
        matrix=matrix,
        world_commitments=tuple(world_commitments),
        commitment=MappingProxyType(commitment_payload),
        _issuer=_PREPARER_ISSUER,
    )
    verify_prepared_split(prepared, expected_world_count=expected_world_count)
    return prepared


def verify_prepared_split(
    prepared: PreparedSplit, *, expected_world_count: int = FORMAL_WORLD_COUNT
) -> None:
    if type(prepared) is not PreparedSplit:
        raise QualityProbePreparerV94Error("Prepared split capability type drift")
    core_v94.verify_frozen_matrix(prepared.matrix)
    expected_fields = (
        "version",
        "split",
        "world_count",
        "row_count",
        "world_source_sha256",
        "noise_signatures_sha256",
        "time_key_commitment_sha256",
        "split_schedule_commitment_sha256",
        "schedule_pair_audit_commitment_sha256",
        "noise_signature_set_commitment_sha256",
        "world_commitments",
        "matrix_raw_f8_c_sha256",
        "matrix_row_keys_sha256",
        "matrix_column_names_sha256",
        "prepared_commitment_sha256",
    )
    upstream_commitment_fields = (
        "world_source_sha256",
        "noise_signatures_sha256",
        "time_key_commitment_sha256",
        "split_schedule_commitment_sha256",
        "schedule_pair_audit_commitment_sha256",
        "noise_signature_set_commitment_sha256",
        "matrix_raw_f8_c_sha256",
        "matrix_row_keys_sha256",
        "matrix_column_names_sha256",
        "prepared_commitment_sha256",
    )
    if (
        type(prepared.commitment) is not MappingProxyType
        or prepared._issuer is not _PREPARER_ISSUER
        or tuple(prepared.commitment) != expected_fields
        or prepared.split not in source_v94.ALLOWED_SPLITS
        or prepared.commitment["version"] != VERSION
        or prepared.commitment["split"] != prepared.split
        or prepared.commitment["world_count"] != expected_world_count
        or prepared.commitment["row_count"] != expected_world_count * 378
        or prepared.matrix.values.shape
        != (expected_world_count * 378, len(matrix_v94.PAIR_FEATURES))
        or prepared.matrix.column_names != matrix_v94.PAIR_FEATURES
        or prepared.world_commitments != prepared.commitment["world_commitments"]
        or len(prepared.world_commitments) != expected_world_count
        or len({value[0] for value in prepared.world_commitments})
        != expected_world_count
        or any(
            type(prepared.commitment[field]) is not str
            or len(prepared.commitment[field]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in prepared.commitment[field]
            )
            for field in upstream_commitment_fields
        )
        or any(
            type(world_uid) is not str
            or not world_uid
            or type(commitment) is not str
            or len(commitment) != 64
            or any(
                character not in "0123456789abcdef"
                for character in commitment
            )
            for world_uid, commitment in prepared.world_commitments
        )
        or prepared.commitment["matrix_raw_f8_c_sha256"]
        != prepared.matrix.commitment["matrix_raw_f8_c_sha256"]
        or prepared.commitment["matrix_row_keys_sha256"]
        != prepared.matrix.commitment["row_keys_sha256"]
        or prepared.commitment["matrix_column_names_sha256"]
        != prepared.matrix.commitment["column_names_sha256"]
        or prepared.commitment["prepared_commitment_sha256"]
        != _canonical_sha256({
            key: prepared.commitment[key]
            for key in expected_fields[:-1]
        })
    ):
        raise QualityProbePreparerV94Error("Prepared split commitment drift")


def contract_payload() -> dict[str, Any]:
    return {
        "version": VERSION,
        "formal_world_count": FORMAL_WORLD_COUNT,
        "labels_available": False,
        "accepted_source": source_v94.VERSION,
        "public_formal_preparer": False,
        "output_view": "model_visible_14",
        "output_columns": list(matrix_v94.PAIR_FEATURES),
    }
