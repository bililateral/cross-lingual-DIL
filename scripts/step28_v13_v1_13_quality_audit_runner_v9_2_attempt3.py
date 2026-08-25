#!/usr/bin/env python3
"""Run V9.2 quality attempt 3 with two frozen wiring repairs only."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any

import step28_v13_common as common
import step28_v13_structure as structure
import step28_v13_v1_13_quality_audit_runner_v9_2 as base
import step28_v13_v1_13_quality_probe_preparer_v9 as preparer
import step28_v13_v1_13_quality_truth_capability_v9_2 as truth_capability


VERSION = "2026-08-25-step28-v13-v1-13-quality-audit-runner-v9-2-attempt3"
ROOT = Path(__file__).resolve().parents[1]
AUTHORIZATION_PATH = (
    ROOT
    / "private_custody"
    / "step28_v13_v1_13_v9_2_quality_run_attempt3_authorization.json"
)
CONSUMED_AUTHORIZATION_PATH = AUTHORIZATION_PATH.with_name(
    AUTHORIZATION_PATH.stem + ".consumed.json"
)
EXPECTED_PAIR_COUNT_PER_WORLD = 378
EXPECTED_WORLD_COUNTS = {
    "train": 500,
    "development": 500,
    "audit_a": 2,
    "audit_b": 2,
}
WORLD_UID_PARENT_MODE = "development_smoke"
EXPECTED_BASE_RUNNER_SHA256 = (
    "60c11b5fcc3ee2e20b1ea92c3f339430f5397aa33f59e4035fecb1b9c4be4357"
)
EXPECTED_PREPARER_SHA256 = (
    "d0c136b712e0aa7c5feba0aa23430c2457efd1e05975a5f5c4245377bc658900"
)
EXPECTED_STRUCTURE_SHA256 = (
    "bf37daba81f77f79fd90825ebb37ad60457ec57bc63eb8dd7a3818f4aeb179d2"
)
EXPECTED_BUILDER_V9_2_SHA256 = (
    "cccd5f79011d392bac5dc6ea14df1f8d50b516e4ec8d495f367e3c0b054390f9"
)

_ORIGINAL_ENDPOINT_VALIDATOR = preparer._validate_endpoints
_ORIGINAL_GLOBAL_ORDINALS = base._global_ordinals
_ORIGINAL_FREEZE_TRAIN_DEVELOPMENT = base._freeze_train_development
_ORIGINAL_LOAD_AUTHORIZATION = base.load_run_authorization
_ORIGINAL_BASE_AUTHORIZATION_PATH = base.AUTHORIZATION_PATH
_ORIGINAL_TRUTH_AUTHORIZATION_PATH = (
    truth_capability.EXPECTED_CONSUMED_QUALITY_AUTHORIZATION_PATH
)
_ORIGINAL_BASE_VERSION = base.VERSION
_ACTIVE_WORLD_ORDINAL_BY_UID: dict[str, int] | None = None


class QualityAuditAttempt3Error(RuntimeError):
    """Raised when the frozen attempt-3 repair boundary drifts."""


def _validate_endpoints_with_frozen_pair_count(
    endpoints: Sequence[Mapping[str, Any]],
    *,
    ordered_world_uids: Sequence[str],
    expected_pairs_per_world: int = EXPECTED_PAIR_COUNT_PER_WORLD,
) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...], dict[str, set[str]]]:
    """Supply the one argument omitted by the attempt-1 public-closure call."""

    if expected_pairs_per_world != EXPECTED_PAIR_COUNT_PER_WORLD:
        raise QualityAuditAttempt3Error("Frozen per-world pair count drift")
    return _ORIGINAL_ENDPOINT_VALIDATOR(
        endpoints,
        ordered_world_uids=ordered_world_uids,
        expected_pairs_per_world=EXPECTED_PAIR_COUNT_PER_WORLD,
    )


def _world_counts(policy: Mapping[str, Any]) -> dict[str, int]:
    try:
        raw = policy["design_scale"]["world_counts"]
    except (KeyError, TypeError) as exc:
        raise QualityAuditAttempt3Error("Frozen world-count authority is absent") from exc
    if not isinstance(raw, Mapping) or set(raw) != set(base.SPLITS):
        raise QualityAuditAttempt3Error("Frozen world-count schema drift")
    counts = {split: raw[split] for split in base.SPLITS}
    if (
        any(type(value) is not int or value <= 0 for value in counts.values())
        or counts != EXPECTED_WORLD_COUNTS
        or counts != base.builder_v9_2.WORLD_COUNTS
    ):
        raise QualityAuditAttempt3Error("Frozen world-count value drift")
    return counts


def _reconstruct_world_ordinals(
    *,
    policy: Mapping[str, Any],
    loaded: Mapping[str, Mapping[str, Any]],
    id_key_hex: str,
) -> dict[str, int]:
    """Rebuild the pre-split global ordinal from the frozen world UID authority."""

    counts = _world_counts(policy)
    if not isinstance(id_key_hex, str):
        raise QualityAuditAttempt3Error("Frozen ID authority type drift")
    total_worlds = sum(counts.values())
    ordinal_by_uid = {
        structure.base_uid(
            key_hex=id_key_hex,
            entity_kind="world",
            parent_uid_or_mode=WORLD_UID_PARENT_MODE,
            ordinal=ordinal,
        ): ordinal
        for ordinal in range(total_worlds)
    }
    if len(ordinal_by_uid) != total_worlds:
        raise QualityAuditAttempt3Error("Reconstructed world UID collision")

    observed_uids: set[str] = set()
    for split in base.SPLITS:
        split_data = loaded.get(split)
        if not isinstance(split_data, Mapping):
            raise QualityAuditAttempt3Error("Loaded split is absent")
        worlds = tuple(split_data.get("worlds", ()))
        if (
            len(worlds) != counts[split]
            or any(
                not isinstance(row, Mapping)
                or set(row) != {"world_uid", "split_ordinal"}
                or type(row["world_uid"]) is not str
                or not row["world_uid"]
                or type(row["split_ordinal"]) is not int
                for row in worlds
            )
            or [row["split_ordinal"] for row in worlds]
            != list(range(counts[split]))
        ):
            raise QualityAuditAttempt3Error("Frozen split world registry drift")
        split_uids = [str(row["world_uid"]) for row in worlds]
        if (
            len(split_uids) != len(set(split_uids))
            or observed_uids.intersection(split_uids)
            or any(value not in ordinal_by_uid for value in split_uids)
        ):
            raise QualityAuditAttempt3Error("Frozen world UID universe drift")
        observed_uids.update(split_uids)
    if observed_uids != set(ordinal_by_uid):
        raise QualityAuditAttempt3Error("Reconstructed world UID closure drift")
    return ordinal_by_uid


def _global_ordinals_from_frozen_identity(
    policy: Mapping[str, Any],
    split: str,
    worlds: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    """Return only ordinals from the already-closed UID bijection."""

    counts = _world_counts(policy)
    active = _ACTIVE_WORLD_ORDINAL_BY_UID
    rows = tuple(worlds)
    if (
        active is None
        or split not in base.SPLITS
        or len(rows) != counts[split]
        or any(
            not isinstance(row, Mapping)
            or set(row) != {"world_uid", "split_ordinal"}
            or row["world_uid"] not in active
            for row in rows
        )
    ):
        raise QualityAuditAttempt3Error("Active world ordinal authority drift")
    return {str(row["world_uid"]): active[str(row["world_uid"])] for row in rows}


def _freeze_train_development_with_reconstructed_world_ordinals(
    *,
    loaded: Mapping[str, Mapping[str, Any]],
    policy: Mapping[str, Any],
    run_capability: truth_capability.ConsumedQualityRunCapabilityV92,
) -> Any:
    """Install the complete UID-to-ordinal bijection only while matrices freeze."""

    global _ACTIVE_WORLD_ORDINAL_BY_UID
    if _ACTIVE_WORLD_ORDINAL_BY_UID is not None:
        raise QualityAuditAttempt3Error("World ordinal reconstruction is reentrant")
    reconstructed = _reconstruct_world_ordinals(
        policy=policy,
        loaded=loaded,
        id_key_hex=run_capability.private_key_hex("id_key_hex"),
    )
    _ACTIVE_WORLD_ORDINAL_BY_UID = reconstructed
    try:
        return _ORIGINAL_FREEZE_TRAIN_DEVELOPMENT(
            loaded=loaded,
            policy=policy,
            run_capability=run_capability,
        )
    finally:
        _ACTIVE_WORLD_ORDINAL_BY_UID = None


def _load_attempt3_authorization() -> tuple[dict[str, Any], dict[str, Any]]:
    return _ORIGINAL_LOAD_AUTHORIZATION(AUTHORIZATION_PATH)


def _verify_frozen_base() -> None:
    if (
        common.sha256_file(Path(base.__file__)) != EXPECTED_BASE_RUNNER_SHA256
        or common.sha256_file(Path(preparer.__file__)) != EXPECTED_PREPARER_SHA256
        or common.sha256_file(Path(structure.__file__)) != EXPECTED_STRUCTURE_SHA256
        or common.sha256_file(Path(base.builder_v9_2.__file__))
        != EXPECTED_BUILDER_V9_2_SHA256
        or base.preparer_v9 is not preparer
        or base.structure is not structure
        or preparer._validate_endpoints is not _ORIGINAL_ENDPOINT_VALIDATOR
        or base._global_ordinals is not _ORIGINAL_GLOBAL_ORDINALS
        or base._freeze_train_development is not _ORIGINAL_FREEZE_TRAIN_DEVELOPMENT
        or base.AUTHORIZATION_PATH.resolve()
        != _ORIGINAL_BASE_AUTHORIZATION_PATH.resolve()
        or truth_capability.EXPECTED_CONSUMED_QUALITY_AUTHORIZATION_PATH.resolve()
        != _ORIGINAL_TRUTH_AUTHORIZATION_PATH.resolve()
        or base.VERSION != _ORIGINAL_BASE_VERSION
        or _ACTIVE_WORLD_ORDINAL_BY_UID is not None
    ):
        raise QualityAuditAttempt3Error("Frozen attempt-1 base drift")


@contextmanager
def _installed_direct_repairs() -> Iterator[None]:
    """Install only the two wiring repairs and fresh attempt-3 receipt paths."""

    global _ACTIVE_WORLD_ORDINAL_BY_UID
    _verify_frozen_base()
    preparer._validate_endpoints = _validate_endpoints_with_frozen_pair_count
    base._global_ordinals = _global_ordinals_from_frozen_identity
    base._freeze_train_development = (
        _freeze_train_development_with_reconstructed_world_ordinals
    )
    base.AUTHORIZATION_PATH = AUTHORIZATION_PATH
    base.load_run_authorization = _load_attempt3_authorization
    base.VERSION = VERSION
    truth_capability.EXPECTED_CONSUMED_QUALITY_AUTHORIZATION_PATH = (
        CONSUMED_AUTHORIZATION_PATH
    )
    try:
        yield
    finally:
        _ACTIVE_WORLD_ORDINAL_BY_UID = None
        truth_capability.EXPECTED_CONSUMED_QUALITY_AUTHORIZATION_PATH = (
            _ORIGINAL_TRUTH_AUTHORIZATION_PATH
        )
        base.VERSION = _ORIGINAL_BASE_VERSION
        base.load_run_authorization = _ORIGINAL_LOAD_AUTHORIZATION
        base.AUTHORIZATION_PATH = _ORIGINAL_BASE_AUTHORIZATION_PATH
        base._freeze_train_development = _ORIGINAL_FREEZE_TRAIN_DEVELOPMENT
        base._global_ordinals = _ORIGINAL_GLOBAL_ORDINALS
        preparer._validate_endpoints = _ORIGINAL_ENDPOINT_VALIDATOR


def run_quality_audit_attempt3() -> dict[str, Any]:
    with _installed_direct_repairs():
        return base.run_formal_quality_audit()


def main() -> None:
    result = run_quality_audit_attempt3()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
