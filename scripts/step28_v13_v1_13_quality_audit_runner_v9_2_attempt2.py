#!/usr/bin/env python3
"""Run V9.2 quality attempt 2 with the frozen endpoint-count repair only."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any

import step28_v13_common as common
import step28_v13_v1_13_quality_audit_runner_v9_2 as base
import step28_v13_v1_13_quality_probe_preparer_v9 as preparer
import step28_v13_v1_13_quality_truth_capability_v9_2 as truth_capability


VERSION = "2026-08-24-step28-v13-v1-13-quality-audit-runner-v9-2-attempt2"
ROOT = Path(__file__).resolve().parents[1]
AUTHORIZATION_PATH = (
    ROOT
    / "private_custody"
    / "step28_v13_v1_13_v9_2_quality_run_attempt2_authorization.json"
)
CONSUMED_AUTHORIZATION_PATH = AUTHORIZATION_PATH.with_name(
    AUTHORIZATION_PATH.stem + ".consumed.json"
)
EXPECTED_PAIR_COUNT_PER_WORLD = 378
EXPECTED_BASE_RUNNER_SHA256 = (
    "60c11b5fcc3ee2e20b1ea92c3f339430f5397aa33f59e4035fecb1b9c4be4357"
)
EXPECTED_PREPARER_SHA256 = (
    "d0c136b712e0aa7c5feba0aa23430c2457efd1e05975a5f5c4245377bc658900"
)

_ORIGINAL_ENDPOINT_VALIDATOR = preparer._validate_endpoints
_ORIGINAL_LOAD_AUTHORIZATION = base.load_run_authorization
_ORIGINAL_BASE_AUTHORIZATION_PATH = base.AUTHORIZATION_PATH
_ORIGINAL_TRUTH_AUTHORIZATION_PATH = (
    truth_capability.EXPECTED_CONSUMED_QUALITY_AUTHORIZATION_PATH
)
_ORIGINAL_BASE_VERSION = base.VERSION


class QualityAuditAttempt2Error(RuntimeError):
    """Raised when the frozen attempt-2 repair boundary drifts."""


def _validate_endpoints_with_frozen_pair_count(
    endpoints: Sequence[Mapping[str, Any]],
    *,
    ordered_world_uids: Sequence[str],
    expected_pairs_per_world: int = EXPECTED_PAIR_COUNT_PER_WORLD,
) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...], dict[str, set[str]]]:
    """Supply the one argument omitted by the attempt-1 public-closure call."""

    if expected_pairs_per_world != EXPECTED_PAIR_COUNT_PER_WORLD:
        raise QualityAuditAttempt2Error("Frozen per-world pair count drift")
    return _ORIGINAL_ENDPOINT_VALIDATOR(
        endpoints,
        ordered_world_uids=ordered_world_uids,
        expected_pairs_per_world=EXPECTED_PAIR_COUNT_PER_WORLD,
    )


def _load_attempt2_authorization() -> tuple[dict[str, Any], dict[str, Any]]:
    return _ORIGINAL_LOAD_AUTHORIZATION(AUTHORIZATION_PATH)


def _verify_frozen_base() -> None:
    if (
        common.sha256_file(Path(base.__file__)) != EXPECTED_BASE_RUNNER_SHA256
        or common.sha256_file(Path(preparer.__file__)) != EXPECTED_PREPARER_SHA256
        or base.preparer_v9 is not preparer
        or preparer._validate_endpoints is not _ORIGINAL_ENDPOINT_VALIDATOR
        or base.AUTHORIZATION_PATH.resolve()
        != _ORIGINAL_BASE_AUTHORIZATION_PATH.resolve()
        or truth_capability.EXPECTED_CONSUMED_QUALITY_AUTHORIZATION_PATH.resolve()
        != _ORIGINAL_TRUTH_AUTHORIZATION_PATH.resolve()
        or base.VERSION != _ORIGINAL_BASE_VERSION
    ):
        raise QualityAuditAttempt2Error("Frozen attempt-1 base drift")


@contextmanager
def _installed_direct_repair() -> Iterator[None]:
    """Install only the missing-argument and fresh-receipt bindings."""

    _verify_frozen_base()
    preparer._validate_endpoints = _validate_endpoints_with_frozen_pair_count
    base.AUTHORIZATION_PATH = AUTHORIZATION_PATH
    base.load_run_authorization = _load_attempt2_authorization
    base.VERSION = VERSION
    truth_capability.EXPECTED_CONSUMED_QUALITY_AUTHORIZATION_PATH = (
        CONSUMED_AUTHORIZATION_PATH
    )
    try:
        yield
    finally:
        truth_capability.EXPECTED_CONSUMED_QUALITY_AUTHORIZATION_PATH = (
            _ORIGINAL_TRUTH_AUTHORIZATION_PATH
        )
        base.VERSION = _ORIGINAL_BASE_VERSION
        base.load_run_authorization = _ORIGINAL_LOAD_AUTHORIZATION
        base.AUTHORIZATION_PATH = _ORIGINAL_BASE_AUTHORIZATION_PATH
        preparer._validate_endpoints = _ORIGINAL_ENDPOINT_VALIDATOR


def run_quality_audit_attempt2() -> dict[str, Any]:
    with _installed_direct_repair():
        return base.run_formal_quality_audit()


def main() -> None:
    result = run_quality_audit_attempt2()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
