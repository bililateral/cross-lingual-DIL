#!/usr/bin/env python3
"""Run unittest discovery and emit one machine-readable result document."""

from __future__ import annotations

import argparse
import json
import sys
import time
import unittest
from typing import Any


class StructuredResult(unittest.TextTestResult):
    """Track exact test IDs, including ambiguous subtest outcomes."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.started_ids: list[str] = []
        self.success_ids: list[str] = []
        self.failed_subtest_ids: list[str] = []
        self.skipped_subtest_ids: list[str] = []

    def startTest(self, test: unittest.case.TestCase) -> None:  # noqa: N802
        self.started_ids.append(test.id())
        super().startTest(test)

    def addSuccess(self, test: unittest.case.TestCase) -> None:  # noqa: N802
        self.success_ids.append(test.id())
        super().addSuccess(test)

    def addSubTest(  # noqa: N802
        self,
        test: unittest.case.TestCase,
        subtest: unittest.case._SubTest,
        err: tuple[type[BaseException], BaseException, Any] | None,
    ) -> None:
        if err is not None:
            self.failed_subtest_ids.append(subtest.id())
        super().addSubTest(test, subtest, err)

    def addSkip(  # noqa: N802
        self, test: unittest.case.TestCase, reason: str
    ) -> None:
        if isinstance(test, unittest.case._SubTest):
            self.skipped_subtest_ids.append(test.id())
        super().addSkip(test, reason)


def _ids(records: list[tuple[Any, str]]) -> list[str]:
    return sorted({test.id() for test, _detail in records})


def _split_skipped(
    result: StructuredResult,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Separate started-test skips from class/module fixture skip events."""

    started = set(result.started_ids)
    rows = sorted(
        ({"id": test.id(), "reason": reason} for test, reason in result.skipped),
        key=lambda row: (row["id"], row["reason"]),
    )
    return (
        [row for row in rows if row["id"] in started],
        [row for row in rows if row["id"] not in started],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-directory", default="tests")
    parser.add_argument("--pattern", default="test*.py")
    parser.add_argument("--top-level-directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suite = unittest.defaultTestLoader.discover(
        start_dir=args.start_directory,
        pattern=args.pattern,
        top_level_dir=args.top_level_directory,
    )
    started = time.perf_counter()
    runner = unittest.TextTestRunner(
        stream=sys.stderr,
        verbosity=2,
        resultclass=StructuredResult,
        buffer=True,
    )
    result = runner.run(suite)
    elapsed = time.perf_counter() - started
    failures = _ids(result.failures)
    errors = _ids(result.errors)
    skipped, fixture_skipped = _split_skipped(result)
    expected_failures = _ids(result.expectedFailures)
    unexpected_successes = sorted(test.id() for test in result.unexpectedSuccesses)
    payload = {
        "version": "2026-08-09-step28-v13-v1-12-unittest-json-v2",
        "tests_run": int(result.testsRun),
        "started_test_ids": sorted(result.started_ids),
        "success_ids": sorted(result.success_ids),
        "failure_ids": failures,
        "error_ids": errors,
        "skipped": skipped,
        "fixture_skipped": fixture_skipped,
        "expected_failure_ids": expected_failures,
        "unexpected_success_ids": unexpected_successes,
        "failed_subtest_ids": sorted(set(result.failed_subtest_ids)),
        "skipped_subtest_ids": sorted(set(result.skipped_subtest_ids)),
        "was_successful": result.wasSuccessful(),
        "wall_seconds": elapsed,
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
