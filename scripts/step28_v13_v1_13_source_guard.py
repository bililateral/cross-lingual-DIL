#!/usr/bin/env python3
"""Source-only launcher for the Step28-v13 v1.13 transaction smoke.

This file is invoked by path, so Python reads it as the main source rather than
loading it from an import cache.  It rejects project bytecode, verifies the
internally pinned implementation bundle, and only then executes the transaction
source or its focused contract tests.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import sysconfig
import types
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
POLICY_PATH = ROOT / "schema" / "step28_v13_v1_13_split_transaction_policy.json"
SOURCE_PATH = SCRIPTS / "step28_v13_v1_13_split_transaction.py"
TEST_PATH = ROOT / "tests" / "step28_v13_v1_13_split_transaction_contracts.py"
POLICY_VERSION = "2026-08-10-step28-v13-v1-13-split-transaction-policy-v1"
ALLOWED_COMMANDS = ("--smoke", "--focused-tests")


class SourceGuardError(RuntimeError):
    """The source-only execution boundary is not clean or pinned."""


def _verify_interpreter_boundary() -> None:
    if not (
        sys.flags.isolated == 1
        and sys.flags.no_site == 1
        and sys.flags.ignore_environment == 1
        and sys.flags.no_user_site == 1
        and sys.dont_write_bytecode
    ):
        raise SourceGuardError("Source guard requires Python -I -S -B")
    if "site" in sys.modules:
        raise SourceGuardError("site module was preloaded before source guard")


def _interpreter_dependency_paths() -> list[str]:
    """Find package directories without importing site or executing .pth files."""

    candidates: list[Path] = []
    default_paths = sysconfig.get_paths()
    for key in ("purelib", "platlib"):
        value = default_paths.get(key)
        if value:
            candidates.append(Path(value))
    user_scheme = "nt_user" if os.name == "nt" else "posix_user"
    if user_scheme in sysconfig.get_scheme_names():
        for key in ("purelib", "platlib"):
            value = sysconfig.get_path(key, scheme=user_scheme)
            if value:
                candidates.append(Path(value))
    executable = Path(sys.executable).resolve()
    if os.name == "nt":
        candidates.append(executable.parent / "Lib" / "site-packages")
    else:
        version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        prefix = executable.parent.parent
        candidates.extend(
            (
                prefix / "lib" / version / "site-packages",
                prefix / "lib64" / version / "site-packages",
            )
        )
    output: list[str] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        output.append(str(resolved))
    return output


def _is_within_any(path: Path, roots: tuple[Path, ...]) -> bool:
    for root in roots:
        try:
            path.relative_to(root)
        except ValueError:
            continue
        return True
    return False


def _install_isolated_import_paths() -> None:
    """Keep stdlib ahead of packages and packages ahead of project source."""

    dependency_paths = _interpreter_dependency_paths()
    dependency_roots = {Path(value).resolve() for value in dependency_paths}
    interpreter_roots = tuple(
        dict.fromkeys(
            Path(value).resolve()
            for value in (
                sys.base_prefix,
                sys.prefix,
            )
        )
    )
    standard_paths: list[str] = []
    seen: set[Path] = set()
    scripts_root = SCRIPTS.resolve()
    for entry in sys.path:
        if not entry:
            raise SourceGuardError("Current-directory import path survived isolation")
        resolved = Path(entry).resolve()
        if resolved == scripts_root or resolved in dependency_roots:
            continue
        if not _is_within_any(resolved, interpreter_roots):
            raise SourceGuardError("Non-interpreter path survived startup isolation")
        if resolved in seen:
            continue
        seen.add(resolved)
        standard_paths.append(str(resolved))
    sys.path[:] = [*standard_paths, *dependency_paths, str(scripts_root)]


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise SourceGuardError(f"Duplicate JSON key: {key}")
        output[key] = value
    return output


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _plain_source_bytes(path: Path, *, label: str) -> bytes:
    try:
        before = os.lstat(path)
    except OSError as exc:
        raise SourceGuardError(f"{label} is unavailable") from exc
    reparse = bool(
        getattr(before, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or reparse:
        raise SourceGuardError(f"{label} is not a plain single-link source file")
    payload = path.read_bytes()
    after = os.lstat(path)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_nlink,
        getattr(value, "st_mtime_ns", None),
    )
    if identity(before) != identity(after) or len(payload) != before.st_size:
        raise SourceGuardError(f"{label} changed while being read")
    return payload


def _load_and_verify_bundle() -> tuple[dict[str, Any], bytes, bytes]:
    _verify_interpreter_boundary()
    if Path(__file__).resolve() != (SCRIPTS / Path(__file__).name).resolve():
        raise SourceGuardError("Source guard was not launched from its canonical path")
    preloaded = sorted(
        name
        for name in sys.modules
        if name.startswith("step28_") or name == "step3_build_seller_profiles"
    )
    if preloaded:
        raise SourceGuardError("Project module was preloaded before source guard")
    cached = sorted(
        path
        for base in (SCRIPTS, ROOT / "tests")
        for path in base.rglob("*.pyc")
    )
    if cached:
        raise SourceGuardError("Project bytecode cache exists before source guard")
    sys.dont_write_bytecode = True
    policy_bytes = _plain_source_bytes(POLICY_PATH, label="transaction policy")
    try:
        policy = json.loads(
            policy_bytes.decode("utf-8"), object_pairs_hook=_reject_duplicate_pairs
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceGuardError("Transaction policy is not strict UTF-8 JSON") from exc
    if not isinstance(policy, dict) or policy.get("version") != POLICY_VERSION:
        raise SourceGuardError("Transaction policy version drift")
    expected_self = policy.get("canonical_self_hash")
    unsigned = dict(policy)
    unsigned.pop("canonical_self_hash", None)
    if (
        not isinstance(expected_self, str)
        or hashlib.sha256(_canonical_bytes(unsigned)).hexdigest() != expected_self
    ):
        raise SourceGuardError("Transaction policy canonical self-hash drift")
    pins = policy.get("implementation_bundle_pins")
    if not isinstance(pins, dict) or set(pins) != {
        "trust_status",
        "external_parent_receipt_required",
        "source_guard",
        "source",
        "contract_test",
    }:
        raise SourceGuardError("Implementation bundle pin set drift")
    if (
        pins["trust_status"]
        != "INTERNAL_CLOSURE_ONLY_PENDING_EXTERNAL_GIT_PARENT"
        or pins["external_parent_receipt_required"] is not True
    ):
        raise SourceGuardError("Implementation bundle trust boundary drift")
    if policy.get("source_execution_boundary") != {
        "entrypoint": "scripts/step28_v13_v1_13_source_guard.py",
        "required_python_flags": ["-I", "-S", "-B"],
        "environment_isolation_required": True,
        "site_initialization_forbidden": True,
        "site_module_preloaded_forbidden": True,
        "third_party_paths_derived_from_interpreter_only": False,
        "user_site_reintroduced": True,
        "user_site_path_may_depend_on_os_environment": True,
        "pth_execution_forbidden": True,
        "interpreter_standard_library_before_scripts": True,
        "third_party_before_canonical_scripts": True,
        "third_party_dependency_bytes_pinned": False,
        "hostile_third_party_payload_protected": False,
        "hostile_or_environment_redirected_third_party_protected": False,
        "formal_environment_attestation_required": True,
        "main_script_source_only": True,
        "project_bytecode_cache_forbidden": True,
        "project_module_preload_forbidden": True,
        "canonical_scripts_path_unique": True,
        "ordinary_unittest_discovery_includes_focused_contracts": False,
        "ordinary_unittest_discovery_authoritative_for_stage4b": False,
        "focused_contracts_authoritative_entrypoint_only": True,
    }:
        raise SourceGuardError("Source execution boundary drift")
    payloads: dict[str, bytes] = {}
    for role, path in (
        ("source_guard", Path(__file__).resolve()),
        ("source", SOURCE_PATH),
        ("contract_test", TEST_PATH),
    ):
        pin = pins[role]
        if not isinstance(pin, dict) or set(pin) != {"path", "size_bytes", "sha256"}:
            raise SourceGuardError(f"Implementation pin is malformed: {role}")
        relative = path.relative_to(ROOT).as_posix()
        payload = _plain_source_bytes(path, label=role)
        if (
            pin["path"] != relative
            or type(pin["size_bytes"]) is not int
            or pin["size_bytes"] != len(payload)
            or pin["sha256"] != hashlib.sha256(payload).hexdigest()
        ):
            raise SourceGuardError(f"Implementation bytes drift: {role}")
        payloads[role] = payload
    _install_isolated_import_paths()
    return policy, payloads["source"], payloads["contract_test"]


def _run_smoke(source_bytes: bytes) -> int:
    namespace = {
        "__builtins__": __builtins__,
        "__file__": str(SOURCE_PATH),
        "__name__": "__main__",
        "__package__": None,
    }
    sys.argv[:] = [str(SOURCE_PATH)]
    exec(compile(source_bytes, str(SOURCE_PATH), "exec"), namespace)
    return 0


def _run_focused_tests(source_bytes: bytes, test_bytes: bytes) -> int:
    transaction_name = "step28_v13_v1_13_split_transaction"
    previous_transaction = sys.modules.get(transaction_name)
    transaction_module = types.ModuleType(transaction_name)
    transaction_module.__file__ = str(SOURCE_PATH)
    transaction_module.__package__ = None
    sys.modules[transaction_name] = transaction_module
    test_name = "guarded_contract_tests_v1_13"
    test_module = types.ModuleType(test_name)
    test_module.__file__ = str(TEST_PATH)
    test_module.__package__ = None
    test_module.__dict__["__stage4b_source_guard__"] = True
    sys.modules[test_name] = test_module
    try:
        exec(
            compile(source_bytes, str(SOURCE_PATH), "exec"),
            transaction_module.__dict__,
        )
        exec(compile(test_bytes, str(TEST_PATH), "exec"), test_module.__dict__)
        suite = unittest.defaultTestLoader.loadTestsFromModule(test_module)
        result = unittest.TextTestRunner(verbosity=1).run(suite)
    finally:
        sys.modules.pop(test_name, None)
        if previous_transaction is None:
            sys.modules.pop(transaction_name, None)
        else:
            sys.modules[transaction_name] = previous_transaction
    return 0 if result.wasSuccessful() else 1


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in ALLOWED_COMMANDS:
        raise SourceGuardError(
            "Usage: step28_v13_v1_13_source_guard.py "
            "{--smoke|--focused-tests}"
        )
    command = sys.argv[1]
    _policy, source_bytes, test_bytes = _load_and_verify_bundle()
    status = (
        _run_smoke(source_bytes)
        if command == "--smoke"
        else _run_focused_tests(source_bytes, test_bytes)
    )
    raise SystemExit(status)


if __name__ == "__main__":
    main()
