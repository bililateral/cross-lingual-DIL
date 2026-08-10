from __future__ import annotations

if (
    __name__ != "guarded_contract_tests_v1_13"
    or globals().get("__stage4b_source_guard__") is not True
):
    raise RuntimeError(
        "Stage4B focused contracts may only execute through the source guard"
    )

import contextlib
import copy
import hashlib
import importlib._bootstrap_external as bootstrap_external
import importlib.machinery
import importlib.util
import inspect
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_split_transaction as subject
import step28_v13_common as common
import step28_v13_v1_13_candidate_selection as selection


class Step28V13V113SplitTransactionContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with subject._exclusive_split_lock():
            selector = selection.DevelopmentSmokeCandidateSelector()
            cls.accepted = selector.select()
            selector.validate_completed_candidate(cls.accepted)
        cls.policy = subject.load_policy()

    @contextlib.contextmanager
    def workspace(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / f"{subject.WORKSPACE_PREFIX}test"
            with subject._exclusive_split_lock() as token:
                _boundary, boundary_bytes = subject._initialize_workspace(
                    token, root, create=True
                )
                yield token, root, boundary_bytes

    def commit(self, token, root, boundary_bytes):
        return subject._commit_world(
            token,
            root,
            self.accepted,
            policy=self.policy,
            boundary_bytes=boundary_bytes,
        )

    def expected_state(self, boundary_bytes):
        return subject._expected_state_from_accepted(
            self.accepted,
            policy=self.policy,
            boundary_bytes=boundary_bytes,
        )

    def write_timestamp_pyc(self, source: Path, code_text: str) -> Path:
        stat_result = source.stat()
        code = compile(code_text, str(source), "exec")
        payload = bootstrap_external._code_to_timestamp_pyc(
            code,
            int(stat_result.st_mtime),
            stat_result.st_size,
        )
        cache = Path(importlib.util.cache_from_source(str(source)))
        cache.parent.mkdir(exist_ok=True)
        cache.write_bytes(payload)
        return cache

    def seal(self, token, root, boundary_bytes):
        marker, marker_bytes = self.commit(token, root, boundary_bytes)
        state = self.expected_state(boundary_bytes)
        self.assertEqual(marker, state.marker)
        self.assertEqual(marker_bytes, state.marker_bytes)
        subject._publish_final_and_seal(token, root, state)
        return state

    def test_policy_is_closed_smoke_only_and_self_hashed(self) -> None:
        policy = self.policy
        self.assertEqual(policy["expected_world_count"], 1)
        self.assertEqual(policy["allowed_mode"], "development_smoke")
        self.assertEqual(policy["allowed_split"], "audit_a")
        self.assertTrue(policy["smoke_projection_only"])
        self.assertFalse(policy["formal_split_semantics"])
        self.assertFalse(policy["formal_canonical_member_set_claimed"])
        self.assertTrue(all(value is False for value in policy["formal_authorizations"].values()))
        self.assertEqual(policy["stable_lock"]["windows_scope"], "same_logon_session_only")
        self.assertFalse(
            policy["stable_lock"][
                "hostile_cross_session_or_repository_root_swap_protected"
            ]
        )
        self.assertFalse(
            policy["publish_protocol"][
                "hostile_concurrent_parent_swap_protected"
            ]
        )
        unsigned = dict(policy)
        expected = unsigned.pop("canonical_self_hash")
        self.assertEqual(common.canonical_sha256(unsigned), expected)

    def test_every_security_policy_section_is_machine_closed(self) -> None:
        mutations = {
            "claim_boundary": lambda p: p.__setitem__("claim_boundary", "broadened"),
            "world_count_bool": lambda p: p.__setitem__("expected_world_count", True),
            "stable_lock": lambda p: p["stable_lock"].__setitem__("nonblocking", False),
            "source_execution": lambda p: p["source_execution_boundary"].__setitem__(
                "project_bytecode_cache_forbidden", False
            ),
            "allocation": lambda p: p["allocation_delta_semantics"].__setitem__(
                "derivation", "changed"
            ),
            "layout": lambda p: p["transaction_layout"].__setitem__(
                "final_directory", "other"
            ),
            "projection": lambda p: p["smoke_final_member_plan"][
                "worlds.jsonl"
            ].__setitem__("allowed_fields", ["world_uid", "extra"]),
            "jsonl": lambda p: p["jsonl_serialization"].__setitem__(
                "bom_forbidden", False
            ),
            "state_machine": lambda p: p["state_machine"].append("UNFROZEN"),
            "publish": lambda p: p["publish_protocol"].__setitem__(
                "replace_calls_forbidden", False
            ),
            "recovery": lambda p: p["recovery_trust"].__setitem__(
                "formal_500_world_authenticity_proven", True
            ),
            "synthetic_bool": lambda p: p["synthetic_architecture_tests"].__setitem__(
                "minimum_marker_chain_length", True
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                damaged = copy.deepcopy(self.policy)
                mutate(damaged)
                damaged.pop("canonical_self_hash")
                damaged["canonical_self_hash"] = common.canonical_sha256(damaged)
                with self.assertRaises(subject.SplitTransactionError):
                    subject._validate_policy(damaged)

    def test_candidate_policy_raw_self_and_source_pins_are_external(self) -> None:
        candidate_policy = subject._verify_frozen_candidate_sources()
        self.assertEqual(
            common.sha256_file(selection.DEFAULT_POLICY_PATH),
            subject.EXPECTED_CANDIDATE_POLICY_RAW_SHA256,
        )
        self.assertEqual(
            candidate_policy["canonical_self_hash"],
            subject.EXPECTED_CANDIDATE_POLICY_SELF_HASH,
        )
        self.assertEqual(
            common.sha256_file(Path(selection.__file__)),
            subject.EXPECTED_CANDIDATE_SOURCE_SHA256,
        )

    def test_transaction_source_and_contract_test_bundle_pins_are_exact(self) -> None:
        pins = self.policy["implementation_bundle_pins"]
        self.assertEqual(
            pins["trust_status"],
            "INTERNAL_CLOSURE_ONLY_PENDING_EXTERNAL_GIT_PARENT",
        )
        self.assertTrue(pins["external_parent_receipt_required"])
        for role, path in (
            ("source_guard", subject.SOURCE_GUARD_PATH),
            ("source", subject.TRANSACTION_SOURCE_PATH),
            ("contract_test", subject.TRANSACTION_TEST_PATH),
        ):
            self.assertEqual(pins[role]["path"], path.relative_to(ROOT).as_posix())
            self.assertEqual(pins[role]["size_bytes"], path.stat().st_size)
            self.assertEqual(pins[role]["sha256"], common.sha256_file(path))

        self.assertFalse(subject.TRANSACTION_TEST_PATH.name.startswith("test_"))

    def test_contract_file_direct_execution_fails_before_running_tests(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                "-B",
                str(subject.TRANSACTION_TEST_PATH),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        combined = completed.stdout + completed.stderr
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(
            "Stage4B focused contracts may only execute through the source guard",
            combined,
        )
        self.assertNotIn("Ran ", combined)

    def test_source_guard_rejects_accepted_timestamp_pyc_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            sentinel = Path(temporary) / "malicious-executed.txt"
            code = f"open(r'{sentinel}', 'w').write('executed')\n"
            cache = self.write_timestamp_pyc(subject.TRANSACTION_SOURCE_PATH, code)
            try:
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-I",
                        "-S",
                        "-B",
                        str(subject.SOURCE_GUARD_PATH),
                        "--smoke",
                    ],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            finally:
                cache.unlink(missing_ok=True)
                if cache.parent.is_dir() and not any(cache.parent.iterdir()):
                    cache.parent.rmdir()
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("bytecode cache exists", completed.stderr)
            self.assertFalse(sentinel.exists())

    def test_internal_bootstrap_rejects_preloaded_project_module(self) -> None:
        code = (
            "import sys,types;"
            f"sys.path.insert(0,r'{SCRIPTS}');"
            "sys.modules['step28_v13_common']=types.ModuleType('step28_v13_common');"
            "import step28_v13_v1_13_split_transaction"
        )
        completed = subprocess.run(
            [sys.executable, "-B", "-c", code],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("preloaded before frozen bootstrap", completed.stderr)

    def test_internal_bootstrap_rejects_dependency_timestamp_pyc(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            sentinel = Path(temporary) / "dependency-pyc-executed.txt"
            code_text = f"open(r'{sentinel}', 'w').write('executed')\n"
            cache = self.write_timestamp_pyc(Path(common.__file__), code_text)
            code = (
                "import sys;"
                f"sys.path.insert(0,r'{SCRIPTS}');"
                "import step28_v13_v1_13_split_transaction"
            )
            try:
                completed = subprocess.run(
                    [sys.executable, "-B", "-c", code],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            finally:
                cache.unlink(missing_ok=True)
                if cache.parent.is_dir() and not any(cache.parent.iterdir()):
                    cache.parent.rmdir()
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("bytecode cache exists", completed.stderr)
            self.assertFalse(sentinel.exists())

    def test_internal_bootstrap_rejects_noncanonical_sys_path_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fake = Path(temporary)
            sentinel = fake / "fake-imported.txt"
            (fake / "step28_v13_common.py").write_text(
                f"open(r'{sentinel}', 'w').write('executed')\n",
                encoding="utf-8",
            )
            code = (
                "import sys;"
                f"sys.path[:0]=[r'{fake}',r'{SCRIPTS}'];"
                "import step28_v13_v1_13_split_transaction"
            )
            completed = subprocess.run(
                [sys.executable, "-B", "-c", code],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("Non-interpreter import path precedes", completed.stderr)
            self.assertFalse(sentinel.exists())

    def test_source_guard_orders_stdlib_then_packages_then_scripts(self) -> None:
        namespace = {
            "__builtins__": __builtins__,
            "__file__": str(subject.SOURCE_GUARD_PATH),
            "__name__": "guard_path_order_contract_subject",
            "__package__": None,
        }
        source = subject.SOURCE_GUARD_PATH.read_bytes()
        exec(
            compile(source, str(subject.SOURCE_GUARD_PATH), "exec"),
            namespace,
        )
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary).resolve()
            package_path = temporary_root / "packages"
            scripts_path = temporary_root / "scripts"
            package_path.mkdir()
            scripts_path.mkdir()
            fake_csv = package_path / "csv.py"
            fake_csv.write_text(
                "raise RuntimeError('shadowed stdlib')\n", encoding="utf-8"
            )
            (package_path / "numpy").mkdir()
            expected_numpy = package_path / "numpy" / "__init__.py"
            expected_numpy.write_text("SOURCE = 'package'\n", encoding="utf-8")
            fake_numpy = scripts_path / "numpy.py"
            fake_numpy.write_text(
                "raise RuntimeError('shadowed package')\n", encoding="utf-8"
            )
            stdlib_path = Path(os.__file__).resolve().parent
            original = list(sys.path)
            try:
                sys.path[:] = [str(stdlib_path)]
                namespace["SCRIPTS"] = scripts_path
                namespace["_interpreter_dependency_paths"] = lambda: [
                    str(package_path)
                ]
                namespace["_install_isolated_import_paths"]()
                resolved = [Path(value).resolve() for value in sys.path]
                self.assertLess(resolved.index(stdlib_path), resolved.index(package_path))
                self.assertLess(
                    resolved.index(package_path), resolved.index(scripts_path)
                )
                spec = importlib.machinery.PathFinder.find_spec("csv", sys.path)
                self.assertIsNotNone(spec)
                self.assertNotEqual(Path(spec.origin).resolve(), fake_csv)
                spec = importlib.machinery.PathFinder.find_spec("numpy", sys.path)
                self.assertIsNotNone(spec)
                self.assertEqual(Path(spec.origin).resolve(), expected_numpy)
                self.assertNotEqual(Path(spec.origin).resolve(), fake_numpy)
            finally:
                sys.path[:] = original

    def test_focused_guard_preloads_exact_transaction_source_over_package(self) -> None:
        namespace = {
            "__builtins__": __builtins__,
            "__file__": str(subject.SOURCE_GUARD_PATH),
            "__name__": "guard_focused_import_contract_subject",
            "__package__": None,
        }
        source = subject.SOURCE_GUARD_PATH.read_bytes()
        exec(
            compile(source, str(subject.SOURCE_GUARD_PATH), "exec"),
            namespace,
        )
        with tempfile.TemporaryDirectory() as temporary:
            fake_root = Path(temporary)
            package = fake_root / "step28_v13_v1_13_split_transaction"
            package.mkdir()
            sentinel = fake_root / "package-executed.txt"
            (package / "__init__.py").write_text(
                f"open(r'{sentinel}', 'w').write('executed')\n",
                encoding="utf-8",
            )
            exact_source = b"EXECUTION_ID = 'verified-source-bytes'\n"
            focused_test = b"""
import unittest
import step28_v13_v1_13_split_transaction as exact_subject

class ExactTransactionImport(unittest.TestCase):
    def test_exact_source_was_preloaded(self):
        self.assertEqual(exact_subject.EXECUTION_ID, 'verified-source-bytes')
"""
            original = list(sys.path)
            try:
                sys.path.insert(0, str(fake_root))
                status = namespace["_run_focused_tests"](
                    exact_source, focused_test
                )
            finally:
                sys.path[:] = original
            self.assertEqual(status, 0)
            self.assertFalse(sentinel.exists())

    def test_public_runner_accepts_no_authority_arguments(self) -> None:
        self.assertEqual(list(inspect.signature(subject.run_development_smoke).parameters), [])
        self.assertEqual(list(inspect.signature(subject.main).parameters), [])

    def test_stable_lock_identity_is_workspace_independent(self) -> None:
        first = subject._stable_lock_identity()
        with tempfile.TemporaryDirectory() as _left, tempfile.TemporaryDirectory() as _right:
            self.assertEqual(subject._stable_lock_identity(), first)

    def test_lock_conflict_precedes_temporary_root_and_selector(self) -> None:
        with subject._exclusive_split_lock():
            with mock.patch.object(subject.tempfile, "TemporaryDirectory") as temporary, mock.patch.object(
                selection, "DevelopmentSmokeCandidateSelector"
            ) as selector:
                with self.assertRaises(subject.SplitLockBusy):
                    subject.run_development_smoke()
                temporary.assert_not_called()
                selector.assert_not_called()

    def test_workspace_boundary_pending_crash_resumes_safely(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / f"{subject.WORKSPACE_PREFIX}boundary-crash"
            with subject._exclusive_split_lock() as token:
                root.mkdir()
                pending = root / f".{subject.WORKSPACE_BOUNDARY_NAME}.pending"
                pending.write_bytes(b"partial boundary")
                _boundary, boundary_bytes = subject._initialize_workspace(
                    token, root, create=False
                )
                self.assertTrue(boundary_bytes)
                self.assertFalse(pending.exists())
                self.assertTrue((root / subject.WORKSPACE_BOUNDARY_NAME).is_file())

    def test_second_process_observes_same_stable_lock(self) -> None:
        code = (
            "import sys;sys.path.insert(0,r'"
            + str(SCRIPTS)
            + "');import step28_v13_v1_13_split_transaction as t;"
            "\ntry:\n with t._exclusive_split_lock(): print('ACQUIRED')"
            "\nexcept t.SplitLockBusy: print('BUSY')"
        )
        with subject._exclusive_split_lock():
            completed = subprocess.run(
                [sys.executable, "-B", "-c", code],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
        self.assertEqual(completed.stdout.strip(), "BUSY")

    def test_no_replace_publish_never_overwrites_destination(self) -> None:
        with self.workspace() as (token, root, _boundary):
            final = root / subject.FINAL_DIRECTORY_NAME
            subject._ensure_plain_directory(
                token, final, parent=root, label="test final"
            )
            target = final / "immutable.bin"
            subject._publish_no_replace(token, target, b"first")
            with self.assertRaises(OSError):
                subject._publish_no_replace(token, target, b"second")
            self.assertEqual(subject._read_plain_file(target, label="immutable"), b"first")
            self.assertFalse((final / ".immutable.bin.pending").exists())

    def test_publish_primitive_requires_the_active_lock_token(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "forbidden.bin"
            with self.assertRaisesRegex(subject.SplitTransactionError, "active lock"):
                subject._publish_no_replace(None, target, b"forbidden")
            self.assertFalse(target.exists())

    def test_directory_and_unlink_mutators_require_active_lock_token(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            child = root / "child"
            with self.assertRaisesRegex(subject.SplitTransactionError, "active lock"):
                subject._ensure_plain_directory(
                    None, child, parent=root, label="forbidden directory"
                )
            self.assertFalse(child.exists())
            existing = root / "existing.bin"
            existing.write_bytes(b"retain")
            with self.assertRaisesRegex(subject.SplitTransactionError, "active lock"):
                subject._unlink_known_plain_file(
                    None, existing, label="forbidden unlink"
                )
            self.assertEqual(existing.read_bytes(), b"retain")

    def test_windows_mutex_cleanup_checks_release_and_close_results(self) -> None:
        kernel = mock.Mock()
        kernel.ReleaseMutex.return_value = 0
        kernel.CloseHandle.return_value = 1
        with self.assertRaisesRegex(subject.SplitTransactionError, "ReleaseMutex"):
            subject._release_windows_mutex(kernel, object(), acquired=True)
        kernel.ReleaseMutex.assert_called_once()
        kernel.CloseHandle.assert_called_once()

        kernel.reset_mock()
        kernel.ReleaseMutex.return_value = 1
        kernel.CloseHandle.return_value = 0
        with self.assertRaisesRegex(subject.SplitTransactionError, "CloseHandle"):
            subject._release_windows_mutex(kernel, object(), acquired=True)

    def test_descriptor_reader_rejects_hardlinks(self) -> None:
        with self.workspace() as (_token, root, _boundary):
            original = root / "original.bin"
            linked = root / "linked.bin"
            original.write_bytes(b"value")
            os.link(original, linked)
            with self.assertRaisesRegex(subject.RecoveryCorruption, "single-link"):
                subject._read_plain_file(original, label="hardlinked")

    def test_descriptor_reader_rejects_symlink_when_supported(self) -> None:
        with self.workspace() as (_token, root, _boundary):
            original = root / "original.bin"
            linked = root / "linked.bin"
            original.write_bytes(b"value")
            try:
                linked.symlink_to(original)
            except OSError as exc:
                self.skipTest(f"symlink unavailable: {exc}")
            with self.assertRaises(subject.RecoveryCorruption):
                subject._read_plain_file(linked, label="symlink")

    def test_world_marker_is_published_after_every_member(self) -> None:
        with self.workspace() as (token, root, boundary):
            calls: list[str] = []
            real = subject._publish_no_replace

            def tracked(passed_token, path, payload):
                calls.append(path.name)
                return real(passed_token, path, payload)

            with mock.patch.object(subject, "_publish_no_replace", side_effect=tracked):
                self.commit(token, root, boundary)
            self.assertEqual(calls[-1], subject.WORLD_MARKER_NAME)
            self.assertEqual(len(calls), len(subject.WORLD_MEMBER_ROLES) + 1)

    def test_safe_uncommitted_directory_is_removed(self) -> None:
        with self.workspace() as (token, root, boundary):
            transactions = root / subject.TRANSACTIONS_DIRECTORY_NAME
            subject._ensure_plain_directory(
                token, transactions, parent=root, label="test transactions"
            )
            world = transactions / subject.WORLD_DIRECTORY_NAME
            world.mkdir()
            filename = self.policy["transaction_layout"]["world_members"]["private_world"]
            subject._publish_no_replace(token, world / filename, b"incomplete")
            recovered = subject._scan_pre_seal_state(
                token,
                root,
                policy=self.policy,
                boundary_bytes=boundary,
            )
            self.assertIsNone(recovered)
            self.assertFalse(world.exists())

    def test_unknown_uncommitted_entry_blocks_cleanup(self) -> None:
        with self.workspace() as (token, root, boundary):
            transactions = root / subject.TRANSACTIONS_DIRECTORY_NAME
            subject._ensure_plain_directory(
                token, transactions, parent=root, label="test transactions"
            )
            world = transactions / subject.WORLD_DIRECTORY_NAME
            world.mkdir()
            (world / "unknown.bin").write_bytes(b"do not delete blindly")
            with self.assertRaisesRegex(subject.RecoveryCorruption, "Unknown entry"):
                subject._scan_pre_seal_state(
                    token,
                    root,
                    policy=self.policy,
                    boundary_bytes=boundary,
                )
            self.assertTrue((world / "unknown.bin").exists())

    def test_unknown_sibling_world_is_not_ignored(self) -> None:
        with self.workspace() as (token, root, boundary):
            self.commit(token, root, boundary)
            (root / subject.TRANSACTIONS_DIRECTORY_NAME / "world_000001").mkdir()
            with self.assertRaisesRegex(subject.RecoveryCorruption, "discontinuous"):
                subject._scan_pre_seal_state(
                    token,
                    root,
                    policy=self.policy,
                    boundary_bytes=boundary,
                )

    def test_valid_marker_recovery_never_constructs_selector(self) -> None:
        with self.workspace() as (token, root, boundary):
            self.commit(token, root, boundary)
            with mock.patch.object(
                selection,
                "DevelopmentSmokeCandidateSelector",
                side_effect=AssertionError("selector construction forbidden"),
            ):
                recovered = subject.verify_committed_world_pre_seal(
                    token,
                    root,
                    policy=self.policy,
                    boundary_bytes=boundary,
                )
            self.assertEqual(recovered[0], self.accepted)

    def test_pre_seal_missing_member_is_corruption(self) -> None:
        with self.workspace() as (token, root, boundary):
            marker, _bytes = self.commit(token, root, boundary)
            path = (
                root
                / subject.TRANSACTIONS_DIRECTORY_NAME
                / subject.WORLD_DIRECTORY_NAME
                / marker["member_manifest"][0]["path"]
            )
            path.unlink()
            with self.assertRaisesRegex(subject.RecoveryCorruption, "missing"):
                subject.verify_committed_world_pre_seal(
                    token, root, policy=self.policy, boundary_bytes=boundary
                )

    def test_marker_rewrite_and_rehashed_self_is_rejected_by_golden(self) -> None:
        with self.workspace() as (token, root, boundary):
            self.commit(token, root, boundary)
            path = (
                root
                / subject.TRANSACTIONS_DIRECTORY_NAME
                / subject.WORLD_DIRECTORY_NAME
                / subject.WORLD_MARKER_NAME
            )
            marker = json.loads(path.read_text(encoding="utf-8"))
            marker["world_uid"] = "w_" + "0" * 64
            marker.pop("canonical_self_hash")
            marker = subject._with_self_hash(marker)
            path.write_bytes(common.canonical_json_bytes(marker))
            with self.assertRaises(subject.RecoveryCorruption):
                subject.verify_committed_world_pre_seal(
                    token, root, policy=self.policy, boundary_bytes=boundary
                )

    def test_member_and_marker_synchronized_rehash_still_fails(self) -> None:
        with self.workspace() as (token, root, boundary):
            marker, _bytes = self.commit(token, root, boundary)
            world = root / subject.TRANSACTIONS_DIRECTORY_NAME / subject.WORLD_DIRECTORY_NAME
            path = world / self.policy["transaction_layout"]["world_members"]["redacted_items"]
            rows = json.loads(path.read_text(encoding="utf-8"))
            rows[0]["title"] += "篡改"
            changed = common.canonical_json_bytes(rows)
            path.write_bytes(changed)
            changed_marker = copy.deepcopy(marker)
            entry = next(
                row for row in changed_marker["member_manifest"] if row["role"] == "redacted_items"
            )
            entry["size_bytes"] = len(changed)
            entry["sha256"] = hashlib.sha256(changed).hexdigest()
            changed_marker["accepted_candidate"]["redacted_items_sha256"] = entry["sha256"]
            changed_marker.pop("canonical_self_hash")
            changed_marker = subject._with_self_hash(changed_marker)
            (world / subject.WORLD_MARKER_NAME).write_bytes(
                common.canonical_json_bytes(changed_marker)
            )
            with self.assertRaises(subject.RecoveryCorruption):
                subject.verify_committed_world_pre_seal(
                    token, root, policy=self.policy, boundary_bytes=boundary
                )

    def test_marker_transplant_to_another_workspace_is_rejected(self) -> None:
        with self.workspace() as (token, root, boundary):
            marker, _bytes = self.commit(token, root, boundary)
            other = root.parent / f"{subject.WORKSPACE_PREFIX}other"
            _b, other_boundary = subject._initialize_workspace(token, other, create=True)
            with self.assertRaisesRegex(subject.RecoveryCorruption, "boundary"):
                subject._validate_marker_envelope(
                    marker, policy=self.policy, boundary_bytes=other_boundary
                )

    def test_pre_seal_marker_bool_cannot_impersonate_plain_integers(self) -> None:
        for field, value in (
            ("expected_world_count", True),
            ("world_ordinal", False),
        ):
            with self.subTest(field=field), self.workspace() as (
                token,
                root,
                boundary,
            ):
                self.commit(token, root, boundary)
                marker_path = (
                    root
                    / subject.TRANSACTIONS_DIRECTORY_NAME
                    / subject.WORLD_DIRECTORY_NAME
                    / subject.WORLD_MARKER_NAME
                )
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
                marker[field] = value
                marker.pop("canonical_self_hash")
                marker_path.write_bytes(
                    common.canonical_json_bytes(subject._with_self_hash(marker))
                )
                with self.assertRaisesRegex(
                    subject.SplitTransactionError, "plain integer"
                ):
                    subject.verify_committed_world_pre_seal(
                        token,
                        root,
                        policy=self.policy,
                        boundary_bytes=boundary,
                    )

    def test_jsonl_projection_is_sorted_newline_terminated_and_private_free(self) -> None:
        with self.workspace() as (_token, _root, boundary):
            state = self.expected_state(boundary)
            for name in (
                "worlds.jsonl",
                "redacted_items.jsonl",
                "seller_profiles.jsonl",
                "identity33.jsonl",
                "document_collision_attempts.jsonl",
            ):
                payload = state.final_payloads[name]
                self.assertTrue(payload.endswith(b"\n"))
                for line in payload.splitlines():
                    self.assertEqual(common.canonical_json_bytes(json.loads(line)), line)
            for name, fields in (
                ("worlds.jsonl", ("world_uid",)),
                ("redacted_items.jsonl", ("world_uid", "seller_uid", "item_uid")),
                ("seller_profiles.jsonl", ("seller_uid",)),
                ("identity33.jsonl", ("world_uid", "canonical_pair_uid")),
                ("document_collision_attempts.jsonl", ("world_ordinal",)),
            ):
                rows = [
                    json.loads(line)
                    for line in state.final_payloads[name].splitlines()
                ]
                expected = sorted(
                    rows,
                    key=lambda row: tuple(
                        str(row[field]).encode("utf-8") for field in fields
                    ),
                )
                self.assertEqual(rows, expected, name)
            private_world = json.loads(self.accepted.world_bytes)
            combined = b"".join(state.final_payloads.values()) + state.seal_bytes + state.cleanup_receipt_bytes
            for asset in private_world["private"]["identity_assets"]:
                self.assertNotIn(str(asset["identity_value"]).encode("utf-8"), combined)
            for membership in private_world["private"]["controller_membership"]:
                self.assertNotIn(
                    str(membership["controller_uid"]).encode("utf-8"), combined
                )
            self.assertNotIn(b'"private"', state.final_payloads["worlds.jsonl"])
            self.assertNotIn(b'"negative_flags"', combined)
            for forbidden_key in (
                b'"label"',
                b'"oracle"',
                b'"identity_value"',
                b'"controller_uid"',
                b'"key_hex"',
            ):
                self.assertNotIn(forbidden_key, combined)

    def test_public_world_projection_rejects_every_unlisted_field(self) -> None:
        private_world = json.loads(self.accepted.world_bytes)
        private_world["public"]["world"]["unexpected_public_field"] = "leak"
        damaged = replace(
            self.accepted,
            world_bytes=common.canonical_json_bytes(private_world),
        )
        with self.assertRaisesRegex(subject.RecoveryCorruption, "field set"):
            subject._final_member_payloads(
                damaged,
                {"canonical_self_hash": "0" * 64},
                policy=self.policy,
            )

    def test_partial_identical_final_files_resume_and_seal_last(self) -> None:
        with self.workspace() as (token, root, boundary):
            self.commit(token, root, boundary)
            state = self.expected_state(boundary)
            final = root / subject.FINAL_DIRECTORY_NAME
            subject._ensure_plain_directory(
                token, final, parent=root, label="test final"
            )
            for name in subject.FINAL_MEMBER_NAMES[:2]:
                subject._publish_no_replace(
                    token, final / name, state.final_payloads[name]
                )
            stale = final / f".{subject.FINAL_MEMBER_NAMES[2]}.pending"
            stale.write_bytes(b"crash residue")
            calls: list[str] = []
            real = subject._publish_no_replace

            def tracked(passed_token, path, payload):
                calls.append(path.name)
                return real(passed_token, path, payload)

            with mock.patch.object(subject, "_publish_no_replace", side_effect=tracked):
                subject._publish_final_and_seal(token, root, state)
            self.assertEqual(calls[-1], subject.SPLIT_SEAL_NAME)
            self.assertFalse(stale.exists())
            subject._verify_final_and_seal(root, state)

    def test_post_seal_pending_file_is_corruption(self) -> None:
        with self.workspace() as (token, root, boundary):
            state = self.seal(token, root, boundary)
            final = root / subject.FINAL_DIRECTORY_NAME
            private_world = json.loads(self.accepted.world_bytes)
            raw_identity = private_world["private"]["identity_assets"][0][
                "identity_value"
            ]
            pending = final / ".worlds.jsonl.pending"
            pending.write_text(raw_identity, encoding="utf-8")
            with self.assertRaisesRegex(subject.RecoveryCorruption, "Unknown"):
                subject._verify_final_and_seal(root, state)
            self.assertTrue(pending.exists())

    def test_different_existing_final_file_is_never_replaced(self) -> None:
        with self.workspace() as (token, root, boundary):
            self.commit(token, root, boundary)
            state = self.expected_state(boundary)
            final = root / subject.FINAL_DIRECTORY_NAME
            subject._ensure_plain_directory(
                token, final, parent=root, label="test final"
            )
            first = subject.FINAL_MEMBER_NAMES[0]
            subject._publish_no_replace(token, final / first, b"wrong")
            with self.assertRaises(subject.PublishConflict):
                subject._publish_final_and_seal(token, root, state)
            self.assertEqual((final / first).read_bytes(), b"wrong")

    def test_unknown_final_entry_fails_closed(self) -> None:
        with self.workspace() as (token, root, boundary):
            self.commit(token, root, boundary)
            state = self.expected_state(boundary)
            final = root / subject.FINAL_DIRECTORY_NAME
            subject._ensure_plain_directory(
                token, final, parent=root, label="test final"
            )
            (final / "unknown.bin").write_bytes(b"unknown")
            with self.assertRaisesRegex(subject.RecoveryCorruption, "Unknown"):
                subject._publish_final_and_seal(token, root, state)

    def test_cleanup_crash_with_missing_subset_resumes_from_seal_plan(self) -> None:
        with self.workspace() as (token, root, boundary):
            state = self.seal(token, root, boundary)
            world = root / subject.TRANSACTIONS_DIRECTORY_NAME / subject.WORLD_DIRECTORY_NAME
            for entry in state.marker["member_manifest"][:3]:
                (world / entry["path"]).unlink()
            with mock.patch.object(
                subject, "_fixed_replay_candidate_zero", return_value=self.accepted
            ), mock.patch.object(
                selection,
                "DevelopmentSmokeCandidateSelector",
                side_effect=AssertionError("selector forbidden post-seal"),
            ):
                recovered = subject.verify_sealed_world_post_seal(
                    token,
                    root,
                    policy=self.policy,
                    boundary_bytes=boundary,
                )
            self.assertEqual(recovered.accepted, self.accepted)
            self.assertTrue((world / subject.CLEANUP_RECEIPT_NAME).is_file())
            self.assertEqual(
                {path.name for path in world.iterdir()},
                {subject.WORLD_MARKER_NAME, subject.CLEANUP_RECEIPT_NAME},
            )

    def test_cleanup_resumes_when_all_large_members_are_gone_before_receipt(self) -> None:
        with self.workspace() as (token, root, boundary):
            state = self.seal(token, root, boundary)
            world = (
                root
                / subject.TRANSACTIONS_DIRECTORY_NAME
                / subject.WORLD_DIRECTORY_NAME
            )
            for entry in state.marker["member_manifest"]:
                (world / entry["path"]).unlink()
            with mock.patch.object(
                subject, "_fixed_replay_candidate_zero", return_value=self.accepted
            ):
                subject.verify_sealed_world_post_seal(
                    token,
                    root,
                    policy=self.policy,
                    boundary_bytes=boundary,
                )
            self.assertEqual(
                {path.name for path in world.iterdir()},
                {subject.WORLD_MARKER_NAME, subject.CLEANUP_RECEIPT_NAME},
            )

    def test_cleanup_receipt_pending_crash_resumes_after_all_member_deletes(self) -> None:
        with self.workspace() as (token, root, boundary):
            state = self.seal(token, root, boundary)
            world = (
                root
                / subject.TRANSACTIONS_DIRECTORY_NAME
                / subject.WORLD_DIRECTORY_NAME
            )
            for entry in state.marker["member_manifest"]:
                (world / entry["path"]).unlink()
            pending = world / f".{subject.CLEANUP_RECEIPT_NAME}.pending"
            pending.write_bytes(b"partial receipt")
            with mock.patch.object(
                subject, "_fixed_replay_candidate_zero", return_value=self.accepted
            ):
                subject.verify_sealed_world_post_seal(
                    token,
                    root,
                    policy=self.policy,
                    boundary_bytes=boundary,
                )
            self.assertFalse(pending.exists())
            self.assertTrue((world / subject.CLEANUP_RECEIPT_NAME).is_file())

    def test_cleanup_receipt_and_pending_together_are_corruption(self) -> None:
        with self.workspace() as (token, root, boundary):
            state = self.seal(token, root, boundary)
            subject._finish_cleanup(token, root, state)
            world = (
                root
                / subject.TRANSACTIONS_DIRECTORY_NAME
                / subject.WORLD_DIRECTORY_NAME
            )
            pending = world / f".{subject.CLEANUP_RECEIPT_NAME}.pending"
            pending.write_bytes(b"must remain for audit")
            with self.assertRaisesRegex(subject.RecoveryCorruption, "Unknown"):
                subject._finish_cleanup(token, root, state)
            self.assertEqual(pending.read_bytes(), b"must remain for audit")

    def test_cleanup_complete_state_is_idempotently_verified_without_writes(self) -> None:
        with self.workspace() as (token, root, boundary):
            state = self.seal(token, root, boundary)
            subject._finish_cleanup(token, root, state)
            with mock.patch.object(
                subject, "_fixed_replay_candidate_zero", return_value=self.accepted
            ), mock.patch.object(
                subject, "_publish_no_replace", wraps=subject._publish_no_replace
            ) as publish:
                subject.verify_sealed_world_post_seal(
                    token,
                    root,
                    policy=self.policy,
                    boundary_bytes=boundary,
                )
            publish.assert_not_called()

    def test_member_reappearing_after_cleanup_receipt_is_preserved_and_rejected(self) -> None:
        with self.workspace() as (token, root, boundary):
            state = self.seal(token, root, boundary)
            world = (
                root
                / subject.TRANSACTIONS_DIRECTORY_NAME
                / subject.WORLD_DIRECTORY_NAME
            )
            first = state.marker["member_manifest"][0]
            original = subject._read_plain_file(
                world / first["path"], label="pre-cleanup test member"
            )
            subject._finish_cleanup(token, root, state)
            subject._publish_no_replace(token, world / first["path"], original)
            with self.assertRaisesRegex(subject.RecoveryCorruption, "reappeared"):
                subject._finish_cleanup(token, root, state)
            self.assertEqual((world / first["path"]).read_bytes(), original)

    def test_tampered_member_reappearing_after_receipt_is_not_deleted(self) -> None:
        with self.workspace() as (token, root, boundary):
            state = self.seal(token, root, boundary)
            world = (
                root
                / subject.TRANSACTIONS_DIRECTORY_NAME
                / subject.WORLD_DIRECTORY_NAME
            )
            first = state.marker["member_manifest"][0]
            subject._finish_cleanup(token, root, state)
            tampered = b"tampered resurrection"
            subject._publish_no_replace(token, world / first["path"], tampered)
            with self.assertRaisesRegex(subject.RecoveryCorruption, "reappeared"):
                subject._finish_cleanup(token, root, state)
            self.assertEqual((world / first["path"]).read_bytes(), tampered)

    def test_post_seal_synchronized_final_seal_receipt_tamper_is_rejected(self) -> None:
        with self.workspace() as (token, root, boundary):
            state = self.seal(token, root, boundary)
            subject._finish_cleanup(token, root, state)
            final = root / subject.FINAL_DIRECTORY_NAME
            target = final / "worlds.jsonl"
            changed = target.read_bytes().replace(b"{", b'{"tampered":true,', 1)
            target.write_bytes(changed)
            seal_path = final / subject.SPLIT_SEAL_NAME
            seal = json.loads(seal_path.read_text(encoding="utf-8"))
            entry = next(row for row in seal["final_members"] if row["path"] == "worlds.jsonl")
            entry["size_bytes"] = len(changed)
            entry["sha256"] = hashlib.sha256(changed).hexdigest()
            seal.pop("canonical_self_hash")
            seal = subject._with_self_hash(seal)
            changed_seal = common.canonical_json_bytes(seal)
            seal_path.write_bytes(changed_seal)
            receipt_path = (
                root
                / subject.TRANSACTIONS_DIRECTORY_NAME
                / subject.WORLD_DIRECTORY_NAME
                / subject.CLEANUP_RECEIPT_NAME
            )
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["split_seal_raw_sha256"] = hashlib.sha256(changed_seal).hexdigest()
            receipt.pop("canonical_self_hash")
            receipt_path.write_bytes(common.canonical_json_bytes(subject._with_self_hash(receipt)))
            with mock.patch.object(
                subject, "_fixed_replay_candidate_zero", return_value=self.accepted
            ), self.assertRaisesRegex(
                subject.RecoveryCorruption, "expected final bytes"
            ):
                subject.verify_sealed_world_post_seal(
                    token,
                    root,
                    policy=self.policy,
                    boundary_bytes=boundary,
                )

    def test_fixed_candidate_zero_replay_matches_frozen_golden(self) -> None:
        with subject._exclusive_split_lock() as token:
            replayed = subject._fixed_replay_candidate_zero(token)
        self.assertEqual(replayed, self.accepted)

    def test_zero_argument_run_returns_only_nonformal_summary(self) -> None:
        fake_selector = mock.Mock()
        fake_selector.select.return_value = self.accepted
        with mock.patch.object(
            selection, "DevelopmentSmokeCandidateSelector", return_value=fake_selector
        ), mock.patch.object(
            subject, "_fixed_replay_candidate_zero", return_value=self.accepted
        ):
            summary = subject.run_development_smoke()
        self.assertTrue(summary["design_smoke_only"])
        self.assertFalse(summary["formal_split_semantics"])
        self.assertFalse(summary["source_candidate_committable"])
        self.assertEqual(summary["formal_seeds_generated"], 0)
        self.assertEqual(summary["formal_candidates_generated"], 0)
        self.assertEqual(summary["formal_rows_generated"], 0)
        self.assertEqual(summary["formal_models_trained"], 0)
        self.assertEqual(summary["formal_metrics_generated"], 0)
        self.assertFalse(summary["ephemeral_workspace_retained"])

    def test_unmocked_zero_argument_runner_completes_ephemerally(self) -> None:
        temporary_root = Path(tempfile.gettempdir())
        summary = subject.run_development_smoke()
        after = {
            path.resolve()
            for path in temporary_root.glob(f"{subject.WORKSPACE_PREFIX}*")
        }
        self.assertEqual(
            summary["status"],
            "PASS_DEVELOPMENT_SMOKE_TRANSACTION_STATE_MACHINE_ONLY",
        )
        self.assertEqual(summary["accepted_candidate_index"], 0)
        self.assertEqual(summary["formal_rows_generated"], 0)
        self.assertFalse(summary["ephemeral_workspace_retained"])
        self.assertEqual(after, set(), "smoke temporary namespace was not empty")

    def test_source_guard_runs_the_real_smoke_from_source_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            startup = Path(temporary) / "sitecustomize.py"
            sentinel = Path(temporary) / "startup-code-executed.txt"
            startup.write_text(
                f"open(r'{sentinel}', 'w').write('executed')\n",
                encoding="utf-8",
            )
            environment = os.environ.copy()
            environment["PYTHONPATH"] = temporary
            completed = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-S",
                    "-B",
                    str(subject.SOURCE_GUARD_PATH),
                    "--smoke",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=300,
                check=True,
                env=environment,
            )
            self.assertFalse(sentinel.exists())
            summary = json.loads(completed.stdout.strip().splitlines()[-1])
            self.assertEqual(
                summary["status"],
                "PASS_DEVELOPMENT_SMOKE_TRANSACTION_STATE_MACHINE_ONLY",
            )
            self.assertEqual(summary["formal_rows_generated"], 0)
            self.assertFalse(summary["ephemeral_workspace_retained"])

    def test_source_guard_refuses_missing_interpreter_isolation_flags(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                str(subject.SOURCE_GUARD_PATH),
                "--smoke",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("requires Python -I -S -B", completed.stderr)

    def test_three_synthetic_markers_accumulate_disjoint_registries(self) -> None:
        genesis = "01" * 32
        payloads = []
        previous = genesis
        for ordinal in range(3):
            payload = subject._synthetic_marker_bytes(
                world_ordinal=ordinal,
                previous_raw_sha256=previous,
                item_hashes=[f"{ordinal + 2:02x}" * 32],
                seller_hashes=[f"{ordinal + 5:02x}" * 32],
                identity_hashes=[f"{ordinal + 8:02x}" * 32],
            )
            payloads.append(payload)
            previous = hashlib.sha256(payload).hexdigest()
        result = subject.validate_synthetic_marker_chain(
            payloads, genesis_sha256=genesis
        )
        self.assertEqual(len(result["item"]), 3)
        self.assertEqual(len(result["seller"]), 3)
        self.assertEqual(len(result["identity"]), 3)

    def test_synthetic_marker_gap_or_wrong_previous_fails(self) -> None:
        genesis = "11" * 32
        first = subject._synthetic_marker_bytes(
            world_ordinal=0,
            previous_raw_sha256=genesis,
            item_hashes=["12" * 32],
            seller_hashes=["13" * 32],
            identity_hashes=["14" * 32],
        )
        gap = subject._synthetic_marker_bytes(
            world_ordinal=2,
            previous_raw_sha256=hashlib.sha256(first).hexdigest(),
            item_hashes=["15" * 32],
            seller_hashes=["16" * 32],
            identity_hashes=["17" * 32],
        )
        with self.assertRaises(subject.RecoveryCorruption):
            subject.validate_synthetic_marker_chain([first, gap], genesis_sha256=genesis)
        wrong = subject._synthetic_marker_bytes(
            world_ordinal=1,
            previous_raw_sha256="18" * 32,
            item_hashes=["19" * 32],
            seller_hashes=["1a" * 32],
            identity_hashes=["1b" * 32],
        )
        with self.assertRaises(subject.RecoveryCorruption):
            subject.validate_synthetic_marker_chain([first, wrong], genesis_sha256=genesis)

    def test_synthetic_current_split_collision_fails(self) -> None:
        genesis = "21" * 32
        first = subject._synthetic_marker_bytes(
            world_ordinal=0,
            previous_raw_sha256=genesis,
            item_hashes=["22" * 32],
            seller_hashes=["23" * 32],
            identity_hashes=["24" * 32],
        )
        second = subject._synthetic_marker_bytes(
            world_ordinal=1,
            previous_raw_sha256=hashlib.sha256(first).hexdigest(),
            item_hashes=["22" * 32],
            seller_hashes=["25" * 32],
            identity_hashes=["26" * 32],
        )
        with self.assertRaisesRegex(subject.RecoveryCorruption, "collision"):
            subject.validate_synthetic_marker_chain([first, second], genesis_sha256=genesis)

    def test_synthetic_marker_plain_integer_gate_rejects_bool(self) -> None:
        with self.assertRaisesRegex(subject.SplitTransactionError, "plain integer"):
            subject._synthetic_marker_bytes(
                world_ordinal=True,
                previous_raw_sha256="31" * 32,
                item_hashes=[],
                seller_hashes=[],
                identity_hashes=[],
            )

        payload = subject._synthetic_marker_bytes(
            world_ordinal=0,
            previous_raw_sha256="32" * 32,
            item_hashes=[],
            seller_hashes=[],
            identity_hashes=[],
        )
        marker = json.loads(payload)
        marker["world_ordinal"] = False
        marker.pop("canonical_self_hash")
        changed = common.canonical_json_bytes(subject._with_self_hash(marker))
        with self.assertRaisesRegex(subject.SplitTransactionError, "plain integer"):
            subject.validate_synthetic_marker_chain(
                [changed], genesis_sha256="32" * 32
            )

    def test_synthetic_predecessor_order_for_all_splits(self) -> None:
        for split in subject.FORMAL_SPLIT_ORDER:
            pins = subject.synthetic_predecessor_fixture_pins(split)
            subject.validate_synthetic_predecessor_pins(split=split, pins=pins)

    def test_synthetic_predecessor_missing_extra_or_wrong_order_fails(self) -> None:
        valid = list(subject.synthetic_predecessor_fixture_pins("audit_a"))
        subject.validate_synthetic_predecessor_pins(split="audit_a", pins=valid)
        extra = list(valid) + [
            {
                "split": "audit_a",
                "split_seal_raw_sha256": "c" * 64,
                "document_registry_raw_sha256": "d" * 64,
            }
        ]
        for broken in (valid[:1], extra, list(reversed(valid))):
            with self.assertRaises(subject.RecoveryCorruption):
                subject.validate_synthetic_predecessor_pins(split="audit_a", pins=broken)

    def test_synthetic_predecessor_wrong_hashes_and_transplant_fail(self) -> None:
        valid = list(subject.synthetic_predecessor_fixture_pins("audit_a"))
        for field in ("split_seal_raw_sha256", "document_registry_raw_sha256"):
            with self.subTest(field=field):
                damaged = copy.deepcopy(valid)
                damaged[0][field] = "0" * 64
                with self.assertRaisesRegex(subject.RecoveryCorruption, "truth"):
                    subject.validate_synthetic_predecessor_pins(
                        split="audit_a", pins=damaged
                    )
        transplanted = copy.deepcopy(valid)
        transplanted[1]["split_seal_raw_sha256"] = transplanted[0][
            "split_seal_raw_sha256"
        ]
        with self.assertRaisesRegex(subject.RecoveryCorruption, "truth"):
            subject.validate_synthetic_predecessor_pins(
                split="audit_a", pins=transplanted
            )

    def test_duplicate_json_keys_are_rejected(self) -> None:
        with self.assertRaisesRegex(subject.SplitTransactionError, "Duplicate"):
            subject._decode_canonical(b'{"a":1,"a":1}', label="duplicate fixture")

    def test_source_has_no_formal_seed_label_or_model_path(self) -> None:
        source = Path(subject.__file__).read_text(encoding="utf-8")
        bootstrap_call = source.index("\n_bootstrap_verify_candidate_import_closure()\n")
        candidate_import = source.index(
            "\nimport step28_v13_v1_13_candidate_selection as selection\n"
        )
        self.assertLess(bootstrap_call, candidate_import)
        for forbidden in (
            "private_custody",
            "classification_labels.csv",
            "retrieval_qrels.csv",
            "formal_seed_key_hex",
            "model.fit(",
        ):
            self.assertNotIn(forbidden, source)
