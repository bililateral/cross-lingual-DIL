from __future__ import annotations

import ast
import copy
import io
import json
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_12_freeze_prelock as freezer
import step28_v13_v1_12_preceremony as preceremony
import step28_v13_v1_12_prelock_evidence as evidence
import step28_v13_v1_12_seed_ceremony as ceremony
import step28_v13_v1_12_unittest_json_runner as json_runner


class Step28V13V112AuthorizationOverlayContracts(unittest.TestCase):
    @staticmethod
    def _valid_full_test_receipt() -> dict[str, object]:
        targeted = [
            f"test_step28_v13_v1_12_alpha.Case.test_{index}"
            for index in range(3)
        ]
        generic = [f"test_generic.Case.test_{index}" for index in range(7)]
        success_ids = sorted(targeted + generic)
        skipped = [{"id": "test_old.Case.test_skip", "reason": "known skip"}]
        started_ids = sorted(
            success_ids + [skipped[0]["id"], evidence.WAIVED_TEST_ID]
        )
        structured = {
            "version": "2026-08-09-step28-v13-v1-12-unittest-json-v1",
            "tests_run": len(started_ids),
            "started_test_ids": started_ids,
            "success_ids": success_ids,
            "failure_ids": [evidence.WAIVED_TEST_ID],
            "error_ids": [],
            "skipped": skipped,
            "expected_failure_ids": [],
            "unexpected_success_ids": [],
            "failed_subtest_ids": [],
            "skipped_subtest_ids": [],
            "was_successful": False,
            "wall_seconds": 1.25,
        }
        members = [
            {
                "path": "tests/example.py",
                "size_bytes": 1,
                "sha256": "a" * 64,
            }
        ]
        return {
            "version": "2026-08-09-step28-v13-v1-12-full-tests-v2",
            "status": "PASS_FULL_REPOSITORY_TESTS",
            "status_semantics": (
                "PASS_WITH_ONE_EXACT_HISTORICAL_MANIFEST_FAILURE_WAIVER"
            ),
            "count_semantics": (
                "raw_counts_are_authoritative; compatibility_skipped_count_"
                "includes_one_accepted_waiver"
            ),
            "command": "synthetic",
            "git_head": "a" * 40,
            "source_closure_canonical_sha256": "b" * 64,
            "source_closure_member_count": 1,
            "test_suite_source_closure": {
                "member_count": 1,
                "canonical_sha256": preceremony.canonical_sha256(members),
                "members": members,
            },
            "test_count": len(started_ids),
            "passed_count": len(success_ids),
            "skipped_count": 2,
            "failed_count": 0,
            "error_count": 0,
            "raw_test_count": len(started_ids),
            "raw_passed_count": len(success_ids),
            "raw_skipped_count": 1,
            "raw_failed_count": 1,
            "raw_error_count": 0,
            "raw_failure_ids": [evidence.WAIVED_TEST_ID],
            "raw_error_ids": [],
            "raw_skipped_ids": [skipped[0]["id"]],
            "raw_expected_failure_ids": [],
            "raw_unexpected_success_ids": [],
            "raw_failed_subtest_ids": [],
            "raw_skipped_subtest_ids": [],
            "accepted_waived_failure_count": 1,
            "accepted_waived_failure_ids": [evidence.WAIVED_TEST_ID],
            "historical_manifest_waiver": dict(evidence.WAIVER_RECEIPT_PIN),
            "current_v1_12_targeted_tests": {
                "tests_run": len(targeted),
                "passed": len(targeted),
                "skipped_ids": [],
                "failure_ids": [],
                "error_ids": [],
            },
            "subprocess_return_code": 1,
            "raw_subprocess_return_code": 1,
            "test_runner_reported_seconds": 1.25,
            "wall_seconds": 1.5,
            "raw_structured_result": structured,
            "structured_result_sha256": preceremony.canonical_sha256(
                structured
            ),
            "captured_output_sha256": "c" * 64,
            "runtime_versions": {},
            "warning_policy": "synthetic",
            "formal_seed_or_key_access": False,
            "formal_rows_produced": 0,
            "canonical_self_hash": "d" * 64,
        }

    def test_current_authorization_receipts_replay_strictly(self) -> None:
        receipts = evidence.load_current_authorization_receipts_exact(
            dereference_waiver_state=False
        )
        self.assertEqual(set(receipts), {
            "text_receipt", "interruption_receipt", "waiver_receipt"
        })

    def test_producer_and_validator_exact_receipt_keysets_match(self) -> None:
        tree = ast.parse(
            (ROOT / "scripts/step28_v13_v1_12_freeze_prelock.py").read_text(
                encoding="utf-8"
            )
        )
        candidates: list[set[str]] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            keys = {
                key.value
                for key in node.keys
                if isinstance(key, ast.Constant)
                and isinstance(key.value, str)
            }
            if {"raw_structured_result", "test_suite_source_closure"}.issubset(
                keys
            ):
                candidates.append(keys)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(
            candidates[0] | {"canonical_self_hash"},
            evidence.FULL_TEST_RECEIPT_KEYS,
        )

    def test_extra_prelock_field_rejected_after_legacy_loader_accepts(self) -> None:
        prelock = {key: None for key in evidence.PRELOCK_KEYS}
        prelock["unexpected_bypass"] = True
        legacy = {"prelock": prelock}
        with mock.patch.object(
            evidence.formal, "load_and_validate_prelock", return_value=legacy
        ):
            with self.assertRaisesRegex(
                evidence.AuthorizationEvidenceError, "keyset drift"
            ):
                evidence.load_and_validate_authorized_prelock(
                    Path("unused.json"), dereference_waiver_state=False
                )

    def test_full_test_receipt_rejects_any_second_failure(self) -> None:
        waiver = preceremony.load_json_strict(evidence.WAIVER_RECEIPT_PATH)
        receipt = self._valid_full_test_receipt()
        evidence.validate_full_test_receipt(receipt, waiver)
        corrupted = copy.deepcopy(receipt)
        structured = corrupted["raw_structured_result"]
        structured["failure_ids"].append("second.failure")
        structured["failure_ids"].sort()
        structured["started_test_ids"].append("second.failure")
        structured["started_test_ids"].sort()
        structured["tests_run"] += 1
        corrupted["structured_result_sha256"] = preceremony.canonical_sha256(
            structured
        )
        corrupted["raw_failed_count"] = 2
        with self.assertRaises(evidence.AuthorizationEvidenceError):
            evidence.validate_full_test_receipt(corrupted, waiver)

    def test_full_test_receipt_rejects_summary_count_reclassification(self) -> None:
        waiver = preceremony.load_json_strict(evidence.WAIVER_RECEIPT_PATH)
        receipt = self._valid_full_test_receipt()
        receipt["raw_passed_count"] -= 1
        receipt["raw_skipped_count"] += 1
        receipt["passed_count"] -= 1
        receipt["skipped_count"] += 1
        with self.assertRaises(evidence.AuthorizationEvidenceError):
            evidence.validate_full_test_receipt(receipt, waiver)

    def test_current_test_environment_rejects_head_or_tree_drift(self) -> None:
        receipt = self._valid_full_test_receipt()
        expected = receipt["test_suite_source_closure"]
        with mock.patch.object(
            evidence, "test_suite_source_closure", return_value=expected
        ), mock.patch.object(
            evidence, "require_tracked_worktree_matches_head"
        ), mock.patch.object(
            evidence, "require_committed_clean_test_tree"
        ), mock.patch.object(
            evidence, "current_git_head", return_value="b" * 40
        ):
            with self.assertRaises(evidence.AuthorizationEvidenceError):
                evidence.validate_current_full_test_environment(receipt)
        changed = copy.deepcopy(expected)
        changed["members"][0]["sha256"] = "e" * 64
        changed["canonical_sha256"] = preceremony.canonical_sha256(
            changed["members"]
        )
        with mock.patch.object(
            evidence, "test_suite_source_closure", return_value=changed
        ):
            with self.assertRaises(evidence.AuthorizationEvidenceError):
                evidence.validate_current_full_test_environment(receipt)

    def test_dirty_tracked_worktree_and_untracked_closure_member_reject(self) -> None:
        with mock.patch.object(
            evidence.subprocess,
            "run",
            return_value=types.SimpleNamespace(returncode=1),
        ):
            with self.assertRaises(evidence.AuthorizationEvidenceError):
                evidence.require_tracked_worktree_matches_head()
        with mock.patch.object(
            evidence, "_git_run", return_value="tests/only_one.py\n"
        ):
            with self.assertRaises(evidence.AuthorizationEvidenceError):
                evidence.require_paths_git_tracked(
                    {"tests/only_one.py", "tests/missing.py"},
                    label="synthetic closure",
                )

    def test_tracked_worktree_check_rejects_staged_and_cancelling_drift(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(
            prefix="step28-v1-12-git-clean-", dir=ROOT
        ) as raw:
            repository = Path(raw)

            def git(*arguments: str) -> None:
                subprocess.run(
                    ["git", *arguments],
                    cwd=repository,
                    check=True,
                    capture_output=True,
                )

            git("init", "--quiet")
            git("config", "user.name", "Step28 Contract Test")
            git("config", "user.email", "step28-contract@example.invalid")
            tracked = repository / "tracked.txt"
            tracked.write_text("A\n", encoding="utf-8")
            git("add", "tracked.txt")
            git("commit", "--quiet", "-m", "baseline")
            with mock.patch.object(evidence, "ROOT", repository):
                evidence.require_tracked_worktree_matches_head()
                tracked.write_text("B\n", encoding="utf-8")
                git("add", "tracked.txt")
                with self.assertRaises(evidence.AuthorizationEvidenceError):
                    evidence.require_tracked_worktree_matches_head()
                tracked.write_text("A\n", encoding="utf-8")
                with self.assertRaises(evidence.AuthorizationEvidenceError):
                    evidence.require_tracked_worktree_matches_head()

    def test_concurrent_ceremony_invocation_has_zero_side_effects(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="step28-v1-12-ceremony-lock-", dir=ROOT
        ) as raw:
            root = Path(raw)
            prelock_path = root / "prelock.json"
            prelock_path.write_bytes(b"{}\n")
            entered = threading.Event()
            release = threading.Event()
            first_errors: list[BaseException] = []

            def hold_first(**_kwargs: object) -> dict[str, object]:
                entered.set()
                if not release.wait(timeout=10):
                    raise AssertionError("ceremony lock test timed out")
                return {"status": "held-for-test"}

            def run_first() -> None:
                try:
                    ceremony.initialize(prelock_path)
                except BaseException as exc:  # pragma: no cover - assertion aid
                    first_errors.append(exc)

            entropy_calls = 0

            def entropy(_size: int) -> bytes:
                nonlocal entropy_calls
                entropy_calls += 1
                return b"x" * 32

            with mock.patch.object(
                ceremony, "_initialize_locked", side_effect=hold_first
            ) as locked_body:
                first = threading.Thread(target=run_first, daemon=True)
                first.start()
                self.assertTrue(entered.wait(timeout=5))
                with self.assertRaisesRegex(
                    ceremony.SeedCeremonyError,
                    "ceremony invocation already active",
                ):
                    ceremony.initialize(prelock_path, random_bytes=entropy)
                self.assertEqual(entropy_calls, 0)
                self.assertEqual(
                    sorted(path.name for path in root.iterdir()),
                    ["prelock.json"],
                )
                release.set()
                first.join(timeout=5)
                self.assertFalse(first.is_alive())
                self.assertEqual(first_errors, [])
                self.assertEqual(locked_body.call_count, 1)

    def test_ceremony_lock_rejects_a_second_real_process(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="step28-v1-12-ceremony-process-lock-", dir=ROOT
        ) as raw:
            root = Path(raw)
            prelock_path = root / "prelock.json"
            prelock_path.write_bytes(b"{}\n")
            holder_code = (
                "import sys\n"
                "from pathlib import Path\n"
                "sys.path.insert(0, sys.argv[2])\n"
                "import step28_v13_v1_12_seed_ceremony as c\n"
                "with c._exclusive_ceremony_invocation(Path(sys.argv[1])):\n"
                "    print('LOCKED', flush=True)\n"
                "    sys.stdin.readline()\n"
            )
            contender_code = (
                "import sys\n"
                "from pathlib import Path\n"
                "sys.path.insert(0, sys.argv[2])\n"
                "import step28_v13_v1_12_seed_ceremony as c\n"
                "try:\n"
                "    with c._exclusive_ceremony_invocation(Path(sys.argv[1])):\n"
                "        pass\n"
                "except c.SeedCeremonyError as exc:\n"
                "    print(str(exc))\n"
                "    raise SystemExit(0)\n"
                "raise SystemExit(9)\n"
            )
            first = subprocess.Popen(
                [
                    sys.executable,
                    "-B",
                    "-c",
                    holder_code,
                    str(prelock_path),
                    str(SCRIPTS),
                ],
                cwd=ROOT,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
            )
            try:
                assert first.stdout is not None
                self.assertEqual(first.stdout.readline().strip(), "LOCKED")
                contender = subprocess.run(
                    [
                        sys.executable,
                        "-B",
                        "-c",
                        contender_code,
                        str(prelock_path),
                        str(SCRIPTS),
                    ],
                    cwd=ROOT,
                    check=False,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    timeout=10,
                )
                self.assertEqual(contender.returncode, 0, contender.stderr)
                self.assertEqual(
                    contender.stdout.strip(),
                    "ceremony invocation already active",
                )
                self.assertEqual(
                    sorted(path.name for path in root.iterdir()),
                    ["prelock.json"],
                )
            finally:
                if first.poll() is None:
                    assert first.stdin is not None
                    first.stdin.write("\n")
                    first.stdin.flush()
                _stdout, stderr = first.communicate(timeout=10)
            self.assertEqual(first.returncode, 0, stderr)

    def test_full_test_receipt_rejects_ambiguous_outcomes(self) -> None:
        waiver = preceremony.load_json_strict(evidence.WAIVER_RECEIPT_PATH)
        for field in (
            "failed_subtest_ids",
            "skipped_subtest_ids",
            "expected_failure_ids",
            "unexpected_success_ids",
        ):
            with self.subTest(field=field):
                receipt = self._valid_full_test_receipt()
                structured = receipt["raw_structured_result"]
                structured[field] = ["ambiguous.Case.test_value"]
                receipt["structured_result_sha256"] = (
                    preceremony.canonical_sha256(structured)
                )
                with self.assertRaises(evidence.AuthorizationEvidenceError):
                    evidence.validate_full_test_receipt(receipt, waiver)

    def test_runner_identifies_skipped_subtest(self) -> None:
        class SkippedSubtest(unittest.TestCase):
            def runTest(self) -> None:
                with self.subTest(case="skip"):
                    self.skipTest("subtest skip")

        result = unittest.TextTestRunner(
            stream=io.StringIO(),
            resultclass=json_runner.StructuredResult,
        ).run(unittest.TestSuite([SkippedSubtest()]))
        self.assertEqual(len(result.skipped_subtest_ids), 1)

    def test_text_semantic_mutations_fail_closed(self) -> None:
        original = preceremony.load_json_strict(evidence.TEXT_RECEIPT_PATH)
        mutations = {
            "assignment_gate": lambda row: row["assignment_null"][
                "development_hard_gate"
            ]["hard_gates"].pop(
                "development_maximum_direct_symmetric_auc"
            ),
            "visible_gate": lambda row: row[
                "counterfactual_visible_text_hard_gate"
            ]["hard_gates"].update(
                {"development_family_model_symmetric_auc": False}
            ),
            "world_count": lambda row: row["rowwise_design_audit"][
                "splits"
            ]["train"].update({"world_count": 499}),
            "original_count": lambda row: row[
                "original_visible_text_descriptive_only"
            ]["development"].update({"pair_count": 188999}),
            "counterfactual_count": lambda row: row[
                "rowwise_design_audit"
            ]["splits"]["development"].update(
                {"counterfactual_pair_rows_recomputed_after_neutral_mask": 185999}
            ),
            "intersection": lambda row: row[
                "original_visible_text_descriptive_only"
            ]["cross_split_exact_intersection_counts"].update(
                {"seller_uid": 1}
            ),
            "residue": lambda row: row[
                "original_visible_text_descriptive_only"
            ]["visible_forbidden_residue_counts"].update({"train": 1}),
            "authorization": lambda row: row[
                "formal_authorizations_after_preflight"
            ].update({"model_training": True}),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                corrupted = copy.deepcopy(original)
                mutate(corrupted)
                with self.assertRaises(evidence.AuthorizationEvidenceError):
                    evidence.validate_text_receipt(corrupted)

    def test_exact_receipt_pin_and_evidence_keyset_fail_closed(self) -> None:
        bad_pin = dict(evidence.TEXT_RECEIPT_PIN)
        bad_pin["size_bytes"] += 1
        with self.assertRaises(evidence.AuthorizationEvidenceError):
            evidence._verify_exact_pin(
                bad_pin, evidence.TEXT_RECEIPT_PIN, label="mutated text pin"
            )
        for keys in (
            {"final_text_shortcut_preflight", "external_interruption"},
            {
                "final_text_shortcut_preflight",
                "external_interruption",
                "historical_manifest_waiver",
                "extra",
            },
        ):
            with self.subTest(keys=sorted(keys)):
                prelock = {key: None for key in evidence.PRELOCK_KEYS}
                prelock["design_evidence_role"] = evidence.DESIGN_EVIDENCE_ROLE
                prelock["authorization_evidence_role"] = (
                    evidence.AUTHORIZATION_EVIDENCE_ROLE
                )
                prelock["authorization_overlay_contract"] = dict(
                    evidence.OVERLAY_PIN
                )
                prelock["authorization_evidence"] = {
                    key: {} for key in keys
                }
                with self.assertRaises(evidence.AuthorizationEvidenceError):
                    evidence.validate_authorization_prelock_document(
                        prelock,
                        legacy_validation={},
                        dereference_waiver_state=False,
                    )

    def test_overlay_failure_precedes_start_stage_and_entropy(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="step28-v1-12-overlay-fail-", dir=ROOT
        ) as raw:
            root = Path(raw)
            relative = root.relative_to(ROOT).as_posix()
            prelock = {
                "custody": {
                    "private_seed_bundle_root": f"{relative}/private/final",
                    "private_seed_stage_root": f"{relative}/private/stage",
                    "seed_ceremony_start_receipt_path": f"{relative}/public/start.json",
                    "public_ceremony_receipt_path": f"{relative}/public/receipt.json",
                    "train_development_execution_lock_path": f"{relative}/public/lock.json",
                    "permanent_failure_receipt_path": f"{relative}/public/failure.json",
                }
            }
            prelock_path = root / "prelock.json"
            prelock_path.write_bytes(b"{}\n")
            calls = 0

            def entropy(_size: int) -> bytes:
                nonlocal calls
                calls += 1
                return b"x" * 32

            with mock.patch.object(
                ceremony.formal,
                "load_and_validate_prelock",
                return_value={"prelock": prelock},
            ), mock.patch.object(
                ceremony.authorization,
                "validate_authorization_prelock_document",
                side_effect=evidence.AuthorizationEvidenceError("closed"),
            ):
                with self.assertRaises(evidence.AuthorizationEvidenceError):
                    ceremony.initialize(prelock_path, random_bytes=entropy)
            self.assertEqual(calls, 0)
            self.assertFalse((root / "public/start.json").exists())
            self.assertFalse((root / "private/stage").exists())
            self.assertFalse((root / "public/failure.json").exists())

    def test_post_start_revalidation_failure_closes_before_stage_and_entropy(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="step28-v1-12-post-start-fail-", dir=ROOT
        ) as raw:
            root = Path(raw)
            relative = root.relative_to(ROOT).as_posix()
            prelock = preceremony.with_canonical_self_hash(
                {
                    "run_id": "unit-post-start-failure",
                    "source_closure": {"canonical_sha256": "a" * 64},
                    "custody": {
                        "private_seed_bundle_root": f"{relative}/private/final",
                        "private_seed_stage_root": f"{relative}/private/stage",
                        "seed_ceremony_start_receipt_path": f"{relative}/public/start.json",
                        "public_ceremony_receipt_path": f"{relative}/public/receipt.json",
                        "train_development_execution_lock_path": f"{relative}/public/lock.json",
                        "permanent_failure_receipt_path": f"{relative}/public/failure.json",
                    },
                }
            )
            prelock_path = root / "prelock.json"
            preceremony.write_bytes_no_replace_long_path(
                prelock_path,
                (
                    json.dumps(
                        prelock,
                        ensure_ascii=False,
                        sort_keys=True,
                        indent=2,
                    ).encode("utf-8")
                    + b"\n"
                ),
            )
            validated = {
                "prelock": prelock,
                "baseline": {"forbidden_master_commitments": frozenset()},
            }
            calls = 0

            def entropy(_size: int) -> bytes:
                nonlocal calls
                calls += 1
                return b"x" * 32

            with mock.patch.object(
                ceremony.formal,
                "load_and_validate_prelock",
                return_value=validated,
            ), mock.patch.object(
                ceremony.authorization,
                "validate_authorization_prelock_document",
                side_effect=[
                    {},
                    evidence.AuthorizationEvidenceError("post-start drift"),
                ],
            ):
                with self.assertRaises(evidence.AuthorizationEvidenceError):
                    ceremony.initialize(prelock_path, random_bytes=entropy)
            self.assertEqual(calls, 0)
            self.assertTrue((root / "public/start.json").exists())
            self.assertFalse((root / "private/stage").exists())
            failure = preceremony.load_json_strict(
                root / "public/failure.json"
            )
            self.assertEqual(
                failure["status"], "FAIL_V1_12_PERMANENTLY_CLOSED_NO_RETRY"
            )
            self.assertEqual(failure["master_draw_count_before_failure"], 0)

    def test_source_closure_contains_mandatory_overlay_members(self) -> None:
        closure = freezer.source_closure()
        paths = {row["path"] for row in closure["members"]}
        self.assertTrue(
            {
                evidence.OVERLAY_PIN["path"],
                evidence.TEXT_RECEIPT_PIN["path"],
                evidence.INTERRUPTION_RECEIPT_PIN["path"],
                evidence.WAIVER_RECEIPT_PIN["path"],
                "scripts/step28_v13_v1_12_prelock_evidence.py",
                "scripts/step28_v13_v1_12_unittest_json_runner.py",
                "tests/test_step28_v13_v1_12_authorization_overlay_contracts.py",
                "tests/test_step28_v13_v1_12_formal_execution_contracts.py",
            }.issubset(paths)
        )


if __name__ == "__main__":
    unittest.main()
