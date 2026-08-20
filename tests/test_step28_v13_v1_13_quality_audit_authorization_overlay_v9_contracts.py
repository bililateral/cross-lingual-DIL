#!/usr/bin/env python3
"""Contracts for the v9 one-shot design-quality audit authorization overlay."""

from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_quality_audit_runner_v9 as frozen_runner
import step28_v13_v1_13_run_quality_audit_once_v9 as overlay


class QualityAuditAuthorizationOverlayV9Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = overlay._load_overlay_policy()
        cls.static = overlay._validate_static_inputs(cls.policy)
        cls.git_commit = "a" * 40
        cls.git_tree = "b" * 40

    def _receipt_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            **overlay._expected_receipt_bindings(
                policy=self.policy,
                static=self.static,
                git_commit=self.git_commit,
                git_tree=self.git_tree,
            ),
            "review_conversation_url": "https://chatgpt.com/c/fixture-review",
            "review_response_sha256": "c" * 64,
            "reviewed_at_utc": "2026-08-20T12:34:56Z",
        }
        payload["canonical_self_hash"] = hashlib.sha256(
            overlay._canonical_json_bytes(payload)
        ).hexdigest()
        return payload

    @staticmethod
    def _write_receipt(path: Path, payload: dict[str, object]) -> bytes:
        raw = (
            json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
        ).encode("utf-8")
        path.write_bytes(raw)
        return raw

    def test_overlay_and_current_static_inputs_close_exactly(self) -> None:
        self.assertEqual(
            self.policy["canonical_self_hash"],
            overlay.OVERLAY_POLICY_CANONICAL_SELF_HASH,
        )
        self.assertEqual(
            overlay._canonical_self_hash(self.policy),
            overlay.OVERLAY_POLICY_CANONICAL_SELF_HASH,
        )
        self.assertEqual(
            self.static["root_manifest"]["canonical_self_hash"],
            "2d453eee6a44ea57fedfe8dd05b28c72b92ed3496f219db1b3d727de9d7969cd",
        )
        self.assertEqual(
            {
                split: binding["canonical_self_hash"]
                for split, binding in self.static["split_manifests"].items()
            },
            {
                "train": "3295bfbf2521eb3cd8f7ef74a53d8552bbe808624a8fc4754305edf5acb5f651",
                "development": "4d55cbb5a413fb96b17a5290c0ac30951d86527802056cf872e47f07ace7a8f3",
                "audit_a": "3daa4a516e0414c36a0d61fdc992d9d344c57eb8f1652261df2dace495d47fd0",
                "audit_b": "5a32a4097949e3c17d731e4e2b75f3cf8c64ea823b7ea75554489ba3c81da61f",
            },
        )

    def test_frozen_policy_bytes_and_closed_authorization_remain_unchanged(self) -> None:
        spec = self.policy["frozen_quality_contract"]["policy"]
        path = ROOT / spec["path"]
        before = path.read_bytes()
        overlay._validate_static_inputs(self.policy)
        after = path.read_bytes()
        self.assertEqual(before, after)
        self.assertEqual(len(after), spec["size_bytes"])
        self.assertEqual(hashlib.sha256(after).hexdigest(), spec["sha256"])
        authorization = self.static["quality_policy"]["authorization"]
        self.assertTrue(authorization["implementation_and_fixture_tests"])
        self.assertTrue(
            all(
                value is False
                for key, value in authorization.items()
                if key != "implementation_and_fixture_tests"
            )
        )

    def test_original_public_runner_remains_unauthorized(self) -> None:
        with self.assertRaisesRegex(
            frozen_runner.QualityAuditRunnerError, "unauthorized"
        ):
            frozen_runner.run_formal_quality_audit()

    def test_public_entry_is_parameterless_and_cli_rejects_arguments(self) -> None:
        self.assertEqual(len(inspect.signature(overlay.run_quality_audit_once).parameters), 0)
        with mock.patch.object(sys, "argv", ["entry.py", "--mode", "other"]), mock.patch.object(
            overlay, "run_quality_audit_once"
        ) as run:
            with self.assertRaisesRegex(
                overlay.QualityAuditAuthorizationError, "accepts no arguments"
            ):
                overlay.main()
        run.assert_not_called()

    def test_strict_json_rejects_duplicate_keys_and_non_finite_values(self) -> None:
        with self.assertRaisesRegex(
            overlay.QualityAuditAuthorizationError, "duplicate"
        ):
            overlay._strict_json_object(b'{"a":1,"a":2}', label="fixture")
        with self.assertRaisesRegex(
            overlay.QualityAuditAuthorizationError, "non-finite"
        ):
            overlay._strict_json_object(b'{"a":NaN}', label="fixture")

    def test_exact_pending_receipt_accepts_and_binding_mutation_fails(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            pending = Path(temp) / "quality_authorization.json"
            payload = self._receipt_payload()
            self._write_receipt(pending, payload)
            with mock.patch.object(
                overlay,
                "_receipt_paths",
                return_value=(pending, pending.relative_to(ROOT).as_posix()),
            ), mock.patch.object(
                overlay,
                "_git_identity",
                return_value=(self.git_commit, self.git_tree),
            ):
                verified = overlay._load_and_validate_pending_receipt(
                    policy=self.policy, static=self.static
                )
                self.assertEqual(
                    verified.receipt_id, payload["canonical_self_hash"]
                )
                self.assertEqual(verified.raw, pending.read_bytes())

                mutated = copy.deepcopy(payload)
                mutated["capabilities"]["audit_a_b_truth_open"] = True
                unsigned = dict(mutated)
                unsigned.pop("canonical_self_hash")
                mutated["canonical_self_hash"] = hashlib.sha256(
                    overlay._canonical_json_bytes(unsigned)
                ).hexdigest()
                self._write_receipt(pending, mutated)
                with self.assertRaisesRegex(
                    overlay.QualityAuditAuthorizationError, "binding drift"
                ):
                    overlay._load_and_validate_pending_receipt(
                        policy=self.policy, static=self.static
                    )

    def test_review_metadata_and_self_hash_are_fail_closed(self) -> None:
        payload = self._receipt_payload()
        payload["review_conversation_url"] = "https://example.invalid/review"
        unsigned = dict(payload)
        unsigned.pop("canonical_self_hash")
        payload["canonical_self_hash"] = hashlib.sha256(
            overlay._canonical_json_bytes(unsigned)
        ).hexdigest()
        with self.assertRaisesRegex(
            overlay.QualityAuditAuthorizationError, "review metadata"
        ):
            overlay._validate_receipt_payload(
                payload=payload,
                policy=self.policy,
                static=self.static,
                git_commit=self.git_commit,
                git_tree=self.git_tree,
            )

    def test_receipt_binds_git_wrapper_policy_runner_validator_and_manifests(self) -> None:
        mutation_paths = (
            ("authorization_entry_source", "sha256"),
            ("frozen_quality_policy", "sha256"),
            ("frozen_quality_runner", "sha256"),
            ("frozen_quality_validator", "sha256"),
            ("root_manifest", "sha256"),
            ("split_manifests", "train", "sha256"),
        )
        for path in mutation_paths:
            payload = self._receipt_payload()
            target = payload
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = "0" * 64
            unsigned = dict(payload)
            unsigned.pop("canonical_self_hash")
            payload["canonical_self_hash"] = hashlib.sha256(
                overlay._canonical_json_bytes(unsigned)
            ).hexdigest()
            with self.subTest(path=path), self.assertRaisesRegex(
                overlay.QualityAuditAuthorizationError, "binding drift"
            ):
                overlay._validate_receipt_payload(
                    payload=payload,
                    policy=self.policy,
                    static=self.static,
                    git_commit=self.git_commit,
                    git_tree=self.git_tree,
                )

        payload = self._receipt_payload()
        payload["git_commit"] = "1" * 40
        unsigned = dict(payload)
        unsigned.pop("canonical_self_hash")
        payload["canonical_self_hash"] = hashlib.sha256(
            overlay._canonical_json_bytes(unsigned)
        ).hexdigest()
        with self.assertRaisesRegex(
            overlay.QualityAuditAuthorizationError, "binding drift"
        ):
            overlay._validate_receipt_payload(
                payload=payload,
                policy=self.policy,
                static=self.static,
                git_commit=self.git_commit,
                git_tree=self.git_tree,
            )

    def test_git_identity_requires_a_fully_clean_worktree(self) -> None:
        dirty = mock.Mock(
            stdout="?? browser-temp\n", stderr="", returncode=0
        )
        with mock.patch.object(overlay.subprocess, "run", return_value=dirty) as run:
            with self.assertRaisesRegex(
                overlay.QualityAuditAuthorizationError, "not clean"
            ):
                overlay._git_identity()
        self.assertEqual(
            run.call_args.args[0],
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        )
        payload = self._receipt_payload()
        payload["canonical_self_hash"] = "d" * 64
        with self.assertRaisesRegex(
            overlay.QualityAuditAuthorizationError, "self-hash"
        ):
            overlay._validate_receipt_payload(
                payload=payload,
                policy=self.policy,
                static=self.static,
                git_commit=self.git_commit,
                git_tree=self.git_tree,
            )

    def test_consumption_is_atomic_and_receipt_cannot_be_reused(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            pending = Path(temp) / "quality_authorization.json"
            payload = self._receipt_payload()
            raw = self._write_receipt(pending, payload)
            receipt = overlay.VerifiedQualityAuditReceipt(
                path=pending,
                raw=raw,
                size_bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),
                receipt_id=str(payload["canonical_self_hash"]),
                payload=payload,
            )
            consumed = overlay._consume_receipt(receipt)
            consumed_path = ROOT / consumed["path"]
            self.assertFalse(pending.exists())
            self.assertEqual(consumed_path.read_bytes(), raw)
            pending.write_bytes(raw)
            with self.assertRaisesRegex(
                overlay.QualityAuditAuthorizationError, "already consumed"
            ):
                overlay._consume_receipt(receipt)

    def test_consumed_path_is_revalidated_before_audit(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            pending = Path(temp) / "quality_authorization.json"
            payload = self._receipt_payload()
            raw = self._write_receipt(pending, payload)
            receipt = overlay.VerifiedQualityAuditReceipt(
                path=pending,
                raw=raw,
                size_bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),
                receipt_id=str(payload["canonical_self_hash"]),
                payload=payload,
            )
            consumed = overlay._consume_receipt(receipt)
            with mock.patch.object(
                overlay,
                "_git_identity",
                return_value=(self.git_commit, self.git_tree),
            ):
                overlay._validate_consumed_receipt(
                    receipt=receipt,
                    consumed_file=consumed,
                    policy=self.policy,
                    static=self.static,
                )
                bad = dict(consumed)
                bad["path"] = pending.relative_to(ROOT).as_posix()
                with self.assertRaisesRegex(
                    overlay.QualityAuditAuthorizationError, "binding drift"
                ):
                    overlay._validate_consumed_receipt(
                        receipt=receipt,
                        consumed_file=bad,
                        policy=self.policy,
                        static=self.static,
                    )

    def test_missing_receipt_fails_before_body_or_result_creation(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            base = Path(temp)
            missing = base / "missing.json"
            result_directory = base / "result"
            result_path = result_directory / "quality_audit_receipt.json"
            temporary = base / ".result.tmp"
            with mock.patch.object(
                overlay, "_load_overlay_policy", return_value=self.policy
            ), mock.patch.object(
                overlay, "_validate_static_inputs", return_value=self.static
            ), mock.patch.object(
                overlay,
                "_result_paths",
                return_value=(result_directory, result_path, temporary),
            ), mock.patch.object(
                overlay,
                "_receipt_paths",
                return_value=(missing, missing.relative_to(ROOT).as_posix()),
            ), mock.patch.object(overlay, "_execute_frozen_body") as body:
                with self.assertRaisesRegex(
                    overlay.QualityAuditAuthorizationError, "remains unauthorized"
                ):
                    overlay.run_quality_audit_once()
            body.assert_not_called()
            self.assertFalse(result_directory.exists())

    def test_existing_result_blocks_before_receipt_lookup(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            base = Path(temp)
            result_directory = base / "result"
            result_directory.mkdir()
            with mock.patch.object(
                overlay, "_load_overlay_policy", return_value=self.policy
            ), mock.patch.object(
                overlay, "_validate_static_inputs", return_value=self.static
            ), mock.patch.object(
                overlay,
                "_result_paths",
                return_value=(
                    result_directory,
                    result_directory / "quality_audit_receipt.json",
                    base / ".result.tmp",
                ),
            ), mock.patch.object(
                overlay, "_load_and_validate_pending_receipt"
            ) as load_receipt:
                with self.assertRaisesRegex(
                    overlay.QualityAuditAuthorizationError, "resume is forbidden"
                ):
                    overlay.run_quality_audit_once()
            load_receipt.assert_not_called()

    def test_public_sequence_consumes_before_frozen_body(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            base = Path(temp)
            pending = base / "quality_authorization.json"
            payload = self._receipt_payload()
            raw = self._write_receipt(pending, payload)
            receipt = overlay.VerifiedQualityAuditReceipt(
                path=pending,
                raw=raw,
                size_bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),
                receipt_id=str(payload["canonical_self_hash"]),
                payload=payload,
            )
            result_directory = base / "result"
            result_path = result_directory / "quality_audit_receipt.json"
            temporary = base / ".result.tmp"
            events: list[str] = []

            def validate_consumed(**_kwargs: object) -> None:
                events.append("consumed_validated")
                self.assertFalse(pending.exists())
                self.assertFalse(result_directory.exists())

            def execute(_policy: object) -> dict[str, object]:
                events.append("body_called")
                self.assertFalse(pending.exists())
                self.assertTrue(any(base.glob("quality_authorization.consumed.*.json")))
                return {
                    "status": "PASS",
                    "formal_500_by_4_generated": False,
                    "training_started": False,
                }

            artifact = {
                "status": "PASS",
                "canonical_self_hash": "e" * 64,
            }
            with mock.patch.object(
                overlay, "_load_overlay_policy", return_value=self.policy
            ), mock.patch.object(
                overlay, "_validate_static_inputs", return_value=self.static
            ), mock.patch.object(
                overlay,
                "_result_paths",
                return_value=(result_directory, result_path, temporary),
            ), mock.patch.object(
                overlay,
                "_load_and_validate_pending_receipt",
                return_value=receipt,
            ), mock.patch.object(
                overlay,
                "_validate_consumed_receipt",
                side_effect=validate_consumed,
            ), mock.patch.object(
                overlay, "_execute_frozen_body", side_effect=execute
            ), mock.patch.object(
                overlay, "_build_result_artifact", return_value=artifact
            ):
                observed = overlay.run_quality_audit_once()
            self.assertEqual(observed, artifact)
            self.assertEqual(events, ["consumed_validated", "body_called"])
            self.assertTrue(result_path.is_file())
            self.assertFalse(temporary.exists())

    def test_frozen_body_is_called_with_unchanged_base_policy(self) -> None:
        quality_policy = self.static["quality_policy"]

        def body(*, policy: object, state: dict[str, str]) -> dict[str, object]:
            self.assertIs(policy, quality_policy)
            self.assertEqual(state["stage"], "authorized_overlay_entry")
            return {
                "status": "PASS",
                "formal_500_by_4_generated": False,
                "training_started": False,
            }

        with mock.patch.object(
            frozen_runner, "_run_authorized_formal_quality_audit", side_effect=body
        ):
            result = overlay._execute_frozen_body(quality_policy)
        self.assertEqual(result["status"], "PASS")
        self.assertFalse(quality_policy["authorization"]["quality_audit_run"])
        self.assertFalse(quality_policy["authorization"]["metric_generation"])

    def test_frozen_body_exception_is_hash_only_and_classified(self) -> None:
        secret = "row-label-secret"
        with mock.patch.object(
            frozen_runner,
            "_run_authorized_formal_quality_audit",
            side_effect=frozen_runner.DatasetGateFailure(secret),
        ):
            result = overlay._execute_frozen_body(self.static["quality_policy"])
        self.assertEqual(result["status"], "DATASET_INVALIDATED")
        self.assertEqual(result["row_level_labels_returned"], 0)
        self.assertNotIn(secret, json.dumps(result, ensure_ascii=False))
        self.assertEqual(
            result["exception_message_sha256"],
            hashlib.sha256(secret.encode("utf-8")).hexdigest(),
        )

        with mock.patch.object(
            frozen_runner,
            "_run_authorized_formal_quality_audit",
            side_effect=frozen_runner.AuditorExecutionFailure(secret),
        ):
            result = overlay._execute_frozen_body(self.static["quality_policy"])
        self.assertEqual(
            result["status"], "AUDITOR_EXECUTION_FAILED_NO_DATASET_CONCLUSION"
        )
        self.assertFalse(result["cleanup_required"])
        self.assertNotIn(secret, json.dumps(result, ensure_ascii=False))

    def test_result_artifact_rejects_formal_or_training_claim_drift(self) -> None:
        payload = self._receipt_payload()
        raw = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
        receipt = overlay.VerifiedQualityAuditReceipt(
            path=ROOT / "private_custody" / "fixture.json",
            raw=raw,
            size_bytes=len(raw),
            sha256=hashlib.sha256(raw).hexdigest(),
            receipt_id=str(payload["canonical_self_hash"]),
            payload=payload,
        )
        consumed = {
            "path": "private_custody/fixture.consumed." + receipt.sha256 + ".json",
            "size_bytes": len(raw),
            "sha256": receipt.sha256,
        }
        for key in ("formal_500_by_4_generated", "training_started"):
            result = {
                "status": "PASS",
                "formal_500_by_4_generated": False,
                "training_started": False,
            }
            result[key] = True
            with self.subTest(key=key), mock.patch.object(
                overlay,
                "_git_identity",
                return_value=(self.git_commit, self.git_tree),
            ), self.assertRaisesRegex(
                overlay.QualityAuditAuthorizationError, "claim boundary"
            ):
                overlay._build_result_artifact(
                    policy=self.policy,
                    static=self.static,
                    receipt=receipt,
                    consumed_file=consumed,
                    audit_result=result,
                )

    def test_result_artifact_requires_sealed_audit_truth_and_no_row_payload(self) -> None:
        payload = self._receipt_payload()
        raw = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
        receipt = overlay.VerifiedQualityAuditReceipt(
            path=ROOT / "private_custody" / "fixture.json",
            raw=raw,
            size_bytes=len(raw),
            sha256=hashlib.sha256(raw).hexdigest(),
            receipt_id=str(payload["canonical_self_hash"]),
            payload=payload,
        )
        consumed = {
            "path": "private_custody/fixture.consumed." + receipt.sha256 + ".json",
            "size_bytes": len(raw),
            "sha256": receipt.sha256,
        }
        base = {
            "status": "PASS",
            "formal_500_by_4_generated": False,
            "training_started": False,
        }
        with mock.patch.object(
            overlay,
            "_git_identity",
            return_value=(self.git_commit, self.git_tree),
        ), self.assertRaisesRegex(
            overlay.QualityAuditAuthorizationError, "truth seal"
        ):
            overlay._build_result_artifact(
                policy=self.policy,
                static=self.static,
                receipt=receipt,
                consumed_file=consumed,
                audit_result=base,
            )
        leaked = {
            **base,
            "audit_a_b_truth_remained_sealed": True,
            "row_level_labels": [0, 1],
        }
        with mock.patch.object(
            overlay,
            "_git_identity",
            return_value=(self.git_commit, self.git_tree),
        ), self.assertRaisesRegex(
            overlay.QualityAuditAuthorizationError, "row-level"
        ):
            overlay._build_result_artifact(
                policy=self.policy,
                static=self.static,
                receipt=receipt,
                consumed_file=consumed,
                audit_result=leaked,
            )

    def test_atomic_result_publication_leaves_no_temporary_file(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            base = Path(temp)
            result = base / "result.json"
            temporary = base / ".result.tmp"
            artifact = {
                "status": "PASS",
                "canonical_self_hash": "f" * 64,
            }
            overlay._publish_result(
                artifact=artifact,
                result_path=result,
                temporary_path=temporary,
            )
            self.assertEqual(
                json.loads(result.read_text(encoding="utf-8")), artifact
            )
            self.assertFalse(temporary.exists())
            with self.assertRaisesRegex(
                overlay.QualityAuditAuthorizationError, "already exists"
            ):
                overlay._publish_result(
                    artifact=artifact,
                    result_path=result,
                    temporary_path=temporary,
                )

    def test_result_publication_never_overwrites_a_concurrent_target(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            base = Path(temp)
            result = base / "result.json"
            temporary = base / ".result.tmp"
            artifact = {
                "status": "PASS",
                "canonical_self_hash": "f" * 64,
            }
            with mock.patch.object(
                overlay.os, "link", side_effect=FileExistsError("fixture race")
            ), self.assertRaisesRegex(
                overlay.QualityAuditAuthorizationError, "could not be published"
            ):
                overlay._publish_result(
                    artifact=artifact,
                    result_path=result,
                    temporary_path=temporary,
                )
            self.assertFalse(result.exists())
            self.assertFalse(temporary.exists())

    def test_wrapper_does_not_read_row_truth_or_generate_authorization(self) -> None:
        source = Path(overlay.__file__).read_text(encoding="utf-8")
        self.assertNotIn("pair_labels", source)
        self.assertNotIn("qrels", source)
        tree = ast.parse(source)
        function_names = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        forbidden = {
            name
            for name in function_names
            if name.startswith("generate_")
            or name.startswith("create_authorization")
            or name.startswith("issue_receipt")
        }
        self.assertEqual(forbidden, set())


if __name__ == "__main__":
    unittest.main()
