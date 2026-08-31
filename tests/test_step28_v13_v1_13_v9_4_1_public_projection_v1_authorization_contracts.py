from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_v9_4_1_public_projection_authority_common_v1 as common
import step28_v13_v1_13_v9_4_1_public_projection_authority_v1 as issuer
import step28_v13_v1_13_v9_4_1_public_projection_authorized_run_v1 as runner


class PublicProjectionAuthorizationContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = common.load_policy()
        cls.paths = common.issued_paths(cls.policy)

    def test_policy_is_implementation_only_and_projection_only(self) -> None:
        self.assertEqual(set(self.policy["authorization_state"].values()), {False})
        self.assertEqual(
            self.policy["claim_boundary"],
            "ONE_TIME_LABEL_FREE_PUBLIC_PROJECTION_ONLY",
        )
        workflow = self.policy["workflow"]
        self.assertFalse(workflow["supervision_or_audit_truth_allowed"])
        self.assertFalse(workflow["model_training_or_threshold_selection_allowed"])
        self.assertTrue(workflow["consume_before_first_formal_row_read"])
        self.assertTrue(workflow["failure_consumes_attempt"])

    def test_review_advice_must_pass_scientific_relevance_gate(self) -> None:
        contract = (
            ROOT
            / "docs/STEP28_V13_V1_13_V9_4_1_PUBLIC_PROJECTION_V1_ONE_TIME_AUTHORIZATION_CONTRACT_20260831.zh.md"
        ).read_text(encoding="utf-8")
        handoff = (ROOT / "docs/AI_RESEARCH_HANDOFF_20260823.zh.md").read_text(
            encoding="utf-8"
        )
        for token in (
            "SCIENTIFIC_BLOCKER",
            "REPRODUCIBILITY_DEFECT",
            "OUT_OF_SCOPE_OVERDESIGN",
            "采用满足合同的最小修复",
            "必须拒绝",
            "采用建议前必须写明它影响哪一项科研结论",
            "网页端模型只有审查权，没有决策权",
        ):
            self.assertIn(token, contract)
        self.assertIn("科研相关性强制审查门", handoff)
        self.assertIn("过度设计", handoff)

    def test_projection_parent_and_all_registered_bytes_are_exact(self) -> None:
        self.assertEqual(
            self.policy["projection_implementation_commit"],
            "a49151d64b8496a75d28420ac51338b3345d81df",
        )
        self.assertEqual(
            self.policy["projection_implementation_tree"],
            "3057823b57ad6b3f9c97743ddc7058c16130f671",
        )
        for role, spec in self.policy["projection_registry"].items():
            self.assertEqual(common.file_record(common.resolve(spec["path"])), spec, role)
            frozen = subprocess.run(
                [
                    "git",
                    "show",
                    f"{self.policy['projection_implementation_commit']}:{spec['path']}",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
            ).stdout
            self.assertEqual(len(frozen), spec["size_bytes"], role)
            self.assertEqual(hashlib.sha256(frozen).hexdigest(), spec["sha256"], role)
        observed_tree = subprocess.run(
            [
                "git",
                "rev-parse",
                f"{self.policy['projection_implementation_commit']}^{{tree}}",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout.strip()
        self.assertEqual(observed_tree, self.policy["projection_implementation_tree"])

    def test_issued_authorization_pins_exact_new_implementation_universe(self) -> None:
        expected = {
            "docs/STEP28_V13_V1_13_V9_4_1_PUBLIC_PROJECTION_V1_ONE_TIME_AUTHORIZATION_CONTRACT_20260831.zh.md",
            "schema/step28_v13_v1_13_v9_4_1_public_projection_authority_policy_v1.json",
            "scripts/step28_v13_v1_13_v9_4_1_public_projection_authority_common_v1.py",
            "scripts/step28_v13_v1_13_v9_4_1_public_projection_authority_v1.py",
            "scripts/step28_v13_v1_13_v9_4_1_public_projection_authorized_run_v1.py",
            "scripts/step28_v13_v1_13_v9_4_1_public_projection_base_prepare_worker_v1.py",
            "scripts/step28_v13_v1_13_v9_4_1_public_projection_identity_prepare_worker_v1.py",
            "scripts/step28_v13_v1_13_v9_4_1_public_projection_isolated_gpu_runner_v1.py",
            "scripts/run_step28_v13_v1_13_v9_4_1_public_projection_v1_linux_20260831.sh",
            "tests/test_step28_v13_v1_13_v9_4_1_public_projection_v1_authorization_contracts.py",
        }
        observed = set(
            self.policy["issued_authorization_contract"][
                "implementation_files_to_pin"
            ]
        )
        self.assertEqual(observed, expected)
        for relative in observed:
            self.assertTrue(common.resolve(relative).is_file())

    def test_attempt_paths_are_unique_and_formal_output_matches_public_policy(self) -> None:
        values = [str(path) for path in self.paths.values()]
        self.assertEqual(len(values), len(set(values)))
        public_policy = common.projection_common.load_policy()
        self.assertEqual(
            self.paths["output"],
            common.resolve(public_policy["formal_outputs"]["root"]),
        )
        self.assertNotEqual(self.paths["output"], self.paths["building"])
        self.assertNotEqual(self.paths["state_root"], self.paths["building"])

    def test_validate_contract_is_explicitly_nonexecuting(self) -> None:
        result = runner.validate_contract()
        self.assertFalse(result["formal_projection_authorized"])
        self.assertFalse(result["formal_projection_executed"])
        self.assertFalse(result["supervision_or_audit_truth_read"])
        self.assertFalse(result["model_training_authorized"])

    def test_missing_authorization_fails_without_creating_terminal_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._toy_issued_paths(Path(temporary))
            before = {name: path.exists() for name, path in paths.items()}
            with (
                mock.patch.object(runner.sys, "platform", "win32"),
                mock.patch.object(runner.authority, "load_policy", return_value=self.policy),
                mock.patch.object(runner, "_roots", return_value=paths),
                mock.patch.object(common, "issued_paths", return_value=paths),
                mock.patch.object(
                    runner.authority,
                    "validate_authorization",
                    side_effect=common.PublicProjectionAuthorityError("absent"),
                ),
            ):
                with self.assertRaises(common.PublicProjectionAuthorityError):
                    runner.prepare_windows()
            after = {name: path.exists() for name, path in paths.items()}
            self.assertEqual(after, before)

    def test_authority_source_never_returns_or_prints_raw_key(self) -> None:
        source = (SCRIPTS / issuer.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        self.assertNotIn("read_text", source)
        self.assertNotIn("formal_dataset", source)
        self.assertNotIn("identity33", source)
        self.assertNotIn("pair_labels", source)
        return_keys = {
            node.slice.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "authorization"
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        }
        self.assertNotIn("raw_key", return_keys)

    def _toy_issued_paths(self, root: Path) -> dict[str, Path]:
        authority_root = root / "private"
        state_root = root / "state"
        return {
            "authorization": root / "authorization.json",
            "authority_root": authority_root,
            "key": authority_root / "projection_key.bin",
            "issuance_claim": authority_root / "issuance_claim.json",
            "issuance_failure": authority_root / "issuance.failed.json",
            "output": root / "output",
            "building": root / "building",
            "state_root": state_root,
            "consumption": state_root / "consumption.json",
            "key_cleanup": state_root / "key_cleanup.json",
            "prepared": state_root / "prepared.json",
            "linux_claim": state_root / "linux_claim.json",
            "linux_completion": state_root / "linux_completion.json",
            "failure": state_root / "failure.json",
            "completion": state_root / "completion.json",
        }

    def _toy_run_paths(self, root: Path) -> dict[str, Path]:
        paths = self._toy_issued_paths(root)
        paths.update(
            {
                "cpu": paths["building"] / ".base_cpu_stage_v1",
                "transfer": paths["building"] / ".base_transfer_v1",
                "gpu_return": paths["building"] / ".base_gpu_return_v1",
                "base": paths["building"] / "base_v1",
                "identity": paths["building"] / "identity_v1",
            }
        )
        return paths

    @staticmethod
    def _write_self_hashed(path: Path, body: dict[str, object]) -> dict[str, object]:
        value = dict(body)
        value["canonical_self_hash"] = common.canonical_sha256(value)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return value

    def _write_valid_consumption_claim(
        self, paths: dict[str, Path], auth: dict[str, object]
    ) -> dict[str, object]:
        return self._write_self_hashed(
            paths["consumption"],
            {
                "version": runner.CONSUMPTION_VERSION,
                "status": "PUBLIC_PROJECTION_AUTHORITY_CONSUMED_BEFORE_FORMAL_ROW_READ",
                "authorization_sha256": common.sha256_file(paths["authorization"]),
                "authorization_canonical_self_hash": auth["canonical_self_hash"],
                "implementation_commit": auth["implementation_commit"],
                "implementation_tree": auth["implementation_tree"],
                "key_commitment_sha256": auth["key_file"]["commitment_sha256"],
                "formal_rows_read_at_consumption": 0,
                "supervision_or_audit_truth_read": False,
                "model_training_authorized": False,
                "rerun_authorized": False,
            },
        )

    def test_issuer_writes_claim_before_one_key_and_returns_only_commitment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._toy_issued_paths(root)
            toy_policy = copy.deepcopy(self.policy)
            fixed = b"p" * 32
            with (
                mock.patch.object(issuer.common, "load_policy", return_value=toy_policy),
                mock.patch.object(issuer.common, "git_status_lines", return_value=[]),
                mock.patch.object(issuer.common, "issued_paths", return_value=paths),
                mock.patch.object(issuer.common, "git_head", return_value="1" * 40),
                mock.patch.object(issuer.common, "git_tree", return_value="2" * 40),
                mock.patch.object(
                    issuer.common, "implementation_file_records", return_value={}
                ),
                mock.patch.object(
                    issuer.projection_common,
                    "load_policy",
                    return_value={"canonical_self_hash": "3" * 64},
                ),
                mock.patch.object(issuer.secrets, "token_bytes", return_value=fixed),
            ):
                result = issuer.issue()
            self.assertTrue(paths["issuance_claim"].is_file())
            self.assertEqual(paths["key"].read_bytes(), fixed)
            self.assertTrue(paths["authorization"].is_file())
            self.assertFalse(result["raw_key_returned"])
            self.assertNotIn(fixed.hex(), json.dumps(result))
            self.assertEqual(
                result["key_commitment_sha256"], hashlib.sha256(fixed).hexdigest()
            )

    def test_resealed_issuance_claim_semantic_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            paths = self._toy_issued_paths(Path(temporary))
            toy_policy = copy.deepcopy(self.policy)
            issued = toy_policy["issued_authorization_contract"]
            execution = toy_policy["execution_paths"]
            issued["authorization_path"] = paths["authorization"].relative_to(
                ROOT
            ).as_posix()
            issued["authority_root"] = paths["authority_root"].relative_to(
                ROOT
            ).as_posix()
            issued["key_path"] = paths["key"].relative_to(ROOT).as_posix()
            issued["issuance_claim_path"] = paths["issuance_claim"].relative_to(
                ROOT
            ).as_posix()
            issued["issuance_failure_path"] = paths[
                "issuance_failure"
            ].relative_to(ROOT).as_posix()
            execution["formal_output_root"] = paths["output"].relative_to(
                ROOT
            ).as_posix()
            execution["building_root"] = paths["building"].relative_to(
                ROOT
            ).as_posix()
            execution["state_root"] = paths["state_root"].relative_to(
                ROOT
            ).as_posix()
            with (
                mock.patch.object(issuer.common, "load_policy", return_value=toy_policy),
                mock.patch.object(issuer.common, "git_status_lines", return_value=[]),
                mock.patch.object(issuer.common, "issued_paths", return_value=paths),
                mock.patch.object(
                    issuer.secrets,
                    "token_bytes",
                    return_value=b"z" * 32,
                ),
            ):
                issuer.issue()
                common.validate_authorization(toy_policy, require_raw_key=True)
                claim = common.load_json(paths["issuance_claim"])
                claim["candidate_draws_at_claim"] = 1
                claim.pop("canonical_self_hash")
                claim["canonical_self_hash"] = common.canonical_sha256(claim)
                paths["issuance_claim"].write_text(
                    json.dumps(claim, ensure_ascii=False, indent=2, sort_keys=True)
                    + "\n",
                    encoding="utf-8",
                    newline="\n",
                )
                auth = common.load_json(paths["authorization"])
                auth["issuance_claim_sha256"] = common.sha256_file(
                    paths["issuance_claim"]
                )
                auth.pop("canonical_self_hash")
                auth["canonical_self_hash"] = common.canonical_sha256(auth)
                paths["authorization"].write_text(
                    json.dumps(auth, ensure_ascii=False, indent=2, sort_keys=True)
                    + "\n",
                    encoding="utf-8",
                    newline="\n",
                )
                with self.assertRaises(common.PublicProjectionAuthorityError):
                    common.validate_authorization(toy_policy, require_raw_key=True)

    def test_issuer_draw_failure_retires_attempt_and_keeps_no_key(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._toy_issued_paths(root)
            with (
                mock.patch.object(issuer.common, "load_policy", return_value=self.policy),
                mock.patch.object(issuer.common, "git_status_lines", return_value=[]),
                mock.patch.object(issuer.common, "issued_paths", return_value=paths),
                mock.patch.object(issuer.common, "git_head", return_value="1" * 40),
                mock.patch.object(issuer.common, "git_tree", return_value="2" * 40),
                mock.patch.object(
                    issuer.secrets, "token_bytes", side_effect=RuntimeError("draw failed")
                ),
            ):
                with self.assertRaises(RuntimeError):
                    issuer.issue()
            self.assertTrue(paths["issuance_claim"].is_file())
            self.assertTrue(paths["issuance_failure"].is_file())
            self.assertFalse(paths["key"].exists())
            self.assertFalse(paths["authorization"].exists())
            failure = common.load_json(paths["issuance_failure"])
            self.assertFalse(failure["rerun_authorized"])
            self.assertFalse(failure["raw_key_material_retained"])

    def test_consumption_deletes_raw_key_before_preparation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._toy_issued_paths(Path(temporary))
            paths["authorization"].write_text("{}\n", encoding="utf-8")
            paths["authority_root"].mkdir(exist_ok=True)
            paths["key"].write_bytes(b"k" * 32)
            auth = {
                "canonical_self_hash": "a" * 64,
                "implementation_commit": "1" * 40,
                "implementation_tree": "2" * 40,
                "key_file": {"commitment_sha256": hashlib.sha256(b"k" * 32).hexdigest()},
            }
            with (
                mock.patch.object(runner, "_roots", return_value=paths),
                mock.patch.object(common, "issued_paths", return_value=paths),
            ):
                consumption, cleanup = runner._consume_authority(self.policy, auth)
            self.assertFalse(paths["key"].exists())
            self.assertTrue(paths["consumption"].is_file())
            self.assertTrue(paths["key_cleanup"].is_file())
            self.assertEqual(consumption["formal_rows_read_at_consumption"], 0)
            self.assertFalse(cleanup["raw_key_material_retained"])
            tampered = common.load_json(paths["consumption"])
            tampered["extra"] = "resealed"
            tampered.pop("canonical_self_hash")
            tampered["canonical_self_hash"] = common.canonical_sha256(tampered)
            paths["consumption"].write_text(
                json.dumps(tampered, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
                newline="\n",
            )
            with (
                mock.patch.object(runner, "_roots", return_value=paths),
                mock.patch.object(common, "issued_paths", return_value=paths),
                self.assertRaises(common.PublicProjectionAuthorityError),
            ):
                runner._validate_consumption(self.policy, auth)

    def test_consumed_failure_deletes_large_staging_and_blocks_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._toy_issued_paths(Path(temporary))
            paths["authorization"].write_text("{}\n", encoding="utf-8")
            paths["building"].mkdir()
            (paths["building"] / "large.bin").write_bytes(b"x" * 1024)
            paths["key"].parent.mkdir(parents=True, exist_ok=True)
            paths["key"].write_bytes(b"k" * 32)
            auth = {
                "canonical_self_hash": "a" * 64,
                "implementation_commit": "1" * 40,
                "implementation_tree": "2" * 40,
                "key_file": {
                    "commitment_sha256": hashlib.sha256(b"k" * 32).hexdigest()
                },
            }
            self._write_valid_consumption_claim(paths, auth)
            with mock.patch.object(runner, "_roots", return_value=paths):
                runner._write_terminal_failure(
                    self.policy,
                    auth,
                    stage="test",
                    exc=RuntimeError("boom"),
                )
            self.assertFalse(paths["building"].exists())
            self.assertFalse(paths["key"].exists())
            failure = common.load_json(paths["failure"])
            self.assertFalse(failure["rerun_authorized"])
            self.assertFalse(failure["scientific_result_valid"])

    def test_corrupt_consumption_file_never_retires_unconsumed_attempt(self) -> None:
        for function, platform in (
            (runner.encode_linux, "linux"),
            (runner.finalize_windows, "win32"),
        ):
            with self.subTest(function=function.__name__), tempfile.TemporaryDirectory() as temporary:
                paths = self._toy_run_paths(Path(temporary))
                paths["authorization"].write_text("{}\n", encoding="utf-8")
                paths["authority_root"].mkdir(parents=True, exist_ok=True)
                paths["key"].write_bytes(b"k" * 32)
                paths["consumption"].parent.mkdir(parents=True)
                paths["consumption"].write_text("{}\n", encoding="utf-8")
                corrupt_consumption_bytes = paths["consumption"].read_bytes()
                output_marker = paths["output"] / "must_not_be_deleted.bin"
                output_marker.parent.mkdir(parents=True)
                output_marker.write_bytes(b"preserve")
                auth = {"canonical_self_hash": "a" * 64}
                with (
                    mock.patch.object(runner.sys, "platform", platform),
                    mock.patch.object(
                        runner.authority, "load_policy", return_value=self.policy
                    ),
                    mock.patch.object(runner, "_roots", return_value=paths),
                    mock.patch.object(
                        runner.authority,
                        "validate_authorization",
                        return_value=auth,
                    ),
                    mock.patch.object(
                        runner,
                        "_validate_consumption",
                        side_effect=common.PublicProjectionAuthorityError("corrupt"),
                    ),
                    self.assertRaises(common.PublicProjectionAuthorityError),
                ):
                    function()
                self.assertEqual(paths["key"].read_bytes(), b"k" * 32)
                self.assertEqual(
                    paths["consumption"].read_bytes(), corrupt_consumption_bytes
                )
                self.assertEqual(output_marker.read_bytes(), b"preserve")
                self.assertFalse(paths["failure"].exists())

    def test_success_state_blocks_prepare_and_encode_without_mutation(self) -> None:
        for function, platform in (
            (runner.prepare_windows, "win32"),
            (runner.encode_linux, "linux"),
        ):
            with self.subTest(function=function.__name__), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                paths = self._toy_run_paths(root)
                output_marker = paths["output"] / "published.bin"
                output_marker.parent.mkdir(parents=True)
                output_marker.write_bytes(b"published")
                self._write_self_hashed(
                    paths["completion"], {"status": "completed"}
                )

                def snapshot() -> dict[str, tuple[str, bytes | None]]:
                    return {
                        path.relative_to(root).as_posix(): (
                            "file" if path.is_file() else "directory",
                            path.read_bytes() if path.is_file() else None,
                        )
                        for path in sorted(root.rglob("*"))
                    }

                before = snapshot()
                with (
                    mock.patch.object(runner.sys, "platform", platform),
                    mock.patch.object(
                        runner.authority, "load_policy", return_value=self.policy
                    ),
                    mock.patch.object(runner, "_roots", return_value=paths),
                    mock.patch.object(
                        runner.authority, "validate_authorization"
                    ) as validate_authorization,
                    self.assertRaises(common.PublicProjectionAuthorityError),
                ):
                    function()
                validate_authorization.assert_not_called()
                self.assertEqual(snapshot(), before)
                self.assertFalse(paths["failure"].exists())

    def test_mismatched_completion_cannot_recover_as_success(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            paths = self._toy_run_paths(Path(temporary))
            paths["authorization"].write_text("{}\n", encoding="utf-8")
            auth = {
                "canonical_self_hash": "a" * 64,
                "implementation_commit": "1" * 40,
                "implementation_tree": "2" * 40,
                "key_file": {"commitment_sha256": "b" * 64},
            }
            self._write_valid_consumption_claim(paths, auth)
            paths["prepared"].write_text("{}\n", encoding="utf-8")
            paths["linux_completion"].write_text("{}\n", encoding="utf-8")
            manifest = self._write_self_hashed(
                paths["output"] / "public_projection_manifest.json",
                {"status": "published"},
            )
            self._write_self_hashed(
                paths["completion"], {"status": "wrong-publication"}
            )

            with (
                mock.patch.object(runner.sys, "platform", "win32"),
                mock.patch.object(
                    runner.authority, "load_policy", return_value=self.policy
                ),
                mock.patch.object(runner, "_roots", return_value=paths),
                mock.patch.object(
                    runner.authority, "validate_authorization", return_value=auth
                ),
                mock.patch.object(runner, "_validate_consumption"),
                mock.patch.object(
                    runner,
                    "_validate_prepared_full",
                    side_effect=common.PublicProjectionAuthorityError(
                        "published staging is absent"
                    ),
                ),
                mock.patch.object(
                    runner.projection_common, "load_policy", return_value={}
                ),
                mock.patch.object(
                    runner.protocol,
                    "validate_publication",
                    return_value=manifest,
                ),
                self.assertRaises(common.PublicProjectionAuthorityError),
            ):
                runner.finalize_windows()

            self.assertFalse(paths["output"].exists())
            self.assertTrue(paths["failure"].is_file())
            failure = common.load_json(paths["failure"])
            self.assertFalse(failure["scientific_result_valid"])
            self.assertFalse(failure["rerun_authorized"])

    def test_unissued_encode_and_finalize_do_not_create_or_delete_state(self) -> None:
        for function, platform in (
            (runner.encode_linux, "linux"),
            (runner.finalize_windows, "win32"),
        ):
            with self.subTest(function=function.__name__), tempfile.TemporaryDirectory() as temporary:
                paths = self._toy_run_paths(Path(temporary))
                marker = paths["building"] / "untouched.bin"
                marker.parent.mkdir(parents=True)
                marker.write_bytes(b"untouched")
                before = {
                    name: (path.exists(), path.read_bytes() if path.is_file() else None)
                    for name, path in paths.items()
                }
                with (
                    mock.patch.object(runner.sys, "platform", platform),
                    mock.patch.object(
                        runner.authority, "load_policy", return_value=self.policy
                    ),
                    mock.patch.object(runner, "_roots", return_value=paths),
                    mock.patch.object(
                        runner.authority,
                        "validate_authorization",
                        side_effect=common.PublicProjectionAuthorityError("absent"),
                    ),
                    self.assertRaises(common.PublicProjectionAuthorityError),
                ):
                    function()
                after = {
                    name: (path.exists(), path.read_bytes() if path.is_file() else None)
                    for name, path in paths.items()
                }
                self.assertEqual(after, before)
                self.assertEqual(marker.read_bytes(), b"untouched")

    def test_issued_unconsumed_encode_and_finalize_preserve_raw_key(self) -> None:
        for function, platform in (
            (runner.encode_linux, "linux"),
            (runner.finalize_windows, "win32"),
        ):
            with self.subTest(function=function.__name__), tempfile.TemporaryDirectory() as temporary:
                paths = self._toy_run_paths(Path(temporary))
                paths["authorization"].write_text("{}\n", encoding="utf-8")
                paths["authority_root"].mkdir(parents=True, exist_ok=True)
                paths["key"].write_bytes(b"k" * 32)
                output_marker = paths["output"] / "must_not_be_deleted.bin"
                output_marker.parent.mkdir(parents=True)
                output_marker.write_bytes(b"preserve")
                auth = {"canonical_self_hash": "a" * 64}
                with (
                    mock.patch.object(runner.sys, "platform", platform),
                    mock.patch.object(
                        runner.authority, "load_policy", return_value=self.policy
                    ),
                    mock.patch.object(runner, "_roots", return_value=paths),
                    mock.patch.object(
                        runner.authority,
                        "validate_authorization",
                        return_value=auth,
                    ),
                    mock.patch.object(
                        runner,
                        "_validate_consumption",
                        side_effect=common.PublicProjectionAuthorityError("unconsumed"),
                    ),
                    self.assertRaises(common.PublicProjectionAuthorityError),
                ):
                    function()
                self.assertEqual(paths["key"].read_bytes(), b"k" * 32)
                self.assertEqual(output_marker.read_bytes(), b"preserve")
                self.assertFalse(paths["failure"].exists())
                self.assertFalse(paths["state_root"].exists())

    def test_prepare_transition_consumes_key_before_any_builder(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            paths = self._toy_run_paths(Path(temporary))
            paths["authorization"].write_text("{}\n", encoding="utf-8")
            paths["authority_root"].mkdir(parents=True)
            paths["key"].write_bytes(b"k" * 32)
            auth = {
                "canonical_self_hash": "a" * 64,
                "implementation_commit": "1" * 40,
                "implementation_tree": "2" * 40,
                "key_file": {
                    "commitment_sha256": hashlib.sha256(b"k" * 32).hexdigest()
                },
            }

            def fake_worker(relative_path):
                self.assertTrue(paths["consumption"].is_file())
                self.assertTrue(paths["key_cleanup"].is_file())
                self.assertFalse(paths["key"].exists())
                if relative_path == runner.BASE_PREPARE_WORKER:
                    self._write_self_hashed(
                        paths["cpu"] / "cpu_stage_manifest.json", {"status": "cpu"}
                    )
                    self._write_self_hashed(
                        paths["transfer"] / "transfer_manifest.json",
                        {"status": "transfer"},
                    )
                elif relative_path == runner.IDENTITY_PREPARE_WORKER:
                    self._write_self_hashed(
                        paths["identity"] / "identity_projection_manifest.json",
                        {"status": "identity"},
                    )
                else:
                    raise AssertionError(relative_path)

            cpu = {"canonical_self_hash": "c" * 64}
            transfer = {"canonical_self_hash": "d" * 64}
            identity = {"canonical_self_hash": "e" * 64}

            expected_status = [
                f"?? {paths['authorization'].relative_to(ROOT).as_posix()}"
            ]
            with (
                mock.patch.object(runner.sys, "platform", "win32"),
                mock.patch.object(runner.authority, "load_policy", return_value=self.policy),
                mock.patch.object(runner, "_roots", return_value=paths),
                mock.patch.object(common, "issued_paths", return_value=paths),
                mock.patch.object(
                    runner.authority, "validate_authorization", return_value=auth
                ),
                mock.patch.object(
                    runner.authority, "git_status_lines", return_value=expected_status
                ),
                mock.patch.object(runner.projection_common, "load_policy", return_value={}),
                mock.patch.object(runner.gpu_common, "load_policy", return_value={}),
                mock.patch.object(
                    runner, "_run_prepare_worker", side_effect=fake_worker
                ),
                mock.patch.object(
                    runner.base_finalizer,
                    "validate_cpu_stage",
                    return_value=(cpu, []),
                ),
                mock.patch.object(
                    runner.gpu_encoder,
                    "validate_transfer",
                    return_value=(transfer, []),
                ),
                mock.patch.object(
                    runner.identity_builder,
                    "validate_identity_output",
                    return_value=identity,
                ),
                mock.patch.object(runner, "_validate_prepared_full"),
            ):
                result = runner.prepare_windows()
            self.assertEqual(
                result["status"], "PREPARED_UNPUBLISHED_LABEL_FREE_PUBLIC_PROJECTION"
            )
            self.assertTrue(paths["prepared"].is_file())
            self.assertFalse(paths["failure"].exists())

    def test_linux_transition_claims_before_materialize_and_cleans_workspace(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            paths = self._toy_run_paths(Path(temporary))
            for path in (
                paths["authorization"],
                paths["consumption"],
                paths["key_cleanup"],
                paths["prepared"],
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n", encoding="utf-8")
            paths["transfer"].mkdir(parents=True)
            self._write_self_hashed(
                paths["transfer"] / "transfer_manifest.json", {"status": "transfer"}
            )
            workspace = Path(temporary) / "workspace"
            transfer_manifest = {"canonical_self_hash": "b" * 64}
            auth = {
                "canonical_self_hash": "a" * 64,
                "implementation_files": {
                    runner.ISOLATED_RUNNER: {
                        "path": runner.ISOLATED_RUNNER,
                        "size_bytes": 1,
                        "sha256": "c" * 64,
                    }
                },
            }

            def fake_mkdtemp(**_kwargs):
                workspace.mkdir()
                return str(workspace)

            def fake_materialize(_policy, _transfer, target, _runner_spec):
                self.assertTrue(paths["linux_claim"].is_file())
                target.mkdir()
                return {}

            def fake_collect(_policy, _transfer, _workspace, destination, _runner):
                self._write_self_hashed(
                    destination / "gpu_return_manifest.json", {"status": "gpu"}
                )
                return {}

            gpu_manifest = {"canonical_self_hash": "d" * 64}
            with (
                mock.patch.object(runner.sys, "platform", "linux"),
                mock.patch.object(runner.authority, "load_policy", return_value=self.policy),
                mock.patch.object(runner, "_roots", return_value=paths),
                mock.patch.object(
                    runner.authority, "validate_authorization", return_value=auth
                ),
                mock.patch.object(runner, "_validate_consumption"),
                mock.patch.object(
                    runner,
                    "_validate_prepared_for_linux",
                    return_value=({}, transfer_manifest, []),
                ),
                mock.patch.object(runner.gpu_common, "load_policy", return_value={}),
                mock.patch.object(runner.tempfile, "mkdtemp", side_effect=fake_mkdtemp),
                mock.patch.object(
                    runner.materializer,
                    "materialize_workspace",
                    side_effect=fake_materialize,
                ),
                mock.patch.object(runner.subprocess, "run", return_value=None),
                mock.patch.object(
                    runner.materializer,
                    "collect_gpu_return",
                    side_effect=fake_collect,
                ),
                mock.patch.object(
                    runner.gpu_encoder,
                    "validate_gpu_return",
                    return_value=(gpu_manifest, []),
                ),
            ):
                result = runner.encode_linux()
            self.assertEqual(
                result["status"], "COMPLETED_ONE_TIME_OPAQUE_LINUX_GPU_PROJECTION"
            )
            self.assertTrue(paths["linux_claim"].is_file())
            self.assertTrue(paths["linux_completion"].is_file())
            self.assertTrue(paths["gpu_return"].is_dir())
            self.assertFalse(workspace.exists())
            self.assertFalse(paths["failure"].exists())

    def test_finalize_transition_removes_intermediates_before_atomic_publish(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            paths = self._toy_run_paths(Path(temporary))
            for path in (
                paths["authorization"],
                paths["consumption"],
                paths["prepared"],
                paths["linux_completion"],
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n", encoding="utf-8")
            for name in ("cpu", "transfer", "gpu_return", "identity"):
                paths[name].mkdir(parents=True, exist_ok=True)
            auth = {"canonical_self_hash": "a" * 64}
            manifest = {"canonical_self_hash": "b" * 64}

            def fake_finalize(_policy, _cpu, _transfer, _gpu, base_root):
                base_root.mkdir()
                return {}

            def fake_freeze(_policy, publication_root):
                for name in ("cpu", "transfer", "gpu_return"):
                    self.assertFalse(paths[name].exists())
                self._write_self_hashed(
                    publication_root / "public_projection_manifest.json", manifest
                )
                return manifest

            with (
                mock.patch.object(runner.sys, "platform", "win32"),
                mock.patch.object(runner.authority, "load_policy", return_value=self.policy),
                mock.patch.object(runner, "_roots", return_value=paths),
                mock.patch.object(
                    runner.authority, "validate_authorization", return_value=auth
                ),
                mock.patch.object(runner, "_validate_consumption"),
                mock.patch.object(runner, "_validate_prepared_full"),
                mock.patch.object(runner, "_validate_linux_completion"),
                mock.patch.object(runner.projection_common, "load_policy", return_value={}),
                mock.patch.object(
                    runner.base_finalizer,
                    "finalize_to_temporary",
                    side_effect=fake_finalize,
                ),
                mock.patch.object(
                    runner.protocol,
                    "freeze_combined_manifest",
                    side_effect=fake_freeze,
                ),
                mock.patch.object(
                    runner.protocol,
                    "validate_publication",
                    return_value=manifest,
                ),
            ):
                result = runner.finalize_windows()
            self.assertEqual(
                result["status"], "COMPLETED_LABEL_FREE_FOUR_SPLIT_PUBLIC_PROJECTION"
            )
            self.assertFalse(paths["building"].exists())
            self.assertTrue(paths["output"].is_dir())
            self.assertEqual(
                {path.name for path in paths["output"].iterdir()},
                {"base_v1", "identity_v1", "public_projection_manifest.json"},
            )
            self.assertTrue(paths["completion"].is_file())
            self.assertFalse(paths["failure"].exists())

    def test_isolated_runner_has_only_gpu_projection_imports(self) -> None:
        path = SCRIPTS / (
            "step28_v13_v1_13_v9_4_1_public_projection_isolated_gpu_runner_v1.py"
        )
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        project_imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
            if alias.name.startswith("step")
        }
        self.assertEqual(
            project_imports,
            {
                "step28_v13_v1_13_v9_4_1_encode_base_projection_linux_v1",
                "step28_v13_v1_13_v9_4_1_public_projection_gpu_common_v1",
            },
        )
        for token in ("pair_labels", "identity33", "controller", "qrels"):
            self.assertNotIn(token, source)
        self.assertIn("encode_transfer_to_temporary", source)

    def test_windows_prepare_workers_have_disjoint_project_imports(self) -> None:
        expected = {
            runner.BASE_PREPARE_WORKER: {
                "step28_v13_v1_13_v9_4_1_prepare_base_projection_v1",
                "step28_v13_v1_13_v9_4_1_public_projection_authority_common_v1",
                "step28_v13_v1_13_v9_4_1_public_projection_common_v1",
            },
            runner.IDENTITY_PREPARE_WORKER: {
                "step28_v13_v1_13_v9_4_1_freeze_identity_projection_v2",
                "step28_v13_v1_13_v9_4_1_public_projection_authority_common_v1",
                "step28_v13_v1_13_v9_4_1_public_projection_common_v1",
            },
        }
        for relative, expected_imports in expected.items():
            source = (ROOT / relative).read_text(encoding="utf-8")
            tree = ast.parse(source)
            observed = {
                alias.name
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
                if alias.name.startswith("step")
            }
            self.assertEqual(observed, expected_imports, relative)
            self.assertLess(
                source.index("authority.validate_consumption"),
                source.index(
                    "base_preparer.prepare_to_temporary"
                    if relative == runner.BASE_PREPARE_WORKER
                    else "identity_builder.build_to_temporary"
                ),
                relative,
            )
        self.assertNotIn("prepare_base_projection", (ROOT / runner.IDENTITY_PREPARE_WORKER).read_text(encoding="utf-8"))
        self.assertNotIn("freeze_identity_projection", (ROOT / runner.BASE_PREPARE_WORKER).read_text(encoding="utf-8"))

    def test_linux_shell_clears_external_libraries_and_is_offline(self) -> None:
        source = (
            SCRIPTS
            / "run_step28_v13_v1_13_v9_4_1_public_projection_v1_linux_20260831.sh"
        ).read_text(encoding="utf-8")
        for fragment in (
            "-u LD_LIBRARY_PATH",
            "-u LD_PRELOAD",
            "CUBLAS_WORKSPACE_CONFIG=:4096:8",
            "HF_HUB_OFFLINE=1",
            "TRANSFORMERS_OFFLINE=1",
            "PYTHONDONTWRITEBYTECODE=1",
            "encode-linux",
        ):
            self.assertIn(fragment, source)

    def test_final_publication_universe_excludes_all_intermediates(self) -> None:
        execution = self.policy["execution_paths"]
        self.assertEqual(execution["base_subdirectory"], "base_v1")
        self.assertEqual(execution["identity_subdirectory"], "identity_v1")
        for key in (
            "cpu_stage_subdirectory",
            "transfer_subdirectory",
            "gpu_return_subdirectory",
        ):
            self.assertTrue(execution[key].startswith("."))
        source = (SCRIPTS / runner.__file__).read_text(encoding="utf-8")
        delete_index = source.index('for name in ("cpu", "transfer", "gpu_return")')
        publish_index = source.index("protocol.freeze_combined_manifest")
        self.assertLess(delete_index, publish_index)
        self.assertIn('shutil.rmtree(paths["output"], ignore_errors=True)', source)

    def test_consumption_and_linux_claim_precede_data_or_model_work(self) -> None:
        source = (SCRIPTS / runner.__file__).read_text(encoding="utf-8")
        self.assertLess(
            source.index('authority.write_json_exclusive(paths["consumption"]'),
            source.index('paths["key"].unlink()'),
        )
        self.assertLess(
            source.index('paths["key"].unlink()'),
            source.index("_run_prepare_worker(BASE_PREPARE_WORKER)"),
        )
        self.assertLess(
            source.index('authority.write_json_exclusive(paths["linux_claim"]'),
            source.index("materializer.materialize_workspace"),
        )

    def test_authorized_runner_does_not_call_training_or_truth_interfaces(self) -> None:
        source = (SCRIPTS / runner.__file__).read_text(encoding="utf-8")
        for token in (
            "pair_labels.csv",
            "qrels.jsonl",
            "controller_membership.jsonl",
            "fit_logistic",
            "fit_lightgbm",
            "model_training_core",
            "audit_a_truth",
            "audit_b_truth",
        ):
            self.assertNotIn(token, source)


if __name__ == "__main__":
    unittest.main()
