from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_v9_4_1_replay_full_english_compatibility_linux_v2 as replay


class FullEnglishCompatibilityV2Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = replay.load_policy()

    def test_policy_is_exact_self_hashed_and_grants_no_authority(self) -> None:
        raw = replay.POLICY_PATH.read_bytes()
        self.assertEqual(len(raw), replay.EXPECTED_POLICY_SIZE_BYTES)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), replay.EXPECTED_POLICY_SHA256)
        unsigned = dict(self.policy)
        claimed = unsigned.pop("canonical_self_hash")
        self.assertEqual(replay.base.canonical_sha256(unsigned), claimed)
        self.assertEqual(claimed, replay.EXPECTED_POLICY_SELF_HASH)
        self.assertFalse(any(self.policy["permissions"].values()))

    def test_policy_replays_complete_original_workload_without_tolerance(self) -> None:
        contract = self.policy["full_replay"]
        self.assertEqual(contract["unique_text_count"], 33434)
        self.assertEqual(contract["shared_chunk_count"], 41808)
        self.assertEqual(contract["opaque_pair_count"], 733)
        self.assertEqual(contract["labse_batch_size"], 24)
        self.assertEqual(contract["labse_dimension"], 768)
        self.assertTrue(contract["embedding_matrix_exact_byte_match_required"])
        self.assertTrue(contract["score_file_exact_byte_match_required"])
        self.assertFalse(contract["numeric_tolerance_or_fixture_reselection_allowed"])

    def test_invalid_v1_fixture_code_and_payload_are_removed(self) -> None:
        removed = [
            "scripts/run_step28_v13_v1_13_v9_4_1_compatibility_linux_20260830.sh",
            "scripts/step28_v13_v1_13_v9_4_1_prepare_compatibility_fixture_v1.py",
            "scripts/step28_v13_v1_13_v9_4_1_replay_compatibility_fixture_linux_v1.py",
            "tests/test_step28_v13_v1_13_v9_4_1_compatibility_fixture_v1_contracts.py",
            "tests/test_step28_v13_v1_13_v9_4_1_linux_fixture_replay_v1_contracts.py",
            "reports/step28_model_experiment/v9_4_1_implementation_v1_20260830/compatibility_fixture",
        ]
        for relative in removed:
            self.assertFalse((ROOT / relative).exists(), relative)

    def test_all_declared_non_model_files_match_size_and_sha256(self) -> None:
        for registry in ("authority", "frozen_label_free_inputs"):
            for role, spec in self.policy[registry].items():
                path = replay.base.resolve(spec["path"])
                self.assertTrue(path.is_file(), role)
                self.assertEqual(path.stat().st_size, spec["size_bytes"], role)
                self.assertEqual(replay.base.sha256_file(path), spec["sha256"], role)

    def test_exact_runtime_and_matrix_pins_are_derived_from_original_manifest(self) -> None:
        paths = {
            name: replay.base.resolve(spec["path"])
            for name, spec in {
                **self.policy["authority"],
                **self.policy["frozen_label_free_inputs"],
            }.items()
        }
        base_policy = replay.base.load_policy(paths["base_model_experiment_policy"])
        replay._validate_original_runtime_contract(self.policy, paths, base_policy)
        altered = json.loads(json.dumps(self.policy))
        altered["exact_runtime"]["numpy"] = "0.0.0"
        with self.assertRaisesRegex(
            replay.FullCompatibilityError, "original runtime binding drift"
        ):
            replay._validate_original_runtime_contract(altered, paths, base_policy)

    def test_replay_source_contains_no_supervision_or_truth_input_paths(self) -> None:
        source = inspect.getsource(replay).casefold()
        for forbidden in (
            "private_labels",
            "pair_labels.csv",
            "private_custody",
            "qrels.jsonl",
            "identity33_all_pairs",
            "controller_uid",
            "membership.jsonl",
        ):
            self.assertNotIn(forbidden, source)

    def test_replay_rebuilds_every_chunk_and_checks_full_matrix_and_score_bytes(self) -> None:
        workload_source = inspect.getsource(replay._load_and_rebuild_workload)
        run_source = inspect.getsource(replay.run_replay)
        self.assertIn("build_shared_chunks", workload_source)
        self.assertIn("rebuilt_chunks != expected_chunks", workload_source)
        self.assertIn("matrix_content_sha256", run_source)
        self.assertIn("observed_score_bytes == expected_score_bytes", run_source)
        self.assertIn("implementation bytes changed during replay", run_source)
        self.assertNotIn("allclose", run_source)

    def test_sentence_transformer_digest_uses_preprocess_not_deprecated_tokenize(self) -> None:
        class FakeModel:
            def preprocess(self, batch):
                self.batch = batch
                return {
                    "input_ids": np.asarray([[101, 7, 102], [101, 8, 102]]),
                    "attention_mask": np.ones((2, 3), dtype=np.int64),
                }

            def tokenize(self, _batch):
                raise AssertionError("deprecated tokenize() must not be called")

        model = FakeModel()
        digest, lengths = replay._preprocess_digest_and_lengths(
            model, ["a", "b"], "", batch_size=2
        )
        self.assertEqual(model.batch, ["a", "b"])
        self.assertEqual(lengths, [3, 3])
        self.assertEqual(len(digest), 64)

    def test_score_diagnostic_reports_exact_first_and_maximum_difference(self) -> None:
        expected = b"pair_uid,a,b\np1,0.100000000000,0.200000000000\n"
        observed = b"pair_uid,a,b\np1,0.100000100000,0.199999000000\n"
        diagnostic = replay._score_diagnostics(expected, observed)
        self.assertEqual(diagnostic["mismatched_numeric_cell_count"], 2)
        self.assertAlmostEqual(diagnostic["maximum_absolute_difference"], 1e-6)
        self.assertEqual(diagnostic["first_mismatch"]["row_index"], 0)
        self.assertEqual(diagnostic["first_mismatch"]["column"], "a")

    def test_failed_run_removes_temporary_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            policy = json.loads(json.dumps(self.policy))
            root = Path(directory) / "success"
            policy["outputs"]["success_root"] = str(root)
            with mock.patch.object(
                replay, "run_replay", side_effect=replay.FullCompatibilityError("boom")
            ):
                with self.assertRaisesRegex(replay.FullCompatibilityError, "boom"):
                    replay.publish(policy)
            self.assertFalse(root.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])
        with tempfile.TemporaryDirectory() as directory:
            policy = json.loads(json.dumps(self.policy))
            root = Path(directory) / "success"
            root.mkdir()
            policy["outputs"]["success_root"] = str(root)
            with mock.patch.object(replay, "run_replay") as run_replay:
                with self.assertRaisesRegex(
                    replay.FullCompatibilityError, "already exists"
                ):
                    replay.publish(policy)
            run_replay.assert_not_called()

    def test_success_validation_requires_exact_original_score_bytes(self) -> None:
        expected_path = replay.base.resolve(
            self.policy["frozen_label_free_inputs"]["expected_labse_scores"]["path"]
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            score_name = self.policy["outputs"]["observed_score_file"]
            score_path = root / score_name
            score_path.write_bytes(expected_path.read_bytes())
            input_records = {}
            for registry_name in ("authority", "frozen_label_free_inputs"):
                for role, spec in self.policy[registry_name].items():
                    source = replay.base.verify_file_pin(spec, label=role)
                    input_records[role] = replay._exact_file_record(source)
            base_policy = replay.base.load_policy(
                replay.base.resolve(
                    self.policy["authority"]["base_model_experiment_policy"]["path"]
                )
            )
            original_runtime = json.loads(
                replay.base.resolve(
                    self.policy["frozen_label_free_inputs"][
                        "original_labse_runtime"
                    ]["path"]
                ).read_text(encoding="utf-8")
            )
            manifest = {
                "step": replay.RESULT_STEP,
                "status": replay.RESULT_STATUS,
                "policy_canonical_self_hash": self.policy["canonical_self_hash"],
                "base_model_experiment_policy_canonical_self_hash": base_policy[
                    "canonical_self_hash"
                ],
                "input_records": input_records,
                "implementation_records": replay._implementation_records(),
                "exact_runtime": self.policy["exact_runtime"],
                "loaded_model_state": {
                    "loaded_default_prompt_name": original_runtime[
                        "loaded_default_prompt_name"
                    ],
                    "loaded_prompts": original_runtime["loaded_prompts"],
                    "loaded_native_max_seq_length": original_runtime[
                        "loaded_native_max_seq_length"
                    ],
                },
                "runtime_labse_token_id_stream_sha256": self.policy["full_replay"][
                    "labse_token_id_stream_sha256"
                ],
                "embedding_matrix_shape": [41808, 768],
                "embedding_matrix_dtype": "float32",
                "embedding_matrix_sha256": self.policy["full_replay"][
                    "embedding_matrix_sha256"
                ],
                "embedding_matrix_exact_byte_match": True,
                "maximum_unit_norm_error": 1e-7,
                "complete_733_pair_score_file_exact_byte_match": True,
                "expected_score_sha256": self.policy["frozen_label_free_inputs"][
                    "expected_labse_scores"
                ]["sha256"],
                "numeric_tolerance_used": False,
                "fixture_reselection_used": False,
                "supervised_labels_or_identity_evidence_read": False,
                "identity33_read": False,
                "controller_or_membership_read": False,
                "qrels_or_retrieval_truth_read": False,
                "audit_truth_read": False,
                "model_parameters_updated": False,
                "model_training_or_threshold_selection_performed": False,
                "m0_m1_m2_m3_training_authorized": False,
                "observed_score_file": {
                    "path": score_name,
                    "size_bytes": score_path.stat().st_size,
                    "sha256": replay.base.sha256_file(score_path),
                },
            }
            manifest["canonical_self_hash"] = replay.base.canonical_sha256(manifest)
            (root / self.policy["outputs"]["success_manifest"]).write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            replay.validate_published(self.policy, root)
            for field, value in (
                ("embedding_matrix_sha256", "0" * 64),
                ("implementation_records", {}),
                ("numeric_tolerance_used", True),
                ("audit_truth_read", True),
                ("model_parameters_updated", True),
            ):
                altered = json.loads(json.dumps(manifest))
                altered[field] = value
                altered.pop("canonical_self_hash", None)
                altered["canonical_self_hash"] = replay.base.canonical_sha256(altered)
                (root / self.policy["outputs"]["success_manifest"]).write_text(
                    json.dumps(altered, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    replay.FullCompatibilityError, "published result drift"
                ):
                    replay.validate_published(self.policy, root)
            (root / self.policy["outputs"]["success_manifest"]).write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            damaged = bytearray(score_path.read_bytes())
            damaged[-2] = ord("1") if damaged[-2] != ord("1") else ord("2")
            score_path.write_bytes(damaged)
            with self.assertRaisesRegex(
                replay.FullCompatibilityError, "published result drift"
            ):
                replay.validate_published(self.policy, root)

    def test_runner_uses_offline_clean_cuda_environment_and_v2_entrypoint(self) -> None:
        runner = (
            ROOT
            / "scripts"
            / "run_step28_v13_v1_13_v9_4_1_full_english_compatibility_v2_linux_20260830.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("env -u LD_LIBRARY_PATH -u LD_PRELOAD", runner)
        self.assertIn("CUBLAS_WORKSPACE_CONFIG=:4096:8", runner)
        self.assertIn("HF_HUB_OFFLINE=1", runner)
        self.assertIn("TRANSFORMERS_OFFLINE=1", runner)
        self.assertNotIn('if [[ "$#" -gt 0 ]]', runner)
        self.assertIn('command_name="${1:-run}"', runner)
        self.assertIn(
            "step28_v13_v1_13_v9_4_1_replay_full_english_compatibility_linux_v2.py",
            runner,
        )

    def test_help_is_non_executing(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-B", str(Path(replay.__file__)), "--help"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("usage:", completed.stdout)


if __name__ == "__main__":
    unittest.main()
