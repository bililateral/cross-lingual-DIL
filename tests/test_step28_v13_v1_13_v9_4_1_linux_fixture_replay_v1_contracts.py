from __future__ import annotations

import inspect
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_v9_4_1_model_experiment_common_v1 as common
import step28_v13_v1_13_v9_4_1_replay_compatibility_fixture_linux_v1 as replay


class LinuxFixtureReplayContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = common.load_policy()

    def test_source_contains_no_supervision_or_training_input_paths(self) -> None:
        source = inspect.getsource(replay).casefold()
        for forbidden in (
            "pair_labels.csv",
            "qrels.jsonl",
            "private_custody",
            "identity33_all_pairs.csv",
            "membership",
        ):
            self.assertNotIn(forbidden, source)

    def test_output_path_is_separate_from_frozen_input_fixture(self) -> None:
        outputs = self.policy["outputs"]
        self.assertNotEqual(
            outputs["compatibility_fixture"],
            outputs["compatibility_fixture_linux_replay"],
        )
        self.assertTrue(outputs["compatibility_fixture_linux_replay"].endswith(
            "/compatibility_fixture_linux_replay"
        ))

    def test_help_does_not_start_the_linux_replay(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-B", str(Path(replay.__file__)), "--help"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("usage:", completed.stdout)
        self.assertNotIn("sentence-transformers is not installed", completed.stderr)

    def test_score_comparison_uses_all_six_twelve_decimal_strings(self) -> None:
        names = self.policy["feature_contract"]["labse6"]
        expected = {"pair_uid": "pair_00000001"}
        expected.update({name: f"{0.1 + index / 100:.12f}" for index, name in enumerate(names)})
        payload = {
            "pairs": [
                {
                    "pair_uid": "pair_00000001",
                    "seller_uid_left": "seller_00000001",
                    "seller_uid_right": "seller_00000002",
                }
            ],
            "sellers": [],
            "chunks": [],
            "expected": [expected],
        }
        observed_with_audit = dict(expected)
        observed_with_audit["raw_labse_extra_audit"] = "0.0"
        with mock.patch.object(
            replay.step7_common,
            "compute_pair_score_rows",
            return_value=[observed_with_audit],
        ):
            rows = replay._score_rows(
                {"embedding_models": {"labse": {}}, "aggregation": {
                    "top_k_item_matches": 3,
                    "serialized_decimal_places": 12,
                    "similarity_block_rows": 256,
                }},
                payload,
                np.zeros((0, 768), dtype=np.float32),
            )
        self.assertEqual(rows, [expected])

    def test_rebuilt_chunks_compare_all_frozen_boundary_fields(self) -> None:
        base = {
            "text_uid": "text_00000001",
            "chunk_index": 0,
            "char_start": 0,
            "char_end": 3,
            "text": "abc",
            "token_lengths": {
                "pcm_multilingual_authorship": 3,
                "mstyledistance": 3,
                "multilingual_e5_large": 3,
                "labse": 3,
            },
        }
        projected = replay._project_chunk_rows([base])
        self.assertEqual(projected, [base])
        for field in replay.CHUNK_COMPARE_FIELDS:
            changed = dict(base)
            changed[field] = "drift"
            self.assertNotEqual(replay._project_chunk_rows([changed]), projected)

    def test_linux_replay_rebuilds_chunks_from_complete_texts(self) -> None:
        source = inspect.getsource(replay.run_replay)
        self.assertIn("_rebuild_and_verify_shared_chunks", source)
        helper = inspect.getsource(replay._rebuild_and_verify_shared_chunks)
        self.assertIn("build_shared_chunks", helper)
        self.assertIn('payload["texts"]', helper)

    def test_linux_runner_removes_ambient_cuda_library_overrides(self) -> None:
        script = (
            ROOT
            / "scripts"
            / "run_step28_v13_v1_13_v9_4_1_compatibility_linux_20260830.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("env -u LD_LIBRARY_PATH -u LD_PRELOAD", script)
        self.assertIn("CUBLAS_WORKSPACE_CONFIG=:4096:8", script)
        self.assertIn("TOKENIZERS_PARALLELISM=false", script)
        self.assertIn("HF_HUB_OFFLINE=1", script)
        self.assertIn("TRANSFORMERS_OFFLINE=1", script)
        self.assertIn("PYTHONDONTWRITEBYTECODE=1", script)

    def test_linux_runner_forwards_help_before_gpu_or_replay_execution(self) -> None:
        script = (
            ROOT
            / "scripts"
            / "run_step28_v13_v1_13_v9_4_1_compatibility_linux_20260830.sh"
        ).read_text(encoding="utf-8")
        argument_gate = script.index('if [[ "$#" -gt 0 ]]')
        gpu_probe = script.index("nvidia-smi")
        self.assertLess(argument_gate, gpu_probe)
        self.assertIn(
            "step28_v13_v1_13_v9_4_1_replay_compatibility_fixture_linux_v1.py \"$@\"",
            script,
        )

    def test_self_resigned_incomplete_linux_pass_is_rejected(self) -> None:
        fixture_root = common.resolve(
            self.policy["outputs"]["compatibility_fixture"]
        )
        expected = (fixture_root / "fixture_expected_labse_scores.csv").read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            score_path = root / replay.OBSERVED_SCORES
            score_path.write_bytes(expected)
            forged = {
                "status": "PASSED_LABEL_FREE_LINUX_LABSE_COMPATIBILITY_REPLAY",
                "policy_canonical_self_hash": self.policy["canonical_self_hash"],
                "fixture_manifest_canonical_self_hash": (
                    replay.fixture.validate_published(self.policy, fixture_root)[
                        "canonical_self_hash"
                    ]
                ),
                "labels_or_identity_evidence_read": False,
                "audit_truth_read": False,
                "model_parameters_updated": False,
                "observed_score_file": {
                    "path": replay.OBSERVED_SCORES,
                    "size_bytes": len(expected),
                    "sha256": hashlib.sha256(expected).hexdigest(),
                },
            }
            forged["canonical_self_hash"] = common.canonical_sha256(forged)
            (root / replay.RESULT_MANIFEST).write_text(
                json.dumps(forged, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            with self.assertRaisesRegex(
                common.ModelExperimentContractError, "boundary drift"
            ):
                replay.validate_published(self.policy, root)

    def test_actual_runtime_revalidation_reopens_all_frozen_gates(self) -> None:
        runtime = {
            "sentence_transformers": "5.6.0",
            "step7_policy_sha256": "a" * 64,
            "payload_count": 4,
        }
        implementation = {
            role: {"path": f"scripts/{role}.py", "size_bytes": 1, "sha256": "b" * 64}
            for role in sorted(replay.STEP7_IMPLEMENTATION_ROLES)
        }
        deterministic = {"deterministic_algorithms_enabled": True}
        loaded = {
            "loaded_default_prompt_name": None,
            "loaded_prompts": {},
            "loaded_native_max_seq_length": 256,
        }
        manifest = {
            "runtime_gate": runtime,
            "step7_implementation_files": implementation,
            "step28_implementation_files": replay._step28_implementation_file_records(),
            "deterministic_gpu_runtime": deterministic,
            "loaded_model_state": loaded,
            "default_prompt_name_cleared_before_encoding": True,
        }
        step7_policy = {
            "embedding_models": {"labse": {"native_max_seq_length": 256}}
        }
        fake_model = mock.Mock(default_prompt_name=None)
        fake_torch = mock.Mock()
        with mock.patch.object(
            replay.common, "validate_encoding_runtime", return_value=runtime
        ), mock.patch.object(
            replay.step7_common,
            "verify_implementation_files",
            return_value=implementation,
        ), mock.patch.object(
            replay.step7_encoder,
            "require_gpu_stack",
            return_value=(fake_torch, mock.Mock(), mock.Mock(), mock.Mock()),
        ), mock.patch.object(
            replay.step7_encoder,
            "configure_deterministic_gpu",
            return_value=deterministic,
        ), mock.patch.object(
            replay.step7_encoder,
            "create_sentence_transformer",
            return_value=(fake_model, loaded),
        ):
            replay._revalidate_actual_runtime_state(
                self.policy, step7_policy, manifest
            )
            changed = dict(manifest)
            changed["step7_implementation_files"] = {}
            with self.assertRaisesRegex(
                common.ModelExperimentContractError, "implementation drift"
            ):
                replay._revalidate_actual_runtime_state(
                    self.policy, step7_policy, changed
                )

    def test_linux_manifest_rebinds_current_step28_implementation_bytes(self) -> None:
        records = replay._step28_implementation_file_records()
        self.assertEqual(list(records), list(replay.STEP28_IMPLEMENTATION_FILES))
        for role, relative in replay.STEP28_IMPLEMENTATION_FILES.items():
            path = common.resolve(relative)
            self.assertEqual(records[role]["size_bytes"], path.stat().st_size)
            self.assertEqual(records[role]["sha256"], common.sha256_file(path))


if __name__ == "__main__":
    unittest.main()
