from __future__ import annotations

import copy
import inspect
import json
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

import step28_v13_v1_13_v9_4_1_encode_base_projection_linux_v1 as gpu
import step28_v13_v1_13_v9_4_1_finalize_base_projection_v1 as finalizer
import step28_v13_v1_13_v9_4_1_freeze_identity_projection_v2 as identity_v2
import step28_v13_v1_13_v9_4_1_materialize_base_gpu_workspace_v1 as materializer
import step28_v13_v1_13_v9_4_1_prepare_base_projection_v1 as prepare
import step28_v13_v1_13_v9_4_1_public_projection_common_v1 as public_common
import step28_v13_v1_13_v9_4_1_public_projection_gpu_common_v1 as gpu_common
import step28_v13_v1_13_v9_4_1_public_projection_protocol_v1 as protocol


def toy_public_rows():
    worlds = [
        {
            "world_uid": "world_0",
            "split": "toy",
            "world_ordinal": 0,
            "seller_count": 3,
            "item_count": 3,
            "pair_count": 3,
        }
    ]
    sellers = [
        {"world_uid": "world_0", "seller_uid": value, "market": "market"}
        for value in ("seller_a", "seller_b", "seller_c")
    ]
    pairs = [
        {
            "canonical_pair_uid": f"{left}||{right}",
            "world_uid": "world_0",
            "seller_uid_left": left,
            "seller_uid_right": right,
        }
        for left, right in (
            ("seller_a", "seller_b"),
            ("seller_a", "seller_c"),
            ("seller_b", "seller_c"),
        )
    ]
    counts = {
        "worlds": 1,
        "sellers": 3,
        "items": 3,
        "pairs": 3,
        "pairs_per_world": 3,
        "sellers_per_world": 3,
    }
    return worlds, sellers, pairs, counts


def toy_text_rows():
    return [
        {
            "world_uid": "world_0",
            "seller_uid": seller,
            "item_uid": f"item_{index}",
            "title": f"标题 {index}",
            "description": f"描述 {index}",
        }
        for index, seller in enumerate(("seller_a", "seller_b", "seller_c"))
    ]


class PublicProjectionV1ImplementationContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = public_common.load_policy()
        cls.gpu_policy = gpu_common.load_policy()

    def test_public_and_gpu_policies_are_exact_and_grant_nothing(self) -> None:
        self.assertEqual(
            self.policy["canonical_self_hash"],
            "bb1849832ffa43efd4135c33e585c81c939843f41a6026e49ae52ddddb0cbaee",
        )
        self.assertEqual(
            self.gpu_policy["canonical_self_hash"],
            "c9d45ec1b5781c4cce3f7209aa6adb8072f82e676b96db2d7a0439d56606831f",
        )
        self.assertEqual(set(self.policy["authorization_state"].values()), {False})
        self.assertEqual(set(self.gpu_policy["permissions"].values()), {False})
        with self.assertRaises(public_common.PublicProjectionContractError):
            public_common.require_formal_projection_authorization(self.policy)

    def test_gpu_encoder_does_not_import_full_semantic_policy(self) -> None:
        source = inspect.getsource(gpu)
        self.assertIn("public_projection_gpu_common_v1", source)
        self.assertNotIn("public_projection_common_v1", source)
        for semantic_split in ("development", "audit_a", "audit_b"):
            self.assertNotIn(semantic_split, source)
        self.assertNotIn("formal_dataset", source)
        score_source = inspect.getsource(gpu.score_part)
        self.assertIn("tokenizer_digest_and_lengths", score_source)
        self.assertIn("runtime_digest != tokenizer_digest", score_source)
        self.assertIn("native_max_seq_length", score_source)

    def test_gpu_workspace_excludes_full_semantic_policy_and_parser(self) -> None:
        staged = set(materializer.STATIC_FILES)
        self.assertNotIn(
            "schema/step28_v13_v1_13_v9_4_1_public_projection_policy_v1.json",
            staged,
        )
        self.assertNotIn(
            "scripts/step28_v13_v1_13_v9_4_1_public_projection_common_v1.py",
            staged,
        )
        self.assertIn(
            "schema/step28_v13_v1_13_v9_4_1_public_projection_gpu_policy_v1.json",
            staged,
        )

    def test_gpu_scoring_rejects_same_lengths_with_different_token_ids(self) -> None:
        chunks = [
            {
                "text": "甲",
                "token_lengths": {"labse": 2},
            }
        ]
        step7_policy = {
            "shared_chunking": {
                "token_budget_including_model_prefix_and_special_tokens": 256,
            },
            "embedding_models": {
                "labse": {
                    "text_prefix": "",
                    "native_max_seq_length": 256,
                }
            },
        }

        class FakeModel:
            @staticmethod
            def preprocess(_texts):
                return {
                    "input_ids": np.asarray([[1, 2]], dtype=np.int64),
                    "attention_mask": np.asarray([[1, 1]], dtype=np.int64),
                }

            @staticmethod
            def encode(*_args, **_kwargs):
                raise AssertionError("encoding must not start after token-ID drift")

        with mock.patch.object(
            gpu.step7_encoder,
            "build_shared_chunks",
            return_value=(chunks, {"chunk_count": 1}),
        ), mock.patch.object(
            gpu.step7_encoder,
            "tokenizer_digest_and_lengths",
            return_value=("0" * 64, [2]),
        ):
            with self.assertRaises(gpu_common.GPUProjectionContractError):
                gpu.score_part(
                    self.gpu_policy,
                    {"text_rows": [{"text": "甲"}]},
                    step7_policy,
                    {"labse": object()},
                    FakeModel(),
                )

    def test_all_command_lines_are_validation_only(self) -> None:
        for module in (
            prepare,
            materializer,
            gpu,
            finalizer,
            identity_v2,
            protocol,
        ):
            source = inspect.getsource(module.main)
            self.assertIn('choices=("validate-contract",)', source)
            self.assertNotIn('"build"', source)
            self.assertNotIn('"run"', source)

    def test_public_row_order_requires_complete_k3_toy_universe(self) -> None:
        worlds, sellers, pairs, counts = toy_public_rows()
        row_keys, seller_uids, seller_worlds = prepare.validate_public_row_order(
            "toy", worlds, sellers, pairs, counts
        )
        self.assertEqual(len(row_keys), 3)
        self.assertEqual(seller_uids, ["seller_a", "seller_b", "seller_c"])
        self.assertEqual(seller_worlds["seller_b"], "world_0")
        with self.assertRaises(public_common.PublicProjectionContractError):
            prepare.validate_public_row_order(
                "toy", worlds, sellers, pairs[:-1] + [pairs[0]], counts
            )

    def test_opaque_workload_drops_canonical_identifiers_and_multiplicity(self) -> None:
        worlds, sellers, pairs, counts = toy_public_rows()
        _keys, seller_uids, seller_worlds = prepare.validate_public_row_order(
            "toy", worlds, sellers, pairs, counts
        )
        texts, mappings, opaque = prepare.build_opaque_text_workload(
            toy_text_rows(),
            valid_worlds={"world_0"},
            seller_uids=seller_uids,
            seller_worlds=seller_worlds,
            expected_item_count=3,
        )
        opaque_pairs = prepare.opaque_pair_rows(pairs, opaque)
        serialized = json.dumps(
            {"texts": texts, "mappings": mappings, "pairs": opaque_pairs},
            ensure_ascii=False,
        )
        for forbidden in ("world_0", "seller_a", "seller_b", "seller_c", "item_0"):
            self.assertNotIn(forbidden, serialized)
        self.assertEqual({row["multiplicity"] for row in mappings}, {1})
        self.assertEqual([row["pair_uid"] for row in opaque_pairs], [
            "pair_000001",
            "pair_000002",
            "pair_000003",
        ])

    def test_opaque_workload_rejects_cross_world_seller_assignment(self) -> None:
        rows = toy_text_rows()
        rows[0] = {**rows[0], "world_uid": "world_wrong"}
        with self.assertRaises(public_common.PublicProjectionContractError):
            prepare.build_opaque_text_workload(
                rows,
                valid_worlds={"world_0", "world_wrong"},
                seller_uids=["seller_a", "seller_b", "seller_c"],
                seller_worlds={
                    "seller_a": "world_0",
                    "seller_b": "world_0",
                    "seller_c": "world_0",
                },
                expected_item_count=3,
            )

    def _write_toy_transfer(self, root: Path):
        policy = copy.deepcopy(self.gpu_policy)
        policy["part_contract"]["seller_count_per_part"] = 3
        policy["part_contract"]["pair_count_per_part"] = 3
        policy["labse_contract"]["output_shape_per_part"] = [3, 6]
        worlds, sellers, pairs, counts = toy_public_rows()
        _keys, seller_uids, seller_worlds = prepare.validate_public_row_order(
            "toy", worlds, sellers, pairs, counts
        )
        texts, mappings, opaque = prepare.build_opaque_text_workload(
            toy_text_rows(),
            valid_worlds={"world_0"},
            seller_uids=seller_uids,
            seller_worlds=seller_worlds,
            expected_item_count=3,
        )
        opaque_pairs = prepare.opaque_pair_rows(pairs, opaque)
        parts = []
        for index in range(4):
            part_id = f"part_{index:03d}"
            part_root = root / part_id
            part_root.mkdir(parents=True)
            text_path = part_root / "opaque_unique_texts.jsonl"
            seller_path = part_root / "opaque_seller_text_index.jsonl"
            pair_path = part_root / "opaque_pair_endpoints.csv"
            prepare.render_jsonl(text_path, texts)
            prepare.render_jsonl(seller_path, mappings)
            prepare.render_csv(pair_path, opaque_pairs, prepare.GPU_PAIR_FIELDS)
            parts.append(
                {
                    "part_id": part_id,
                    "unique_text_count": len(texts),
                    "seller_text_row_count": len(mappings),
                    "opaque_seller_count": 3,
                    "opaque_pair_count": 3,
                    "files": {
                        "opaque_unique_texts": prepare.file_record(text_path, root),
                        "opaque_seller_text_index": prepare.file_record(seller_path, root),
                        "opaque_pair_endpoints": prepare.file_record(pair_path, root),
                    },
                }
            )
        manifest = {
            "step": "toy",
            "status": "FROZEN_OPAQUE_LABEL_FREE_TRANSFER_NO_CANONICAL_IDENTITIES",
            "public_policy_canonical_self_hash": self.policy["canonical_self_hash"],
            "gpu_policy_canonical_self_hash": policy["canonical_self_hash"],
            "cpu_stage_canonical_self_hash": "1" * 64,
            "parts": parts,
            "part_count": 4,
            "split_names_or_canonical_identifiers_present": False,
            "labels_controllers_membership_qrels_or_audit_truth_present": False,
            "identity33_or_legacy18_present": False,
        }
        manifest["canonical_self_hash"] = gpu_common.canonical_sha256(manifest)
        prepare.render_json(root / "transfer_manifest.json", manifest)
        return policy, manifest

    def _write_toy_return(
        self,
        root: Path,
        policy,
        transfer_manifest,
        transfer_parts,
    ):
        records = []
        for transfer in transfer_parts:
            part_id = transfer["part_id"]
            part_root = root / part_id
            part_root.mkdir(parents=True)
            values = np.ascontiguousarray(np.arange(18).reshape(3, 6) / 20, dtype="<f8")
            value_path = part_root / "labse6.npy"
            np.save(value_path, values, allow_pickle=False)
            part_manifest = {
                "step": "toy",
                "status": "FROZEN_OPAQUE_LABSE6_PART_NO_MODEL_TRAINING",
                "part_id": part_id,
                "gpu_policy_canonical_self_hash": policy["canonical_self_hash"],
                "public_policy_canonical_self_hash": transfer_manifest[
                    "public_policy_canonical_self_hash"
                ],
                "transfer_manifest_canonical_self_hash": transfer_manifest[
                    "canonical_self_hash"
                ],
                "input_files": transfer["input_files"],
                "labse6_file": gpu.file_record(value_path, root),
                "labse6_shape": [3, 6],
                "labse6_dtype": "<f8",
                "labse6_value_sha256": gpu.matrix_value_sha256(values),
                "audit": {
                    "part_id": part_id,
                    "shared_labse_token_id_stream_sha256": "a" * 64,
                    "runtime_labse_token_id_stream_sha256": "a" * 64,
                    "runtime_token_id_stream_replays_shared_tokenizer": True,
                    "maximum_unit_norm_error": 0.0,
                },
                "canonical_identifiers_or_split_names_read": False,
                "labels_controllers_membership_qrels_or_audit_truth_read": False,
                "identity33_or_legacy18_read": False,
                "model_parameters_updated": False,
            }
            part_manifest["canonical_self_hash"] = gpu_common.canonical_sha256(
                part_manifest
            )
            manifest_path = part_root / "labse6_manifest.json"
            gpu.render_json(manifest_path, part_manifest)
            records.append(
                {
                    "part_id": part_id,
                    "manifest_file": gpu.file_record(manifest_path, root),
                    "manifest_canonical_self_hash": part_manifest[
                        "canonical_self_hash"
                    ],
                }
            )
        root_manifest = {
            "step": "toy",
            "status": "FROZEN_OPAQUE_FOUR_PART_LABSE6_RETURN_NO_MODEL_TRAINING",
            "gpu_policy_canonical_self_hash": policy["canonical_self_hash"],
            "public_policy_canonical_self_hash": transfer_manifest[
                "public_policy_canonical_self_hash"
            ],
            "transfer_manifest_canonical_self_hash": transfer_manifest[
                "canonical_self_hash"
            ],
            "parts": records,
            "exact_runtime": policy["exact_runtime"],
            "loaded_model_state": policy["loaded_model_state"],
            "temporary_chunks_retained": False,
            "temporary_embeddings_retained": False,
            "canonical_identifiers_or_split_names_read": False,
            "labels_controllers_membership_qrels_or_audit_truth_read": False,
            "model_parameters_updated": False,
        }
        root_manifest["canonical_self_hash"] = gpu_common.canonical_sha256(root_manifest)
        gpu.render_json(root / "gpu_return_manifest.json", root_manifest)

    def _reseal_toy_return_part(self, root: Path, part_id: str) -> None:
        part_path = root / part_id / "labse6_manifest.json"
        part_manifest = json.loads(part_path.read_text(encoding="utf-8"))
        part_manifest.pop("canonical_self_hash", None)
        part_manifest["canonical_self_hash"] = gpu_common.canonical_sha256(
            part_manifest
        )
        gpu.render_json(part_path, part_manifest)
        root_path = root / "gpu_return_manifest.json"
        root_manifest = json.loads(root_path.read_text(encoding="utf-8"))
        record = next(
            row for row in root_manifest["parts"] if row["part_id"] == part_id
        )
        record["manifest_file"] = gpu.file_record(part_path, root)
        record["manifest_canonical_self_hash"] = part_manifest[
            "canonical_self_hash"
        ]
        root_manifest.pop("canonical_self_hash", None)
        root_manifest["canonical_self_hash"] = gpu_common.canonical_sha256(
            root_manifest
        )
        gpu.render_json(root_path, root_manifest)

    def test_opaque_transfer_and_return_reopen_semantically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transfer_root = root / "transfer"
            return_root = root / "return"
            transfer_root.mkdir()
            return_root.mkdir()
            policy, transfer_manifest = self._write_toy_transfer(transfer_root)
            observed_manifest, parts = gpu.validate_transfer(policy, transfer_root)
            self.assertEqual(observed_manifest, transfer_manifest)
            self._write_toy_return(
                return_root, policy, transfer_manifest, parts
            )
            _manifest, values = gpu.validate_gpu_return(
                policy, transfer_manifest, parts, return_root
            )
            self.assertEqual(len(values), 4)
            self.assertEqual(values[0]["values"].shape, (3, 6))

    def test_gpu_return_rejects_missing_mismatched_or_false_token_replay(self) -> None:
        mutations = (
            lambda audit: audit.pop("shared_labse_token_id_stream_sha256"),
            lambda audit: audit.__setitem__(
                "runtime_labse_token_id_stream_sha256", "b" * 64
            ),
            lambda audit: audit.__setitem__(
                "runtime_token_id_stream_replays_shared_tokenizer", False
            ),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation.__code__.co_firstlineno):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    transfer_root = root / "transfer"
                    return_root = root / "return"
                    transfer_root.mkdir()
                    return_root.mkdir()
                    policy, transfer_manifest = self._write_toy_transfer(
                        transfer_root
                    )
                    _observed, parts = gpu.validate_transfer(
                        policy, transfer_root
                    )
                    self._write_toy_return(
                        return_root, policy, transfer_manifest, parts
                    )
                    part_path = return_root / "part_000" / "labse6_manifest.json"
                    part_manifest = json.loads(part_path.read_text(encoding="utf-8"))
                    mutation(part_manifest["audit"])
                    part_manifest.pop("canonical_self_hash", None)
                    part_manifest["canonical_self_hash"] = (
                        gpu_common.canonical_sha256(part_manifest)
                    )
                    gpu.render_json(part_path, part_manifest)
                    self._reseal_toy_return_part(return_root, "part_000")
                    with self.assertRaises(gpu_common.GPUProjectionContractError):
                        gpu.validate_gpu_return(
                            policy, transfer_manifest, parts, return_root
                        )

    def test_opaque_transfer_rejects_an_extra_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy, _manifest = self._write_toy_transfer(root)
            (root / "unexpected.txt").write_text("x", encoding="utf-8")
            with self.assertRaises(gpu_common.GPUProjectionContractError):
                gpu.validate_transfer(policy, root)

    def test_m0_imputation_and_probability_contract(self) -> None:
        raw = np.asarray([[1.0, np.nan], [2.0, 3.0]], dtype="<f8")
        observed = finalizer.impute(raw, np.asarray([10.0, 20.0], dtype="<f8"))
        np.testing.assert_array_equal(observed, [[1.0, 20.0], [2.0, 3.0]])

        class FakeModel:
            def predict_proba(self, matrix):
                positive = np.asarray([0.2, 0.8], dtype=np.float64)
                return np.column_stack((1.0 - positive, positive))

        probability = finalizer.predict_positive(FakeModel(), observed)
        np.testing.assert_array_equal(probability, [0.2, 0.8])

    def test_required_metrics_remain_in_parent_training_contract(self) -> None:
        training = public_common.load_json(
            public_common.resolve(
                self.policy["authority_registry"]["training_policy_v3"]["path"]
            )
        )
        metrics = training["metric_contract"]
        self.assertIn("average_precision", metrics["pooled_classification_metrics"])
        self.assertIn("trapezoidal_pr_auc", metrics["pooled_classification_metrics"])
        self.assertIn("map", metrics["retrieval_metrics"])
        self.assertIn("mrr", metrics["retrieval_metrics"])


if __name__ == "__main__":
    unittest.main()
