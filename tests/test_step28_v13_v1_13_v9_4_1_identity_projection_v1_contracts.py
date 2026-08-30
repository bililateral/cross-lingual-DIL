from __future__ import annotations

import inspect
import csv
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

import step28_v13_v1_13_v9_4_1_freeze_identity_projection_v1 as identity
import step28_v13_v1_13_v9_4_1_model_experiment_common_v1 as common


class IdentityProjectionContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = common.load_policy()

    def test_identity_builder_source_has_no_text_or_supervision_paths(self) -> None:
        source = inspect.getsource(identity).casefold()
        for forbidden in (
            "pair_labels.csv",
            "qrels.jsonl",
            "private_custody",
            "redacted_items.jsonl",
            "model_seller_profiles.jsonl",
        ):
            self.assertNotIn(forbidden, source)

    def test_world_m1_source_indices_are_exact_derangements(self) -> None:
        sellers = [f"seller_{index:02d}" for index in range(28)]
        edges = [
            common.canonical_pair_endpoints(left, right)
            for index, left in enumerate(sellers)
            for right in sellers[index + 1 :]
        ]
        digests = []
        for repeat_id in self.policy["m1"]["repeat_ids"]:
            source = identity.m1_source_indices_for_world(
                "world_000", edges, repeat_id
            )
            self.assertEqual(source.dtype.str, "<i8")
            self.assertEqual(set(int(value) for value in source), set(range(378)))
            self.assertTrue(np.all(source != np.arange(378)))
            for destination, origin in enumerate(source):
                self.assertFalse(set(edges[destination]) & set(edges[int(origin)]))
            digests.append(common.matrix_value_sha256(source.reshape(-1, 1)))
        self.assertEqual(len(set(digests)), 5)

    def test_world_k28_validation_applies_independently_of_m1(self) -> None:
        sellers = [f"seller_{index:02d}" for index in range(28)]
        edges = [
            common.canonical_pair_endpoints(left, right)
            for index, left in enumerate(sellers)
            for right in sellers[index + 1 :]
        ]
        self.assertEqual(identity.validate_world_k28(edges), sellers)
        with self.assertRaises(common.ModelExperimentContractError):
            identity.validate_world_k28(edges[:-1] + [edges[0]])

    def test_published_m1_validator_rejects_fixed_duplicate_and_cross_world(self) -> None:
        sellers = [f"seller_{index:02d}" for index in range(28)]
        edges = [
            common.canonical_pair_endpoints(left, right)
            for index, left in enumerate(sellers)
            for right in sellers[index + 1 :]
        ]
        valid = identity.m1_source_indices_for_world("world_000", edges, "r01")
        digest = identity.validate_m1_global_indices(valid, edges)
        self.assertEqual(len(digest), 64)
        self.assertEqual(
            identity.validate_exact_m1_repeat(
                valid, edges, ["world_000"], "r01"
            ),
            digest,
        )
        with self.assertRaisesRegex(
            common.ModelExperimentContractError, "frozen r02"
        ):
            identity.validate_exact_m1_repeat(
                valid, edges, ["world_000"], "r02"
            )
        for changed in (
            np.arange(378, dtype="<i8"),
            np.zeros(378, dtype="<i8"),
            valid + 378,
        ):
            with self.assertRaises(common.ModelExperimentContractError):
                identity.validate_m1_global_indices(changed, edges)

    def test_publish_validation_reopens_semantic_payloads(self) -> None:
        source = inspect.getsource(identity.validate_published)
        self.assertIn("_validate_split_payload", source)
        split_source = inspect.getsource(identity._validate_split_payload)
        self.assertIn("_validate_row_keys", split_source)
        self.assertIn("validate_exact_m1_repeat", split_source)
        self.assertIn("_validate_formal_source_binding", split_source)
        self.assertIn("np.load", split_source)

    def test_formal_source_binding_rejects_self_consistent_wrong_identity_values(self) -> None:
        names = list(self.policy["feature_contract"]["identity33"])
        pair_uid = identity._canonical_pair_uid("seller_a", "seller_b")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worlds_path = root / "worlds.jsonl"
            endpoints_path = root / "endpoints.csv"
            identities_path = root / "identities.csv"
            row_keys_path = root / "row_keys.csv"
            worlds_path.write_text("{}\n", encoding="utf-8")
            with endpoints_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "canonical_pair_uid", "world_uid", "seller_uid_left", "seller_uid_right"
                ])
                writer.writeheader()
                writer.writerow({
                    "canonical_pair_uid": pair_uid,
                    "world_uid": "world_0",
                    "seller_uid_left": "seller_a",
                    "seller_uid_right": "seller_b",
                })
            with identities_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["canonical_pair_uid", "world_uid", *names]
                )
                writer.writeheader()
                writer.writerow({
                    "canonical_pair_uid": pair_uid,
                    "world_uid": "world_0",
                    **{name: "0" for name in names},
                })
            with row_keys_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(identity.ROW_KEY_FIELDS))
                writer.writeheader()
                writer.writerow({
                    "split": "train",
                    "world_ordinal": "0",
                    "world_uid": "world_0",
                    "canonical_pair_uid": pair_uid,
                    "seller_uid_left": "seller_a",
                    "seller_uid_right": "seller_b",
                })
            paths = {
                "worlds.jsonl": worlds_path,
                "complete_model_pair_endpoints.csv": endpoints_path,
                "identity33_all_pairs.csv": identities_path,
            }
            with mock.patch.object(
                identity.projection, "verify_split_public_inputs", return_value=paths
            ), mock.patch.object(
                identity.projection,
                "load_worlds",
                return_value=[{"world_uid": "world_0"}],
            ):
                identity._validate_formal_source_binding(
                    self.policy,
                    "train",
                    row_keys_path,
                    np.zeros((1, 33), dtype="<f8"),
                    expected_row_count=1,
                    pairs_per_world=1,
                )
                changed = np.zeros((1, 33), dtype="<f8")
                changed[0, 0] = 1.0
                with self.assertRaisesRegex(
                    common.ModelExperimentContractError, "not bound to the formal source"
                ):
                    identity._validate_formal_source_binding(
                        self.policy,
                        "train",
                        row_keys_path,
                        changed,
                        expected_row_count=1,
                        pairs_per_world=1,
                    )

    def test_output_subdirectory_is_new_and_versioned(self) -> None:
        self.assertEqual(identity.OUTPUT_SUBDIRECTORY, "identity_v1")
        output = common.resolve(self.policy["outputs"]["public_projection"])
        self.assertEqual(
            (output / identity.OUTPUT_SUBDIRECTORY).relative_to(common.ROOT).as_posix(),
            "reports/step28_model_experiment/v9_4_1_implementation_v1_20260830/"
            "public_projection/identity_v1",
        )

    def test_identity_manifest_rebinds_current_implementation_bytes(self) -> None:
        records = identity.implementation_file_records()
        self.assertEqual(list(records), list(identity.IMPLEMENTATION_FILES))
        for role, relative in identity.IMPLEMENTATION_FILES.items():
            path = common.resolve(relative)
            self.assertEqual(records[role]["size_bytes"], path.stat().st_size)
            self.assertEqual(records[role]["sha256"], common.sha256_file(path))
        source = inspect.getsource(identity.validate_published)
        self.assertIn("implementation_file_records()", source)


if __name__ == "__main__":
    unittest.main()
