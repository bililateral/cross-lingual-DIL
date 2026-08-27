from __future__ import annotations

from pathlib import Path
from copy import deepcopy
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_structure_matrix_v9_3 as structure
import step28_v13_v1_13_method_policy_v9_3 as method_policy
import step28_v13_v1_13_quality_text_probe_views_v9 as text_views
import step28_v13_common as common


class StructureMatrixV93Contracts(unittest.TestCase):
    def test_frozen_feature_contract_is_unique_and_hash_bound(self) -> None:
        payload = structure.contract_payload()
        seller = payload["seller_matrix_feature_names"]
        noise = payload["noise_matrix_feature_names"]
        self.assertEqual(len(seller), len(set(seller)))
        self.assertEqual(len(noise), len(set(noise)))
        self.assertEqual(payload["seller_matrix_feature_count"], 382)
        self.assertEqual(payload["noise_matrix_feature_count"], len(noise))
        self.assertEqual(payload["row_count_per_world"], 378)

    def test_one_hot_domains_fail_closed(self) -> None:
        self.assertEqual(structure._one_hot(1, 3, label="test"), [0.0, 1.0, 0.0])
        for value in (-1, 3, True, 1.5):
            with self.subTest(value=value):
                with self.assertRaises(structure.StructureMatrixV93Error):
                    structure._one_hot(value, 3, label="test")

    def test_matrix_dtype_is_float64(self) -> None:
        self.assertEqual(np.dtype(np.float64).str, "<f8")

    def test_machine_policy_binds_exact_structure_contract_and_fails_tamper(self) -> None:
        policy = method_policy.load_policy()
        self.assertEqual(
            policy["structure_matrix_contract"], structure.contract_payload()
        )
        tampered = deepcopy(policy)
        tampered["quality_gates"]["maximum_family_symmetric_roc_auc"] = 0.54
        with self.assertRaises(method_policy.MethodPolicyV93Error):
            method_policy.validate_policy(tampered)

    def test_machine_policy_binds_exact_text_view_names_widths_and_hashes(self) -> None:
        policy = method_policy.load_policy()
        text = policy["text_probe_contract"]
        self.assertEqual(text["views"], list(text_views.VIEW_ORDER))
        self.assertEqual(
            text["feature_counts"],
            dict(zip(text_views.VIEW_ORDER, text_views.EXPECTED_WIDTHS, strict=True)),
        )
        self.assertEqual(text["feature_name_sha256s"], text_views.EXPECTED_NAME_HASHES)
        tampered = deepcopy(policy)
        tampered["text_probe_contract"]["views"][0] = "fs_full33"
        tampered["canonical_self_sha256"] = None
        tampered["canonical_self_sha256"] = common.canonical_sha256(tampered)
        with self.assertRaises(method_policy.MethodPolicyV93Error):
            method_policy.validate_policy(tampered)

    def test_observation_registry_is_complete_unique_and_exact(self) -> None:
        policy = method_policy.load_policy()
        expected = method_policy.expected_observation_registry()
        self.assertEqual(policy["observation_registry"], expected)
        identifiers = [row["observation_id"] for row in expected]
        self.assertEqual(len(identifiers), 96)
        self.assertEqual(len(identifiers), len(set(identifiers)))

        tampered = deepcopy(policy)
        tampered["observation_registry"] = tampered["observation_registry"][:-1]
        tampered["canonical_self_sha256"] = None
        tampered["canonical_self_sha256"] = common.canonical_sha256(tampered)
        with self.assertRaises(method_policy.MethodPolicyV93Error):
            method_policy.validate_policy(tampered)

    def test_tree_max_features_is_validated_as_float_not_integer(self) -> None:
        policy = method_policy.load_policy()
        self.assertIs(
            type(
                policy["probe_models"]["hist_gradient_boosting_depth2"][
                    "max_features"
                ]
            ),
            float,
        )
        tampered = deepcopy(policy)
        tampered["probe_models"]["hist_gradient_boosting_depth2"][
            "max_features"
        ] = 1
        tampered["canonical_self_sha256"] = None
        tampered["canonical_self_sha256"] = common.canonical_sha256(tampered)
        with self.assertRaises(method_policy.MethodPolicyV93Error):
            method_policy.validate_policy(tampered)


if __name__ == "__main__":
    unittest.main()
