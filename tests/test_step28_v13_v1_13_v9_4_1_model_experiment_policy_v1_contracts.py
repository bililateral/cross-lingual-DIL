from __future__ import annotations

import copy
import hashlib
import json
import sys
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_v9_4_1_model_experiment_common_v1 as common


class ModelExperimentPolicyContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = common.load_policy()

    def test_policy_self_hash_and_pretraining_boundary(self) -> None:
        policy = self.policy
        unsigned = dict(policy)
        claimed = unsigned.pop("canonical_self_hash")
        self.assertEqual(common.canonical_sha256(unsigned), claimed)
        self.assertEqual(policy["status"], common.EXPECTED_STATUS)
        self.assertFalse(policy["m0_m1_m2_m3_training_authorized"])
        self.assertFalse(policy["audit_truth_authorized"])
        raw = common.DEFAULT_POLICY.read_bytes()
        self.assertEqual(len(raw), common.EXPECTED_POLICY_SIZE_BYTES)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), common.EXPECTED_POLICY_SHA256)
        self.assertEqual(claimed, common.EXPECTED_POLICY_CANONICAL_SELF_HASH)

    def test_resigned_policy_drift_is_rejected_before_semantic_validation(self) -> None:
        mutations = []

        def add_label_file(value):
            value["public_projection"]["allowed_observed_files"].append(
                "pair_labels.csv"
            )

        def remove_identity_forbidden(value):
            value["public_projection"]["m0_forbidden_inputs"] = []

        def open_audit_truth(value):
            value["model_stage_truth_boundary"][
                "audit_truth_open_before_blind_prediction_allowed"
            ] = True

        def change_estimand(value):
            value["metric_registry"]["primary_confirmatory_estimand"] = "wrong"

        def change_weight(value):
            value["folds_and_weights"]["m1_m2_train_row_weight"] = "1/(500*378)"

        def change_bootstrap(value):
            value["bootstrap"]["replicates"] = 9998

        mutations.extend(
            [
                add_label_file,
                remove_identity_forbidden,
                open_audit_truth,
                change_estimand,
                change_weight,
                change_bootstrap,
            ]
        )
        for mutate in mutations:
            altered = copy.deepcopy(self.policy)
            mutate(altered)
            altered.pop("canonical_self_hash", None)
            altered["canonical_self_hash"] = common.canonical_sha256(altered)
            raw = json.dumps(altered, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
            with self.assertRaisesRegex(
                common.ModelExperimentContractError, "exact byte-size|exact SHA-256"
            ):
                common.parse_exact_policy_bytes(raw)

    def test_formal_policy_loader_never_opens_historical_draft(self) -> None:
        original = common.verify_file_pin

        def guarded(spec, *, label):
            if spec.get("role") == "HISTORICAL_DRAFT_NO_MODEL_AUTHORITY":
                raise AssertionError("formal loader opened the historical draft")
            return original(spec, label=label)

        with mock.patch.object(common, "verify_file_pin", side_effect=guarded):
            common.load_policy()

    def test_ordinary_test_can_verify_rejected_draft_hash_only(self) -> None:
        spec = self.policy["authority_registry"]["historical_draft"]
        path = common.verify_file_pin(spec, label="rejected historical draft test pin")
        self.assertEqual(path.stat().st_size, spec["size_bytes"])
        self.assertEqual(
            spec["sha256"],
            "ce18015199c864df0f76a240df782c331020e5e76d483c5440cea6a673c74729",
        )
        self.assertFalse(hasattr(common, "validate_historical_draft_rejection_pin"))

    def test_dataset_build_quality_and_model_authority_are_distinct(self) -> None:
        qualification = self.policy["dataset_qualification"]
        self.assertEqual(
            qualification["root_manifest"]["status"],
            "BUILT_NOT_TRAINING_QUALIFIED",
        )
        self.assertFalse(qualification["root_manifest"]["training_qualified"])
        self.assertEqual(
            qualification["quality_result"]["status"],
            "PASSED_FORMAL_500X4_ROOT_QUALITY_TRAINING_QUALIFIED",
        )
        self.assertTrue(qualification["quality_result"]["training_qualified"])
        self.assertFalse(
            qualification["quality_result"]["m0_m1_m2_m3_training_authorized"]
        )

    def test_truth_read_counts_are_stage_separated(self) -> None:
        boundary = self.policy["model_stage_truth_boundary"]
        self.assertEqual(
            boundary["initial_semantic_read_counts"],
            {"train": 0, "development": 0, "audit_a": 0, "audit_b": 0},
        )
        historical = boundary["historical_quality_stage_semantic_read_counts"]
        self.assertEqual(historical["train_labels"], 1)
        self.assertEqual(historical["development_labels"], 1)
        self.assertEqual(historical["audit_a_truth"], 0)
        self.assertEqual(historical["audit_b_truth"], 0)
        self.assertFalse(boundary["train_qrels_model_stage_read_allowed"])

    def test_frozen_m0_and_c0_payloads_replay_internal_contracts(self) -> None:
        models = common.validate_frozen_model_payloads(self.policy)
        self.assertEqual(models["m0"]["feature_count"], 24)
        self.assertEqual(models["c0"]["feature_count"], 18)
        self.assertEqual(models["m0"]["threshold"], 0.2324060118538871)
        self.assertEqual(models["c0"]["threshold"], 0.32706942161832925)
        self.assertEqual(
            self.policy["frozen_models"]["c0"]["role"],
            "DIRECT_SENSITIVITY_ONLY_NO_ADAPTER_NO_GATE",
        )

    def test_bootstrap_indices_have_frozen_bytes(self) -> None:
        expected = {
            "development": "111b1338cc607c6bd78bad88efe47606ffa2230e9cc764eec940e84f86e56661",
            "audit_a": "617be9200ad55b45eda8b1800989d7e0b50579bb53ecee675713f8ba2cd4c3e4",
            "audit_b": "12565157b109301070a3648989e74a1faab05d015b5ac0dbcd772c38a5a91a87",
        }
        for split, digest in expected.items():
            indices = common.bootstrap_indices(self.policy, split)
            self.assertEqual(indices.shape, (9999, 500))
            self.assertEqual(indices.dtype.str, "<i8")
            self.assertTrue(indices.flags.c_contiguous)
            self.assertEqual(
                hashlib.sha256(indices.tobytes(order="C")).hexdigest(), digest
            )

    def test_fold_assignment_is_deterministic_and_balanced(self) -> None:
        worlds = [f"train_world_{index:03d}" for index in range(500)]
        first = common.assign_world_folds(worlds)
        second = common.assign_world_folds(list(reversed(worlds)))
        self.assertEqual(first, second)
        self.assertEqual(Counter(first.values()), Counter({i: 100 for i in range(5)}))

    def test_weight_contract_rejects_old_fixed_one_over_500_semantics(self) -> None:
        altered = copy.deepcopy(self.policy)
        altered["folds_and_weights"]["m1_m2_train_row_weight"] = "1/(500*378)"
        with self.assertRaisesRegex(common.ModelExperimentContractError, "weight"):
            common._validate_static_scientific_contract(altered)

    def test_m1_each_repeat_is_a_zero_overlap_whole_row_bijection(self) -> None:
        sellers = [f"seller_{index:02d}" for index in range(28)]
        mappings = []
        for repeat in self.policy["m1"]["repeat_ids"]:
            mapping = common.build_m1_mapping("world_000", sellers, repeat)
            self.assertEqual(len(mapping), 378)
            self.assertEqual(len(set(mapping.values())), 378)
            for destination, source in mapping.items():
                self.assertNotEqual(destination, source)
                self.assertFalse(set(destination) & set(source))
            mappings.append(
                common.canonical_sha256(
                    [
                        [list(destination), list(source)]
                        for destination, source in sorted(mapping.items())
                    ]
                )
            )
        self.assertEqual(len(set(mappings)), 5)

    def test_m1_active_mask_moves_with_mapped_identity_row(self) -> None:
        sellers = [f"seller_{index:02d}" for index in range(28)]
        mapping = common.build_m1_mapping("world_000", sellers, "r01")
        source_rows = {}
        for index, edge in enumerate(sorted(mapping)):
            row = np.zeros(33, dtype=np.float64)
            if index % 2:
                row[0] = float(index + 1)
            source_rows[edge] = row
        mapped = np.vstack([source_rows[mapping[edge]] for edge in sorted(mapping)])
        expected_active = np.asarray(
            [np.any(source_rows[mapping[edge]] != 0.0) for edge in sorted(mapping)]
        )
        actual_active = common.active_mask(mapped)
        np.testing.assert_array_equal(actual_active, expected_active)

    def test_transform_is_world_equal_not_active_row_equal(self) -> None:
        matrix = np.zeros((6, 33), dtype=np.float64)
        worlds = ["a", "a", "a", "b", "b", "b"]
        matrix[0, 0] = 1.0
        matrix[3, 0] = 3.0
        matrix[4, 0] = 3.0
        scale, mu = common.fit_identity_transform(matrix, worlds)
        self.assertAlmostEqual(scale[0], math_sqrt_five := np.sqrt(5.0))
        self.assertAlmostEqual(mu[0], 2.0 / math_sqrt_five)
        self.assertNotAlmostEqual(scale[0], np.sqrt(19.0 / 3.0))
        phi, active = common.apply_identity_transform(matrix, scale, mu)
        self.assertTrue(np.all(phi[~active] == 0.0))

    def test_shared_l2_tie_rule_uses_global_minimum_set(self) -> None:
        losses = {value: 2.0 for value in common.FROZEN_L2_GRID}
        losses.update({0.01: 1.0, 0.1: 1.0 - 0.5e-12, 1.0: 1.0 + 0.4e-12})
        selected = common.select_shared_l2(losses)
        self.assertEqual(selected, 1.0)

    def test_shared_l2_rejects_missing_extra_and_float_duplicate_keys(self) -> None:
        full = {value: 1.0 for value in common.FROZEN_L2_GRID}
        missing = dict(full)
        missing.pop(100.0)
        with self.assertRaisesRegex(common.ModelExperimentContractError, "frozen nine"):
            common.select_shared_l2(missing)
        with self.assertRaisesRegex(common.ModelExperimentContractError, "frozen nine"):
            common.select_shared_l2({**full, 999.0: 0.0})
        duplicate = dict(full)
        duplicate["0.01"] = 1.0
        with self.assertRaisesRegex(common.ModelExperimentContractError, "duplicate"):
            common.select_shared_l2(duplicate)

    def test_finite_median_is_exact_and_all_missing_fails(self) -> None:
        matrix = np.asarray(
            [[1.0, np.nan], [4.0, 10.0], [2.0, 30.0], [3.0, np.nan]],
            dtype=np.float64,
        )
        medians = common.finite_median_impute_fit(matrix)
        np.testing.assert_array_equal(medians, np.asarray([2.5, 20.0]))
        imputed = common.impute_with_medians(matrix, medians)
        self.assertTrue(np.isfinite(imputed).all())
        with self.assertRaisesRegex(common.ModelExperimentContractError, "entirely missing"):
            common.finite_median_impute_fit(np.full((3, 1), np.nan))

    def test_metric_and_bootstrap_registry_forbids_reselection(self) -> None:
        registry = self.policy["metric_registry"]
        self.assertIn("average_precision", registry["pooled_score_metrics"])
        self.assertIn("trapezoidal_pr_auc", registry["pooled_score_metrics"])
        self.assertIn("map", registry["retrieval_metrics"])
        self.assertIn("mrr", registry["retrieval_metrics"])
        self.assertFalse(self.policy["bootstrap"]["refit_or_reselect_inside_bootstrap_allowed"])
        self.assertFalse(self.policy["thresholds"]["bootstrap_reselect_threshold"])
        self.assertEqual(
            registry["comparison_direction"]["higher_is_better"],
            "metric(target_model)-metric(control_model)",
        )
        self.assertEqual(
            registry["comparison_direction"]["lower_is_better"],
            "metric(control_model)-metric(target_model)",
        )
        self.assertIn("sum_i(w_i*(p_i-y_i)^2)", registry["brier_formula"])
        self.assertIn("p_i_clipped", registry["log_loss_formula"])

    def test_column_name_hashes_match_exact_order(self) -> None:
        features = self.policy["feature_contract"]
        self.assertEqual(
            common.canonical_sha256([*features["legacy18"], *features["labse6"]]),
            features["column_name_hashes"]["base24"],
        )
        self.assertEqual(
            common.canonical_sha256(features["identity33"]),
            features["column_name_hashes"]["identity33"],
        )
        self.assertEqual(
            common.canonical_sha256(
                [*features["legacy18"], *features["labse6"], *features["identity33"]]
            ),
            features["column_name_hashes"]["joint57"],
        )

    def test_supervised_runtime_is_a_mechanical_gate(self) -> None:
        self.assertEqual(
            common.validate_supervised_cpu_runtime(self.policy),
            self.policy["runtime"]["supervised_cpu"],
        )
        altered = copy.deepcopy(self.policy)
        altered["runtime"]["supervised_cpu"]["numpy"] = "0.0.0"
        with self.assertRaisesRegex(common.ModelExperimentContractError, "runtime drift"):
            common.validate_supervised_cpu_runtime(altered)

    def test_encoding_runtime_replays_step7_and_all_four_payload_pins(self) -> None:
        payloads = self.policy["labse_encoding"]["chunk_tokenizer_payloads"]

        def fake_fingerprint(path):
            for pin in payloads.values():
                if common.resolve(pin["path"]) == path:
                    return {
                        "file_count": pin["file_count"],
                        "total_size_bytes": pin["total_size_bytes"],
                        "content_sha256": pin["content_sha256"],
                    }
            raise AssertionError(path)

        with mock.patch.object(
            common.importlib.metadata, "version", return_value="5.6.0"
        ), mock.patch.object(
            common, "model_content_fingerprint", side_effect=fake_fingerprint
        ):
            record = common.validate_encoding_runtime(self.policy)
        self.assertEqual(record["payload_count"], 4)


if __name__ == "__main__":
    unittest.main()
