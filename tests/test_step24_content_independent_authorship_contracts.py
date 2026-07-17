from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import step24_common as common  # noqa: E402
import step15_v7_common as v7_common  # noqa: E402


POLICY_PATH = ROOT / "schema" / "step24_content_independent_authorship_policy.json"


class Step24ContentIndependentAuthorshipContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))

    def test_policy_validates_and_primary_is_fixed_three_feature_fusion(self) -> None:
        common.validate_policy(self.policy)
        evaluation = self.policy["evaluation"]
        self.assertEqual(evaluation["primary_model"], "semantic_style_lr_l2_primary")
        self.assertEqual(evaluation["matched_baseline_model"], "e5_lr_l2_control")
        self.assertEqual(
            evaluation["model_feature_sets"][evaluation["primary_model"]],
            [
                "identifier_redacted_e5_cosine",
                "pcm_multilingual_authorship_cosine",
                "mstyledistance_cosine",
            ],
        )
        self.assertTrue(evaluation["candidate_selection_forbidden"])
        self.assertTrue(evaluation["valid_or_test_selection_forbidden"])

    def test_encoders_are_frozen_local_inference_only(self) -> None:
        encoders = self.policy["frozen_style_encoders"]
        self.assertEqual(set(encoders), {"pcm_multilingual_authorship", "mstyledistance"})
        self.assertEqual(encoders["pcm_multilingual_authorship"]["expected_dimension"], 1024)
        self.assertEqual(encoders["mstyledistance"]["expected_dimension"], 768)
        for cfg in encoders.values():
            self.assertEqual(cfg["loader"], "sentence_transformers")
            self.assertRegex(cfg["revision"], r"^[0-9a-f]{40}$")
            self.assertTrue(cfg["local_finetuning_forbidden"])
            self.assertTrue(cfg["inference_time_content_masking_forbidden"])
            self.assertTrue(cfg["local_path"].startswith("models/step24/"))

    def test_clean_text_replays_v7_identifier_redaction_and_train_only_scope(self) -> None:
        cfg = self.policy["clean_text_contract"]
        self.assertEqual(cfg["source"], "step15_v7_identifier_redacted_clean_text_exact_replay")
        self.assertEqual(cfg["encode_split"], "train")
        self.assertFalse(cfg["encode_valid_or_test"])
        self.assertTrue(cfg["require_exact_v7_corpus_hash_replay"])
        self.assertNotIn("profile_text", cfg["text_fields"])
        self.assertIn("profile_text", cfg["excluded_profile_fields"])

    def test_pair_features_forbid_previous_shortcuts_and_high_dimensional_projection(self) -> None:
        cfg = self.policy["pair_features"]
        self.assertTrue(cfg["identifiers_forbidden"])
        self.assertTrue(cfg["candidate_rule_features_forbidden"])
        self.assertTrue(cfg["random_projection_forbidden"])
        self.assertTrue(cfg["item_distribution_features_forbidden"])
        declared = {
            cfg["identifier_redacted_e5_cosine"],
            cfg["pcm_multilingual_authorship_cosine"],
            cfg["mstyledistance_cosine"],
        }
        self.assertEqual(len(declared), 3)

    def test_all_non_model_inputs_exist(self) -> None:
        for value in self.policy["inputs"].values():
            self.assertTrue((ROOT / value).is_file(), value)
        for pool_cfg in self.policy["pools"].values():
            for key in (
                "frozen_labels",
                "evidence_labels",
                "seller_profiles",
                "item_identity_signals",
                "identifier_redacted_e5_metadata",
                "identifier_redacted_e5_matrix",
            ):
                value = pool_cfg[key]
                self.assertTrue((ROOT / value).is_file(), f"{key}:{value}")

    def test_balanced_component_folds_keep_components_intact_and_both_classes(self) -> None:
        rows = []
        for component_index in range(15):
            component = f"component-{component_index:02d}"
            rows.append(
                {
                    "pair_uid": f"{component}-positive",
                    "step24_component_id": component,
                    "review_label": "positive",
                }
            )
            rows.append(
                {
                    "pair_uid": f"{component}-negative",
                    "step24_component_id": component,
                    "review_label": "negative",
                }
            )
        assignment = common.balanced_component_folds(rows, 5, 20260717)
        self.assertEqual(set(assignment), {row["step24_component_id"] for row in rows})
        for fold in range(5):
            held = [row for row in rows if assignment[row["step24_component_id"]] == fold]
            self.assertEqual({row["review_label"] for row in held}, {"positive", "negative"})

    def test_current_canonical_train_inputs_form_valid_grouped_folds(self) -> None:
        rows_by_pool = common.load_canonical_train_rows(self.policy)
        self.assertEqual(set(rows_by_pool), {"en_content_train_pool", "zh_target_strict"})
        for rows in rows_by_pool.values():
            self.assertTrue(rows)
            self.assertEqual({row["split_name"] for row in rows}, {"train"})
            self.assertEqual({row["review_label"] for row in rows}, {"positive", "negative"})
        zh_rows = rows_by_pool["zh_target_strict"]
        assignment = common.balanced_component_folds(
            zh_rows,
            self.policy["evaluation"]["fold_count"],
            self.policy["evaluation"]["fold_seed"],
        )
        self.assertEqual(
            set(assignment), {row["step24_component_id"] for row in zh_rows}
        )
        for fold in range(self.policy["evaluation"]["fold_count"]):
            held = [
                row
                for row in zh_rows
                if assignment[row["step24_component_id"]] == fold
            ]
            self.assertEqual({row["review_label"] for row in held}, {"positive", "negative"})

        v7_policy_path = ROOT / self.policy["inputs"]["v7_policy"]
        v7_policy = json.loads(v7_policy_path.read_text(encoding="utf-8"))
        all_rows = rows_by_pool["en_content_train_pool"] + zh_rows
        weights, diagnostics = v7_common.factorized_evidence_weights(
            all_rows, v7_policy["factorized_evidence_weighting"]
        )
        self.assertEqual(len(weights), len(all_rows))
        self.assertGreater(diagnostics["min"], 0.0)

    def test_immutable_writer_rejects_changed_payload(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            path = Path(temporary) / "artifact.json"
            common.write_json_immutable(path, {"value": 1})
            common.write_json_immutable(path, {"value": 1})
            with self.assertRaises(ValueError):
                common.write_json_immutable(path, {"value": 2})

    def test_scientific_constraints_forbid_synthetic_truth_and_test_selection(self) -> None:
        constraints = " ".join(self.policy["scientific_constraints"])
        self.assertIn("creates no local synthetic seller", constraints)
        self.assertIn("valid/test labels, scores and pair features are never used", constraints)
        self.assertIn("Step20", constraints)


if __name__ == "__main__":
    unittest.main()
