from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import step15_build_v7_inductive_pair_features as features  # noqa: E402
import step15_build_v7_clean_embedding_cache as clean_embeddings  # noqa: E402
import step15_v7_common as common  # noqa: E402
import step9_run_few_shot_adaptation as step9  # noqa: E402
import step20_build_representative_validation as validation  # noqa: E402
import step20_evaluate_prospective_holdout as prospective_eval  # noqa: E402
import step20_freeze_prospective_holdout as prospective_freeze  # noqa: E402


class Step15V7TwoStageContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.v7_policy = json.loads(
            (ROOT / "schema" / "step15_v7_two_stage_policy.json").read_text(encoding="utf-8")
        )
        cls.step12_policy = json.loads(
            (ROOT / "schema" / "step12_v7_statistical_robustness_policy.json").read_text(
                encoding="utf-8"
            )
        )
        cls.step20_policy = json.loads(
            (ROOT / "schema" / "step20_prospective_holdout_policy.json").read_text(
                encoding="utf-8"
            )
        )

    def test_strict_clean_is_20d_and_contaminated_semantics_cannot_reenter(self) -> None:
        stable = self.v7_policy["inductive_features"]["stable_strict_clean_features"]
        removed = self.v7_policy["inductive_features"]["removed_from_strict_clean"]
        self.assertEqual(len(stable), 20)
        self.assertEqual(len(stable), self.v7_policy["inductive_features"]["stable_strict_clean_feature_count"])
        self.assertEqual(len(set(stable)), 20)
        self.assertFalse(set(stable) & set(removed))
        self.assertNotIn("candidate_rule_count_raw", stable)
        self.assertNotIn("candidate_rule_count_non_identifier", stable)
        self.assertIn("embedding_cosine_multilingual_e5_large_identifier_redacted", stable)
        for contaminated in (
            "embedding_cosine_multilingual_e5_large",
            "embedding_cosine_bge_m3",
            "embedding_cosine_labse",
            "embedding_cosine_gte_multilingual_base",
            "embedding_cosine_paraphrase_multilingual_mpnet",
            "reranker_score_gte_multilingual_reranker_base",
        ):
            self.assertNotIn(contaminated, stable)

    def test_clean_semantic_encoder_excludes_profile_identity_sections(self) -> None:
        cfg = self.v7_policy["clean_semantic_encoder"]
        excluded = set(cfg["excluded_profile_fields"])
        self.assertTrue(
            {"source_seller_raw", "alias_normalized", "contact_concat_top", "profile_text"}
            .issubset(excluded)
        )
        self.assertNotIn("profile_text", cfg["text_fields"])
        self.assertNotIn("contact_concat_top", cfg["text_fields"])
        redacted, diagnostics = clean_embeddings.redact_identifiers(
            "商品联系 Telegram: @seller_demo，邮箱 seller@example.com",
            ["seller_demo"],
        )
        self.assertNotIn("seller_demo", redacted.casefold())
        self.assertNotIn("example.com", redacted.casefold())
        self.assertGreater(diagnostics["generic_identifier_match_count"], 0)

    def test_clean_model_directory_fingerprint_is_content_addressed(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            directory = Path(temporary)
            (directory / "config.json").write_text("{}", encoding="utf-8")
            first = clean_embeddings.directory_fingerprint(directory)
            (directory / "config.json").write_text('{"changed": true}', encoding="utf-8")
            second = clean_embeddings.directory_fingerprint(directory)
            self.assertEqual(first["file_count"], 1)
            self.assertNotEqual(first["files_sha256"], second["files_sha256"])

    def test_oov_signature_uses_df_floor_but_is_not_rare(self) -> None:
        profile = {"signature_titles": [{"value": "novel signature"}]}
        reference = {
            "boilerplate_config": {
                "minimum_signature_value_length": 1,
                "boilerplate_seller_share_threshold": 0.1,
                "rare_seller_share_threshold": 0.01,
            },
            "train_seller_count": 100,
            "title_df": {},
        }
        state = features.reference_signature_state(
            profile,
            reference,
            "signature_titles",
            "title_df",
            self.v7_policy["inductive_features"]["oov_policy"],
        )
        norm = next(iter(state["norms"]))
        self.assertIn(norm, state["oov"])
        self.assertNotIn(norm, state["rare"])
        self.assertAlmostEqual(state["idf"][norm], math.log(101 / 3) + 1.0)

    def test_seller_graph_components_close_transitive_paths(self) -> None:
        rows = [
            {"pair_uid": "ab", "seller_uid_left": "a", "seller_uid_right": "b"},
            {"pair_uid": "bc", "seller_uid_left": "b", "seller_uid_right": "c"},
            {"pair_uid": "de", "seller_uid_left": "d", "seller_uid_right": "e"},
        ]
        validation.attach_seller_graph_components(rows)
        self.assertEqual(rows[0]["v7_component_id"], rows[1]["v7_component_id"])
        self.assertNotEqual(rows[0]["v7_component_id"], rows[2]["v7_component_id"])

    def test_pair_latent_projection_is_endpoint_order_invariant(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            directory = Path(temporary)
            matrix_path = directory / "embeddings.npy"
            metadata_path = directory / "embeddings.json"
            matrix = np.asarray([[1.0, 0.0, 0.5], [0.0, 1.0, -0.5]], dtype=np.float32)
            np.save(matrix_path, matrix)
            metadata_path.write_text(
                json.dumps(
                    {
                        "model_key": "fixture",
                        "identifier_redacted": True,
                        "seller_uids": ["left", "right"],
                        "shape": [2, 3],
                    }
                ),
                encoding="utf-8",
            )
            pool = {
                "clean_e5_cache_metadata": str(metadata_path),
                "clean_e5_cache_matrix": str(matrix_path),
            }
            rows = [
                {"pair_uid": "lr", "seller_uid_left": "left", "seller_uid_right": "right"},
                {"pair_uid": "rl", "seller_uid_left": "right", "seller_uid_right": "left"},
            ]
            projected = common.projected_pair_latents(
                rows, pool, self.v7_policy["latent_pair_representation"]
            )
            np.testing.assert_allclose(projected[0], projected[1], rtol=0.0, atol=1e-12)

    def test_factorized_weights_use_domain_evidence_confidence_and_never_eight_x(self) -> None:
        rows = [
            {
                "domain": "zh",
                "v7_component_id": "a",
                "evidence_type": "same_controller_direct_identifier",
                "training_sample_weight": "1.0",
            },
            {
                "domain": "zh",
                "v7_component_id": "b",
                "evidence_type": "same_controller_style_structural_soft",
                "training_sample_weight": "0.2",
            },
            {
                "domain": "en",
                "v7_component_id": "c",
                "evidence_type": "ordinary_negative",
                "training_sample_weight": "1.0",
            },
        ]
        weights, diagnostics = common.factorized_evidence_weights(
            rows, self.v7_policy["factorized_evidence_weighting"]
        )
        self.assertGreater(weights[0], weights[1])
        self.assertLessEqual(float(np.max(weights)), 2.5)
        self.assertLess(float(np.max(weights)), 8.0)
        self.assertIn("zh:same_controller_direct_identifier", diagnostics["by_domain_evidence"])

    def _mixup_fixture(self) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray]:
        rows = [
            {
                "pair_uid": "p1",
                "review_label": "positive",
                "domain": "zh",
                "evidence_type": "same_controller_direct_identifier",
                "training_sample_weight": "1.0",
            },
            {
                "pair_uid": "p2",
                "review_label": "positive",
                "domain": "zh",
                "evidence_type": "same_controller_direct_identifier",
                "training_sample_weight": "0.8",
            },
            {
                "pair_uid": "n1",
                "review_label": "negative",
                "domain": "zh",
                "evidence_type": "ordinary_negative",
                "training_sample_weight": "1.0",
            },
            {
                "pair_uid": "n2",
                "review_label": "negative",
                "domain": "zh",
                "evidence_type": "ordinary_negative",
                "training_sample_weight": "1.0",
            },
            {
                "pair_uid": "n3",
                "review_label": "negative",
                "domain": "en",
                "evidence_type": "ordinary_negative",
                "training_sample_weight": "1.0",
            },
        ]
        clean = np.arange(10, dtype=float).reshape(5, 2)
        latent = np.asarray([[0.0, 0.0], [1.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0]])
        weights = np.asarray([1.0, 0.8, 1.0, 1.0, 1.0])
        return rows, clean, latent, weights

    def test_mixup_schedule_never_crosses_domain_or_evidence_type(self) -> None:
        rows, _, latent, weights = self._mixup_fixture()
        schedule, diagnostics = common.build_mixup_schedule(
            rows,
            latent,
            weights,
            self.v7_policy["step9_latent_mixup"]["mixup"],
            20260320,
        )
        self.assertTrue(schedule)
        self.assertEqual(diagnostics["cross_domain_parent_pairs"], 0)
        self.assertEqual(diagnostics["gap_reference_scope"], "target_domain_support_only")
        self.assertEqual(diagnostics["target_domain_real_row_count"], 4)
        self.assertIs(diagnostics["schedule_budget_satisfied"], True)
        self.assertAlmostEqual(
            diagnostics["actual_synthetic_effective_weight"],
            diagnostics["target_additional_positive_weight"],
            places=12,
        )
        for item in schedule:
            left = rows[item["anchor_index"]]
            right = rows[item["partner_index"]]
            self.assertEqual(left["domain"], right["domain"])
            self.assertEqual(left["evidence_type"], right["evidence_type"])

    def test_mixup_changes_only_latent_and_duplication_matches_effective_weight(self) -> None:
        rows, clean, latent, weights = self._mixup_fixture()
        schedule, _ = common.build_mixup_schedule(
            rows,
            latent,
            weights,
            self.v7_policy["step9_latent_mixup"]["mixup"],
            20260320,
        )
        dup_clean, dup_latent, dup_weight, _ = common.augment_from_schedule(
            clean, latent, rows, schedule, "equal_effective_weight_duplication"
        )
        mix_clean, mix_latent, mix_weight, _ = common.augment_from_schedule(
            clean, latent, rows, schedule, "latent_pair_embedding_mixup"
        )
        for index, item in enumerate(schedule):
            anchor = item["anchor_index"]
            np.testing.assert_array_equal(dup_clean[index], clean[anchor])
            np.testing.assert_array_equal(mix_clean[index], clean[anchor])
            np.testing.assert_array_equal(dup_latent[index], latent[anchor])
            expected = (1.0 - item["lambda_partner"]) * latent[anchor] + item[
                "lambda_partner"
            ] * latent[item["partner_index"]]
            np.testing.assert_allclose(mix_latent[index], expected)
        np.testing.assert_array_equal(dup_weight, mix_weight)
        self.assertAlmostEqual(float(np.sum(dup_weight)), float(np.sum(mix_weight)), places=12)

    def test_uniform_class_weight_keeps_augmentation_control_interpretable(self) -> None:
        cfg = self.v7_policy["step9_latent_mixup"]
        self.assertEqual(cfg["logistic"]["class_weight"], "none")
        self.assertEqual(
            cfg["experiments"],
            [
                "no_augmentation",
                "equal_effective_weight_duplication",
                "latent_pair_embedding_mixup",
            ],
        )
        self.assertEqual(
            cfg["logistic"]["standardization_reference"],
            "real_train_rows_only_shared_across_all_three_controls",
        )
        self.assertEqual(
            cfg["logistic"]["sample_weight_total_normalization"],
            "real_train_row_count_shared_across_all_three_controls",
        )

    def test_zero_percent_support_is_a_matched_source_only_control(self) -> None:
        cfg = self.v7_policy["step9_latent_mixup"]
        self.assertIn(0.0, cfg["support_ratios"])
        self.assertEqual(common.stratified_support_sample(self._mixup_fixture()[0], 0.0, 1), [])
        self.assertIn("step9_v7_source_only_clean_fusion", self.step12_policy["models"])
        self.assertIn(
            "step9_v7_source_only_clean_fusion",
            self.step20_policy["evaluation"]["models"],
        )

    def test_augmented_controls_can_share_a_fixed_real_train_weight_total(self) -> None:
        base = np.ones(5, dtype=float)
        multipliers = np.asarray([1.0, 0.8, 1.2, 0.7, 1.5], dtype=float)
        adjusted, summary = step9.apply_logistic_row_sample_weights(
            base,
            multipliers,
            target_total=3.0,
        )
        self.assertAlmostEqual(float(np.sum(adjusted)), 3.0, places=12)
        self.assertEqual(summary["target_total"], 3.0)
        self.assertEqual(summary["normalized_total"], 3.0)

    def test_clean_ranker_has_no_auxiliary_evidence_head(self) -> None:
        stage_a = self.v7_policy["two_stage_method"]["stage_a"]
        self.assertIs(stage_a["auxiliary_evidence_head"], False)
        self.assertNotIn(
            "evidence_type",
            self.v7_policy["inductive_features"]["stable_strict_clean_features"],
        )

    def test_reliability_veto_uses_raw_context_and_public_precedence_without_labels(self) -> None:
        token = ("telegram", "shared")
        direct_occurrence = {
            "direct_identity_eligible": "1",
            "seller_facing_context": "1",
            "product_data_risk_context": "0",
            "support_only": "0",
        }
        risky_occurrence = {
            "direct_identity_eligible": "0",
            "seller_facing_context": "0",
            "product_data_risk_context": "1",
            "support_only": "0",
        }
        cfg = self.v7_policy["two_stage_method"]["stage_b"]
        direct = common.relation_reliability(
            {"seller_uid_left": "a", "seller_uid_right": "b"},
            {"a": {token: [direct_occurrence]}, "b": {token: [direct_occurrence]}},
            CounterFixture({token: 2}),
            cfg,
        )
        public = common.relation_reliability(
            {"seller_uid_left": "a", "seller_uid_right": "b"},
            {"a": {token: [risky_occurrence]}, "b": {token: [risky_occurrence]}},
            CounterFixture({token: 2}),
            cfg,
        )
        self.assertEqual(direct["decision"], "verified_seller_facing_direct")
        self.assertEqual(public["decision"], "public_or_product_contact_veto")
        self.assertEqual(public["score_multiplier"], 0.1)

    def test_validation_and_internal_test_selection_are_separated(self) -> None:
        self.assertEqual(
            self.v7_policy["evaluation"]["current_zh_test_role"],
            "internal_development_test_only",
        )
        self.assertIs(self.v7_policy["evaluation"]["selection_uses_current_zh_test"], False)
        self.assertIs(self.step12_policy["publication_promotion"]["eligible"], False)
        self.assertIs(self.step12_policy["selection"]["test_metrics_used_for_selection"], False)

    def test_old_candidates_are_not_prospective_final_eligible(self) -> None:
        sources = self.step20_policy["candidate_sources"]
        self.assertIs(sources[0]["prospective_final_eligible"], False)
        self.assertTrue(any(source["prospective_final_eligible"] for source in sources))
        for source in sources:
            if source["prospective_final_eligible"]:
                self.assertIs(source["requires_collection_after_model_freeze"], True)

    def test_prospective_candidate_schema_requires_post_freeze_provenance(self) -> None:
        schema = json.loads(
            (ROOT / self.step20_policy["candidate_schema"]).read_text(encoding="utf-8")
        )
        self.assertIn("collection_timestamp_utc", schema["required_non_empty_fields"])
        self.assertIn("model_score", schema["forbidden_fields"])

    def test_prospective_scoring_physically_uses_pair_universe_not_label_file(self) -> None:
        source = (ROOT / "scripts" / "step20_score_prospective_holdout.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('outputs["frozen_pair_universe"]', source)
        self.assertNotIn('outputs["frozen_labels"]', source)

    def test_step12_model_freeze_is_in_the_atomic_step12_directory(self) -> None:
        output_root = Path(self.step12_policy["outputs_root"])
        model_freeze = Path(self.step12_policy["model_freeze_manifest_output"])
        self.assertEqual(model_freeze.parent, output_root)

    def test_prospective_stage_outputs_are_directory_isolated(self) -> None:
        outputs = {key: Path(value) for key, value in self.step20_policy["outputs"].items()}
        self.assertEqual(outputs["blind_mapping"].parent.name, "preparation_v2")
        self.assertEqual(outputs["frozen_pair_universe"].parent.name, "freeze_v2")
        self.assertEqual(outputs["prospective_pair_features"].parent.name, "features_v2")
        self.assertEqual(outputs["frozen_model_scores"].parent.name, "scores_v2")
        self.assertEqual(outputs["metrics"].parent.name, "evaluation_v2")

    def test_prospective_positive_requires_direct_or_component_evidence(self) -> None:
        with self.assertRaisesRegex(ValueError, "lacks direct/component"):
            prospective_freeze.validate_decision(
                "positive", "template_clone_not_controller", self.step20_policy
            )
        prospective_freeze.validate_decision(
            "positive", "same_controller_direct_identifier", self.step20_policy
        )

    def test_prospective_evaluation_lock_is_exclusive(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            path = Path(temporary) / "lock.json"
            prospective_eval.create_lock_once(path, {"status": "in_progress"})
            with self.assertRaises(FileExistsError):
                prospective_eval.create_lock_once(path, {"status": "second_attempt"})

    def test_map_and_mrr_are_not_claimed_without_query_groups(self) -> None:
        self.assertEqual(
            self.step12_policy["metrics"]["map_mrr_status"],
            "not_applicable_without_preregistered_query_groups",
        )


class CounterFixture(dict):
    def __missing__(self, key):
        return 0


if __name__ == "__main__":
    unittest.main()
