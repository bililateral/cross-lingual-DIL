from __future__ import annotations

import ast
import inspect
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import step27_common as common  # noqa: E402
import step27_build_parent_manifest as parent_builder  # noqa: E402
import step27_encode_profiles as encoding  # noqa: E402
import step27_train_residual_models as training  # noqa: E402


POLICY_PATH = ROOT / "schema" / "step27_english_pretrained_synthetic_adaptation_policy.json"


class Step27EnglishPretrainedSyntheticContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))

    def test_frozen_english_source_artifact_is_exact(self) -> None:
        source_path = common.resolve(self.policy["inputs"]["step24_model_artifacts"])
        self.assertEqual(
            common.sha256_file(source_path),
            self.policy["inputs"]["step24_model_artifacts_sha256"],
        )
        payload = common.load_json(source_path)
        artifact = payload["artifacts"]["source_only"]["e5_lr_l2_control"]
        expected = self.policy["frozen_english_source_scorer"]
        self.assertEqual(artifact["train_row_count"], 401)
        self.assertEqual(artifact["train_positive_count"], 116)
        self.assertEqual(artifact["train_negative_count"], 285)
        self.assertEqual(artifact["feature_names"], ["identifier_redacted_e5_cosine"])
        self.assertEqual(expected["source_logit_coefficient_in_primary_models"], 1.0)
        training.validate_frozen_source_contract(self.policy, source_path, artifact)

    def test_canonical_chinese_boundary_is_unchanged(self) -> None:
        rows = common.canonical_rows(self.policy, {"train", "valid", "test"})
        observed = {}
        for split_name in ("train", "valid", "test"):
            selected = [row for row in rows if row["split_name"] == split_name]
            observed[split_name] = {
                "rows": len(selected),
                "positive": sum(row["review_label"] == "positive" for row in selected),
                "negative": sum(row["review_label"] == "negative" for row in selected),
            }
        self.assertEqual(observed, self.policy["canonical_chinese_boundary"]["split_counts"])
        self.assertEqual(observed["train"], {"rows": 573, "positive": 229, "negative": 344})
        self.assertEqual(observed["valid"], {"rows": 120, "positive": 30, "negative": 90})
        self.assertEqual(observed["test"], {"rows": 200, "positive": 50, "negative": 150})

    def test_primary_and_silver_parent_truth_boundaries_are_explicit(self) -> None:
        rows = common.canonical_rows(self.policy, {"train"})
        primary = [
            row
            for row in rows
            if row["review_label"] == "positive"
            and not common.bool_value(row.get("silver_train_only"))
        ]
        self.assertEqual(len(primary), 16)
        self.assertEqual(len({row["component_id"] for row in primary}), 13)
        self.assertEqual(
            Counter(row["review_stratum"] for row in primary),
            Counter({"identifier_plus_text": 1, "semantic_structural": 6, "semantic_only": 9}),
        )
        silver_direct = [
            row
            for row in rows
            if row["review_label"] == "positive"
            and common.bool_value(row.get("silver_train_only"))
            and row.get("review_stratum") == "silver_direct_or_contact"
        ]
        self.assertEqual(len(silver_direct), 56)

    def test_primary_and_silver_have_same_fold_negative_parent_capacity(self) -> None:
        rows = common.canonical_rows(self.policy, {"train"})
        fold_count, fold_seed = common.fold_config(self.policy)
        folds = common.build_fixed_component_folds(rows, fold_count, fold_seed)
        limits = common.generation_limits(self.policy)
        cohorts = (
            {
                "track": "primary",
                "silver": False,
                "positive_cap": limits["primary_positive_parents"],
                "selection_seed": fold_seed,
                "evidence_allowlist": None,
                "review_stratum_allowlist": None,
            },
            {
                "track": "silver_sensitivity",
                "silver": True,
                "positive_cap": limits["silver_positive_parents"],
                "selection_seed": fold_seed + 1,
                "evidence_allowlist": {
                    "same_controller_direct_identifier",
                    "same_controller_component_anchor",
                },
                "review_stratum_allowlist": {"silver_direct_or_contact"},
            },
        )
        for cohort in cohorts:
            candidates = [
                row
                for row in rows
                if common.bool_value(row.get("silver_train_only")) is cohort["silver"]
            ]
            positives = [
                row
                for row in candidates
                if row["review_label"] == "positive"
                and (
                    cohort["evidence_allowlist"] is None
                    or row.get("evidence_type") in cohort["evidence_allowlist"]
                )
                and (
                    cohort["review_stratum_allowlist"] is None
                    or row.get("review_stratum") in cohort["review_stratum_allowlist"]
                )
            ]
            selected = common.balanced_parent_select(
                positives,
                cohort["positive_cap"],
                cohort["selection_seed"],
                f"{cohort['track']}:positive",
            )
            self.assertEqual(len(selected), cohort["positive_cap"])
            selected_pair_uids = {row["pair_uid"] for row in selected}
            eligible_negatives = [
                row
                for row in candidates
                if row["review_label"] == "negative"
                and row["pair_uid"] not in selected_pair_uids
            ]
            for fold in range(fold_count):
                demand = sum(folds[row["component_id"]] == fold for row in selected)
                capacity = sum(
                    folds[row["component_id"]] == fold for row in eligible_negatives
                )
                with self.subTest(track=cohort["track"], fold=fold):
                    self.assertGreaterEqual(
                        capacity,
                        demand,
                        "same-fold reviewed-negative pair capacity must cover every "
                        "selected positive; distinct components are preferred, not required",
                    )

        negative_selection = self.policy["parent_cohorts"]["primary_non_silver"][
            "negative_parent_selection"
        ]
        with self.subTest(contract="policy_component_fallback"):
            self.assertTrue(negative_selection["one_parent_pair_per_component_preferred"])
            self.assertFalse(
                common.bool_value(
                    negative_selection.get("forbid_positive_parent_components")
                ),
                "policy must permit a reviewed same-component negative when a giant "
                "component exhausts distinct-component capacity",
            )
        docs = (
            ROOT / "docs" / "STEP27_ENGLISH_PRETRAINED_SYNTHETIC_ADAPTATION_PLAN_20260718.zh.md"
        ).read_text(encoding="utf-8")
        with self.subTest(contract="docs_component_fallback"):
            self.assertNotRegex(
                docs,
                r"(?:排除|禁止使用|不得使用)正父样本所在\s*component",
                "Step27 documentation must describe distinct components as a preference, "
                "not an absolute prohibition",
            )

    def test_actual_parent_selector_executes_same_fold_reviewed_fallback(self) -> None:
        rows = common.canonical_rows(self.policy, {"train"})
        fold_count, fold_seed = common.fold_config(self.policy)
        folds = common.build_fixed_component_folds(rows, fold_count, fold_seed)
        limits = common.generation_limits(self.policy)
        shapes = {
            row["pair_uid"]: {
                "stratum_family": "unit",
                "text_length_bin": 0,
                "segment_count_bin": 0,
                "missingness_pattern": (),
            }
            for row in rows
        }
        tracks = (
            dict(
                track="primary",
                positive_cap=limits["primary_positive_parents"],
                negative_cap=limits["primary_negative_parents"],
                seed=fold_seed,
                folds=folds,
                silver=False,
                positive_evidence_allowlist=None,
                positive_review_stratum_allowlist=None,
                shapes=shapes,
            ),
            dict(
                track="silver_sensitivity",
                positive_cap=limits["silver_positive_parents"],
                negative_cap=limits["silver_negative_parents"],
                seed=fold_seed + 1,
                folds=folds,
                silver=True,
                positive_evidence_allowlist={
                    "same_controller_direct_identifier",
                    "same_controller_component_anchor",
                },
                positive_review_stratum_allowlist={"silver_direct_or_contact"},
                shapes=shapes,
            ),
        )
        selected_tracks = {}
        for arguments in tracks:
            selected = parent_builder.select_matched_track(rows, **arguments)
            selected_tracks[arguments["track"]] = selected
            self.assertEqual(len(selected), 2 * arguments["positive_cap"])
            by_match = {}
            for row in selected:
                by_match.setdefault(row["matched_set_id"], []).append(row)
            self.assertEqual(len(by_match), arguments["positive_cap"])
            for matched_rows in by_match.values():
                self.assertEqual(
                    {row["review_label"] for row in matched_rows}, {"positive", "negative"}
                )
                self.assertEqual(len({row["fold"] for row in matched_rows}), 1)
                self.assertEqual(
                    len({row["matched_component_relation"] for row in matched_rows}), 1
                )
            self.assertGreater(
                sum(
                    matched_rows[0]["matched_component_relation"]
                    == "same_component_fallback"
                    for matched_rows in by_match.values()
                ),
                0,
            )
        common.ensure_track_isolation(
            selected_tracks["primary"], selected_tracks["silver_sensitivity"]
        )

    def test_parent_track_isolation_uses_parent_manifest_identity(self) -> None:
        common.ensure_track_isolation(
            [{"parent_pair_uid": "primary::pair"}],
            [{"parent_pair_uid": "silver::pair"}],
        )
        with self.assertRaisesRegex(ValueError, "primary/silver tracks overlap"):
            common.ensure_track_isolation(
                [{"parent_pair_uid": "shared::pair"}],
                [{"parent_pair_uid": "shared::pair"}],
            )
        with self.assertRaisesRegex(ValueError, "missing parent_pair_uid"):
            common.ensure_track_isolation(
                [{"pair_uid": "wrong-schema::pair"}],
                [{"parent_pair_uid": "silver::pair"}],
            )
        with self.assertRaisesRegex(ValueError, "repeats a parent pair"):
            common.ensure_track_isolation(
                [
                    {"parent_pair_uid": "duplicate::pair"},
                    {"parent_pair_uid": "duplicate::pair"},
                ],
                [{"parent_pair_uid": "silver::pair"}],
            )

    def test_m1_and_m2_additions_preserve_parent_budget_metadata(self) -> None:
        real = [
            {
                "pair_uid": "real::negative",
                "review_label": "negative",
                "split_name": "train",
                "component_id": "component::negative",
                "fold": "0",
                "training_sample_weight": "2.0",
            },
            {
                "pair_uid": "real::positive",
                "review_label": "positive",
                "split_name": "train",
                "component_id": "component::positive",
                "fold": "1",
                "training_sample_weight": "1.5",
            },
        ]
        synthetic = [
            {
                **row,
                "pair_uid": f"synthetic::{row['pair_uid']}",
                "parent_pair_uid": row["pair_uid"],
                "training_sample_weight": str(float(row["training_sample_weight"]) * 0.25),
                "synthetic_train_only": "1",
            }
            for row in real
        ]
        _, m1 = training.model_training_rows(training.MODEL_IDS[1], real, synthetic, None)
        _, m2 = training.model_training_rows(training.MODEL_IDS[2], real, synthetic, None)

        def signature(rows: list[dict]) -> Counter:
            return Counter(
                (
                    row.get("parent_pair_uid") or row.get("step27_duplicate_of"),
                    row["component_id"],
                    row["fold"],
                    row["review_label"],
                    float(row["training_sample_weight"]),
                )
                for row in rows
            )

        self.assertEqual(signature(m1), signature(m2))

    def test_generation_caps_weights_and_seed_semantics_are_preregistered(self) -> None:
        limits = common.generation_limits(self.policy)
        self.assertEqual(limits["primary_child_cap"], 64)
        self.assertEqual(limits["silver_child_cap"], 112)
        self.assertEqual(common.generation_seeds(self.policy), list(range(20260320, 20260330)))
        self.assertFalse(self.policy["replication"]["seed_is_inferential_unit"])
        self.assertEqual(
            self.policy["weighting"]["maximum_child_total_weight_relative_to_parent"],
            0.5,
        )
        self.assertTrue(self.policy["weighting"]["M1_and_M2_effective_weight_must_match_exactly"])

    def test_transforms_preserve_fields_and_content_segments(self) -> None:
        fields = {
            "category_concat_top": "甲 || 乙",
            "signature_title_concat": "标题一 || 标题二",
            "title_concat_top": "商品甲 || 商品乙",
            "signature_description_concat": "描述甲 || 描述乙",
            "description_concat_top": "段落甲 || 段落乙",
        }
        before = {
            key: Counter(common.split_segments(value)) for key, value in fields.items()
        }
        for name in self.policy["generation"]["matched_recipe_schedule_for_positive_and_negative"]:
            transformed = common.transform_fields(
                fields, name, common.deterministic_rng(20260320, name)
            )
            self.assertEqual(set(transformed), set(fields))
            after = {
                key: Counter(common.split_segments(value))
                for key, value in transformed.items()
            }
            self.assertEqual(after, before)

    def test_synthetic_field_order_changes_rendering_without_changing_content(self) -> None:
        parent_fields = {
            "category_concat_top": "甲 || 乙",
            "signature_title_concat": "标题一 || 标题二",
            "title_concat_top": "商品甲 || 商品乙",
            "signature_description_concat": "描述甲 || 描述乙",
            "description_concat_top": "段落甲 || 段落乙",
        }
        transformed = common.transform_fields(
            parent_fields,
            "section_order_rotation",
            common.deterministic_rng(20260320, "synthetic_field_order"),
        )
        profile = common.make_synthetic_profile(
            synthetic_uid="synthetic://step27/unit/left",
            parent_uid="real::left",
            parent_pair_uid="real::left||real::right",
            component_id="component::unit",
            fold=0,
            label="positive",
            track="primary",
            seed=20260320,
            variant_index=0,
            transform_name="section_order_rotation",
            fields=transformed,
        )
        self.assertNotEqual(profile["synthetic_field_order"], list(parent_fields))
        self.assertNotEqual(
            profile["profile_text"], common.render_profile_text(parent_fields)
        )
        parent_segments = Counter(
            (field, segment)
            for field, value in parent_fields.items()
            for segment in common.split_segments(value)
        )
        synthetic_segments = Counter(
            (field, segment)
            for field in profile["synthetic_field_order"]
            for segment in common.split_segments(profile[field])
        )
        self.assertEqual(synthetic_segments, parent_segments)

        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "synthetic_profiles.jsonl"
            common.write_jsonl_immutable(source_path, [profile])
            cleaned, _ = encoding.clean_synthetic_profiles(
                source_path, list(parent_fields)
            )
        self.assertEqual(cleaned[0]["profile_text"], profile["profile_text"])

    def test_training_validate_rows_rejects_exposed_valid_or_test_rows(self) -> None:
        training_path = ROOT / "scripts" / "step27_train_residual_models.py"
        tree = ast.parse(
            training_path.read_text(encoding="utf-8"), filename=str(training_path)
        )
        main = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        validation_calls = [
            node
            for node in ast.walk(main)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "validate_rows"
        ]
        self.assertTrue(validation_calls, "training main must validate its train inputs")
        for call in validation_calls:
            required_splits = next(
                (
                    keyword.value
                    for keyword in call.keywords
                    if keyword.arg == "required_real_splits"
                ),
                None,
            )
            self.assertIsNotNone(
                required_splits,
                "training validate_rows calls must declare their visible real splits",
            )
            self.assertEqual(ast.literal_eval(required_splits), ("train",))

        feature_name = "clean_token_jaccard"
        real_rows = [
            {
                "pair_uid": "real::negative",
                "review_label": "negative",
                "split_name": "train",
                "component_id": "component::negative",
                "fold": "0",
                feature_name: "0.1",
            },
            {
                "pair_uid": "real::positive",
                "review_label": "positive",
                "split_name": "train",
                "component_id": "component::positive",
                "fold": "1",
                feature_name: "0.9",
            },
        ]
        synthetic_row = {
            "pair_uid": "synthetic::positive",
            "parent_pair_uid": "real::positive",
            "review_label": "positive",
            "split_name": "train",
            "synthetic_split_name": "synthetic_train_only",
            "synthetic_train_only": "1",
            "component_id": "component::positive",
            "fold": "1",
            "seed": "20260320",
            feature_name: "0.8",
        }
        kwargs = {
            "names": [feature_name],
            "fold_count": 4,
            "seeds": (20260320,),
            "required_real_splits": ("train",),
        }
        training.validate_rows(real_rows, [synthetic_row], **kwargs)
        for leaked_split in ("valid", "test"):
            leaked_real = {
                **real_rows[0],
                "pair_uid": f"real::{leaked_split}",
                "split_name": leaked_split,
            }
            with self.subTest(kind="real", split=leaked_split):
                with self.assertRaisesRegex(ValueError, "exposes splits"):
                    training.validate_rows(
                        [*real_rows, leaked_real], [synthetic_row], **kwargs
                    )
            leaked_synthetic = {**synthetic_row, "split_name": leaked_split}
            with self.subTest(kind="synthetic", split=leaked_split):
                with self.assertRaisesRegex(ValueError, "not train-only"):
                    training.validate_rows(real_rows, [leaked_synthetic], **kwargs)

    def test_feature_contract_excludes_identity_and_parent_copy_shortcuts(self) -> None:
        configured = tuple(self.policy["pair_representation"]["residual_feature_names"])
        self.assertEqual(configured, common.FEATURE_NAMES[1:])
        forbidden = ("contact", "identifier", "pgp", "wallet", "uppercase", "candidate_rule")
        for feature in configured:
            self.assertFalse(any(token in feature.casefold() for token in forbidden))
        self.assertFalse(self.policy["generation"]["copy_parent_pair_features"])
        self.assertTrue(self.policy["generation"]["recompute_all_pair_features_from_transformed_views"])
        audit_source = (ROOT / "scripts" / "step27_audit_synthetic_data.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("M1_M2_parent_component_fold_label_weight_parity", audit_source)
        self.assertIn("distinguishability_not_estimable", audit_source)

    def test_shared_real_fold_standardization_is_supported(self) -> None:
        signature = inspect.signature(training.fit_offset_logistic)
        self.assertIn("standardization", signature.parameters)
        source = (ROOT / "scripts" / "step27_train_residual_models.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("real_fold_train_rows_shared_by_M0_M1_M2", source)
        self.assertIn("standardization={\"mean\": real_mean.tolist()", source)

    def test_fixed_learned_and_zero_source_modes_have_consistent_dimensions(self) -> None:
        names = ["clean_token_jaccard", "clean_text_length_gap_ratio"]
        rows = []
        for index in range(8):
            rows.append(
                {
                    "pair_uid": f"unit::{index}",
                    "review_label": "positive" if index % 2 else "negative",
                    "training_sample_weight": "1",
                    "identifier_redacted_e5_cosine": str(0.80 + index * 0.02),
                    "clean_token_jaccard": str(index / 8),
                    "clean_text_length_gap_ratio": str((7 - index) / 8),
                }
            )
        source = {
            "logistic_artifact": {
                "parameter_intercept": -1.0,
                "parameter_coefficients": [0.8],
                "standardization": {"mean": [0.9], "scale": [0.05]},
            }
        }
        cfg = {"l2_penalty": 10.0, "max_iter": 400, "tolerance": 1e-8}
        for mode, expected_width in (
            ("fixed_unit_offset", 2),
            ("target_only_alpha_zero", 2),
            ("learned_source_alpha", 3),
        ):
            artifact = training.train_one(
                rows,
                [],
                names,
                source,
                "identifier_redacted_e5_cosine",
                cfg,
                source_mode=mode,
            )
            self.assertEqual(len(artifact["coefficients"]), expected_width)
            scores = training.predict_rows(
                rows, artifact, names, source, "identifier_redacted_e5_cosine"
            )
            self.assertEqual(scores.shape, (8,))
            self.assertTrue(np.isfinite(scores).all())
        self.assertIn("learned_source_alpha", artifact)

    def test_pr_auc_trapezoid_is_reported_separately_from_average_precision(self) -> None:
        y = np.asarray([1, 0, 1, 0], dtype=int)
        scores = np.asarray([0.9, 0.8, 0.7, 0.1], dtype=float)
        result = training.metrics(y, scores, 0.5)
        self.assertIn("average_precision", result)
        self.assertIn("pr_auc_trapezoidal", result)
        self.assertGreaterEqual(result["pr_auc_trapezoidal"], 0.0)
        self.assertLessEqual(result["pr_auc_trapezoidal"], 1.0)

    def test_prediction_serialization_preserves_near_threshold_decisions(self) -> None:
        threshold = 0.5000000000004
        source_rows = [
            {
                "pair_uid": "unit::below",
                "review_label": "negative",
                "component_id": "component::below",
                "evidence_type": "ordinary_negative",
            },
            {
                "pair_uid": "unit::above",
                "review_label": "positive",
                "component_id": "component::above",
                "evidence_type": "same_controller_component_anchor",
            },
        ]
        records = training.prediction_rows(
            source_rows,
            np.asarray([0.5000000000003, 0.5000000000005]),
            "unit_model",
            "train_oof",
            20260320,
            threshold,
        )
        for record in records:
            self.assertEqual(
                int(record["predicted_label"]),
                int(
                    float(record["prob_positive"])
                    >= float(record["frozen_oof_threshold"])
                ),
            )
        self.assertEqual([row["predicted_label"] for row in records], [0, 1])
        self.assertNotEqual(
            float(records[0]["prob_positive"]),
            float(records[0]["frozen_oof_threshold"]),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "near_threshold_predictions.csv"
            training.write_csv_immutable(path, records)
            persisted = training.load_csv(path)
        for record in persisted:
            self.assertEqual(
                int(record["predicted_label"]),
                int(
                    float(record["prob_positive"])
                    >= float(record["frozen_oof_threshold"])
                ),
            )

    def test_bootstrap_and_permutation_seeds_are_distinct(self) -> None:
        statistics = self.policy["statistics"]
        bootstrap_seed = statistics["grouped_bootstrap_seed"]
        permutation_seed = statistics["paired_permutation_seed"]
        self.assertIsInstance(bootstrap_seed, int)
        self.assertIsInstance(permutation_seed, int)
        self.assertNotEqual(bootstrap_seed, permutation_seed)

    def test_oof_gate_precedes_valid_and_internal_test_in_runner(self) -> None:
        runner = (
            ROOT / "scripts" / "run_step27_english_pretrained_synthetic_linux_20260718.sh"
        ).read_text(encoding="utf-8")
        oof_gate = runner.index("--mode oof_gate")
        score_valid = runner.index("--score-valid")
        valid_gate = runner.index("--mode valid_gate")
        score_test = runner.index("--score-internal-test")
        final = runner.index("--mode final_diagnostic")
        self.assertLess(oof_gate, score_valid)
        self.assertLess(score_valid, valid_gate)
        self.assertLess(valid_gate, score_test)
        self.assertLess(score_test, final)
        self.assertIn("eligible_for_valid", runner)
        self.assertIn("eligible_for_internal_test", runner)

    def test_source_dependence_controls_are_implemented_and_diagnostic_only(self) -> None:
        controls = self.policy["models"]["exploratory_controls"]
        self.assertEqual(
            training.EXPLORATORY_MODEL_IDS,
            (
                controls["learned_source_logit_coefficient"]["model_id"],
                controls["target_only_alpha_zero"]["model_id"],
            ),
        )
        self.assertEqual(training.source_mode_for_model(training.EXPLORATORY_MODEL_IDS[0]), "learned_source_alpha")
        self.assertEqual(training.source_mode_for_model(training.EXPLORATORY_MODEL_IDS[1]), "target_only_alpha_zero")
        self.assertEqual(training.training_model_id_for(training.EXPLORATORY_MODEL_IDS[0]), training.MODEL_IDS[2])
        self.assertIn("diagnostic_only", controls["learned_source_logit_coefficient"]["publication_role"])

    def test_policy_output_paths_match_runtime_namespaces(self) -> None:
        outputs = self.policy["outputs"]
        self.assertEqual(outputs["fold_manifest"], "parent_manifest/fixed_four_fold_components.csv")
        self.assertEqual(outputs["oof_gate_audit"], "statistical_audit/oof_gate/step12_step27_statistical_audit.json")
        self.assertEqual(outputs["valid_gate_audit"], "statistical_audit/valid_gate/step12_step27_statistical_audit.json")
        self.assertEqual(outputs["sync_manifest"], "manifests/step27_sync_manifest.json")

    def test_immutable_writer_refuses_changed_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            common.write_json_immutable(path, {"value": 1})
            common.write_json_immutable(path, {"value": 1})
            with self.assertRaises(ValueError):
                common.write_json_immutable(path, {"value": 2})

    def test_publication_claim_remains_step20_only(self) -> None:
        self.assertFalse(self.policy["scientific_position"]["current_valid_test_are_confirmatory"])
        confirmatory = self.policy["scientific_position"]["only_confirmatory_evaluation"].casefold()
        self.assertIn("prospective", confirmatory)
        self.assertIn("frozen", confirmatory)
        step20 = self.policy["development_promotion_gates"]["step20"]
        self.assertTrue(step20["required_for_publication_claim"])
        self.assertFalse(step20["evaluation_currently_authorized"])
        self.assertTrue(step20["requires_new_step27_specific_policy_and_model_freeze_manifest"])

    def test_internal_valid_gate_does_not_directly_grant_step20_eligibility(self) -> None:
        step20 = self.policy["development_promotion_gates"]["step20"]
        self.assertEqual(step20["evaluation_count"], 1)
        self.assertTrue(step20["requires_frozen_code_policy_model_and_manifest_hashes"])

        audit_path = ROOT / "scripts" / "step12_step27_statistical_audit.py"
        tree = ast.parse(audit_path.read_text(encoding="utf-8"), filename=str(audit_path))
        direct_aliases = {"eligible_for_internal_test"}
        changed = True
        while changed:
            changed = False
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                    continue
                value = node.value
                if not isinstance(value, ast.Name) or value.id not in direct_aliases:
                    continue
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name) and target.id not in direct_aliases:
                        direct_aliases.add(target.id)
                        changed = True

        step20_values = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            for key, value in zip(node.keys, node.values, strict=True):
                if (
                    isinstance(key, ast.Constant)
                    and key.value == "eligible_for_step20_prospective_evaluation"
                ):
                    step20_values.append(value)
        self.assertTrue(step20_values, "statistical audit must report Step20 eligibility")
        direct_bindings = [
            value
            for value in step20_values
            if isinstance(value, ast.Name) and value.id in direct_aliases
        ]
        self.assertFalse(
            direct_bindings,
            "internal-valid eligibility cannot directly authorize prospective Step20 evaluation",
        )


if __name__ == "__main__":
    unittest.main()
