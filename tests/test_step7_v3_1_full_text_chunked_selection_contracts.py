from __future__ import annotations

import copy
import json
import math
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

import step7_v3_1_source_data as source  # noqa: E402
import step7_v3_1_build_sync_manifest as sync_builder  # noqa: E402
import step7_v3_1_common as common  # noqa: E402
import step7_v3_1_encode_chunked_models as encoder  # noqa: E402
import step7_v3_1_materialize_gpu_workspace as materializer  # noqa: E402
import step7_v3_1_prepare_source_data as preparation  # noqa: E402
import step7_v3_1_select_source_model as selector  # noqa: E402
import step7_v3_1_selection_core as core  # noqa: E402


class FakeTokenizer:
    def __init__(self, width: int):
        self.width = width

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        if pair:
            raise AssertionError("pair tokenization is not used")
        return 2

    def _one(self, text: str, add_special_tokens: bool, offsets: bool):
        spans = [
            (start, min(start + self.width, len(text)))
            for start in range(0, len(text), self.width)
        ]
        ids = [
            1000 + sum(ord(character) for character in text[start:stop])
            for start, stop in spans
        ]
        if add_special_tokens:
            ids = [101, *ids, 102]
        result = {"input_ids": ids}
        if offsets:
            result["offset_mapping"] = spans
        return result

    def __call__(
        self,
        text,
        *,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        return_offsets_mapping=False,
    ):
        self.assert_flags(padding, truncation)
        if isinstance(text, list):
            records = [
                self._one(value, add_special_tokens, return_offsets_mapping)
                for value in text
            ]
            output = {"input_ids": [record["input_ids"] for record in records]}
            if return_offsets_mapping:
                output["offset_mapping"] = [
                    record["offset_mapping"] for record in records
                ]
            return output
        return self._one(text, add_special_tokens, return_offsets_mapping)

    @staticmethod
    def assert_flags(padding, truncation):
        if padding is not False or truncation is not False:
            raise AssertionError("tests require unpadded, untruncated tokenization")


class Step7V31Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = json.loads(common.DEFAULT_POLICY.read_text(encoding="utf-8"))

    def test_policy_and_standalone_source_contract_are_frozen_and_valid(self):
        common.validate_policy(self.policy)
        source_policy = common.source_policy(self.policy)
        self.assertEqual(source_policy["version"], "2026-07-22-step7-v3.1-source-data-v1")
        self.assertEqual(
            self.policy["source_data_policy"]["sha256"],
            source.sha256_file(source.DEFAULT_POLICY),
        )
        self.assertEqual(self.policy["outputs"]["root"], source_policy["outputs"]["root"])

    def test_standalone_public_and_development_artifacts_replay(self):
        public = common.validate_source_public_artifacts(self.policy)
        development = common.validate_source_development_artifacts(self.policy)
        self.assertEqual(
            set(public),
            {
                "preparation_manifest",
                "pair_manifest",
                "field_corpus",
                "train_feature_reference",
                "safe_pair_features",
            },
        )
        self.assertEqual(set(development), {"development_labels_manifest", "train_labels", "valid_labels"})

    def test_policy_requires_complete_shared_chunks_and_no_reranker(self):
        chunk = self.policy["shared_chunking"]
        self.assertIsNone(chunk["maximum_chunks_per_seller"])
        self.assertFalse(chunk["long_history_sampling_or_dropping_allowed"])
        self.assertTrue(chunk["require_exact_character_reconstruction_per_field"])
        self.assertEqual(chunk["token_budget_including_model_prefix_and_special_tokens"], 480)
        self.assertNotIn("shared_reranker", self.policy)
        self.assertNotIn("reranker", " ".join(self.policy["candidate_tiers"]))

    def test_candidate_matrix_is_ten_encoder_pipelines_plus_three_controls(self):
        specs = selector.candidate_specs(self.policy)
        self.assertEqual(len(specs), 13)
        self.assertEqual(sum(row["m0_pipeline_eligible"] for row in specs), 10)
        self.assertEqual(sum(row["encoder_comparison_eligible"] for row in specs), 5)
        self.assertEqual(sum(row["attribution_control_only"] for row in specs), 2)
        self.assertEqual(sum(row["shortcut_audit_only"] for row in specs), 1)
        shortcuts = set(self.policy["pair_feature_roles"]["shortcut_audit_only_features"])
        audit_names = {
            "chunk_count",
            "token_length",
            "field_missingness",
            "char_length",
        }
        for spec in specs:
            if spec["m0_pipeline_eligible"]:
                self.assertTrue(shortcuts.isdisjoint(spec["feature_names"]))
                self.assertTrue(audit_names.isdisjoint(spec["feature_names"]))

    def test_each_encoder_pipeline_has_same_six_aggregates(self):
        specs = selector.candidate_specs(self.policy)
        for model_key, cfg in self.policy["embedding_models"].items():
            expected = common.aggregate_feature_names(cfg)
            aggregate_only = next(
                row
                for row in specs
                if row["candidate_id"]
                == f"{model_key}__encoder_aggregates_only"
            )
            self.assertEqual(aggregate_only["feature_names"], expected)
            self.assertEqual(len(expected), 6)
            self.assertEqual(common.primary_feature_name(cfg), expected[1])

    def test_shared_chunker_preserves_every_character_under_all_tokenizers(self):
        policy = copy.deepcopy(self.policy)
        policy["shared_chunking"][
            "token_budget_including_model_prefix_and_special_tokens"
        ] = 10
        for cfg in policy["embedding_models"].values():
            cfg["text_prefix"] = ""
        tokenizers = {
            key: FakeTokenizer(1 if index == 0 else 2 + index)
            for index, key in enumerate(policy["embedding_models"])
        }
        fields = policy["clean_text_contract"]["fields_in_order"]
        field_texts = {field: "" for field in fields}
        field_texts["category_concat_top"] = "alpha beta gamma delta"
        field_texts["title_concat_top"] = "one two three four five"
        field_rows = [
            {
                "seller_uid": "seller-a",
                "split_name": "train",
                "field_texts": field_texts,
            }
        ]
        chunks, audit = encoder.build_shared_chunks(policy, field_rows, tokenizers)
        self.assertGreater(len(chunks), 2)
        self.assertTrue(audit["exact_character_reconstruction"])
        for field, text in field_texts.items():
            observed = "".join(
                row["text"] for row in chunks if row["field_name"] == field
            )
            self.assertEqual(observed, text)
        for row in chunks:
            self.assertLessEqual(max(row["token_lengths"].values()), 10)

    def test_chunk_validation_rejects_gap_and_over_budget(self):
        policy = copy.deepcopy(self.policy)
        fields = policy["clean_text_contract"]["fields_in_order"]
        texts = {field: "" for field in fields}
        texts[fields[0]] = "abc"
        field_rows = [{"seller_uid": "s", "split_name": "train", "field_texts": texts}]
        lengths = {key: 5 for key in policy["embedding_models"]}
        identity = {
            "seller_uid": "s",
            "field_name": fields[0],
            "chunk_index": 0,
            "char_start": 0,
            "char_end": 3,
            "text_sha256": common.sha256_text("abc"),
        }
        row = {
            "chunk_uid": common.canonical_hash(identity),
            "seller_uid": "s",
            "split_name": "train",
            "field_name": fields[0],
            "field_group": "category",
            "chunk_index": 0,
            "char_start": 0,
            "char_end": 3,
            "text": "abc",
            "text_sha256": identity["text_sha256"],
            "token_lengths": lengths,
        }
        common.validate_shared_chunk_rows(policy, field_rows, [row])
        broken = copy.deepcopy(row)
        broken["char_start"] = 1
        broken["chunk_uid"] = common.canonical_hash(
            {
                "seller_uid": broken["seller_uid"],
                "field_name": broken["field_name"],
                "chunk_index": broken["chunk_index"],
                "char_start": broken["char_start"],
                "char_end": broken["char_end"],
                "text_sha256": broken["text_sha256"],
            }
        )
        with self.assertRaisesRegex(ValueError, "overlapping or discontinuous"):
            common.validate_shared_chunk_rows(policy, field_rows, [broken])
        broken = copy.deepcopy(row)
        broken["token_lengths"][next(iter(lengths))] = 481
        with self.assertRaisesRegex(ValueError, "exceeds common token budget"):
            common.validate_shared_chunk_rows(policy, field_rows, [broken])

    def test_fixed_aggregates_match_hand_computation(self):
        fields = self.policy["clean_text_contract"]["fields_in_order"]
        chunk_rows = [
            {"seller_uid": "a", "field_name": fields[0], "field_group": "category"},
            {"seller_uid": "a", "field_name": fields[2], "field_group": "title"},
            {"seller_uid": "b", "field_name": fields[0], "field_group": "category"},
            {"seller_uid": "b", "field_name": fields[2], "field_group": "title"},
        ]
        matrix = np.asarray([[1, 0], [0, 1], [1, 0], [0, -1]], dtype=np.float32)
        summaries = common.seller_embedding_summaries(self.policy, matrix, chunk_rows)
        result = common.aggregate_pair(self.policy, summaries["a"], summaries["b"])
        self.assertAlmostEqual(result["all_chunk_mean_cosine"], 0.0, places=7)
        self.assertAlmostEqual(result["field_equal_mean_cosine"], 0.0, places=7)
        self.assertAlmostEqual(result["symmetric_top3_block_cosine"], 0.0, places=7)
        self.assertAlmostEqual(result["category_mean_cosine"], 1.0, places=7)
        self.assertAlmostEqual(result["title_mean_cosine"], -1.0, places=7)
        self.assertEqual(result["description_mean_cosine"], 0.0)

    def test_pair_score_schema_and_primary_feature_are_fixed(self):
        cfg = self.policy["embedding_models"]["gte_multilingual_base"]
        fields = self.policy["clean_text_contract"]["fields_in_order"]
        chunk_rows = [
            {"seller_uid": "a", "field_name": fields[0], "field_group": "category"},
            {"seller_uid": "b", "field_name": fields[0], "field_group": "category"},
        ]
        matrix = np.asarray([[1, 0], [1, 0]], dtype=np.float32)
        rows = common.compute_pair_score_rows(
            self.policy,
            cfg,
            matrix,
            chunk_rows,
            [{"pair_uid": "p", "seller_uid_left": "a", "seller_uid_right": "b"}],
        )
        self.assertEqual(list(rows[0]), ["pair_uid", *common.aggregate_feature_names(cfg)])
        self.assertEqual(float(rows[0][common.primary_feature_name(cfg)]), 1.0)

    def test_tokenizer_digest_is_order_sensitive_and_batched(self):
        tokenizer = FakeTokenizer(2)
        first = encoder.tokenizer_digest(tokenizer, ["abc", "def"], "query: ")
        second = encoder.tokenizer_digest(tokenizer, ["def", "abc"], "query: ")
        replay = encoder.tokenizer_digest(tokenizer, ["abc", "def"], "query: ")
        self.assertNotEqual(first, second)
        self.assertEqual(first, replay)

    def test_gpu_payload_allowlist_excludes_labels_raw_sources_and_safe_features(self):
        paths = set(sync_builder.gpu_payload_paths(self.policy, common.DEFAULT_POLICY))
        source_policy = common.source_policy(self.policy)
        forbidden = {spec["path"] for spec in source_policy["inputs"].values()}
        forbidden.update(
            {
                source_policy["outputs"]["train_labels"],
                source_policy["outputs"]["valid_labels"],
                source_policy["outputs"]["safe_pair_features"],
                source_policy["outputs"]["train_feature_reference"],
            }
        )
        self.assertTrue(paths.isdisjoint(forbidden))
        self.assertIn(self.policy["outputs"]["field_corpus"], paths)
        self.assertIn(self.policy["outputs"]["pair_manifest"], paths)

    def test_materializer_rejects_absolute_and_parent_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for value in ("../escape", "/absolute", "C:/absolute"):
                with self.assertRaises(ValueError):
                    materializer.safe_relative_path(root, value)

    def test_historical_test_is_not_an_executable_stage(self):
        source = (SCRIPTS / "step7_v3_1_select_source_model.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('choices=("select",)', source)
        self.assertNotIn("run_historical_test(", source)
        self.assertNotIn("historical_test_labels", self.policy["outputs"])

    def test_frozen_source_artifact_hashes_replay(self):
        source_policy = common.source_policy(self.policy)
        for role, record in source_policy["expected_artifacts"].items():
            path = ROOT / source_policy["outputs"][role]
            self.assertEqual(path.stat().st_size, record["size_bytes"])
            self.assertEqual(source.sha256_file(path), record["sha256"])

    def test_source_preparation_has_no_historical_test_stage(self):
        parser_source = (SCRIPTS / "step7_v3_1_prepare_source_data.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('choices=("public", "development-labels")', parser_source)
        self.assertNotIn("acknowledge-historical-test", parser_source)

    def test_field_only_profile_does_not_mutate_source(self):
        fields = self.policy["clean_text_contract"]["fields_in_order"]
        profile = {field: field for field in fields}
        isolated = preparation.field_only_profile(profile, fields[2], fields)
        self.assertEqual(profile, {field: field for field in fields})
        self.assertEqual(isolated[fields[2]], fields[2])
        self.assertTrue(all(not isolated[field] for field in fields if field != fields[2]))

    def test_public_artifact_schemas_exclude_labels_and_identity_columns(self):
        outputs = self.policy["outputs"]
        pair_rows = source.load_csv(common.resolve(outputs["pair_manifest"]))
        safe_rows = source.load_csv(common.resolve(outputs["safe_pair_features"]))
        field_rows = source.load_jsonl(common.resolve(outputs["field_corpus"]))
        self.assertEqual(len(pair_rows), 734)
        self.assertEqual(len(safe_rows), 734)
        self.assertEqual(len(field_rows), 855)
        self.assertEqual(
            list(pair_rows[0]),
            [
                "pair_uid",
                "split_name",
                "component_id",
                "seller_uid_left",
                "seller_uid_right",
            ],
        )
        self.assertEqual(list(safe_rows[0]), ["pair_uid", *source.SAFE_FEATURE_NAMES])
        self.assertEqual(
            list(field_rows[0]),
            [
                "seller_uid",
                "split_name",
                "field_texts",
                "field_text_sha256",
                "model_text",
                "model_text_sha256",
            ],
        )
        forbidden = {
            "review_label",
            "evidence_type",
            "source_seller_raw",
            "alias_normalized",
            "source_market_raw",
            "contact_concat_top",
        }
        self.assertTrue(forbidden.isdisjoint(pair_rows[0]))
        self.assertTrue(forbidden.isdisjoint(safe_rows[0]))
        self.assertTrue(forbidden.isdisjoint(field_rows[0]))

    def test_source_preparation_manifest_replays_label_isolation_and_field_counts(self):
        manifest = common.load_json(
            common.resolve(self.policy["outputs"]["preparation_manifest"])
        )
        self.assertFalse(manifest["feature_generation_uses_review_label_values"])
        self.assertFalse(manifest["feature_generation_uses_evidence_type_values"])
        self.assertTrue(manifest["complete_field_text_replay"])
        self.assertEqual(manifest["seller_count"], 855)
        self.assertEqual(
            manifest["nonempty_field_seller_counts"],
            {
                "category_concat_top": 855,
                "signature_title_concat": 854,
                "title_concat_top": 854,
                "signature_description_concat": 782,
                "description_concat_top": 853,
            },
        )
        self.assertEqual(manifest["identity_residue_scan"]["status"], "pass")
        self.assertEqual(manifest["identity_residue_scan"]["total_residue_count"], 0)
        self.assertFalse(
            manifest["identity_residue_scan"]["unknown_identifier_absence_proven"]
        )
        self.assertEqual(manifest["split_isolation"]["cross_split_seller_count"], 0)
        self.assertEqual(manifest["split_isolation"]["cross_split_component_count"], 0)

    def test_redaction_removes_identity_handles_but_preserves_ordinary_content(self):
        for value, forbidden in (
            ("TELEGRAM: @limestone420", "limestone420"),
            ("Wickr: drugstores1", "drugstores1"),
            ("contact us at nevadakay3gmailc", "nevadakay3gmailc"),
            ("call/text +1 669 228 0192", "669 228 0192"),
        ):
            cleaned, diagnostics = source.redact_identifiers(value, [])
            self.assertNotIn(forbidden, cleaned.casefold(), value)
            self.assertGreater(diagnostics["generic_identifier_match_count"], 0)
        for value in (
            "premium private support",
            "DMT is also called Dimitri or the Spirit Molecule",
            "Delivery format: John Wick 72 Garden Close",
            "price 0.1905617980645162 BTC",
        ):
            cleaned, _ = source.redact_identifiers(value, [])
            self.assertEqual(cleaned, value)

    def test_metrics_include_auc_ap_f1_recall_and_mrr(self):
        rows = [
            {"seller_uid_left": "q", "seller_uid_right": f"c{index}"}
            for index in range(4)
        ]
        labels = np.asarray([1, 0, 1, 0], dtype=np.int8)
        scores = np.asarray([0.9, 0.8, 0.7, 0.1], dtype=np.float64)
        metrics = core.full_metrics(rows, labels, scores, threshold=0.75)
        self.assertAlmostEqual(metrics["roc_auc"], 0.75)
        self.assertAlmostEqual(metrics["average_precision"], 5.0 / 6.0)
        for name in ("precision", "recall", "f1", "balanced_accuracy"):
            self.assertIn(name, metrics)
        ranking = metrics["labelled_candidate_ranking"]
        self.assertEqual(ranking["status"], "diagnostic_labelled_pair_set_only")
        self.assertIn("mrr", ranking)
        self.assertIn("recall_at_3", ranking)

    def test_shortcut_conditioned_ap_ignores_between_stratum_offsets(self):
        strata = (
            "same_market=0|same_source=1",
            "same_market=0|same_source=1",
            "same_market=1|same_source=1",
            "same_market=1|same_source=1",
        )
        rows = [
            {
                "pair_uid": f"p{index}",
                "split_name": "valid",
                "component_id": f"c{index}",
                "review_label": "positive" if index % 2 == 0 else "negative",
                core.SHORTCUT_CONTROL_STRATUM_FIELD: stratum,
            }
            for index, stratum in enumerate(strata)
        ]
        labels = np.asarray([1, 0, 1, 0], dtype=np.int8)
        original = np.asarray([0.90, 0.80, 0.40, 0.30], dtype=np.float64)
        shifted = np.asarray([0.90, 0.80, 0.85, 0.75], dtype=np.float64)
        first = core.shortcut_conditioned_component_equal_average_precision(
            rows, labels, original, self.policy, require_expected_strata=True
        )
        second = core.shortcut_conditioned_component_equal_average_precision(
            rows, labels, shifted, self.policy, require_expected_strata=True
        )
        self.assertEqual(first["macro_average_precision"], 1.0)
        self.assertEqual(second["macro_average_precision"], 1.0)
        self.assertNotEqual(
            core.average_precision(labels, original),
            core.average_precision(labels, shifted),
        )

    def test_grouped_folds_and_logistic_oof_are_finite(self):
        policy = copy.deepcopy(self.policy)
        policy["training"]["l2_grid"] = [0.1, 1.0]
        rows, values = [], []
        for component_index in range(20):
            for label_index, label in enumerate(("negative", "positive")):
                rows.append(
                    {
                        "pair_uid": f"p{component_index}_{label_index}",
                        "split_name": "train",
                        "component_id": f"c{component_index}",
                        "review_label": label,
                        "seller_uid_left": f"l{component_index}_{label_index}",
                        "seller_uid_right": f"r{component_index}_{label_index}",
                        core.SHORTCUT_CONTROL_STRATUM_FIELD: (
                            "same_market=0|same_source=1"
                            if component_index % 2 == 0
                            else "same_market=1|same_source=1"
                        ),
                    }
                )
                values.append([float(label_index), float(component_index % 4)])
        folds = core.balanced_component_folds(rows, 5, 20260721)
        for fold in range(5):
            labels = {
                row["review_label"]
                for row in rows
                if folds[row["component_id"]] == fold
            }
            self.assertEqual(labels, {"positive", "negative"})
        result = core.tune_and_fit(
            rows,
            np.asarray(values, dtype=np.float64),
            ["signal", "nuisance"],
            policy,
            "component_equal_normalized_to_row_count",
        )
        artifact = result["final_train_artifact"]
        self.assertTrue(artifact["solver_converged"])
        self.assertLessEqual(
            artifact["solver_final_normalized_gradient_inf_norm"],
            policy["training"]["tolerance"],
        )
        self.assertTrue(math.isfinite(result["selected_threshold"]))
        self.assertEqual(len(result["train_oof_scores"]), len(rows))
        self.assertGreater(result["train_oof_metrics"]["average_precision"], 0.95)

    def test_development_label_projection_never_reads_test_values(self):
        source_policy = common.source_policy(self.policy)
        pair_rows = source.load_csv(common.resolve(source_policy["outputs"]["pair_manifest"]))
        labels_path = common.resolve(source_policy["inputs"]["frozen_labels"]["path"])
        evidence_path = common.resolve(source_policy["inputs"]["evidence_labels"]["path"])
        original_load_csv = source.load_csv

        class GuardedTestRow(dict):
            guarded = {
                "review_label",
                "usable_for_supervision",
                "evidence_type",
                "shared_contact_count",
                "shared_pgp_fingerprint_count",
            }

            def get(self, key, default=None):
                if key in self.guarded:
                    raise AssertionError(f"test value was accessed: {key}")
                return super().get(key, default)

            def __getitem__(self, key):
                if key in self.guarded:
                    raise AssertionError(f"test value was accessed: {key}")
                return super().__getitem__(key)

        def guarded_load(path):
            rows = original_load_csv(path)
            if Path(path).resolve() in {labels_path.resolve(), evidence_path.resolve()}:
                return [
                    row
                    if row.get("split_name") in {"train", "valid"}
                    else GuardedTestRow(row)
                    for row in rows
                ]
            return rows

        with mock.patch.object(source, "load_csv", side_effect=guarded_load):
            prepared = preparation.prepare_private_labels(
                source_policy, pair_rows, ("train", "valid")
            )
        self.assertEqual(prepared["label_counts"]["train"]["total"], 401)
        self.assertEqual(prepared["label_counts"]["valid"]["total"], 152)

    def test_linux_runner_enforces_isolated_workspace(self):
        runner = (
            SCRIPTS / "run_step7_v3_1_full_text_chunked_linux_20260722.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("step7_v3_1_materialize_gpu_workspace.py", runner)
        self.assertIn("STEP7_V3_1_ISOLATED_WORKSPACE=1", runner)
        self.assertIn("step7_v3_1_build_sync_manifest.py", runner)
        self.assertIn("--validate-only", runner)
        self.assertIn("collect --workspace", runner)


if __name__ == "__main__":
    unittest.main()
