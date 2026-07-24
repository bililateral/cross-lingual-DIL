from __future__ import annotations

import copy
import inspect
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections import UserDict
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step7_v4_common as common  # noqa: E402
import step7_v4_encode_item_models as encoder  # noqa: E402
import step7_v3_1_source_data as source  # noqa: E402
import step7_v3_1_selection_core as parent_solver  # noqa: E402
import step7_v4_selection_core as v4_solver  # noqa: E402
import step7_v4_select_source_model as selector  # noqa: E402
import step7_v4_resume_selection_after_solver_fix as selection_resume  # noqa: E402
import step7_v4_prepare_source_data as preparation  # noqa: E402
import step7_v4_build_sync_manifest as sync_builder  # noqa: E402


class FakeTokenizer:
    def __init__(self, width: int):
        self.width = width

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        if pair:
            raise AssertionError("pair tokenization is not used")
        return 2

    def _one(self, text: str, add_special_tokens: bool, offsets: bool) -> dict:
        spans = [
            (start, min(start + self.width, len(text)))
            for start in range(0, len(text), self.width)
        ]
        ids = [1000 + sum(ord(character) for character in text[start:stop]) for start, stop in spans]
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
        if padding is not False or truncation is not False:
            raise AssertionError("fake tokenizer forbids padding/truncation")
        if isinstance(text, list):
            rows = [
                self._one(value, add_special_tokens, return_offsets_mapping)
                for value in text
            ]
            result = {"input_ids": [row["input_ids"] for row in rows]}
            if return_offsets_mapping:
                result["offset_mapping"] = [row["offset_mapping"] for row in rows]
            return result
        return self._one(text, add_special_tokens, return_offsets_mapping)


class FakeFeatureFactory:
    @staticmethod
    def design(fit_rows, hold_rows, names):
        def matrix(rows):
            return np.asarray(
                [[float(row[name]) for name in names] for row in rows],
                dtype=np.float64,
            ).reshape(len(rows), len(names))

        return matrix(fit_rows), matrix(hold_rows), [0.0] * len(names), {
            "fake_label_free_reference": True
        }


class FakeSentenceTransformer:
    def __init__(self, path, device, local_files_only):
        self.path = path
        self.device = device
        self.local_files_only = local_files_only
        self.prompts = {"document": "document: ", "query": "query: "}
        self.default_prompt_name = "document"
        self.max_seq_length = 512
        self.eval_called = False

    def eval(self):
        self.eval_called = True

    def encode(self, sentences, *, prompt=None, **kwargs):
        raise AssertionError("model loading contract test must not encode")


class FakeSentenceTransformer256(FakeSentenceTransformer):
    def __init__(self, path, device, local_files_only):
        super().__init__(path, device, local_files_only)
        self.max_seq_length = 256


class FakeSentenceTransformerTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def tokenize(self, texts):
        rows = self.tokenizer(
            texts,
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )["input_ids"]
        width = max(len(row) for row in rows)
        return UserDict({
            "input_ids": np.asarray(
                [row + [0] * (width - len(row)) for row in rows],
                dtype=np.int64,
            ),
            "attention_mask": np.asarray(
                [[1] * len(row) + [0] * (width - len(row)) for row in rows],
                dtype=np.int64,
            ),
        })


class FakeSentenceTransformerWithoutPrompt:
    def __init__(self, path, device, local_files_only):
        raise AssertionError("incompatible class must fail before construction")

    def encode(self, sentences):
        raise AssertionError("incompatible class must not encode")


class FakeSmokeSentenceTransformer(FakeSentenceTransformer):
    def __init__(self, path, device, local_files_only):
        super().__init__(path, device, local_files_only)
        self._tokenizer = FakeTokenizer(2)

    def tokenize(self, texts):
        return FakeSentenceTransformerTokenizer(self._tokenizer).tokenize(texts)

    def parameters(self):
        class Parameter:
            dtype = "torch.float32"

        return iter([Parameter()])

    def encode(
        self,
        sentences,
        *,
        prompt=None,
        convert_to_numpy=True,
        normalize_embeddings=True,
        **kwargs,
    ):
        if prompt != "" or not convert_to_numpy or not normalize_embeddings:
            raise AssertionError("smoke encode contract drift")
        matrix = np.zeros((len(sentences), 1024), dtype=np.float32)
        matrix[:, 0] = 1.0
        return matrix


class FakeTorchForSmoke:
    class cuda:
        @staticmethod
        def empty_cache():
            return None


def brute_symmetric_top_k(left: np.ndarray, right: np.ndarray, k: int) -> float:
    left_unit = left / np.linalg.norm(left, axis=1, keepdims=True)
    right_unit = right / np.linalg.norm(right, axis=1, keepdims=True)
    similarities = np.clip(left_unit @ right_unit.T, -1.0, 1.0)

    def row_score(row: np.ndarray) -> float:
        count = min(k, len(row))
        return float(np.mean(np.sort(row)[-count:]))

    return float(
        (
            np.mean([row_score(row) for row in similarities])
            + np.mean([row_score(row) for row in similarities.T])
        )
        / 2.0
    )


class Step7V4Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = json.loads(
            common.DEFAULT_POLICY.read_text(encoding="utf-8")
        )

    def test_unfrozen_policy_is_internally_valid_and_keeps_labels_private(self):
        common.validate_policy(self.policy, require_frozen=False)
        self.assertEqual(
            self.policy["shared_chunking"][
                "token_budget_including_model_prefix_and_special_tokens"
            ],
            256,
        )
        self.assertEqual(
            {
                key: value["native_max_seq_length"]
                for key, value in self.policy["embedding_models"].items()
            },
            common.MODEL_NATIVE_MAX_SEQ_LENGTHS,
        )
        self.assertEqual(
            common.PRIVATE_OUTPUT_ROLES,
            ["train_labels", "valid_labels", "train_evidence", "valid_evidence"],
        )
        self.assertEqual(
            self.policy["gpu_execution"][
                "required_sentence_transformers_version"
            ],
            common.REQUIRED_SENTENCE_TRANSFORMERS_VERSION,
        )
        self.assertTrue(
            set(common.PUBLIC_OUTPUT_ROLES).isdisjoint(common.PRIVATE_OUTPUT_ROLES)
        )
        self.assertFalse(
            self.policy["supervision_boundary"][
                "historical_test_labels_may_be_materialized"
            ]
        )
        self.assertEqual(
            self.policy["clean_text_contract"][
                "multiline_identity_redaction_rule_allowlist"
            ],
            sorted(common.MULTILINE_IDENTITY_REDACTION_RULE_NAMES),
        )

        self.assertEqual(
            self.policy["clean_text_contract"][
                "non_pgp_identity_matching_scope"
            ],
            "single_logical_line_except_two_fixed_split_plus_phone_rules_no_other_non_pgp_lf_or_cr_crossing",
        )
        self.assertEqual(
            self.policy["clean_text_contract"]["identity_match_evaluation"],
            "single_non_cascading_sweep_on_normalized_original_text_then_merge_spans_and_redact",
        )
        self.assertEqual(
            self.policy["clean_text_contract"]["identity_bridge_evaluation"],
            "only_mask_preexisting_seller_local_identity_spans_to_detect_audited_global_aliases_no_other_cascade",
        )
        self.assertEqual(
            self.policy["clean_text_contract"][
                "v4_additional_identity_rule_names"
            ],
            [
                "repeated_email_domain_chain",
                "contact_service_at_mixed_handle",
                "contact_cued_split_plus_phone",
                "split_plus_country_area_phone",
                "market_identity_list_before_vendor_cue",
                "google_voice_attached_phone",
                "concatenated_whatsapp_phone_wickr_mixed_handle",
            ],
        )
        fixed_collisions = common.validated_fixed_final_audit_content_collisions(
            self.policy["clean_text_contract"][
                "fixed_final_audit_content_collision_handling"
            ]
        )
        self.assertEqual(len(fixed_collisions), 1)
        self.assertEqual(
            sum(row["expected_match_count"] for row in fixed_collisions), 1
        )
        self.assertTrue(
            common.FIXED_FINAL_AUDIT_CONTENT_COLLISION_ALLOWED_RULE_NAMES
            <= {
                rule_name
                for rule_name, _pattern in source.FINAL_CORPUS_AUDIT_RULES
            }
        )
        specs = {item["id"]: item for item in common.candidate_specs(self.policy)}
        self.assertEqual(len(specs), 22)
        self.assertEqual(len(specs["control__legacy18"]["feature_names"]), 18)
        self.assertEqual(len(specs["control__stylometry"]["feature_names"]), 22)
        self.assertEqual(len(specs["style__pcm_mstyle_stylometry"]["feature_names"]), 34)
        for raw_id, matched_id, block in (
            ("encoder__e5", "matched__e5_stylometry", "e5_6"),
            ("encoder__labse", "matched__labse_stylometry", "labse6"),
            ("style__pcm", "style__pcm_stylometry", "pcm6"),
            ("style__mstyle", "style__mstyle_stylometry", "mstyle6"),
        ):
            self.assertEqual(specs[raw_id]["blocks"], [block])
            self.assertEqual(specs[matched_id]["blocks"], [block, "stylometry22"])
        self.assertEqual(
            specs["control__legacy_stylometry"]["blocks"],
            ["legacy18", "stylometry22"],
        )
        for candidate_id, block in (
            ("fusion__legacy_e5_stylometry", "e5_6"),
            ("fusion__legacy_labse_stylometry", "labse6"),
            ("fusion__legacy_pcm_stylometry", "pcm6"),
            ("fusion__legacy_mstyle_stylometry", "mstyle6"),
        ):
            self.assertEqual(
                specs[candidate_id]["blocks"],
                ["legacy18", block, "stylometry22"],
            )
        self.assertEqual(
            specs["fusion__legacy_style_e5"]["blocks"],
            ["legacy18", "pcm6", "mstyle6", "stylometry22", "e5_6"],
        )
        self.assertEqual(
            specs["fusion__legacy_style_labse"]["blocks"],
            ["legacy18", "pcm6", "mstyle6", "stylometry22", "labse6"],
        )
        self.assertTrue(
            self.policy["outputs"]["blind_valid_predictions"].endswith(
                ".blind.no_labels.csv"
            )
        )
        self.assertNotEqual(
            self.policy["outputs"]["blind_valid_predictions"],
            self.policy["outputs"]["valid_predictions"],
        )

    def test_policy_rejects_shared_budget_or_native_window_expansion(self):
        expanded_budget = copy.deepcopy(self.policy)
        expanded_budget["shared_chunking"][
            "token_budget_including_model_prefix_and_special_tokens"
        ] = 257
        with self.assertRaisesRegex(ValueError, "chunk token budget drift"):
            common.validate_policy(expanded_budget, require_frozen=False)

        expanded_labse = copy.deepcopy(self.policy)
        expanded_labse["embedding_models"]["labse"][
            "native_max_seq_length"
        ] = 512
        with self.assertRaisesRegex(ValueError, "model dimensional contract drift"):
            common.validate_policy(expanded_labse, require_frozen=False)

        drifted_library = copy.deepcopy(self.policy)
        drifted_library["gpu_execution"][
            "required_sentence_transformers_version"
        ] = "5.6.1"
        with self.assertRaisesRegex(ValueError, "deterministic GPU contract"):
            common.validate_policy(drifted_library, require_frozen=False)
        self.assertEqual(
            encoder.validate_sentence_transformers_version(
                self.policy, common.REQUIRED_SENTENCE_TRANSFORMERS_VERSION
            ),
            common.REQUIRED_SENTENCE_TRANSFORMERS_VERSION,
        )
        with self.assertRaisesRegex(RuntimeError, "version drift"):
            encoder.validate_sentence_transformers_version(
                self.policy, "5.6.1"
            )

    def test_blind_validation_predictions_exclude_all_supervision_columns(self):
        valid_rows = [
            {
                "pair_uid": "pair-a",
                "component_id": "component-secret-a",
                "review_label": "positive",
                "evidence_type": "secret-evidence-a",
            },
            {
                "pair_uid": "pair-b",
                "component_id": "component-secret-b",
                "review_label": "negative",
                "evidence_type": "secret-evidence-b",
            },
        ]
        rows = selector.blind_valid_prediction_rows(
            valid_rows,
            ["candidate-z", "candidate-a"],
            {
                "candidate-z": np.asarray([0.8, 0.2]),
                "candidate-a": np.asarray([0.6, 0.4]),
            },
        )
        self.assertEqual(len(rows), 4)
        self.assertTrue(
            all(
                list(row) == ["pair_uid", "candidate_id", "probability"]
                for row in rows
            )
        )
        serialized = json.dumps(rows, sort_keys=True)
        for forbidden in (
            "component-secret",
            "review_label",
            "evidence_type",
            "secret-evidence",
        ):
            self.assertNotIn(forbidden, serialized)
        replayed = selector.replay_blind_valid_scores(
            valid_rows, ["candidate-z", "candidate-a"], rows
        )
        np.testing.assert_array_equal(
            replayed["candidate-z"], np.asarray([0.8, 0.2])
        )
        np.testing.assert_array_equal(
            replayed["candidate-a"], np.asarray([0.6, 0.4])
        )
        difficult = float(np.nextafter(0.1, 1.0))
        self.assertEqual(float(selector.serialize_probability(difficult)), difficult)
        noncanonical = copy.deepcopy(rows)
        noncanonical[0]["probability"] = "0.80000000000000004"
        with self.assertRaisesRegex(ValueError, "canonical exact-round-trip"):
            selector.replay_blind_valid_scores(
                valid_rows, ["candidate-z", "candidate-a"], noncanonical
            )
        reordered = [rows[1], rows[0], *rows[2:]]
        with self.assertRaisesRegex(ValueError, "order/key drift"):
            selector.replay_blind_valid_scores(
                valid_rows, ["candidate-z", "candidate-a"], reordered
            )

    def test_physical_blind_locks_precede_any_diagnostic_label_open(self):
        body = inspect.getsource(selector.run_selection)
        self.assertNotIn("validate_private_label_artifacts", body)
        selection_write = body.index(
            "common.write_json_immutable(selection_lock_path"
        )
        artifact_write = body.index(
            "common.write_json_immutable(artifact_path"
        )
        blind_prediction_write = body.index(
            "common.write_csv_immutable(blind_prediction_path"
        )
        blind_lock_write = body.index(
            "common.write_json_immutable(blind_lock_path"
        )
        locked_replay = body.index("locked_valid_scores = replay_blind_valid_scores")
        original_scores_deleted = body.index("del original")
        in_memory_scores_deleted = body.index("del unlabelled_valid_scores")
        train_evidence_open = body.index("train_rows = load_evidence_split")
        valid_label_open = body.index(
            'valid_label_rows = load_label_split(policy, pair_rows, "valid")'
        )
        self.assertLess(selection_write, artifact_write)
        self.assertLess(artifact_write, blind_lock_write)
        self.assertLess(blind_prediction_write, blind_lock_write)
        self.assertLess(blind_prediction_write, locked_replay)
        self.assertLess(locked_replay, blind_lock_write)
        self.assertLess(blind_lock_write, original_scores_deleted)
        self.assertLess(original_scores_deleted, in_memory_scores_deleted)
        self.assertLess(blind_lock_write, in_memory_scores_deleted)
        self.assertLess(in_memory_scores_deleted, train_evidence_open)
        self.assertLess(blind_lock_write, train_evidence_open)
        self.assertLess(blind_lock_write, valid_label_open)
        post_delete = body[
            in_memory_scores_deleted + len("del unlabelled_valid_scores") :
        ]
        self.assertNotIn("unlabelled_valid_scores", post_delete)
        self.assertIn("locked_valid_scores", post_delete)

    def test_mojibake_repair_decodes_complete_utf8_sequence_only(self):
        garbled = "before " + "".join(chr(value) for value in "❂".encode("utf-8")) + " after"
        clean, diagnostics = common._normalize_raw_field_with_diagnostics(garbled)
        self.assertEqual(clean, "before ❂ after")
        self.assertEqual(diagnostics["utf8_mojibake_sequence_repair_count"], 1)
        self.assertEqual(diagnostics["windows_1252_c1_repair_count"], 0)
        self.assertEqual(common.normalize_raw_field("café"), "café")

    def test_undefined_c1_control_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "undefined Windows-1252"):
            common.normalize_raw_field("left\u0081right")
        self.assertEqual(common.normalize_raw_field("left\u0093right"), "left“right")

    def test_contact_redaction_cannot_cross_line_into_product_content(self):
        cases = (
            (
                "Call/Text Whats-app.....+1 978-406-9577 "
                "Quality ATM PREPAID cards",
                "Quality ATM PREPAID cards",
            ),
            (
                "Call/Text Whats-app.....+1 978-406-9577\n"
                "Quality ATM PREPAID cards",
                "Quality ATM PREPAID cards",
            ),
            (
                "TELEGRAM----- +1 864 939 8764\n----------------\n"
                "Master Kush is a strain",
                "Master Kush is a strain",
            ),
            (
                "WICKR ME......Master444\nWICKR ME......Master444\n"
                "Master Kush is a strain",
                "Master Kush is a strain",
            ),
        )
        for raw, expected_content in cases:
            with self.subTest(expected_content=expected_content):
                clean, diagnostics = common.redact_raw_field(
                    raw,
                    seller_uid="seller-one",
                    seller_literals=[],
                    seller_phrase_tokens=set(),
                    global_tokens=set(),
                    contextual_aliases=set(),
                    contextual_alias_deletions=set(),
                    seller_contextual_collision_tokens=set(),
                    audited_global_phrases=set(),
                )
                self.assertIn(expected_content, clean)
                self.assertGreater(
                    diagnostics["generic_identifier_match_count"], 0
                )

    def test_multiline_redaction_is_restricted_to_fixed_allowlist(self):
        raw = (
            "ordinary prefix\n"
            "-----BEGIN PGP PUBLIC KEY BLOCK-----\n"
            "Version: test\nYWJjZA==\n"
            "-----END PGP PUBLIC KEY BLOCK-----\n"
            "ordinary suffix"
        )
        clean, diagnostics = common.redact_raw_field(
            raw,
            seller_uid="seller-one",
            seller_literals=[],
            seller_phrase_tokens=set(),
            global_tokens=set(),
            contextual_aliases=set(),
            contextual_alias_deletions=set(),
            seller_contextual_collision_tokens=set(),
            audited_global_phrases=set(),
        )
        self.assertEqual(clean, "ordinary prefix\nordinary suffix")
        self.assertGreater(diagnostics["generic_identifier_match_count"], 0)

    def test_contact_cued_split_plus_phone_is_removed_without_overreach(self):
        cases = (
            (
                "Ordinary product details\nyou can text at +\n\n"
                "1214) 702-9822\nKeep this product sentence",
                "Ordinary product details\nyou can\nKeep this product sentence",
                "1214",
            ),
            (
                "Ordinary product details\nContact on how it works via +1(315)\n"
                "696-1570\nKeep this product sentence",
                "Ordinary product details\nContact on how it works via\n"
                "Keep this product sentence",
                "696",
            ),
            (
                "Ordinary product details\nWelcome. +1(315)\n"
                "696-1570 or\nKeep this product sentence",
                "Ordinary product details\nWelcome. or\nKeep this product sentence",
                "696",
            ),
        )
        for raw, expected, forbidden in cases:
            with self.subTest(raw=raw):
                clean, diagnostics = common.redact_raw_field(
                    raw,
                    seller_uid="seller-one",
                    seller_literals=[],
                    seller_phrase_tokens=set(),
                    global_tokens=set(),
                    contextual_aliases=set(),
                    contextual_alias_deletions=set(),
                    seller_contextual_collision_tokens=set(),
                    audited_global_phrases=set(),
                )
                self.assertEqual(clean, expected)
                self.assertNotIn(forbidden, clean)
                self.assertGreater(
                    diagnostics["generic_identifier_match_count"], 0
                )

        ordinary, ordinary_diagnostics = common.redact_raw_field(
            "The text says add +\n\n1214 products to the catalogue",
            seller_uid="seller-one",
            seller_literals=[],
            seller_phrase_tokens=set(),
            global_tokens=set(),
            contextual_aliases=set(),
            contextual_alias_deletions=set(),
            seller_contextual_collision_tokens=set(),
            audited_global_phrases=set(),
        )
        self.assertEqual(
            ordinary,
            "The text says add +\n1214 products to the catalogue",
        )
        self.assertEqual(
            ordinary_diagnostics["generic_identifier_match_count"], 0
        )

    def test_repeated_email_domain_chain_is_removed_without_cascade(self):
        raw = (
            "Roxicodone 30mg. 30 pills $300 "
            "maryjohanna@gmail.com@gmail.com"
        )
        clean, diagnostics = common.redact_raw_field(
            raw,
            seller_uid="seller-one",
            seller_literals=[],
            seller_phrase_tokens=set(),
            global_tokens=set(),
            contextual_aliases=set(),
            contextual_alias_deletions=set(),
            seller_contextual_collision_tokens=set(),
            audited_global_phrases=set(),
        )
        self.assertEqual(clean, "Roxicodone 30mg. 30 pills $300")
        self.assertNotIn("@", clean)
        self.assertGreater(diagnostics["generic_identifier_match_count"], 0)

    def test_service_at_mixed_handle_is_removed_from_original_text(self):
        for service in ("Wickr", "vvickr", "wckr", "wikr", "wicker"):
            with self.subTest(service=service):
                raw = (
                    "Buy Phentermine 37.5 mg 2X100 Caps Bottle\n"
                    f"contact us on {service} at express74 after placing an order"
                )
                clean, diagnostics = common.redact_raw_field(
                    raw,
                    seller_uid="seller-one",
                    seller_literals=[],
                    seller_phrase_tokens=set(),
                    global_tokens=set(),
                    contextual_aliases=set(),
                    contextual_alias_deletions=set(),
                    seller_contextual_collision_tokens=set(),
                    audited_global_phrases=set(),
                )
                self.assertIn(
                    "Buy Phentermine 37.5 mg 2X100 Caps Bottle", clean
                )
                self.assertNotIn("express74", clean.casefold())
                self.assertNotIn(service.casefold(), clean.casefold())
                self.assertGreater(
                    diagnostics["generic_identifier_match_count"], 0
                )

    def test_market_identity_list_is_removed_on_original_text_only(self):
        raw = (
            "Dream market,agora market,wallstreet,grey,samsara,"
            "alphabay market verified vendor.\nOrdinary product description"
        )
        clean, diagnostics = common.redact_raw_field(
            raw,
            seller_uid="seller-one",
            seller_literals=[],
            seller_phrase_tokens=set(),
            global_tokens=set(),
            contextual_aliases=set(),
            contextual_alias_deletions=set(),
            seller_contextual_collision_tokens=set(),
            audited_global_phrases={
                "dreammarket",
                "agoramarket",
                "wallstreet",
                "samsara",
                "alphabaymarket",
            },
        )
        self.assertEqual(clean, "Ordinary product description")
        self.assertGreater(diagnostics["generic_identifier_match_count"], 0)

        ordinary, _diagnostics = common.redact_raw_field(
            "Grey pigment for an ordinary verified vendor product",
            seller_uid="seller-one",
            seller_literals=[],
            seller_phrase_tokens=set(),
            global_tokens=set(),
            contextual_aliases=set(),
            contextual_alias_deletions=set(),
            seller_contextual_collision_tokens=set(),
            audited_global_phrases=set(),
        )
        self.assertEqual(
            ordinary, "Grey pigment for an ordinary verified vendor product"
        )

    def test_compound_contact_sequences_are_removed_on_original_text(self):
        cases = (
            (
                "Product details. contact us here. w.i.c.k.'r......love2024// "
                "whatsapp// +1(662) 709 7015//....google voice+1(662) 709 7015",
                ("google voice", "+1(662) 709 7015"),
            ),
            (
                "Product details. contact me on\u00a0"
                "WhatsApp:+17203347285Wickr: doli2\u00a0",
                ("whatsapp", "wickr", "doli2", "17203347285"),
            ),
        )
        for raw, forbidden in cases:
            with self.subTest(raw=raw):
                clean, diagnostics = common.redact_raw_field(
                    raw,
                    seller_uid="seller-one",
                    seller_literals=[],
                    seller_phrase_tokens=set(),
                    global_tokens=set(),
                    contextual_aliases=set(),
                    contextual_alias_deletions=set(),
                    seller_contextual_collision_tokens=set(),
                    audited_global_phrases=set(),
                )
                self.assertIn("Product details", clean)
                for value in forbidden:
                    self.assertNotIn(value, clean.casefold())
                self.assertGreater(
                    diagnostics["generic_identifier_match_count"], 0
                )

    def test_only_seller_identity_may_bridge_an_audited_alias(self):
        bridged, diagnostics = common.redact_raw_field(
            "WELCOME Hi Tech Programmers Hackers Services",
            seller_uid="seller-one",
            seller_literals=["programmers"],
            seller_phrase_tokens={"programmers"},
            global_tokens=set(),
            contextual_aliases=set(),
            contextual_alias_deletions=set(),
            seller_contextual_collision_tokens=set(),
            audited_global_phrases={"techhackers"},
        )
        self.assertEqual(bridged, "WELCOME Hi Services")
        self.assertGreaterEqual(
            diagnostics["audited_global_identity_phrase_match_count"], 1
        )
        ordinary, _diagnostics = common.redact_raw_field(
            "Tech Programmers Guide",
            seller_uid="seller-one",
            seller_literals=["programmers"],
            seller_phrase_tokens={"programmers"},
            global_tokens=set(),
            contextual_aliases=set(),
            contextual_alias_deletions=set(),
            seller_contextual_collision_tokens=set(),
            audited_global_phrases={"techhackers"},
        )
        self.assertEqual(ordinary, "Tech Guide")

    def test_post_redaction_fuzzy_alias_collision_is_censused_not_deleted(self):
        filtered_deletions = common.v4_contextual_alias_deletion_tokens(
            {"withouth"}
        )
        self.assertNotIn("without", filtered_deletions)
        raw = (
            "With a simple process, become a seller on Empire Market "
            "without having to"
        )
        clean, diagnostics = common.redact_raw_field(
            raw,
            seller_uid="seller-one",
            seller_literals=[],
            seller_phrase_tokens=set(),
            global_tokens=set(),
            contextual_aliases={"withouth"},
            contextual_alias_deletions=filtered_deletions,
            seller_contextual_collision_tokens=set(),
            audited_global_phrases={"empiremarket"},
        )
        self.assertEqual(
            clean,
            "With a simple process, become a seller without having to",
        )
        self.assertEqual(diagnostics["contextual_alias_match_count"], 0)
        self.assertEqual(
            diagnostics[
                "post_redaction_one_character_omission_collision_count"
            ],
            0,
        )

        emergent, emergent_diagnostics = common.redact_raw_field(
            "Become a seller on Empire Market ordinary products",
            seller_uid="seller-one",
            seller_literals=[],
            seller_phrase_tokens=set(),
            global_tokens=set(),
            contextual_aliases={"ordinarry"},
            contextual_alias_deletions={"ordinary"},
            seller_contextual_collision_tokens=set(),
            audited_global_phrases={"empiremarket"},
        )
        self.assertEqual(emergent, "Become a seller ordinary products")
        self.assertEqual(
            emergent_diagnostics[
                "post_redaction_one_character_omission_collision_count"
            ],
            1,
        )

        original_match, original_diagnostics = common.redact_raw_field(
            "With a simple process, become a seller withouth having to",
            seller_uid="seller-one",
            seller_literals=[],
            seller_phrase_tokens=set(),
            global_tokens=set(),
            contextual_aliases={"withouth"},
            contextual_alias_deletions=filtered_deletions,
            seller_contextual_collision_tokens=set(),
            audited_global_phrases=set(),
        )
        self.assertNotIn("withouth", original_match.casefold())
        self.assertEqual(original_diagnostics["contextual_alias_match_count"], 1)
        self.assertEqual(
            original_diagnostics[
                "post_redaction_one_character_omission_collision_count"
            ],
            0,
        )

    def test_collision_prone_local_alias_is_context_gated_not_unconditional(self):
        parent_policy = {
            "clean_text_contract": {
                "quality_gates": {
                    "protected_content_words": ["master"],
                    "protected_identity_collision_terms": [],
                }
            }
        }
        policy = copy.deepcopy(self.policy)
        cfg = policy["clean_text_contract"][
            "seller_local_content_collision_handling"
        ]
        mapping = {"seller-one": ["master"]}
        cfg.update(
            {
                "collision_compact_count": 1,
                "collision_compact_registry_canonical_sha256": common.canonical_hash(
                    ["master"]
                ),
                "expected_affected_seller_count": 1,
                "expected_affected_seller_uid_to_tokens_canonical_sha256": common.canonical_hash(
                    mapping
                ),
            }
        )
        literals, phrases, contextual, audit = (
            preparation.split_seller_local_content_collisions(
                policy,
                parent_policy,
                {"seller-one"},
                {"seller-one": ["Master", "master"]},
                {"seller-one": {"master"}},
            )
        )
        self.assertEqual(literals["seller-one"], [])
        self.assertEqual(phrases["seller-one"], set())
        self.assertEqual(contextual["seller-one"], {"master"})
        self.assertFalse(audit["one_character_omission_matching_allowed"])

        clean, _diagnostics = common.redact_raw_field(
            "Master Kush. Welcome to Master shop. Contact Master444",
            seller_uid="seller-one",
            seller_literals=[],
            seller_phrase_tokens=set(),
            global_tokens={"master444"},
            contextual_aliases=set(),
            contextual_alias_deletions=set(),
            seller_contextual_collision_tokens={"master"},
            audited_global_phrases=set(),
        )
        self.assertIn("Master Kush", clean)
        self.assertNotIn("Master shop", clean)
        self.assertNotIn("Master444", clean)

    def test_protected_occurrences_exclude_identifier_substrings(self):
        patterns = {
            term: re.compile(
                rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])",
                flags=re.IGNORECASE,
            )
            for term in ("master", "amazon prime store", "prime")
        }
        counts = preparation._protected_occurrences(
            "Master Kush grandmaster Master444 master card Amazon Prime Store",
            patterns,
        )
        self.assertEqual(counts["master"], 2)
        self.assertEqual(counts["amazon prime store"], 1)
        self.assertEqual(counts["prime"], 1)

        text = "ordinary private listing; contact us via wickr me-private message"
        private_pattern = {
            "private": re.compile(
                r"(?<![a-z0-9])private(?![a-z0-9])", re.IGNORECASE
            )
        }
        spans = preparation._intentional_identity_spans(
            text,
            seller_literals=[],
            seller_phrase_tokens=set(),
            global_tokens=set(),
            contextual_aliases=set(),
            contextual_alias_deletions=set(),
            seller_contextual_collision_tokens=set(),
            audited_global_phrases=set(),
        )
        self.assertEqual(
            preparation._protected_occurrences(text, private_pattern)["private"],
            2,
        )
        self.assertEqual(
            preparation._protected_occurrences_overlapping_spans(
                text, private_pattern, spans
            )["private"],
            1,
        )

        cross_line = "Whats-app.....+1 978-406-9577\nQuality ATM cards"
        quality_pattern = {
            "quality": re.compile(
                r"(?<![a-z0-9])quality(?![a-z0-9])", re.IGNORECASE
            )
        }
        cross_line_spans = preparation._intentional_identity_spans(
            cross_line,
            seller_literals=[],
            seller_phrase_tokens=set(),
            global_tokens=set(),
            contextual_aliases=set(),
            contextual_alias_deletions=set(),
            seller_contextual_collision_tokens=set(),
            audited_global_phrases=set(),
        )
        self.assertEqual(
            preparation._protected_occurrences_overlapping_spans(
                cross_line, quality_pattern, cross_line_spans
            )["quality"],
            0,
        )

    def test_protected_loss_is_accumulated_per_field_without_cancellation(self):
        policy = copy.deepcopy(self.policy)
        parent_policy = {
            "clean_text_contract": {
                "quality_gates": {
                    "protected_content_words": ["master"],
                    "protected_identity_collision_terms": [],
                }
            }
        }
        public = {
            "seller_identity_literals": {"seller-one": []},
            "seller_identity_phrase_tokens": {"seller-one": set()},
            "global_identity_tokens": set(),
            "contextual_global_alias_tokens": set(),
            "contextual_alias_deletion_tokens": set(),
            "seller_contextual_collision_tokens": {"seller-one": set()},
            "audited_global_identity_phrase_tokens": set(),
        }
        builder = preparation.RawCorpusBuilder(
            policy=policy,
            parent_policy=parent_policy,
            public=public,
            seller_split={"seller-one": "train"},
        )
        meta = {
            "seller_uid": "seller-one",
            "source_dataset": "synthetic.xlsx",
            "source_row_number": 2,
        }

        def fake_redact(value, **_kwargs):
            raw = str(value)
            clean = "" if raw == "Master item" else "Master"
            return clean, {
                "raw_character_count": len(raw),
                "clean_character_count": len(clean),
            }

        with mock.patch.object(common, "redact_raw_field", side_effect=fake_redact):
            builder._process_field(meta, "title", "Master item")
            builder._process_field(meta, "title", "other")
        self.assertEqual(builder.protected_unexplained_removed["master"], 1)
        self.assertEqual(builder.protected_created_surplus["master"], 1)

    def test_text_diagnostics_validator_enforces_protected_arithmetic(self):
        policy = copy.deepcopy(self.policy)
        policy["raw_item_boundary"]["expected_selected_item_count"] = 1
        policy["clean_text_contract"][
            "one_character_omission_content_collision_handling"
        ]["expected_retained_surface_sha256_counts"] = {}
        policy["clean_text_contract"][
            "one_character_omission_content_collision_handling"
        ]["expected_retained_match_occurrence_count"] = 0
        parent_policy = {
            "clean_text_contract": {
                "quality_gates": {
                    "protected_content_words": ["master"],
                    "protected_identity_collision_terms": [],
                    "minimum_protected_word_retention": 0.8,
                    "minimum_protected_identity_collision_term_retention": 0.8,
                }
            }
        }
        field_counts = {
            f"{field}_{suffix}": 1 if suffix != "empty_after_redaction_count" else 0
            for field in common.FIELD_NAMES
            for suffix in (
                "source_occurrence_count",
                "raw_nonempty_count",
                "clean_nonempty_count",
                "empty_after_redaction_count",
            )
        }
        redaction_counts = {
            key: 0
            for key in (
                "generic_identifier_match_count",
                "seller_local_alias_match_count",
                "seller_local_alias_phrase_match_count",
                "audited_global_identity_phrase_match_count",
                "global_identifier_token_match_count",
                "contextual_alias_match_count",
                "post_redaction_one_character_omission_collision_count",
                "empty_input",
                "empty_after_redaction",
                "utf8_mojibake_sequence_repair_count",
                "windows_1252_c1_repair_count",
            )
        }
        redaction_counts.update(
            {
                "raw_character_count": 100,
                "clean_character_count": 90,
                "maximum_redaction_pass_count": 2,
            }
        )
        diagnostics = {
            "raw_character_count": 100,
            "clean_character_count_occurrence_weighted": 90,
            "aggregate_character_retention": 0.9,
            "field_counts": field_counts,
            "redaction_counts": redaction_counts,
            "removed_global_identifier_token_sha256_counts": {},
            "removed_audited_phrase_sha256_counts": {},
            "removed_one_character_omission_surface_sha256_counts": {},
            "protected_content_occurrence_matching": policy[
                "clean_text_contract"
            ]["protected_content_occurrence_matching"],
            "protected_content_retention_aggregation": policy[
                "clean_text_contract"
            ]["protected_content_retention_aggregation"],
            "protected_content_retention": {
                "master": {
                    "raw_count": 2,
                    "intentional_identity_span_count": 1,
                    "eligible_content_count": 1,
                    "clean_count": 1,
                    "total_removed_count": 1,
                    "unexplained_removed_count": 0,
                    "created_surplus_count": 0,
                    "retention": 1.0,
                }
            },
        }
        common.validate_text_diagnostics(policy, parent_policy, diagnostics)
        diagnostics["protected_content_retention"]["master"][
            "unexplained_removed_count"
        ] = 1
        with self.assertRaisesRegex(ValueError, "arithmetic drift"):
            common.validate_text_diagnostics(policy, parent_policy, diagnostics)

    def test_canonical_manifest_self_hash_fails_closed_on_tamper(self):
        payload = {"step": "example", "count": 2}
        payload["content_sha256"] = common.canonical_hash(payload)
        self.assertEqual(
            common.verify_canonical_self_hash(
                payload, "content_sha256", "test manifest"
            ),
            payload["content_sha256"],
        )
        payload["count"] = 3
        with self.assertRaisesRegex(ValueError, "self-hash drift"):
            common.verify_canonical_self_hash(
                payload, "content_sha256", "test manifest"
            )

    def test_selection_labels_physically_exclude_evidence_and_rule_score(self):
        parent_rows = [
            {
                "pair_uid": "p1",
                "review_label": "positive",
                "evidence_type": "same_controller_direct_identifier",
                "component_id": "c1",
                "identity_rule_control_score": "1",
            }
        ]
        labels, evidence = preparation.project_private_label_rows(
            parent_rows, "train"
        )
        self.assertEqual(
            list(labels[0]), ["pair_uid", "review_label", "component_id"]
        )
        self.assertEqual(list(evidence[0]), ["pair_uid", "evidence_type"])
        self.assertNotIn("evidence_type", labels[0])
        self.assertNotIn("identity_rule_control_score", labels[0])
        self.assertNotIn("identity_rule_control_score", evidence[0])

        quarantined = {
            **parent_rows[0],
            "pair_uid": "quarantined-pair",
            "review_label": "negative",
        }
        labels, evidence = preparation.project_private_label_rows(
            [parent_rows[0], quarantined],
            "train",
            expected_pair_uids=["p1"],
            allowed_excluded_pair_uid_sha256={
                common.sha256_text("quarantined-pair")
            },
        )
        self.assertEqual([row["pair_uid"] for row in labels], ["p1"])
        self.assertEqual([row["pair_uid"] for row in evidence], ["p1"])
        with self.assertRaisesRegex(ValueError, "quarantine projection drift"):
            preparation.project_private_label_rows(
                [parent_rows[0], quarantined],
                "train",
                expected_pair_uids=["p1"],
                allowed_excluded_pair_uid_sha256=set(),
            )

    def test_private_artifact_validation_replays_frozen_parent_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "schema").mkdir()
            (root / "scripts").mkdir()
            (root / "inputs").mkdir()
            (root / "reports").mkdir()
            policy_path = root / "schema" / "policy.json"
            producer_path = root / "scripts" / "prepare.py"
            producer_path.write_text("# frozen producer\n", encoding="utf-8")
            public_manifest_path = root / "reports" / "public.json"
            public_manifest_path.write_text("{}\n", encoding="utf-8")
            train_parent_path = root / "inputs" / "train.csv"
            valid_parent_path = root / "inputs" / "valid.csv"
            parent_schema = (
                "pair_uid,review_label,evidence_type,component_id,"
                "identity_rule_control_score\n"
            )
            train_parent_path.write_text(
                parent_schema + "p1,positive,e1,c1,0\n", encoding="utf-8"
            )
            valid_parent_path.write_text(
                parent_schema
                + "p2,negative,e2,c2,0\n"
                + "quarantined,negative,e3,cq,0\n",
                encoding="utf-8",
            )
            output_paths = {
                "train_labels": root / "reports" / "train_labels.csv",
                "valid_labels": root / "reports" / "valid_labels.csv",
                "train_evidence": root / "reports" / "train_evidence.csv",
                "valid_evidence": root / "reports" / "valid_evidence.csv",
            }
            development_manifest_path = root / "reports" / "private.json"
            policy = {
                "version": common.EXPECTED_VERSION,
                "inputs": {
                    "parent_train_labels": {
                        "path": "inputs/train.csv",
                        "sha256": common.sha256_file(train_parent_path),
                    },
                    "parent_valid_labels": {
                        "path": "inputs/valid.csv",
                        "sha256": common.sha256_file(valid_parent_path),
                    },
                },
                "outputs": {
                    **{
                        role: path.relative_to(root).as_posix()
                        for role, path in output_paths.items()
                    },
                    "development_labels_manifest": development_manifest_path.relative_to(
                        root
                    ).as_posix(),
                    "preparation_manifest": public_manifest_path.relative_to(
                        root
                    ).as_posix(),
                },
                "implementation": {
                    "preparation": {
                        "path": producer_path.relative_to(root).as_posix(),
                        "sha256": common.sha256_file(producer_path),
                    }
                },
                "parent_fragment_quarantine": {
                    "decision_basis": "label_free_structure",
                    "pair_uid_sha256": common.sha256_text("quarantined"),
                    "split_name": "valid",
                },
                "supervision_boundary": {
                    "expected_counts": {
                        "train": {"positive": 1, "negative": 0},
                        "valid": {"positive": 0, "negative": 1},
                    }
                },
            }
            pair_rows = [
                {"pair_uid": "p1", "component_id": "c1", "split_name": "train"},
                {"pair_uid": "p2", "component_id": "c2", "split_name": "valid"},
            ]
            with (
                mock.patch.object(common, "ROOT", root),
                mock.patch.object(common, "DEFAULT_POLICY", policy_path),
            ):
                train_parent = common.load_csv(train_parent_path)
                valid_parent = common.load_csv(valid_parent_path)
                train_labels, train_evidence = common.project_private_label_rows(
                    train_parent, "train", expected_pair_uids=["p1"]
                )
                valid_labels, valid_evidence = common.project_private_label_rows(
                    valid_parent,
                    "valid",
                    expected_pair_uids=["p2"],
                    allowed_excluded_pair_uid_sha256={
                        common.sha256_text("quarantined")
                    },
                )
                for role, rows in (
                    ("train_labels", train_labels),
                    ("valid_labels", valid_labels),
                    ("train_evidence", train_evidence),
                    ("valid_evidence", valid_evidence),
                ):
                    common.write_csv_immutable(output_paths[role], rows)
                policy_path.write_text(
                    json.dumps(policy, sort_keys=True) + "\n", encoding="utf-8"
                )
                manifest = {
                    "step": "step7_v4_prepare_source_data_private_labels",
                    "version": common.EXPECTED_VERSION,
                    "public_preparation_manifest_sha256": common.sha256_file(
                        public_manifest_path
                    ),
                    "label_inputs": common.verify_inputs(
                        policy,
                        ("parent_train_labels", "parent_valid_labels"),
                    ),
                    "selection_label_columns": [
                        "pair_uid",
                        "review_label",
                        "component_id",
                    ],
                    "diagnostic_evidence_columns": ["pair_uid", "evidence_type"],
                    "identity_rule_control_score_materialized": False,
                    "evidence_is_physically_separate_from_selection_labels": True,
                    "historical_test_labels_materialized": False,
                    "parent_fragment_quarantine_projection": {
                        "decision_basis": "label_free_structure",
                        "excluded_pair_uid_sha256": common.sha256_text(
                            "quarantined"
                        ),
                        "excluded_pair_count_by_split": {"train": 0, "valid": 1},
                        "labels_or_evidence_types_used_to_choose_exclusion": False,
                    },
                    "policy_sha256": common.sha256_file(policy_path),
                    "producer_sha256": common.sha256_file(producer_path),
                    "outputs": {
                        role: common.file_record(path)
                        for role, path in output_paths.items()
                    },
                }
                manifest["manifest_content_sha256"] = common.canonical_hash(manifest)
                development_manifest_path.write_text(
                    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
                )
                common.validate_private_label_artifacts(policy, pair_rows)

                tampered_manifest = copy.deepcopy(manifest)
                tampered_manifest["label_inputs"]["parent_train_labels"][
                    "sha256"
                ] = "0" * 64
                tampered_manifest.pop("manifest_content_sha256")
                tampered_manifest["manifest_content_sha256"] = common.canonical_hash(
                    tampered_manifest
                )
                development_manifest_path.write_text(
                    json.dumps(tampered_manifest, indent=2) + "\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, "manifest step drift"):
                    common.validate_private_label_artifacts(policy, pair_rows)

                output_paths["valid_evidence"].write_text(
                    "pair_uid,evidence_type\np2,tampered-evidence\n",
                    encoding="utf-8",
                )
                tampered_output_manifest = copy.deepcopy(manifest)
                tampered_output_manifest["outputs"]["valid_evidence"] = (
                    common.file_record(output_paths["valid_evidence"])
                )
                tampered_output_manifest.pop("manifest_content_sha256")
                tampered_output_manifest["manifest_content_sha256"] = (
                    common.canonical_hash(tampered_output_manifest)
                )
                development_manifest_path.write_text(
                    json.dumps(tampered_output_manifest, indent=2) + "\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, "parent replay drift"):
                    common.validate_private_label_artifacts(policy, pair_rows)

    def test_stylometry_is_fixed_numeric_and_pairwise(self):
        values = common.text_stylometry("plain 1!!\n- next")
        self.assertEqual(list(values), common.STYLOMETRY_STATISTICS)
        self.assertTrue(all(math.isfinite(value) for value in values.values()))
        self.assertGreater(values["digit_character_ratio"], 0.0)
        self.assertGreater(values["repeated_punctuation_character_ratio"], 0.0)
        self.assertEqual(values["bullet_line_share"], 0.5)

    def test_batched_top_k_exactly_matches_full_matrix_and_replication(self):
        rng = np.random.default_rng(20260722)
        left = rng.normal(size=(17, 9))
        right = rng.normal(size=(13, 9))
        left_weights = rng.integers(1, 5, size=len(left), dtype=np.int64)
        right_weights = rng.integers(1, 5, size=len(right), dtype=np.int64)
        observed = common.symmetric_top_k_cosine(
            left, right, 3, similarity_block_rows=4
        )
        self.assertAlmostEqual(observed, brute_symmetric_top_k(left, right, 3), places=14)

        weighted = common.symmetric_top_k_cosine(
            left,
            right,
            3,
            left_multiplicities=left_weights,
            right_multiplicities=right_weights,
            similarity_block_rows=4,
        )
        expanded_left = np.repeat(left, left_weights, axis=0)
        expanded_right = np.repeat(right, right_weights, axis=0)
        self.assertAlmostEqual(
            weighted,
            brute_symmetric_top_k(expanded_left, expanded_right, 3),
            places=14,
        )

    def test_primary_aggregation_is_multiplicity_neutral(self):
        left = {
            "title": (
                np.asarray([[1.0, 0.0], [0.0, 1.0]]),
                np.asarray([1, 9]),
            )
        }
        right = {
            "title": (
                np.asarray([[1.0, 0.0], [-0.5, 0.8660254037844386]]),
                np.asarray([8, 1]),
            )
        }
        primary, audit = common.aggregate_pair_vectors(
            left, right, top_k=1, similarity_block_rows=1
        )
        changed = {
            "title": (left["title"][0], np.asarray([500, 1]))
        }
        primary_changed, audit_changed = common.aggregate_pair_vectors(
            changed, right, top_k=1, similarity_block_rows=1
        )
        self.assertEqual(primary, primary_changed)
        self.assertNotEqual(audit, audit_changed)
        self.assertIsNone(primary["description_centroid_cosine"])

    def test_gpu_indices_remove_identity_split_component_and_lineage(self):
        pairs = [
            {
                "pair_uid": "raw-pair-a",
                "split_name": "train",
                "component_id": "component-secret",
                "seller_uid_left": "market|seller_raw:alice",
                "seller_uid_right": "market|seller_raw:bob",
            }
        ]
        sellers = [
            {
                "seller_uid": "market|seller_raw:alice",
                "split_name": "train",
                "field_name": "title",
                "text_uid": "a" * 64,
                "multiplicity": 2,
                "source_lineage": [{"source_dataset": "raw.xlsx", "source_row_numbers": [2, 3]}],
            },
            {
                "seller_uid": "market|seller_raw:bob",
                "split_name": "train",
                "field_name": "title",
                "text_uid": "b" * 64,
                "multiplicity": 1,
                "source_lineage": [{"source_dataset": "raw.xlsx", "source_row_numbers": [4]}],
            },
        ]
        gpu_pairs, gpu_sellers = common.build_opaque_gpu_indices(pairs, sellers)
        serialized = json.dumps([gpu_pairs, gpu_sellers], sort_keys=True)
        for forbidden in ("alice", "bob", "train", "component-secret", "raw.xlsx"):
            self.assertNotIn(forbidden, serialized)
        self.assertEqual(
            list(gpu_pairs[0]), ["pair_uid", "seller_uid_left", "seller_uid_right"]
        )
        self.assertEqual(
            list(gpu_sellers[0]),
            ["seller_uid", "field_name", "text_uid", "multiplicity"],
        )
        self.assertEqual(
            sync_builder.GPU_PUBLIC_OUTPUT_ROLES,
            ["gpu_pair_manifest", "unique_text_corpus", "gpu_seller_text_index"],
        )
        self.assertEqual(
            sync_builder.GPU_IMPLEMENTATION_ROLES,
            ["common", "sync_builder", "encoder"],
        )
        forbidden_text = "\n".join(sync_builder.FORBIDDEN_WORKSPACE_PATHS)
        self.assertIn("pair_manifest.no_labels.csv", forbidden_text)
        self.assertIn("source_preparation_manifest.json", forbidden_text)
        for role in (
            "pair_manifest",
            "raw_item_lineage",
            "seller_text_index",
            "preparation_manifest",
            "train_labels",
            "valid_labels",
            "train_evidence",
            "valid_evidence",
            "development_labels_manifest",
        ):
            self.assertIn(
                self.policy["outputs"][role],
                sync_builder.FORBIDDEN_WORKSPACE_PATHS,
            )
        for role in ("parent_train_labels", "parent_valid_labels"):
            self.assertIn(
                self.policy["inputs"][role]["path"],
                sync_builder.FORBIDDEN_WORKSPACE_PATHS,
            )
        encoder_sync_check = inspect.getsource(
            encoder.verify_label_free_gpu_sync
        )
        self.assertIn(
            'manifest.get("forbidden_workspace_paths")', encoder_sync_check
        )
        self.assertIn(
            "sync_builder.FORBIDDEN_WORKSPACE_PATHS", encoder_sync_check
        )

    def test_public_preparation_never_opens_labelled_component_assignments(self):
        replay_text = inspect.getsource(preparation.replay_parent_public)
        self.assertNotIn(
            'parent_policy["inputs"]["component_assignments"]', replay_text
        )
        self.assertNotIn("eligible_assignment_rows", replay_text)
        self.assertIn('policy["inputs"]["parent_pair_manifest"]', replay_text)
        validator_text = inspect.getsource(common.validate_preparation_artifacts)
        start = validator_text.index("parent_replayed_roles")
        stop = validator_text.index("parent_records", start)
        self.assertNotIn("component_assignments", validator_text[start:stop])
        self.assertNotIn(
            "component_assignments",
            inspect.getsource(selector.replay_legacy_context),
        )
        parser_text = inspect.getsource(preparation.parse_args)
        self.assertNotIn('"all"', parser_text)
        self.assertIn('default="public"', parser_text)
        main_text = inspect.getsource(preparation.main)
        self.assertNotIn('args.stage == "all"', main_text)
        public_text = inspect.getsource(preparation.prepare_public)
        self.assertIn(
            '"labelled_component_assignment_file_opened": False',
            public_text,
        )
        self.assertIn(
            '"frozen_label_free_parent_pair_projection_used": True',
            public_text,
        )

    def test_redaction_does_not_encode_removed_identity_length_in_spaces(self):
        common_arguments = {
            "seller_phrase_tokens": frozenset(),
            "global_tokens": frozenset(),
            "contextual_aliases": frozenset(),
            "contextual_alias_deletions": frozenset(),
            "audited_global_phrases": frozenset(),
        }
        short, _ = common._redact_identifiers_non_cascading_line_bounded(
            "same ShortID remaining text",
            seller_literals=("ShortID",),
            **common_arguments,
        )
        long, _ = common._redact_identifiers_non_cascading_line_bounded(
            "same ExtremelyLongIdentityHandle remaining text",
            seller_literals=("ExtremelyLongIdentityHandle",),
            **common_arguments,
        )
        self.assertEqual(short, "same remaining text")
        self.assertEqual(long, short)

    def test_gpu_sync_keeps_file_and_canonical_manifest_hashes_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            payload = {"step": "test", "value": 7}
            payload["manifest_content_sha256"] = common.canonical_hash(payload)
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            observed = sync_builder.preparation_manifest_hash_contract(
                payload, path
            )
            self.assertEqual(
                observed["source_preparation_manifest_file_sha256"],
                common.sha256_file(path),
            )
            self.assertEqual(
                observed["source_preparation_manifest_content_sha256"],
                payload["manifest_content_sha256"],
            )
            self.assertNotEqual(
                observed["source_preparation_manifest_file_sha256"],
                observed["source_preparation_manifest_content_sha256"],
            )

    def test_minimal_gpu_modules_import_without_raw_data_stack(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script_root = root / "scripts"
            script_root.mkdir()
            schema_root = root / "schema"
            schema_root.mkdir()
            for name in (
                "step7_v4_common.py",
                "step7_v4_build_sync_manifest.py",
                "step7_v4_encode_item_models.py",
            ):
                shutil.copy2(SCRIPTS / name, script_root / name)
            shutil.copy2(common.DEFAULT_POLICY, schema_root / common.DEFAULT_POLICY.name)
            code = (
                "import sys;"
                f"sys.path.insert(0, {str(script_root)!r});"
                "import step7_v4_encode_item_models;"
                "import step7_v4_common;"
                "step7_v4_common.load_policy();"
                "assert 'step7_v3_1_source_data' not in sys.modules;"
                "assert 'step3_build_seller_profiles' not in sys.modules"
            )
            completed = subprocess.run(
                [sys.executable, "-c", code],
                cwd=directory,
                check=False,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONPATH": ""},
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_local_model_fingerprint_exactly_replays_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "nested").mkdir()
            (root / "config.json").write_text("{}\n", encoding="utf-8")
            (root / "nested" / "weights.bin").write_bytes(b"abc\x00def")
            (root / "__pycache__").mkdir()
            (root / "__pycache__" / "ignored.pyc").write_bytes(b"ignored")
            self.assertEqual(
                common.model_content_fingerprint(root),
                source.model_content_fingerprint(root),
            )

    def test_item_manifest_boundary_selects_only_declared_raw_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            path.write_text(
                "item_uid,seller_uid,source_dataset,source_row_number,data_bucket,eligibility_status\n"
                "i1,s1,market_item.xlsx,2,en_content_train_pool,content_train_eligible\n"
                "i2,s1,2017-12-05-philipjames11-darknetmarketplacedataagora20142015.xlsx,3,en_content_train_pool,content_train_eligible\n"
                "i3,s1,market_item.xlsx,4,other_bucket,content_train_eligible\n",
                encoding="utf-8",
            )
            policy = copy.deepcopy(self.policy)
            policy["raw_item_boundary"]["expected_selected_item_count"] = 2
            policy["raw_item_boundary"]["expected_selected_item_count_by_source"] = {
                "market_item.xlsx": 1,
                "2017-12-05-philipjames11-darknetmarketplacedataagora20142015.xlsx": 1,
            }
            policy["parent_fragment_quarantine"] = {
                "raw_source_dataset": "2017-12-05-philipjames11-darknetmarketplacedataagora20142015.xlsx",
                "raw_rows": [],
            }
            with mock.patch.object(common, "resolve", return_value=path):
                selected, quarantined, audit = preparation.selected_item_manifest_rows(
                    policy, {"s1"}, set()
                )
            self.assertEqual(sum(len(rows) for rows in selected.values()), 2)
            self.assertEqual(sum(len(rows) for rows in quarantined.values()), 0)
            self.assertEqual(audit["ignored_rows_for_pair_universe_sellers"], 1)
            self.assertEqual(selected["market_item.xlsx"][2]["item_uid"], "i1")

    def test_label_free_fragment_quarantine_requires_exact_isolated_component(self):
        pairs = [
            {
                "pair_uid": "train-pair",
                "split_name": "train",
                "component_id": "train-component",
                "seller_uid_left": "train-left",
                "seller_uid_right": "train-right",
            },
            {
                "pair_uid": "valid-pair",
                "split_name": "valid",
                "component_id": "valid-component",
                "seller_uid_left": "valid-left",
                "seller_uid_right": "valid-right",
            },
            {
                "pair_uid": "quarantine-pair",
                "split_name": "valid",
                "component_id": "quarantine-component",
                "seller_uid_left": "fragment-left",
                "seller_uid_right": "fragment-right",
            },
            {
                "pair_uid": "test-pair",
                "split_name": "test",
                "component_id": "test-component",
                "seller_uid_left": "test-left",
                "seller_uid_right": "test-right",
            },
        ]
        safe = [{"pair_uid": row["pair_uid"], "x": "0"} for row in pairs]
        policy = copy.deepcopy(self.policy)
        policy["parent_fragment_quarantine"].update(
            {
                "expected_parent_pair_count": 4,
                "excluded_pair_count": 1,
                "pair_uid_sha256": common.sha256_text("quarantine-pair"),
                "split_name": "valid",
                "component_id": "quarantine-component",
                "seller_uid_sha256": sorted(
                    [
                        common.sha256_text("fragment-left"),
                        common.sha256_text("fragment-right"),
                    ]
                ),
                "must_be_an_isolated_component": True,
            }
        )
        policy["supervision_boundary"] = {
            "expected_counts": {
                "train": {"positive": 0, "negative": 1, "total": 1},
                "valid": {"positive": 0, "negative": 1, "total": 1},
                "test": {"positive": 0, "negative": 1, "total": 1},
                "total": 3,
            },
            "expected_component_count_by_split": {
                "train": 1,
                "valid": 1,
                "test": 1,
            },
            "expected_seller_count_by_split": {
                "train": 2,
                "valid": 2,
                "test": 2,
            },
            "expected_total_unique_sellers": 6,
            "historical_test_labels_may_be_materialized": False,
        }
        effective, effective_safe, excluded_sellers, audit = (
            preparation.project_label_free_parent_quarantine(
                policy, pairs, safe
            )
        )
        self.assertEqual(
            [row["pair_uid"] for row in effective],
            ["train-pair", "valid-pair", "test-pair"],
        )
        self.assertEqual(
            [row["pair_uid"] for row in effective_safe],
            ["train-pair", "valid-pair", "test-pair"],
        )
        self.assertEqual(excluded_sellers, {"fragment-left", "fragment-right"})
        self.assertFalse(audit["labels_or_evidence_types_read"])
        contaminated = [*pairs, {**pairs[2], "pair_uid": "second-fragment-pair"}]
        with self.assertRaises(ValueError):
            preparation.project_label_free_parent_quarantine(
                policy, contaminated, [*safe, {"pair_uid": "second-fragment-pair", "x": "0"}]
            )

    def test_raw_workbook_row_must_replay_step2_item_and_seller_identity(self):
        parts = (2, "market-a", "vendor-a", "title", "description", 3.5, "cat")
        meta = {
            "source_dataset": "market_item.xlsx",
            "source_row_number": 2,
            "seller_uid": "market_item.xlsx|market-a|seller_raw:vendor-a",
            "item_uid": preparation._step2_item_uid("market_item.xlsx", *parts),
        }
        preparation._verify_step2_row_identity(
            meta,
            vendor="vendor-a",
            market="market-a",
            title="title",
            description="description",
            price=3.5,
            category="cat",
        )
        changed = dict(meta)
        changed["item_uid"] = "market_item.xlsx|wrong"
        with self.assertRaisesRegex(ValueError, "does not replay"):
            preparation._verify_step2_row_identity(
                changed,
                vendor="vendor-a",
                market="market-a",
                title="title",
                description="description",
                price=3.5,
                category="cat",
            )

    def test_quarantined_raw_row_requires_exact_full_row_signature(self):
        values = ("fragment", "0.1 BTC", "NL", "EU", "4.9/5", None, None, None, None)
        canonical = ["" if value is None else str(value) for value in values]
        meta = {
            "source_dataset": "agora.xlsx",
            "source_row_number": 7,
            "quarantine_contract": {
                "canonical_cell_values_sha256": common.canonical_hash(canonical),
                "cell_string_lengths": [len(value) for value in canonical],
            },
        }
        preparation._verify_quarantined_raw_workbook_values(meta, values)
        with self.assertRaisesRegex(ValueError, "fragment structure drift"):
            preparation._verify_quarantined_raw_workbook_values(
                meta, ("changed", *values[1:])
            )

    def test_raw_builder_retains_lineage_but_deduplicates_model_text(self):
        parent_policy = json.loads(
            (ROOT / "schema" / "step7_v3_1_source_data_policy.json").read_text(
                encoding="utf-8"
            )
        )
        public = {
            "seller_identity_literals": {"s1": []},
            "seller_identity_phrase_tokens": {"s1": set()},
            "global_identity_tokens": set(),
            "contextual_global_alias_tokens": set(),
            "contextual_alias_deletion_tokens": set(),
            "seller_contextual_collision_tokens": {"s1": set()},
            "audited_global_identity_phrase_tokens": set(),
        }
        builder = preparation.RawCorpusBuilder(
            policy=self.policy,
            parent_policy=parent_policy,
            public=public,
            seller_split={"s1": "train"},
        )

        def fake_redact(value, **_kwargs):
            clean = common.normalize_raw_field(value)
            return clean, {
                "raw_character_count": len(clean),
                "clean_character_count": len(clean),
                "empty_input": not bool(clean),
                "empty_after_redaction": False,
                "utf8_mojibake_sequence_repair_count": 0,
                "windows_1252_c1_repair_count": 0,
                "redaction_pass_count": 0,
            }

        with mock.patch.object(common, "redact_raw_field", side_effect=fake_redact):
            for row_number in (2, 3):
                builder.add_item(
                    {
                        "item_uid": f"i{row_number}",
                        "seller_uid": "s1",
                        "source_dataset": "market_item.xlsx",
                        "source_row_number": row_number,
                    },
                    title="same clean title",
                    description="",
                )
        lineage, unique_rows, seller_rows, diagnostics = builder.finalize()
        self.assertEqual(len(lineage), 2)
        self.assertEqual(len(unique_rows), 1)
        self.assertEqual(len(seller_rows), 1)
        self.assertEqual(seller_rows[0]["multiplicity"], 2)
        self.assertEqual(
            seller_rows[0]["source_lineage"][0]["source_row_numbers"], [2, 3]
        )
        self.assertEqual(diagnostics["aggregate_character_retention"], 1.0)

    def test_chunk_to_text_rejects_noncontiguous_order(self):
        rows = [
            {"text_uid": "a", "chunk_index": 0},
            {"text_uid": "a", "chunk_index": 1},
            {"text_uid": "b", "chunk_index": 0},
        ]
        matrix = np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32
        )
        vectors = common.text_vectors_from_chunks(rows, matrix)
        self.assertEqual(set(vectors), {"a", "b"})
        self.assertAlmostEqual(float(np.linalg.norm(vectors["a"])), 1.0)
        invalid = [rows[1], rows[0], rows[2]]
        with self.assertRaisesRegex(ValueError, "chunk order"):
            common.text_vectors_from_chunks(invalid, matrix)

    def test_shared_chunks_are_complete_and_identical_for_all_tokenizers(self):
        tokenizers = {
            model_key: FakeTokenizer(width)
            for model_key, width in zip(common.MODEL_KEYS, (1, 2, 3, 4), strict=True)
        }
        text = "first line.\n" + ("middle-word " * 70) + "final"
        unique_rows = [
            {
                "text_uid": common.sha256_text(text),
                "text": text,
                "text_sha256": common.sha256_text(text),
            }
        ]
        chunks, audit = encoder.build_shared_chunks(
            self.policy, unique_rows, tokenizers
        )
        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(row["text"] for row in chunks), text)
        self.assertTrue(audit["exact_character_reconstruction"])
        self.assertTrue(
            all(
                max(row["token_lengths"].values())
                <= common.SHARED_TOKEN_BUDGET
                for row in chunks
            )
        )
        digest, lengths = encoder.tokenizer_digest_and_lengths(
            tokenizers[common.MODEL_KEYS[0]],
            [row["text"] for row in chunks],
            self.policy["embedding_models"][common.MODEL_KEYS[0]]["text_prefix"],
            batch_size=2,
        )
        self.assertEqual(len(digest), 64)
        self.assertEqual(
            lengths,
            [row["token_lengths"][common.MODEL_KEYS[0]] for row in chunks],
        )
        model = FakeSentenceTransformerTokenizer(
            tokenizers[common.MODEL_KEYS[0]]
        )
        runtime_digest, runtime_lengths = (
            encoder.sentence_transformer_tokenizer_digest_and_lengths(
                model,
                [row["text"] for row in chunks],
                self.policy["embedding_models"][common.MODEL_KEYS[0]][
                    "text_prefix"
                ],
                batch_size=2,
            )
        )
        self.assertEqual(runtime_digest, digest)
        self.assertEqual(runtime_lengths, lengths)

    def test_sentence_transformer_default_prompt_is_explicitly_disabled(self):
        cfg = self.policy["embedding_models"][common.MODEL_KEYS[0]]
        model, loaded = encoder.create_sentence_transformer(
            FakeSentenceTransformer, cfg
        )
        self.assertIsNone(model.default_prompt_name)
        self.assertTrue(model.eval_called)
        self.assertEqual(model.max_seq_length, 512)
        self.assertEqual(loaded["loaded_default_prompt_name"], "document")
        self.assertEqual(loaded["loaded_native_max_seq_length"], 512)
        self.assertEqual(
            loaded["loaded_prompts"],
            {"document": "document: ", "query": "query: "},
        )
        self.assertEqual(cfg["sentence_transformer_prompt"], "")

        labse_cfg = self.policy["embedding_models"]["labse"]
        labse_model, labse_loaded = encoder.create_sentence_transformer(
            FakeSentenceTransformer256, labse_cfg
        )
        self.assertEqual(labse_model.max_seq_length, 256)
        self.assertEqual(labse_loaded["loaded_native_max_seq_length"], 256)
        with self.assertRaisesRegex(ValueError, "native max_seq_length drift"):
            encoder.create_sentence_transformer(
                FakeSentenceTransformer, labse_cfg
            )

    def test_sentence_transformer_without_prompt_support_fails_closed(self):
        cfg = self.policy["embedding_models"][common.MODEL_KEYS[0]]
        with self.assertRaisesRegex(RuntimeError, "hidden model-default prompts"):
            encoder.create_sentence_transformer(
                FakeSentenceTransformerWithoutPrompt, cfg
            )

    def test_real_model_smoke_replays_actual_tokens_and_encoding(self):
        model_key = "pcm_multilingual_authorship"
        cfg = self.policy["embedding_models"][model_key]
        tokenizer = FakeTokenizer(2)
        text = "abcdefgh"
        length = len(
            tokenizer(
                cfg["text_prefix"] + text,
                add_special_tokens=True,
                padding=False,
                truncation=False,
            )["input_ids"]
        )
        result = encoder.smoke_test_one_model(
            self.policy,
            model_key,
            cfg,
            tokenizer,
            [
                {
                    "text": text,
                    "token_lengths": {model_key: length},
                }
            ],
            FakeTorchForSmoke,
            FakeSmokeSentenceTransformer,
        )
        self.assertEqual(result["longest_shared_chunk_token_length"], length)
        self.assertEqual(result["loaded_native_max_seq_length"], 512)
        self.assertEqual(result["embedding_shape"], [1, 1024])
        self.assertTrue(result["repeated_encoding_byte_identical"])

    def test_linux_runner_preflights_runtime_before_encoding(self):
        runner = (
            SCRIPTS / "run_step7_v4_raw_item_authorship_linux_20260722.sh"
        ).read_text(encoding="utf-8")
        preparation_check = (
            '"$PYTHON_BIN" scripts/step7_v4_prepare_source_data.py \\\n'
            "  --stage validate-existing"
        )
        sync_check = (
            '"$PYTHON_BIN" scripts/step7_v4_build_sync_manifest.py '
            "--validate-only"
        )
        selector_check = (
            '"$PYTHON_BIN" scripts/step7_v4_select_source_model.py '
            "--validate-config-only"
        )
        preflight = (
            '"$PYTHON_BIN" scripts/step7_v4_encode_item_models.py '
            "--validate-inputs-only"
        )
        formal = '"$PYTHON_BIN" scripts/step7_v4_encode_item_models.py\n'
        nvidia = "nvidia-smi --query-gpu"
        for check in (preparation_check, sync_check, selector_check):
            self.assertIn(check, runner)
            self.assertLess(runner.index(check), runner.index(nvidia))
        self.assertIn(preflight, runner)
        self.assertLess(runner.index(preflight), runner.index(formal))

    def test_v4_l2_strength_is_invariant_to_exact_row_replication(self):
        rows = [
            {
                "pair_uid": f"p{index}",
                "component_id": f"c{index // 2}",
                "review_label": label,
            }
            for index, label in enumerate(
                ("negative", "negative", "positive", "positive")
            )
        ]
        matrix = np.asarray([[-2.0], [-1.0], [1.0], [2.0]], dtype=np.float64)
        base = selector.fit_logistic(matrix, rows, 0.3, self.policy)
        repeated_rows = [
            {**row, "pair_uid": f"{row['pair_uid']}-{copy_index}"}
            for copy_index in range(3)
            for row in rows
        ]
        repeated_matrix = np.tile(matrix, (3, 1))
        repeated = selector.fit_logistic(
            repeated_matrix, repeated_rows, 0.3, self.policy
        )
        self.assertAlmostEqual(base["intercept"], repeated["intercept"], places=10)
        self.assertAlmostEqual(
            base["coefficients"][0], repeated["coefficients"][0], places=10
        )
        self.assertEqual(base["l2_penalty"], 0.3)
        self.assertEqual(
            base["l2_parameterization"],
            "weighted_mean_logloss_plus_half_l2_squared_coefficient_norm",
        )
        self.assertAlmostEqual(
            repeated["solver_sum_loss_l2_penalty"],
            3.0 * base["solver_sum_loss_l2_penalty"],
        )

    def test_v4_solver_crosses_float64_armijo_plateau_without_relaxing_gradient_gate(self):
        row_count = 16
        rng = np.random.default_rng(81)
        matrix = rng.normal(size=(row_count, 1))
        beta = rng.normal(size=1)
        logits = matrix @ beta
        labels = (
            logits + rng.normal(scale=1.0, size=row_count)
            > np.median(logits)
        ).astype(np.int8)
        weights = np.exp(rng.normal(scale=1.0, size=row_count))
        weights *= row_count / float(np.sum(weights))
        arguments = (
            matrix,
            labels,
            weights,
            0.247,
            500,
            1e-9,
            1e-4,
            2**-30,
        )
        with self.assertRaisesRegex(ValueError, "did not converge"):
            parent_solver.fit_logistic(*arguments)
        observed = v4_solver.fit_logistic(*arguments)
        repeated = v4_solver.fit_logistic(*arguments)
        relaxed_reference = parent_solver.fit_logistic(
            matrix,
            labels,
            weights,
            0.247,
            500,
            1e-8,
            1e-4,
            2**-30,
        )
        self.assertTrue(observed["solver_converged"])
        self.assertFalse(
            observed["solver_used_float64_stationarity_fallback"]
        )
        self.assertGreaterEqual(
            observed["solver_float64_objective_resolution_step_count"], 1
        )
        self.assertLessEqual(
            observed["solver_final_normalized_gradient_inf_norm"],
            1e-9,
        )
        self.assertLessEqual(
            observed["solver_final_objective"],
            relaxed_reference["solver_final_objective"]
            + observed["solver_float64_objective_resolution"],
        )
        self.assertEqual(observed, repeated)
        self.assertLess(
            observed["solver_final_normalized_gradient_inf_norm"],
            relaxed_reference[
                "solver_final_normalized_gradient_inf_norm"
            ],
        )

    def test_v4_solver_still_rejects_genuine_nonconvergence(self):
        matrix = np.asarray(
            [[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]],
            dtype=np.float64,
        )
        labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int8)
        weights = np.ones(len(labels), dtype=np.float64)
        with self.assertRaisesRegex(
            ValueError, "did not reach certified convergence"
        ):
            v4_solver.fit_logistic(
                matrix,
                labels,
                weights,
                0.001,
                1,
                1e-12,
                1e-4,
                2**-30,
            )

    def test_selection_solver_patch_reuses_frozen_gpu_contract_only(self):
        patch = selection_resume.load_patch_policy()
        self.assertFalse(
            patch["parent_contract"]["gpu_reencoding_required"]
        )
        self.assertFalse(
            patch["parent_contract"][
                "training_or_validation_labels_changed"
            ]
        )
        run_policy = selection_resume.execution_policy(
            self.policy, patch
        )
        for key, value in self.policy["outputs"].items():
            if key in selection_resume.SELECTION_OUTPUT_KEYS:
                self.assertEqual(
                    run_policy["outputs"][key],
                    patch["outputs"][key],
                )
            else:
                self.assertEqual(run_policy["outputs"][key], value)
        self.assertEqual(
            run_policy["training"]["tolerance"],
            self.policy["training"]["tolerance"],
        )
        self.assertEqual(
            run_policy["training"]["l2_initial_grid"],
            self.policy["training"]["l2_initial_grid"],
        )
        source_text = inspect.getsource(
            selection_resume.install_verified_patch
        )
        self.assertIn(
            "original_verify_gpu_outputs", source_text
        )
        self.assertIn(
            "parent_policy, preparation_manifest, preparation_bundle",
            source_text.replace("\n", " ").replace("  ", " "),
        )
        self.assertIn(
            "artifact = corrected_solver.fit_logistic",
            source_text,
        )
        self.assertIn(
            "parent_selector.solver.fit_logistic = audited_fit_logistic",
            source_text,
        )

    def test_formal_solver_patch_outputs_are_hash_closed_and_strictly_converged(self):
        patch = selection_resume.load_patch_policy()
        manifest = common.load_json(
            common.resolve(patch["outputs"]["patch_manifest"])
        )
        common.verify_canonical_self_hash(
            manifest,
            "manifest_content_sha256",
            "Step7-v4 solver patch manifest",
        )
        self.assertFalse(manifest["gpu_reencoding_performed"])
        for record in manifest["outputs"].values():
            common.verify_file_record(record, "Step7-v4 solver patch output")

        audit = manifest["solver_execution_audit"]
        self.assertEqual(audit["fit_count"], 32423)
        self.assertEqual(
            audit["convergence_criterion_counts"],
            {
                "normalized_gradient_inf_norm_at_most_requested_tolerance": (
                    32423
                )
            },
        )
        self.assertEqual(
            audit["float64_stationarity_fallback_fit_count"], 0
        )
        self.assertLessEqual(
            audit["maximum_final_normalized_gradient_inf_norm"], 1e-9
        )

        summary = common.load_json(
            common.resolve(patch["outputs"]["selection_summary"])
        )
        common.verify_canonical_self_hash(
            summary,
            "summary_content_sha256",
            "Step7-v4 corrected selection summary",
        )
        decision = summary["selection_decision"]
        self.assertEqual(
            decision["selection_status"],
            "no_stable_unique_provisional_m0",
        )
        self.assertFalse(decision["unique_provisional_m0_gate_passed"])
        self.assertFalse(
            decision["matched_single_encoder_comparison"][
                "stable_unique_encoder_at_0_95"
            ]
        )
        self.assertEqual(
            decision["winner_no_exact_clone_robustness"][
                "original_winner_rate_across_no_clone_outer_seeds"
            ],
            0.0,
        )
        self.assertFalse(summary["historical_test_labels_read"])

    def test_no_clone_gate_requires_complete_nested_retraining(self):
        selection_text = inspect.getsource(selector.run_selection)
        assessment_text = inspect.getsource(selector.assess_selection)
        self.assertIn("no_clone_nested_results", selection_text)
        self.assertIn(
            "run_nested_selection(policy, factory, no_clone_train_rows)",
            selection_text,
        )
        self.assertNotIn("no_clone_indices", assessment_text)
        gate = self.policy["selection_rule"][
            "unique_provisional_m0_requires"
        ]
        self.assertEqual(
            gate[
                "no_exact_clone_nested_winner_rate_across_outer_repeats_at_least"
            ],
            0.6,
        )

    def test_chunk_manifest_locks_visible_prefix_and_empty_prompt(self):
        source_text = inspect.getsource(encoder.main)
        self.assertIn('"model_input_contracts"', source_text)
        self.assertIn('"sentence_transformer_prompt"', source_text)
        verifier_text = inspect.getsource(selector.verify_gpu_outputs)
        self.assertIn("expected_input_contracts", verifier_text)
        for cfg in self.policy["embedding_models"].values():
            self.assertEqual(cfg["sentence_transformer_prompt"], "")

    def test_optimized_embedded_identity_census_exactly_replays_parent(self):
        registry = frozenset({"dark", "darkm", "wallstreet", "abc", "bc"})
        rows = [
            {
                "seller_uid": "seller-a",
                "model_text": "darkmarkets antidarkm xabcx abc abcs abcshop",
            },
            {
                "seller_uid": "seller-b",
                "model_text": "wallstreetbet darkmarkets xabcx",
            },
        ]
        expected = source.audited_identity_embedded_residual_census(rows, registry)
        observed = common.audited_identity_embedded_residual_census(rows, registry)
        for key, value in expected.items():
            self.assertEqual(observed[key], value, key)
        self.assertEqual(
            observed["implementation_audit"]["algorithm"],
            "exact_alias_trie_with_repeated_surface_cache",
        )

    def test_optimized_full_alias_census_exactly_replays_parent(self):
        registry = frozenset({"alpha", "alphashop", "beta", "gamma", "abc"})
        rows = [
            {
                "seller_uid": "seller-a",
                "model_text": "alpha alpha-s beta shop || gamma_vendor\nalphas abc",
            },
            {
                "seller_uid": "seller-b",
                "model_text": "alpha alpha-s beta shop || gamma_vendor\nalphas abc",
            },
        ]
        expected = source.full_known_alias_residual_census(rows, registry)
        observed = common.full_known_alias_residual_census(rows, registry)
        for key, value in expected.items():
            self.assertEqual(observed[key], value, key)
        self.assertEqual(
            observed["implementation_audit"]["algorithm"],
            "exact_precompiled_surface_lookup_with_segment_cache",
        )

    def test_compiled_final_identity_matchers_exactly_replay_parent_helpers(self):
        phrase_registry = frozenset(
            {"alpha", "alphashop", "mrcodez", "betateam"}
        )
        contextual_registry = frozenset({"alpha", "betateam", "gamma"})
        deletion_registry = frozenset({"betatem"})
        texts = [
            "alpha alpha-s alpha.shop m r codez",
            "contact: alpha; betateam shop; welcome to betatem",
            "gamma vendor and ordinary product words",
            "alphashop alpha store m.r.codez",
        ]
        phrase_counter = common._compile_unconditional_alias_count(
            phrase_registry
        )
        contextual_counter = common._compile_contextual_alias_count(
            contextual_registry, deletion_registry
        )
        for text in texts:
            self.assertEqual(
                phrase_counter(text),
                len(source.unconditional_alias_spans(text, phrase_registry)),
                text,
            )
            self.assertEqual(
                contextual_counter(text),
                len(
                    source.contextual_alias_spans(
                        text, contextual_registry, deletion_registry
                    )
                ),
                text,
            )

    def test_exact_final_identity_scan_replays_all_parent_counters(self):
        rows = [
            {"seller_uid": "s1", "model_text": "contact: alpha alpha-s @user007"},
            {"seller_uid": "s2", "model_text": "safe product unionstore beta shop"},
        ]
        seller_literals = {"s1": ["alpha"], "s2": ["never"]}
        global_tokens = frozenset({"user007"})
        contextual = frozenset({"alpha", "beta"})
        deletions = frozenset()
        seller_phrases = {"s1": {"alpha"}, "s2": set()}
        global_phrases = frozenset({"beta"})

        # This is the parent implementation's counting body, retained here as
        # an independent oracle because the parent public function raises as
        # soon as any nonzero residue is summarized.
        from collections import Counter

        pattern_counts = Counter()
        local = global_count = contextual_count = local_phrase = global_phrase = 0
        for row in rows:
            text = row["model_text"]
            seller_uid = row["seller_uid"]
            for rule_name, pattern in source.FINAL_CORPUS_AUDIT_RULES:
                pattern_counts[rule_name] += sum(1 for _ in pattern.finditer(text))
            for literal in seller_literals.get(seller_uid, []):
                local += sum(
                    1
                    for _ in source.identity_literal_pattern(literal).finditer(text)
                )
            local_phrase += len(
                source.unconditional_alias_spans(
                    text, seller_phrases.get(seller_uid, set())
                )
            )
            global_phrase += len(
                source.unconditional_alias_spans(text, global_phrases)
            )
            global_count += sum(
                1
                for match in source.IDENTIFIER_TOKEN_RE.finditer(text)
                if source.matches_global_identity_token(
                    match.group(0), global_tokens
                )
            )
            contextual_count += len(
                source.contextual_alias_spans(text, contextual, deletions)
            )
        expected = {
            "pattern_residue_count": int(sum(pattern_counts.values())),
            "pattern_residue_count_by_rule": {
                key: int(value)
                for key, value in sorted(pattern_counts.items())
                if value
            },
            "seller_local_identity_literal_residue_count": local,
            "seller_local_separator_variant_residue_count": local_phrase,
            "audited_global_identity_phrase_residue_count": global_phrase,
            "known_global_high_confidence_handle_residue_count": global_count,
            "context_gated_known_alias_residue_count": contextual_count,
        }
        observed = common.exact_final_corpus_identity_residue_scan(
            rows,
            seller_literals,
            global_tokens,
            contextual,
            deletions,
            seller_phrases,
            global_phrases,
            fail_on_residue=False,
        )
        for key, value in expected.items():
            self.assertEqual(observed[key], value, key)
        self.assertEqual(
            observed["seller_local_context_gated_collision_residue_count"], 0
        )
        self.assertEqual(
            observed["context_gated_exact_known_alias_residue_count"],
            expected["context_gated_known_alias_residue_count"],
        )
        self.assertEqual(
            observed[
                "post_redaction_one_character_omission_collision_census_count"
            ],
            0,
        )
        self.assertEqual(
            observed["total_residue_count"],
            sum(
                value
                for key, value in expected.items()
                if key != "pattern_residue_count_by_rule"
            ),
        )
        with self.assertRaisesRegex(ValueError, "identity scan failed"):
            common.exact_final_corpus_identity_residue_scan(
                rows,
                seller_literals,
                global_tokens,
                contextual,
                deletions,
                seller_phrases,
                global_phrases,
            )

    def test_final_identity_scan_excludes_only_exact_pinned_content_collision(self):
        seller_uid = "tutorial-seller"
        text = "025: Unlisted Phone Numbers (NEW Revision, 4.14)\n026: Fuses"
        matches = []
        for rule_name, pattern in source.FINAL_CORPUS_AUDIT_RULES:
            matches.extend(
                (rule_name, match.group(0))
                for match in pattern.finditer(text)
            )
        self.assertEqual(
            matches,
            [
                (
                    "audit_contact_cued_phone",
                    "Phone Numbers (NEW Revision, 4.14)\n026",
                )
            ],
        )
        contract = {
            "review_basis": (
                common.FIXED_FINAL_AUDIT_CONTENT_COLLISION_REVIEW_BASIS
            ),
            "expected_match_count": 1,
            "collisions": [
                {
                    "rule_name": matches[0][0],
                    "seller_uid_sha256": common.sha256_text(seller_uid),
                    "clean_text_sha256": common.sha256_text(text),
                    "matched_surface_sha256": common.sha256_text(matches[0][1]),
                    "expected_match_count": 1,
                    "reason": "numbered_tutorial_revision_and_next_entry",
                }
            ],
            "labels_or_evidence_types_read": False,
        }
        baseline = common.exact_final_corpus_identity_residue_scan(
            [{"seller_uid": seller_uid, "model_text": text}],
            {seller_uid: []},
            fail_on_residue=False,
        )
        self.assertEqual(baseline["pattern_residue_count"], 1)

        observed = common.exact_final_corpus_identity_residue_scan(
            [{"seller_uid": seller_uid, "model_text": text}],
            {seller_uid: []},
            fixed_content_collision_contract=contract,
        )
        self.assertEqual(observed["status"], "pass")
        self.assertEqual(observed["pattern_residue_count"], 0)
        self.assertEqual(
            observed["pinned_non_identity_pattern_collision_count"], 1
        )
        self.assertEqual(
            observed[
                "independent_pattern_match_count_including_pinned_content_collisions"
            ],
            1,
        )

        with self.assertRaisesRegex(ValueError, "replay drift"):
            common.exact_final_corpus_identity_residue_scan(
                [
                    {
                        "seller_uid": seller_uid,
                        "model_text": text.replace("026", "027"),
                    }
                ],
                {seller_uid: []},
                fixed_content_collision_contract=contract,
                fail_on_residue=False,
            )

    def test_final_identity_scan_censuses_fuzzy_new_adjacency_separately(self):
        observed = common.exact_final_corpus_identity_residue_scan(
            [
                {
                    "seller_uid": "seller-one",
                    "model_text": "Become a seller ordinary products",
                }
            ],
            {"seller-one": []},
            contextual_alias_tokens={"ordinarry"},
            contextual_alias_deletion_registry={"ordinary"},
        )
        self.assertEqual(observed["status"], "pass")
        self.assertEqual(observed["context_gated_known_alias_residue_count"], 1)
        self.assertEqual(
            observed["context_gated_exact_known_alias_residue_count"], 0
        )
        self.assertEqual(
            observed[
                "post_redaction_one_character_omission_collision_census_count"
            ],
            1,
        )
        self.assertEqual(observed["total_residue_count"], 0)

    def test_local_contextual_collision_scan_and_uncued_census_are_separate(self):
        rows = [
            {
                "seller_uid": "seller-one",
                "model_text": "Master Kush",
                "multiplicity": 3,
            },
            {
                "seller_uid": "seller-one",
                "model_text": "Welcome to Master shop",
                "multiplicity": 2,
            },
        ]
        observed = common.exact_final_corpus_identity_residue_scan(
            rows,
            {"seller-one": []},
            seller_contextual_collision_tokens_by_uid={
                "seller-one": {"master"}
            },
            fail_on_residue=False,
        )
        self.assertEqual(
            observed["seller_local_context_gated_collision_residue_count"], 1
        )
        census = common.seller_local_collision_residual_census(
            [rows[0]], {"seller-one": {"master"}}
        )
        self.assertEqual(census["retained_unique_text_occurrence_count"], 1)
        self.assertEqual(census["retained_source_weighted_occurrence_count"], 3)
        self.assertFalse(census["unknown_or_ambiguous_identifier_absence_proven"])

    def test_weighted_metrics_replay_unweighted_metrics_for_unit_weights(self):
        labels = np.asarray([0, 1, 0, 1, 1, 0], dtype=np.int8)
        scores = np.asarray([0.1, 0.8, 0.4, 0.4, 0.9, 0.2], dtype=np.float64)
        weights = np.ones(len(labels), dtype=np.float64)
        self.assertAlmostEqual(
            selector.weighted_roc_auc(labels, scores, weights),
            selector.solver.roc_auc(labels, scores),
            places=14,
        )
        self.assertTrue(
            0.0 <= selector.trapezoidal_pr_auc(labels, scores, weights) <= 1.0
        )

    def test_ranking_requires_both_positive_and_negative_candidates(self):
        rows = [
            {"seller_uid_left": "a", "seller_uid_right": "b"},
            {"seller_uid_left": "a", "seller_uid_right": "c"},
        ]
        result = selector.strict_ranking_metrics(
            rows,
            np.asarray([1, 0], dtype=np.int8),
            np.asarray([0.8, 0.2], dtype=np.float64),
        )
        self.assertEqual(result["eligible_query_count"], 1)
        self.assertGreaterEqual(result["excluded_no_negative_count"], 1)

    def test_adaptive_l2_tuning_is_component_isolated_and_converges(self):
        rows = []
        for index in range(16):
            positive = index % 2 == 0
            rows.append(
                {
                    "pair_uid": f"p{index:02d}",
                    "component_id": f"c{index:02d}",
                    "seller_uid_left": f"l{index:02d}",
                    "seller_uid_right": f"r{index:02d}",
                    "review_label": "positive" if positive else "negative",
                    "x": 2.0 if positive else -2.0,
                }
            )
        result = selector.tune_l2(
            self.policy,
            FakeFeatureFactory(),
            rows,
            ["x"],
            fold_count=4,
            fold_seed=20260722,
        )
        self.assertGreater(result["selected_l2_penalty"], 0.0)
        self.assertTrue(np.all(np.isfinite(result["oof_scores"])))
        self.assertEqual(len(result["fold_diagnostics"]), 4)
        self.assertGreaterEqual(result["formal_fit_count"], 28)

    def test_grouped_bootstrap_identical_scores_has_zero_delta(self):
        rows = []
        for index in range(8):
            rows.append(
                {
                    "component_id": f"c{index}",
                    "review_label": "positive" if index % 2 == 0 else "negative",
                }
            )
        scores = np.linspace(0.1, 0.8, len(rows))
        result = selector.grouped_bootstrap_delta(
            rows,
            scores,
            scores.copy(),
            resamples=31,
            seed=7,
            confidence=0.95,
        )
        self.assertEqual(result["observed_delta"], 0.0)
        self.assertEqual(result["ci_lower"], 0.0)
        self.assertEqual(result["ci_upper"], 0.0)
        self.assertEqual(result["probability_delta_above_zero"], 0.0)

        simultaneous = selector.grouped_bootstrap_winner_above_all(
            rows,
            {"winner": scores, "other": scores.copy()},
            "winner",
            resamples=31,
            seed=8,
            confidence=0.95,
        )
        self.assertEqual(
            simultaneous[
                "probability_winner_strictly_above_all_candidates"
            ],
            0.0,
        )
        self.assertEqual(
            simultaneous["observed_minimum_delta_above_any_competitor"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
