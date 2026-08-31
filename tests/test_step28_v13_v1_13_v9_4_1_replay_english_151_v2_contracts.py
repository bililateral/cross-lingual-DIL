from __future__ import annotations

import csv
import json
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

import step28_v13_v1_13_v9_4_1_model_training_common_v2 as common
import step28_v13_v1_13_v9_4_1_replay_english_151_v2 as replay
import step7_v3_1_source_data as step7_source


class ReplayEnglish151V2Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = common.load_policy()

    def test_label_free_replay_matches_all_four_frozen_hashes(self) -> None:
        result, rows = replay.run_replay(self.policy)
        self.assertEqual(result["status"], replay.STATUS)
        self.assertTrue(result["all_four_exact_matches"])
        self.assertEqual(result["valid_pair_count"], 151)
        self.assertEqual(len(rows), 151)
        self.assertEqual(len({row["opaque_pair_uid"] for row in rows}), 151)
        self.assertEqual(result["labels_or_identity_evidence_read"], 0)
        self.assertEqual(result["controller_or_membership_read"], 0)
        self.assertEqual(result["qrels_or_retrieval_truth_read"], 0)
        self.assertEqual(result["audit_truth_read"], 0)
        self.assertFalse(result["model_parameters_updated"])
        self.assertFalse(result["model_training_or_threshold_selection_performed"])

    def test_shared_profile_projection_uses_only_frozen_public_fields(self) -> None:
        row = {
            "seller_uid": "seller_01",
            "category_concat_top": "类别甲 || 类别乙",
            "signature_title_concat": "标题甲 || 标题乙",
            "title_concat_top": "标题乙 || 标题丙",
            "signature_description_concat": "描述甲",
            "description_concat_top": "描述乙 || 描述甲",
            "item_count": 3,
            "title_length_stats": {"median": 10},
            "description_length_stats": {"median": 20},
            "style_stats": {
                "digit_ratio_mean": 0.1,
                "punct_ratio_mean": 0.2,
                "repeated_title_share": 0.3,
                "repeated_description_share": 0.4,
                "max_category_share": 0.5,
            },
        }
        projected = replay.shared_base24.project_model_profile(row)
        self.assertEqual(projected["seller_uid"], "seller_01")
        self.assertEqual(set(projected["clean_titles"]), {"标题甲", "标题乙", "标题丙"})
        self.assertEqual(set(projected["clean_descriptions"]), {"描述甲", "描述乙"})
        self.assertNotIn("model_text", projected)

    def test_shared_legacy18_formula_matches_frozen_step7_formula(self) -> None:
        sellers = {}
        for seller_uid, shift in (("left", 0.0), ("right", 1.0)):
            sellers[seller_uid] = {
                "seller_uid": seller_uid,
                "clean_categories": ["shared", seller_uid],
                "clean_titles": ["same-title", seller_uid],
                "clean_descriptions": ["same-description", seller_uid],
                "source_dataset": "",
                "source_market_raw": "",
                "numeric_profile": {
                    name: float(index) + shift
                    for index, name in enumerate(step7_source.NUMERIC_PROFILE_FIELDS)
                },
            }
        reference = {
            "train_seller_count": 4,
            "title_df": {"same-title": 2},
            "description_df": {"same-description": 2},
            "numeric_references": {
                name: [float(value) for value in range(12)]
                for name in step7_source.NUMERIC_PROFILE_FIELDS
            },
        }
        pair = {
            "pair_uid": "pair_test",
            "seller_uid_left": "left",
            "seller_uid_right": "right",
        }
        predecessor = replay.predecessor_common.load_policy()
        feature_names = predecessor["feature_contract"]["legacy18"]
        expected_row = step7_source.build_safe_pair_rows(
            [pair], sellers, reference
        )[0]
        expected = np.asarray(
            [expected_row[name] for name in feature_names], dtype="<f8"
        )
        observed = replay.shared_base24.legacy18_row(
            pair, sellers, reference, feature_names
        )
        np.testing.assert_array_equal(observed, expected)

    def test_publish_is_immutable_and_validate_does_not_score(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "english_151_replay"
            with mock.patch.object(replay, "OUTPUT_ROOT", output):
                published = replay.publish(self.policy)
                with self.assertRaisesRegex(
                    common.ModelTrainingContractError, "already exists"
                ):
                    replay.publish(self.policy)
                with mock.patch.object(
                    replay, "run_replay", side_effect=AssertionError("must not score")
                ):
                    validated = replay.validate_output(self.policy)
            self.assertEqual(
                validated["canonical_self_hash"], published["canonical_self_hash"]
            )

    def test_tampered_prediction_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "english_151_replay"
            with mock.patch.object(replay, "OUTPUT_ROOT", output):
                replay.publish(self.policy)
                prediction = output / replay.PREDICTIONS_NAME
                prediction.write_bytes(prediction.read_bytes() + b"\n")
                with self.assertRaisesRegex(
                    common.ModelTrainingContractError, "size drift"
                ):
                    replay.validate_output(self.policy)

    def test_resigned_manifest_cannot_hide_probability_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "english_151_replay_attempt2"
            with mock.patch.object(replay, "OUTPUT_ROOT", output):
                replay.publish(self.policy)
                prediction = output / replay.PREDICTIONS_NAME
                with prediction.open("r", encoding="utf-8", newline="") as handle:
                    rows = list(csv.DictReader(handle))
                rows[0]["m0_probability"] = format(
                    float(rows[0]["m0_probability"]) + 1e-6,
                    ".17g",
                )
                replay._write_predictions(prediction, rows)
                manifest_path = output / replay.MANIFEST_NAME
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["prediction_file"]["size_bytes"] = prediction.stat().st_size
                manifest["prediction_file"]["sha256"] = common.sha256_file(prediction)
                manifest.pop("canonical_self_hash")
                manifest["canonical_self_hash"] = common.canonical_sha256(manifest)
                replay._write_json(manifest_path, manifest)
                with self.assertRaisesRegex(
                    common.ModelTrainingContractError,
                    "does not round-trip",
                ):
                    replay.validate_output(self.policy)

    def test_resigned_manifest_cannot_hide_opaque_row_reordering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "english_151_replay_attempt2"
            with mock.patch.object(replay, "OUTPUT_ROOT", output):
                replay.publish(self.policy)
                prediction = output / replay.PREDICTIONS_NAME
                with prediction.open("r", encoding="utf-8", newline="") as handle:
                    rows = list(csv.DictReader(handle))
                rows[0], rows[1] = rows[1], rows[0]
                replay._write_predictions(prediction, rows)
                manifest_path = output / replay.MANIFEST_NAME
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["prediction_file"]["size_bytes"] = prediction.stat().st_size
                manifest["prediction_file"]["sha256"] = common.sha256_file(prediction)
                manifest.pop("canonical_self_hash")
                manifest["canonical_self_hash"] = common.canonical_sha256(manifest)
                replay._write_json(manifest_path, manifest)
                with self.assertRaisesRegex(
                    common.ModelTrainingContractError,
                    "opaque pair order drift",
                ):
                    replay.validate_output(self.policy)

    def test_published_rows_expose_only_opaque_pair_ids_and_probabilities(self) -> None:
        _, rows = replay.run_replay(self.policy)
        self.assertEqual(
            set(rows[0]),
            {"opaque_pair_uid", "m0_probability", "c0_probability"},
        )
        for row in rows:
            self.assertRegex(row["opaque_pair_uid"], r"^pair_[0-9]{6}$")
            self.assertGreater(float(row["m0_probability"]), 0.0)
            self.assertLess(float(row["m0_probability"]), 1.0)
            self.assertGreater(float(row["c0_probability"]), 0.0)
            self.assertLess(float(row["c0_probability"]), 1.0)


if __name__ == "__main__":
    unittest.main()
