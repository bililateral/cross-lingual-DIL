from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import unittest
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step7_v4_1_select_style_free_m0 as selection


REPORT_ROOT = (
    ROOT
    / "reports"
    / "step7_v4_1_style_free_classifier_selection"
    / "v1_20260724"
)
SUMMARY_PATH = REPORT_ROOT / "selection_summary.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class Step7V41FormalResultContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = read_json(SUMMARY_PATH)
        cls.train_lock = read_json(
            REPORT_ROOT / "train_only_selection_lock.json"
        )
        cls.blind_lock = read_json(
            REPORT_ROOT / "blind_valid_scoring_lock.json"
        )
        cls.artifacts = read_json(
            REPORT_ROOT / "final_train_model_artifacts.json"
        )

    def assert_self_hash(self, payload: dict, field: str) -> None:
        content = dict(payload)
        expected = content.pop(field)
        self.assertEqual(canonical_hash(content), expected)

    def assert_file_record(self, record: dict) -> None:
        path = ROOT / record["path"]
        self.assertTrue(path.is_file(), path)
        self.assertEqual(path.stat().st_size, int(record["size_bytes"]))
        self.assertEqual(sha256_file(path), record["sha256"])

    def test_self_hashes_and_frozen_implementation_match(self) -> None:
        self.assert_self_hash(
            self.summary, "summary_content_sha256"
        )
        self.assert_self_hash(
            self.train_lock, "lock_content_sha256"
        )
        self.assert_self_hash(
            self.blind_lock, "lock_content_sha256"
        )
        self.assert_self_hash(
            self.artifacts, "artifact_content_sha256"
        )
        policy_path = (
            ROOT / "schema" / "step7_v4_1_style_free_classifier_policy.json"
        )
        producer_path = (
            ROOT / "scripts" / "step7_v4_1_select_style_free_m0.py"
        )
        self.assertEqual(
            sha256_file(policy_path), self.summary["policy_sha256"]
        )
        self.assertEqual(
            sha256_file(producer_path),
            self.summary["producer_sha256"],
        )

    def test_all_output_and_input_file_records_match(self) -> None:
        for role, record in self.summary["outputs"].items():
            if role == "selected_models":
                for model_record in record.values():
                    self.assert_file_record(model_record)
            else:
                self.assert_file_record(record)
        for record in self.summary["pinned_input_audit"].values():
            self.assert_file_record(record)
        self.assertEqual(
            self.summary["train_selection_lock"], self.train_lock
        )
        self.assertEqual(
            self.summary["blind_scoring_lock"], self.blind_lock
        )
        self.assertEqual(
            self.blind_lock["train_selection_lock"],
            self.summary["outputs"]["train_selection_lock"],
        )
        self.assertEqual(
            self.blind_lock["model_artifacts"],
            self.summary["outputs"]["model_artifacts"],
        )

    def test_claim_boundary_and_style_exclusion_are_fail_closed(
        self,
    ) -> None:
        decision = self.summary["selection_decision"]
        self.assertEqual(
            decision["overall_selection_status"],
            "no_stable_unique_current_best_style_free_pipeline",
        )
        self.assertEqual(
            decision["transfer_capable_m0_status"],
            "no_transfer_capable_m0",
        )
        self.assertFalse(decision["formal_m0_certified"])
        self.assertFalse(self.summary["formal_m0_certified"])
        self.assertFalse(self.summary["historical_test_labels_read"])
        self.assertFalse(
            self.summary["valid_metrics_may_change_selection"]
        )
        audit = self.summary["style_free_data_audit"]
        self.assertFalse(
            audit["author_style_encoder_score_or_runtime_files_opened"]
        )
        self.assertEqual(audit["selectable_style_feature_count"], 0)
        forbidden = ("style_", "raw_pcm_", "raw_mstyle_")
        self.assertFalse(
            any(
                name.startswith(forbidden)
                for name in audit["retained_feature_names"]
            )
        )

    def check_oof_file(
        self,
        filename: str,
        ranking_key: str,
        results_key: str,
        expected_pair_count: int,
    ) -> dict[str, list[dict[str, str]]]:
        rows = read_csv(REPORT_ROOT / filename)
        ranking = self.summary[ranking_key]
        results = self.summary[results_key]
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            grouped[row["candidate_id"]].append(row)
        self.assertEqual(set(grouped), set(ranking))
        self.assertEqual(
            {len(candidate_rows) for candidate_rows in grouped.values()},
            {expected_pair_count},
        )
        reference = {
            (
                row["pair_uid"],
                row["component_id"],
                row["review_label"],
            )
            for row in grouped[ranking[0]]
        }
        self.assertEqual(len(reference), expected_pair_count)

        for candidate_id in ranking:
            candidate_rows = grouped[candidate_id]
            observed = {
                (
                    row["pair_uid"],
                    row["component_id"],
                    row["review_label"],
                )
                for row in candidate_rows
            }
            self.assertEqual(observed, reference)
            labels = np.asarray(
                [
                    int(row["review_label"] == "positive")
                    for row in candidate_rows
                ],
                dtype=np.int64,
            )
            probabilities = np.asarray(
                [
                    float(row["mean_repeated_nested_oof_probability"])
                    for row in candidate_rows
                ],
                dtype=np.float64,
            )
            self.assertTrue(np.all(np.isfinite(probabilities)))
            self.assertTrue(
                np.all((probabilities >= 0.0) & (probabilities <= 1.0))
            )
            component_counts = Counter(
                row["component_id"] for row in candidate_rows
            )
            weights = np.asarray(
                [
                    1.0 / component_counts[row["component_id"]]
                    for row in candidate_rows
                ],
                dtype=np.float64,
            )
            reported = results[candidate_id]["metrics"]
            comparisons = (
                (
                    average_precision_score(labels, probabilities),
                    reported["row"]["average_precision"],
                ),
                (
                    average_precision_score(
                        labels,
                        probabilities,
                        sample_weight=weights,
                    ),
                    reported["component_equal"]["average_precision"],
                ),
                (
                    roc_auc_score(labels, probabilities),
                    reported["row"]["roc_auc"],
                ),
                (
                    roc_auc_score(
                        labels,
                        probabilities,
                        sample_weight=weights,
                    ),
                    reported["component_equal"]["roc_auc"],
                ),
            )
            for observed_metric, reported_metric in comparisons:
                self.assertAlmostEqual(
                    observed_metric, reported_metric, places=12
                )
            seed_fields = [
                f"outer_seed_{seed}_oof_probability"
                for seed in (2026072201, 2026072202, 2026072203,
                             2026072204, 2026072205)
            ]
            for row in candidate_rows:
                repeated_mean = sum(
                    float(row[field]) for field in seed_fields
                ) / len(seed_fields)
                self.assertTrue(
                    math.isclose(
                        repeated_mean,
                        float(
                            row[
                                "mean_repeated_nested_oof_probability"
                            ]
                        ),
                        rel_tol=0.0,
                        abs_tol=2e-16,
                    )
                )
        return grouped

    def test_main_and_no_clone_oof_metrics_replay(self) -> None:
        main = self.check_oof_file(
            "train_nested_oof_predictions.csv",
            "train_only_candidate_ranking",
            "candidate_train_results",
            401,
        )
        no_clone = self.check_oof_file(
            "train_no_exact_clone_nested_oof_predictions.csv",
            "no_exact_clone_candidate_ranking",
            "no_exact_clone_candidate_train_results",
            286,
        )
        main_id = self.summary["train_only_candidate_ranking"][0]
        no_clone_id = self.summary[
            "no_exact_clone_candidate_ranking"
        ][0]
        main_rows = main[main_id]
        no_clone_rows = no_clone[no_clone_id]
        no_clone_pair_ids = {
            row["pair_uid"] for row in no_clone_rows
        }
        removed = [
            row
            for row in main_rows
            if row["pair_uid"] not in no_clone_pair_ids
        ]
        self.assertEqual(len(removed), 115)
        self.assertEqual(
            sum(row["review_label"] == "positive" for row in removed),
            89,
        )
        self.assertEqual(
            sum(row["review_label"] == "negative" for row in removed),
            26,
        )

    def test_blind_valid_scores_are_unchanged_after_labels_open(self) -> None:
        blind = read_csv(
            REPORT_ROOT / "valid_predictions.blind.no_labels.csv"
        )
        labelled = read_csv(
            REPORT_ROOT / "valid_predictions.labelled_diagnostic.csv"
        )
        self.assertEqual(len(blind), 151 * 22)
        self.assertEqual(len(labelled), 151 * 22)
        self.assertEqual(
            list(blind[0]), ["pair_uid", "candidate_id", "probability"]
        )
        self.assertNotIn("review_label", blind[0])
        self.assertNotIn("component_id", blind[0])
        for blind_row, labelled_row in zip(
            blind, labelled, strict=True
        ):
            self.assertEqual(
                (
                    blind_row["pair_uid"],
                    blind_row["candidate_id"],
                    blind_row["probability"],
                ),
                (
                    labelled_row["pair_uid"],
                    labelled_row["candidate_id"],
                    labelled_row["probability"],
                ),
            )

    def test_selected_joblib_metadata_is_label_safe(self) -> None:
        records = self.summary["outputs"]["selected_models"]
        self.assertEqual(
            set(records),
            {
                "lightgbm__legacy18_labse",
                "lightgbm__legacy18",
            },
        )
        for candidate_id, record in records.items():
            payload = joblib.load(ROOT / record["path"])
            self.assertEqual(
                payload["step"],
                "step7_v4_1_selected_style_free_pipeline",
            )
            self.assertEqual(
                payload["candidate"]["candidate_id"], candidate_id
            )
            self.assertFalse(
                payload["valid_label_values_read_for_fit_or_scoring"]
            )
            self.assertFalse(
                payload["historical_test_label_values_read"]
            )
            self.assertEqual(
                payload["policy_sha256"],
                self.summary["policy_sha256"],
            )
            self.assertEqual(
                payload["producer_sha256"],
                self.summary["producer_sha256"],
            )


if __name__ == "__main__":
    unittest.main()
