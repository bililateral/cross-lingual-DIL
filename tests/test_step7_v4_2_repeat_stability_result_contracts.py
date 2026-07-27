from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import unittest
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step7_v4_2_repeat_stability_audit as repeat


REPORT_ROOT = (
    ROOT
    / "reports"
    / "step7_v4_2_repeat_stability"
    / "v1_20260727"
)
SUMMARY_PATH = REPORT_ROOT / "repeat_stability_summary.json"
SEEDS = (
    2026072701,
    2026072702,
    2026072703,
    2026072704,
    2026072705,
)


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class Step7V42RepeatStabilityResultContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = json.loads(
            SUMMARY_PATH.read_text(encoding="utf-8")
        )

    def test_summary_self_hash_and_output_records(self) -> None:
        content = dict(self.summary)
        expected = content.pop("summary_content_sha256")
        self.assertEqual(canonical_hash(content), expected)
        self.assertEqual(
            sha256_file(
                ROOT
                / "schema"
                / "step7_v4_2_repeat_stability_policy.json"
            ),
            self.summary["policy_sha256"],
        )
        self.assertEqual(
            sha256_file(
                ROOT
                / "scripts"
                / "step7_v4_2_repeat_stability_audit.py"
            ),
            self.summary["producer_sha256"],
        )
        for record in self.summary["outputs"].values():
            path = ROOT / record["path"]
            self.assertTrue(path.is_file(), path)
            self.assertEqual(path.stat().st_size, record["size_bytes"])
            self.assertEqual(sha256_file(path), record["sha256"])

    def check_oof(
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
        self.assertEqual(len(ranking), 22)
        self.assertEqual(len(rows), expected_pair_count * 22)
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
            self.assertEqual(
                {
                    (
                        row["pair_uid"],
                        row["component_id"],
                        row["review_label"],
                    )
                    for row in candidate_rows
                },
                reference,
            )
            labels = np.asarray(
                [
                    int(row["review_label"] == "positive")
                    for row in candidate_rows
                ],
                dtype=np.int64,
            )
            probabilities = np.asarray(
                [
                    float(
                        row["mean_repeated_nested_oof_probability"]
                    )
                    for row in candidate_rows
                ],
                dtype=np.float64,
            )
            self.assertTrue(np.all(np.isfinite(probabilities)))
            self.assertTrue(
                np.all((probabilities >= 0.0) & (probabilities <= 1.0))
            )
            for row in candidate_rows:
                seed_mean = sum(
                    float(
                        row[
                            f"outer_seed_{seed}_oof_probability"
                        ]
                    )
                    for seed in SEEDS
                ) / len(SEEDS)
                self.assertTrue(
                    math.isclose(
                        seed_mean,
                        float(
                            row[
                                "mean_repeated_nested_oof_probability"
                            ]
                        ),
                        rel_tol=0.0,
                        abs_tol=2e-16,
                    )
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
            for observed, expected in comparisons:
                self.assertAlmostEqual(observed, expected, places=12)
            retrieval = repeat.strict_retrieval_metrics(
                [
                    {
                        "review_label": row["review_label"],
                        "seller_uid_left": row["pair_uid"].split(
                            "||", maxsplit=1
                        )[0],
                        "seller_uid_right": row["pair_uid"].split(
                            "||", maxsplit=1
                        )[1],
                    }
                    for row in candidate_rows
                ],
                probabilities,
            )
            self.assertEqual(
                retrieval,
                results[candidate_id]["retrieval_metrics_extended"],
            )
        return grouped

    def test_main_and_no_clone_metrics_replay(self) -> None:
        main = self.check_oof(
            "train_nested_oof_predictions.csv",
            "train_candidate_ranking",
            "train_candidate_results",
            401,
        )
        no_clone = self.check_oof(
            "train_no_exact_clone_nested_oof_predictions.csv",
            "no_exact_clone_candidate_ranking",
            "no_exact_clone_candidate_results",
            286,
        )
        main_pairs = {
            row["pair_uid"]
            for row in main[self.summary["train_candidate_ranking"][0]]
        }
        no_clone_rows = no_clone[
            self.summary["no_exact_clone_candidate_ranking"][0]
        ]
        no_clone_pairs = {row["pair_uid"] for row in no_clone_rows}
        removed = main_pairs - no_clone_pairs
        main_reference = main[
            self.summary["train_candidate_ranking"][0]
        ]
        removed_rows = [
            row for row in main_reference if row["pair_uid"] in removed
        ]
        self.assertEqual(len(removed_rows), 115)
        self.assertEqual(
            sum(
                row["review_label"] == "positive"
                for row in removed_rows
            ),
            89,
        )
        self.assertEqual(
            sum(
                row["review_label"] == "negative"
                for row in removed_rows
            ),
            26,
        )

    def test_recorded_selection_outcome_is_fail_closed(self) -> None:
        comparison = self.summary["comparison"]
        self.assertEqual(
            self.summary["train_candidate_ranking"][0],
            "lightgbm__legacy18_e5_labse",
        )
        self.assertEqual(
            self.summary["no_exact_clone_candidate_ranking"][0],
            "l2_logistic__legacy18",
        )
        self.assertFalse(
            comparison["primary_remains_main_aggregate_winner"]
        )
        self.assertEqual(comparison["primary_main_rank"], 2)
        self.assertEqual(comparison["primary_no_clone_rank"], 10)
        self.assertFalse(
            self.summary["selection_decision"][
                "overall_unique_current_best_gate_passed"
            ]
        )
        self.assertFalse(
            self.summary["selection_decision"][
                "transfer_capable_internal_gate_passed"
            ]
        )
        self.assertFalse(
            self.summary["selection_decision"]["formal_m0_certified"]
        )
        self.assertFalse(self.summary["old_valid_label_values_read"])
        self.assertFalse(
            self.summary["historical_test_label_values_read"]
        )


if __name__ == "__main__":
    unittest.main()
