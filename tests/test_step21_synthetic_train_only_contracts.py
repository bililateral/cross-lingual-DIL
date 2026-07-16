from __future__ import annotations

import csv
import json
import random
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import step21_build_synthetic_zh_train as step21  # noqa: E402
import step21_evaluate_synthetic_augmentation as step21_eval  # noqa: E402


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class Step21SyntheticTrainOnlyContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = json.loads(
            (ROOT / "schema" / "step21_synthetic_train_only_policy.json").read_text(
                encoding="utf-8"
            )
        )

    def test_policy_never_treats_synthetic_rows_as_benchmark_data(self) -> None:
        constraints = " ".join(self.policy["scientific_constraints"])
        self.assertIn("never enter Step5 frozen labels", constraints)
        self.assertTrue(self.policy["evaluation"]["forbid_valid_or_test_selection"])
        self.assertEqual(
            self.policy["evaluation"]["selection_scope"],
            "five_fold_grouped_oof_on_train_seller_components_only",
        )
        self.assertTrue(
            self.policy["weighting"]["equal_effective_weight_duplication_control_required"]
        )
        self.assertFalse(self.policy["generation"]["fabricate_identifiers"])
        self.assertFalse(self.policy["generation"]["fabricate_market_provenance"])

    def test_section_rotation_changes_order_without_swapping_field_values(self) -> None:
        fields = {
            "category_concat_top": "category",
            "title_concat_top": "title",
            "description_concat_top": "description",
        }
        transformed = step21.transform_sections(
            fields, "section_rotation", random.Random(3)
        )
        self.assertEqual(transformed, fields)
        self.assertNotEqual(list(transformed), list(fields))

    def test_synthetic_profile_clears_identifiers_and_provenance(self) -> None:
        parent = {
            "seller_uid": "real-seller",
            "source_dataset": "market_item.xlsx",
            "source_market_raw": "real-market",
            "source_seller_raw": "real-alias",
            "source_seller_id_raw": "42",
            "alias_normalized": "real-alias",
            "contact_type_count": 2,
            "contact_token_count_total": 2,
            "contact_signals": {"telegram": ["real_handle"]},
            "contact_concat_top": "real_handle",
            "structured_snapshot_examples": ["x"],
            "structured_snapshot_concat_top": "x",
        }
        output = step21.synthetic_profile(
            parent,
            "synthetic://step21/v1/track/p0000v00/left",
            {"title_concat_top": "clean title", "description_concat_top": "clean text"},
            "SYNTHETIC_TRAIN_ONLY",
        )
        self.assertEqual(output["data_bucket"], "zh_synthetic_train_only")
        self.assertEqual(output["source_market_raw"], "SYNTHETIC_TRAIN_ONLY")
        self.assertEqual(output["alias_normalized"], "")
        self.assertEqual(output["contact_token_count_total"], 0)
        self.assertEqual(output["contact_concat_top"], "")
        self.assertNotIn("real_handle", output["profile_text"])
        self.assertTrue(output["synthetic_train_only"])

    def test_two_sided_no_op_uses_deterministic_section_rotation(self) -> None:
        fields = {
            "category_concat_top": "category",
            "title_concat_top": "title",
            "description_concat_top": "description",
        }
        left, right, transform, left_changed, right_changed = step21.ensure_pair_text_change(
            fields,
            fields,
            dict(fields),
            dict(fields),
            random.Random(11),
            random.Random(12),
            "segment_subsample",
        )
        self.assertIn("fallback_section_rotation", transform)
        self.assertTrue(left_changed or right_changed)
        self.assertEqual(left, fields)
        self.assertEqual(right, fields)
        self.assertNotEqual(list(left), list(fields))

    def test_step5_compatible_label_is_train_only_and_not_benchmark_eligible(self) -> None:
        fields = [
            "pair_uid",
            "data_bucket",
            "candidate_scope",
            "review_status",
            "review_label",
            "reviewer_id",
            "review_notes",
            "usable_for_supervision",
            "usable_for_core_transfer",
            "split_name",
            "split_component_id",
            "seller_uid_left",
            "seller_uid_right",
            "source_market_raw_left",
            "source_market_raw_right",
            "source_seller_raw_left",
            "source_seller_raw_right",
            "same_market_raw",
            "benchmark_eligible",
            "silver_train_only",
            "training_sample_weight",
            "shared_contact_count",
            "shared_contact_values",
            "shared_pgp_fingerprint_count",
            "shared_pgp_fingerprint_values",
        ]
        parent = {field: "" for field in fields}
        parent.update(
            {
                "pair_uid": "parent",
                "split_name": "train",
                "split_component_id": "component-1",
            }
        )
        output = step21.label_row(
            parent,
            fields,
            "synthetic-pair",
            "synthetic-left",
            "synthetic-right",
            "SYNTHETIC_TRAIN_ONLY",
            0.25,
            "section_rotation",
        )
        self.assertEqual(set(output), set(fields))
        self.assertEqual(output["split_name"], "train")
        self.assertEqual(output["usable_for_core_transfer"], "0")
        self.assertEqual(output["benchmark_eligible"], "0")
        self.assertEqual(output["training_sample_weight"], "0.250000000000")

    def test_current_parent_tracks_use_only_final_v7_train_components(self) -> None:
        labels = load_csv(ROOT / self.policy["inputs"]["frozen_labels"])
        evidence = {
            row["pair_uid"]: row
            for row in load_csv(ROOT / self.policy["inputs"]["evidence_labels"])
        }
        assignments = {
            row["pair_uid"]: row
            for row in load_csv(ROOT / self.policy["inputs"]["component_assignments"])
        }
        observed = {}
        for track_name, track_cfg in self.policy["tracks"].items():
            parents = step21.select_parent_rows(
                labels,
                evidence,
                assignments,
                track_cfg,
                self.policy["eligibility"],
            )
            observed[track_name] = len(parents)
            self.assertTrue(parents)
            self.assertTrue(
                all(assignments[row["pair_uid"]]["split_name"] == "train" for row, _ in parents)
            )
        self.assertEqual(
            observed,
            {"primary_non_silver": 16, "sensitivity_silver_anchor": 85},
        )

    def test_grouped_oof_balances_labels_despite_the_large_train_component(self) -> None:
        labels = [
            row
            for row in load_csv(ROOT / self.policy["inputs"]["frozen_labels"])
            if row["split_name"] == "train"
            and row["review_label"] in {"positive", "negative"}
        ]
        assignments = {
            row["pair_uid"]: row
            for row in load_csv(ROOT / self.policy["inputs"]["component_assignments"])
        }
        rows = [
            {
                "pair_uid": row["pair_uid"],
                "review_label": row["review_label"],
                "v7_component_id": assignments[row["pair_uid"]]["recomputed_component_id"],
            }
            for row in labels
        ]
        folds = step21_eval.grouped_folds(rows, 5, 20260716)
        counts = {fold: {"positive": 0, "negative": 0} for fold in range(5)}
        for row in rows:
            counts[folds[row["v7_component_id"]]][row["review_label"]] += 1
        positives = [count["positive"] for count in counts.values()]
        totals = [count["positive"] + count["negative"] for count in counts.values()]
        # One immutable component contains 175 rows (11 positive / 164 negative),
        # so perfect five-fold balance is mathematically impossible.
        self.assertGreaterEqual(min(positives), 30)
        self.assertLessEqual(max(positives), 50)
        self.assertGreaterEqual(min(totals), 90)
        self.assertLessEqual(max(totals), 200)
        self.assertTrue(all(count["negative"] > 0 for count in counts.values()))


if __name__ == "__main__":
    unittest.main()
