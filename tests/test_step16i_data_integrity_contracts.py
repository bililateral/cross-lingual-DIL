from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import step16i_audit_data_integrity as integrity  # noqa: E402
import step16i_prepare_retrospective_dev2 as dev2  # noqa: E402


class Step16IDataIntegrityContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.integrity_policy = json.loads(
            (ROOT / "schema" / "step16i_data_integrity_policy.json").read_text(
                encoding="utf-8"
            )
        )
        cls.dev2_policy = json.loads(
            (ROOT / "schema" / "step16i_retrospective_dev2_policy.json").read_text(
                encoding="utf-8"
            )
        )

    def test_v8_readiness_split_aliases_are_canonical(self) -> None:
        cfg = self.integrity_policy["v8_readiness_assignment_check"]
        self.assertEqual(integrity.normalize_readiness_split("train", cfg), "train")
        self.assertEqual(
            integrity.normalize_readiness_split("representative_valid", cfg), "valid"
        )
        self.assertEqual(
            integrity.normalize_readiness_split("internal_development_test", cfg), "test"
        )

    def test_integrity_and_dev2_default_manifest_paths_match(self) -> None:
        run_id = "step16i_integrity_20260716_v1"
        integrity_outputs = self.integrity_policy["outputs"]
        expected = Path(
            integrity_outputs["root_template"].format(run_id=run_id)
        ) / integrity_outputs["permanent_exclusion_manifest"]
        configured = Path(self.dev2_policy["inputs"]["permanent_exclusion_manifest"])
        self.assertEqual(configured, expected)

    def test_exclusion_csv_entity_id_is_loaded_by_entity_type(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            path = Path(temporary) / "exclusions.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["entity_type", "entity_id"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"entity_type": "pair", "entity_id": "pair:1"},
                        {"entity_type": "seller", "entity_id": "seller:1"},
                        {"entity_type": "seller_alias", "entity_id": "shared-alias"},
                        {"entity_type": "seller_component", "entity_id": "component:1"},
                    ]
                )
            loaded = dev2.load_permanent_exclusions(
                path, self.dev2_policy["permanent_exclusion"]
            )
        self.assertIn("pair:1", loaded["pair_uids"])
        self.assertIn("seller:1", loaded["seller_uids"])
        self.assertNotIn("component:1", loaded["seller_uids"])
        self.assertIn("shared-alias", loaded["seller_aliases"])
        self.assertIn("component:1", loaded["component_ids"])

    def test_cross_market_seller_aliases_are_normalized(self) -> None:
        self.assertEqual(integrity.normalize_seller_alias(" Shared-Alias "), "shared-alias")
        self.assertEqual(integrity.normalize_seller_alias("ＡＢＣ"), "abc")
        self.assertEqual(integrity.portable_seller_alias("/shop/441160"), "")
        self.assertEqual(integrity.portable_seller_alias("605194"), "")
        self.assertEqual(
            integrity.alias_from_seller_uid("source|market|seller_raw:Shared-Alias"),
            "shared-alias",
        )

    def test_cross_split_alias_reuse_is_reported_as_leakage(self) -> None:
        rows = [
            {
                "pair_uid": "train-pair",
                "split_name": "train",
                "split_component_id": "old-train",
                "split_component_size": "1",
                "seller_uid_left": "market-a|seller_raw:Shared-Alias",
                "seller_uid_right": "market-a|seller_raw:train-peer",
                "source_seller_raw_left": "Shared-Alias",
                "source_seller_raw_right": "train-peer",
                "review_label": "positive",
                "review_notes": "",
            },
            {
                "pair_uid": "valid-pair",
                "split_name": "valid",
                "split_component_id": "old-valid",
                "split_component_size": "1",
                "seller_uid_left": "market-b|seller_raw:shared-alias",
                "seller_uid_right": "market-b|seller_raw:valid-peer",
                "source_seller_raw_left": "shared-alias",
                "source_seller_raw_right": "valid-peer",
                "review_label": "positive",
                "review_notes": "",
            },
        ]
        _, summary, _, _ = integrity.audit_dataset(
            "synthetic", rows, {"train", "valid", "test"}, []
        )
        self.assertEqual(summary["leakage"]["seller_alias_cross_split_count"], 1)
        self.assertIs(summary["leakage"]["detected"], True)

    def _audit_synthetic_readiness(self, rows: list[dict[str, str]]) -> dict:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            path = Path(temporary) / "readiness.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            summary, _, _, _ = integrity.audit_v8_readiness(
                path,
                self.integrity_policy["v8_readiness_assignment_check"],
                {"train", "valid", "test"},
            )
        return summary

    def test_readiness_conservative_component_merge_is_warning_not_failure(self) -> None:
        rows = [
            {
                "pair_uid": "pair-ab",
                "seller_uid_left": "market|seller_raw:alpha",
                "seller_uid_right": "market|seller_raw:beta",
                "v7_split_name": "train",
                "split_component_id": "coarse-component",
            },
            {
                "pair_uid": "pair-cd",
                "seller_uid_left": "market|seller_raw:charlie",
                "seller_uid_right": "market|seller_raw:delta",
                "v7_split_name": "train",
                "split_component_id": "coarse-component",
            },
        ]
        summary = self._audit_synthetic_readiness(rows)
        self.assertIs(summary["passed"], True)
        self.assertEqual(summary["status"], "warning")
        self.assertEqual(summary["persisted_components_overmerging_recomputed_count"], 1)
        self.assertIs(summary["partition_is_safe_conservative_coarsening"], True)

    def test_readiness_connected_component_fragmentation_still_fails(self) -> None:
        rows = [
            {
                "pair_uid": "pair-ab",
                "seller_uid_left": "market|seller_raw:alpha",
                "seller_uid_right": "market|seller_raw:beta",
                "v7_split_name": "train",
                "split_component_id": "component-one",
            },
            {
                "pair_uid": "pair-bc",
                "seller_uid_left": "market|seller_raw:beta",
                "seller_uid_right": "market|seller_raw:charlie",
                "v7_split_name": "train",
                "split_component_id": "component-two",
            },
        ]
        summary = self._audit_synthetic_readiness(rows)
        self.assertIs(summary["passed"], False)
        self.assertEqual(summary["status"], "fail")
        self.assertEqual(summary["recomputed_components_split_across_persisted_count"], 1)

    def test_reviewer_queue_contract_hides_selection_and_model_fields(self) -> None:
        row = {
            "source_market_raw_left": "market-a",
            "source_market_raw_right": "market-b",
            "source_seller_raw_left": "left",
            "source_seller_raw_right": "right",
            "left_preview": "left evidence",
            "right_preview": "right evidence",
        }
        reviewer_row = dev2.reviewer_row(row, "blind-1", "[]")
        dev2.assert_reviewer_blinding(
            list(reviewer_row),
            self.dev2_policy["blinding"]["forbidden_field_fragments"],
        )
        self.assertNotIn("pair_uid", reviewer_row)
        self.assertNotIn("candidate_rank_score", reviewer_row)
        self.assertNotIn("review_label", reviewer_row)

    def test_reviewer_csv_contract_includes_review_index(self) -> None:
        base = dev2.reviewer_row({}, "blind-1", "[]")
        fields = list(base)
        row = dict(base)
        row["review_index"] = 1
        payload = dev2.render_csv([row], fields)
        self.assertIn(b"review_index", payload)

    def test_dev2_is_explicitly_retrospective_and_label_free(self) -> None:
        self.assertIs(self.dev2_policy["prospective_claim_allowed"], False)
        source = (ROOT / "scripts" / "step16i_prepare_retrospective_dev2.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"prospective_claim_allowed": False', source)
        self.assertNotIn('"review_label": "positive"', source)
        self.assertNotIn('"review_label": "negative"', source)


if __name__ == "__main__":
    unittest.main()
