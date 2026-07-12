from __future__ import annotations

import csv
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import step5_freeze_silver_labels as step5  # noqa: E402


def load_supervision_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [
            row
            for row in csv.DictReader(handle)
            if row.get("usable_for_supervision") == "1"
            and row.get("split_name") in {"train", "valid", "test"}
        ]


class FixedSplitLeakageContractTests(unittest.TestCase):
    def test_seller_alias_and_component_sets_do_not_cross_splits(self) -> None:
        for filename in (
            "step5_en_frozen_silver_labels.csv",
            "step5_zh_target_strict_frozen_silver_labels.csv",
        ):
            rows = load_supervision_rows(ROOT / "reports" / filename)
            self.assertTrue(all(value == 0 for value in step5.split_seller_overlap_counts(rows).values()))
            self.assertTrue(all(value == 0 for value in step5.split_alias_overlap_counts(rows).values()))
            components = {
                split: {
                    row["split_component_id"]
                    for row in rows
                    if row["split_name"] == split and row.get("split_component_id")
                }
                for split in ("train", "valid", "test")
            }
            self.assertFalse(components["train"] & components["valid"])
            self.assertFalse(components["train"] & components["test"])
            self.assertFalse(components["valid"] & components["test"])


if __name__ == "__main__":
    unittest.main()
