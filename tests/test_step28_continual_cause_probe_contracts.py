"""Expression intervention semantics and saved-result alignment; no truth reads."""
from __future__ import annotations

from collections import Counter
import itertools
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import step28_continual_cause_probe as probe
import step28_continual_population_data as data


class Contracts(unittest.TestCase):
    def setUp(self):
        self.catalog = data.read_json(probe.CATALOG)

    def fields(self, intro="主要提供"):
        product = self.catalog["topics"][0]["subcategories"][0]["products"][0]
        return {"product": product, "qualifier": self.catalog["qualifiers"][0],
                "intro": intro + product, "detail": self.catalog["topics"][0]["subcategories"][0]["details"][0],
                "packaging": self.catalog["packaging"][0], "delivery": self.catalog["delivery"][0],
                "service_position": 2}

    def group(self):
        ids = tuple(f"seller_{i:02d}" for i in range(28))
        items = []
        for i in range(28):
            style = (i % 3, (i // 3) % 3, i % 4)
            fields = self.fields()
            title, description = probe.render(style, fields, self.catalog)
            items.append(tuple((f"item_{i:02d}_{j}", title, description) for j in range(2 + i % 3)))
        return data.Group("toy_group", ids, tuple(items))

    def test_all_style_combinations_exactly_decode_ambiguous_intro(self):
        for style in itertools.product(range(3), range(3), range(4)):
            for intro in ("提供", "主要提供"):
                fields = self.fields(intro)
                title, description = probe.render(style, fields, self.catalog)
                decoded, parsed = probe.decode(title, description, self.catalog)
                self.assertEqual(decoded, style)
                self.assertEqual(parsed["intro"], fields["intro"])
                self.assertEqual(probe.render(decoded, parsed, self.catalog), (title, description))

    def test_interventions_preserve_content_and_control_relations(self):
        original = self.group()
        views, record = probe.views(original, self.catalog)
        self.assertIs(views["original"], original)
        self.assertGreater(record["pair_axis_relations_changed"], 0)
        observed = {}
        for name, group in views.items():
            self.assertEqual(group.sellers, original.sellers)
            self.assertIsNone(group.labels)
            styles = []
            for old, new in zip(original.items, group.items):
                self.assertEqual(len(old), len(new))
                current_styles = []
                for a, b in zip(old, new):
                    self.assertEqual(a[0], b[0])
                    _, f = probe.decode(a[1], a[2], self.catalog)
                    style, g = probe.decode(b[1], b[2], self.catalog)
                    self.assertEqual({k:v for k,v in f.items() if k != "service"},
                                     {k:v for k,v in g.items() if k != "service"})
                    current_styles.append(style)
                self.assertEqual(len(set(current_styles)), 1)
                styles.append(current_styles[0])
            observed[name] = styles
        self.assertEqual(Counter(observed["renamed"]), Counter(observed["reassigned"]))
        for i, j in itertools.combinations(range(28), 2):
            for axis in range(3):
                self.assertEqual(observed["original"][i][axis] == observed["original"][j][axis],
                                 observed["renamed"][i][axis] == observed["renamed"][j][axis])
        self.assertEqual(probe.views(original, self.catalog)[1], record)

    def test_unrecognized_text_and_inconsistent_account_rejected(self):
        with self.assertRaises(ValueError):
            probe.decode("unknown", "unknown", self.catalog)
        g = self.group()
        changed = list(g.items)
        changed[0] = (g.items[0][0], g.items[1][0])
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            probe.views(data.Group(g.uid, g.sellers, tuple(changed)), self.catalog)

    def test_first_domain_contrasts_and_interaction_by_hand(self):
        arrays = {}
        for oi, order in enumerate(probe.ORDERS):
            for role in probe.ROLES:
                for vi, view in enumerate(probe.VIEWS):
                    # Irrelevant domains must not dilute the selected domain's change.
                    x = np.zeros((60, 1))
                    x[oi*20:(oi+1)*20, 0] = 10 + (oi+1)*(1 if role == "sequential" else 0) + vi*(2 if role == "sequential" else 1)
                    arrays[(order+"_"+role, view)] = x
        contrasts = probe.contrast_rows(arrays, 0)
        np.testing.assert_array_equal(contrasts["retention_original"], np.repeat([[1], [2], [3]], 20, axis=1))
        np.testing.assert_array_equal(contrasts["reassignment_retention_interaction"], np.ones((3, 20)))

    def test_failed_probe_and_missing_points_stop_before_supervision(self):
        with tempfile.TemporaryDirectory(prefix="seller_alias_cause_") as folder:
            root = Path(folder)
            (root/"failure.json").write_text("{}", encoding="utf-8")
            with patch.object(data, "align_labels", side_effect=AssertionError("Truth must not be reached")) as parse:
                with self.assertRaisesRegex(ValueError, "Failed run"):
                    probe.validated_scores(root, {}, {}, [])
                (root/"failure.json").unlink()
                data.write_json(root/"manifest.json", {"status":probe.COMPLETE, "source_files":probe.sources(),
                    "run_sha256":probe.RUN_SHA, "inputs":{}, "groups":[], "label_parses":0,
                    "optimizer_updates":0, "points":{}, "total_seconds":1})
                with self.assertRaisesRegex(ValueError, "Incomplete"):
                    probe.validated_scores(root, {}, {}, [])
                parse.assert_not_called()

    def test_all_actual_valid_public_records_without_labels(self):
        original = probe.original_manifest()
        with patch.object(data, "align_labels", side_effect=AssertionError("No labels in preparation")):
            views, metadata, record = probe.inputs(original)
        self.assertEqual(len(metadata), 60)
        self.assertEqual(sum(len(rows) for group in views["original"] for rows in group.items), 6120)
        self.assertEqual(set(record["files"]), {"groups.csv", "development/items.jsonl"})
        self.assertEqual(len(set(record["view_sha256"].values())), 3)
        self.assertTrue(all(g.labels is None for groups in views.values() for g in groups))

    def test_changed_relations_can_preserve_label_association(self):
        # Swapping two isomorphic 14-account label subgraphs can change expression
        # equality on particular edges while preserving all label-conditioned counts.
        owners = [owner for owner, size in enumerate([2, 2, 2, 2, 3, 3] * 2) for _ in range(size)]
        combinations = list(itertools.product(range(3), range(3), range(4)))
        records = []
        for i, owner in enumerate(owners):
            title, desc = probe.render(combinations[owner], self.fields(), self.catalog)
            records.append(tuple((f"{i:02d}_{j}", title, desc) for j in range(2)))
        group = data.Group("isomorphic_halves", tuple(f"s{i:02d}" for i in range(28)), tuple(records))
        truth = np.asarray([[int(owners[i] == owners[j]) for i,j in itertools.combinations(range(28), 2)]], dtype=np.uint8)
        with patch.object(data, "seed_for", return_value=13):
            changed, record = probe.views(group, self.catalog)
        self.assertEqual(record["offset"], 14)
        self.assertGreater(record["pair_axis_relations_changed"], 0)
        counts = probe.manipulation_counts({k:[v] for k,v in changed.items()}, truth, self.catalog)
        original = counts["by_view"]["original"][0]
        self.assertEqual(original["positive_pattern_counts"], [0,0,0,0,0,0,0,20])
        self.assertEqual(sum(original["negative_pattern_counts"]), 358)
        self.assertEqual(counts["by_view"]["original"], counts["by_view"]["renamed"])
        self.assertEqual(counts["by_view"]["renamed"], counts["by_view"]["reassigned"])


if __name__ == "__main__":
    unittest.main()
