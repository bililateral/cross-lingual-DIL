"""Hand-checkable training intervention and gate checks; no project label reads."""
from __future__ import annotations

from collections import Counter
import copy
import csv
import itertools
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import step28_continual_cause_train as trial
import step28_continual_cause_probe as probe
import step28_continual_population_data as data
import test_step28_continual_cause_probe_contracts as fixture


class Contracts(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture.Contracts()
        self.fixture.setUp()
        self.catalog = self.fixture.catalog

    def group(self):
        g = self.fixture.group()
        # Eight pairs and four triples give 20 positive edges, using toy indices.
        owners = [i // 2 for i in range(16)] + [8 + i // 3 for i in range(12)]
        labels = tuple(int(owners[i] == owners[j]) for i, j in itertools.combinations(range(28), 2))
        return data.Group(g.uid, g.sellers, g.items, labels)

    def test_direct_original_cycle_preserves_recipient_labels_and_content(self):
        g = self.group()
        changed, record = trial.reassign(g, self.catalog)
        self.assertIs(changed.labels, g.labels)
        self.assertEqual(changed.sellers, g.sellers)
        styles = [probe.decode(rows[0][1], rows[0][2], self.catalog)[0] for rows in g.items]
        selected = [probe.decode(rows[0][1], rows[0][2], self.catalog)[0] for rows in changed.items]
        expected_offset = data.seed_for(20260910, g.uid, "style_cycle") % 27 + 1
        self.assertEqual(record["offset"], expected_offset)
        self.assertEqual(selected, [styles[(i + expected_offset) % 28] for i in range(28)])
        self.assertEqual(Counter(selected), Counter(styles))
        for old, new in zip(g.items, changed.items):
            self.assertEqual([r[0] for r in old], [r[0] for r in new])
            for a, b in zip(old, new):
                _, before = probe.decode(a[1], a[2], self.catalog)
                _, after = probe.decode(b[1], b[2], self.catalog)
                self.assertEqual({k: v for k, v in before.items() if k != "service"},
                                 {k: v for k, v in after.items() if k != "service"})
        # The previous inference view included a rename; it must not be used here.
        old_views, _ = probe.views(g, self.catalog)
        self.assertNotEqual(old_views["reassigned"].items, changed.items)
        self.assertEqual(trial.reassign(g, self.catalog), (changed, record))

    def test_manipulation_counts_against_independent_pair_loop(self):
        g = self.group()
        changed, _ = trial.reassign(g, self.catalog)
        actual = trial.manipulation([g], [changed], self.catalog)
        for name, item in (("original", g), ("reassigned", changed)):
            styles = [probe.decode(rows[0][1], rows[0][2], self.catalog)[0] for rows in item.items]
            expected = [[0] * 8 for _ in range(2)]
            for label, (i, j) in zip(g.labels, itertools.combinations(range(28), 2)):
                bits = "".join("1" if a == b else "0" for a, b in zip(styles[i], styles[j]))
                expected[label][int(bits, 2)] += 1
            self.assertEqual(actual["tables"][name][0], expected)
        with self.assertRaisesRegex(ValueError, "supervision moved"):
            trial.manipulation([g], [data.Group(changed.uid, changed.sellers, changed.items, None)], self.catalog)

    def test_relation_change_does_not_guarantee_label_association_weakening(self):
        # A cyclic label graph is invariant to any fixed account cycle.
        g = self.group()
        labels = tuple(int((j - i) % 28 in (1, 27)) for i, j in itertools.combinations(range(28), 2))
        changed, _ = trial.reassign(g, self.catalog)
        left, right = np.triu_indices(28, 1)
        tables = []
        codes_list = []
        for item in (g, changed):
            styles = np.asarray([probe.decode(rows[0][1], rows[0][2], self.catalog)[0] for rows in item.items])
            codes = (styles[left] == styles[right]).astype(int) @ np.asarray([4, 2, 1])
            table = np.zeros((2, 8), dtype=int)
            np.add.at(table, (labels, codes), 1)
            tables.append(table)
            codes_list.append(codes)
        self.assertFalse(np.array_equal(*codes_list))
        np.testing.assert_array_equal(*tables)

    def test_first_domain_contrast_sign_and_exclusion_by_hand(self):
        arrays = {}
        for i, order in enumerate(probe.ORDERS):
            for key in trial.POINTS:
                values = np.full((60, 1), 999.0)
                amount = 10 if key == "shared" else 13 if key.startswith("original") else 11
                values[i * 20:(i + 1) * 20] = amount + i
                arrays[(order, key)] = values
        results = trial.contrasts(arrays, 0)
        np.testing.assert_array_equal(results["original_first_domain_change_stage3"], np.full((3, 20), 3))
        np.testing.assert_array_equal(results["reassigned_first_domain_change_stage3"], np.ones((3, 20)))
        np.testing.assert_array_equal(results["original_minus_reassigned_first_domain_change_stage3"], np.full((3, 20), 2))

    def test_complete_pair_alignment_and_duplicate_rejection(self):
        g = self.group()
        pairs = [{"group_uid": g.uid, "seller_uid_left": a, "seller_uid_right": b, "label": str(y)}
                 for (a, b), y in zip(itertools.combinations(g.sellers, 2), g.labels)]
        with tempfile.TemporaryDirectory(prefix="seller_alias_cause_train_") as folder:
            root = Path(folder)
            target = root / "train/supervision/pairs.csv"
            target.parent.mkdir(parents=True)
            with target.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(pairs[0]))
                writer.writeheader()
                writer.writerows(reversed(pairs))
            data.write_json(root / "manifest.json", {"files": {"train/supervision/pairs.csv": data.record(target, root)}})
            self.assertEqual(trial.attach_labels([data.Group(g.uid, g.sellers, g.items)], root, "train"), [g])
        with self.assertRaises(ValueError):
            data.align_labels(g, pairs[:-1] + [pairs[0]])
        with self.assertRaises(ValueError):
            trial.attach_labels([], Path("unused"), "heldout")

    def test_actual_public_train_valid_reader_cannot_open_supervision_or_test(self):
        original = probe.original_manifest()
        real_open = Path.open
        opened = []
        def guarded(path, *args, **kwargs):
            p = path.as_posix()
            if "/heldout/" in p or "/supervision/" in p or "owners" in p:
                raise AssertionError("Unapproved payload read")
            opened.append(p)
            return real_open(path, *args, **kwargs)
        with patch.object(Path, "open", guarded):
            groups, meta, checked = trial.public_inputs(original)
        self.assertEqual({k: len(v) for k, v in groups.items()}, {"train": 180, "development": 60})
        self.assertEqual(len(meta), 240)
        self.assertEqual(set(checked), {"groups.csv", "train/items.jsonl", "development/items.jsonl"})
        self.assertTrue(all(g.labels is None for rows in groups.values() for g in rows))
        self.assertTrue(opened)

    def test_failed_or_missing_results_gate_precedes_valid_access(self):
        with tempfile.TemporaryDirectory(prefix="seller_alias_cause_gate_") as folder:
            root = Path(folder)
            data.write_json(root / "failure.json", {})
            with patch.object(trial, "attach_labels", side_effect=AssertionError("Must not parse")) as attach:
                with self.assertRaisesRegex(ValueError, "Failed run"):
                    trial.validated_scores(root, {}, {}, [], {})
                attach.assert_not_called()

    def test_changed_model_or_adam_rejected_before_score_read(self):
        point = {"model_state_sha256": "same", "checkpoint": {"state_sha256": "new_adam"}}
        old = {"model_state_sha256": "same", "checkpoint": {"state_sha256": "old_adam"}}
        with self.assertRaisesRegex(ValueError, "model/Adam"):
            trial.same_original_point(point, old, Path("unused"))

    def test_current_sampling_matches_both_original_stages(self):
        original = probe.original_manifest()
        groups, meta, _ = trial.public_inputs(original)
        for old in original["orders"]:
            log = {"order": old["order"], "training": {}}
            for key in trial.POINTS:
                row = old["training"]["shared"] if key == "shared" else old["training"]["sequential"][int(key[-1]) - 2]
                row = copy.deepcopy(row)
                row["first_update_modules"] = {k: {"finite_nonzero_gradient": True, "parameters_changed": True} for k in ("encoder", "head")}
                log["training"][key] = row
            trial.validate_training(log, meta, original["config"], old)
            log["training"]["reassigned_stage3"]["current_dropout_stream"] += 1
            with self.assertRaisesRegex(ValueError, "schedule"):
                trial.validate_training(log, meta, original["config"], old)

    def test_evaluation_saved_arrays_match_independent_ap_with_one_toy_parse(self):
        from sklearn.metrics import average_precision_score
        g = self.group()
        labelled = [data.Group(f"toy_{i:02d}", g.sellers, g.items, g.labels) for i in range(60)]
        public = {"development": [data.Group(x.uid, x.sellers, x.items) for x in labelled]}
        rng = np.random.default_rng(451)
        scores = {(order, key): rng.normal(size=(60, 378)).astype(np.float32)
                  for order in probe.ORDERS for key in trial.POINTS}
        with tempfile.TemporaryDirectory(prefix="seller_alias_cause_evaluation_") as folder:
            root = Path(folder)
            out = root / "run"
            out.mkdir()
            data.write_json(out / "manifest.json", {"toy": True})
            (root / "reports").mkdir()
            destination = root / "reports" / "evaluation"
            with patch.object(data, "ROOT", root), patch.object(probe, "original_manifest", return_value={"config": {"data_root": "toy"}}), \
                    patch.object(trial, "public_inputs", return_value=(public, [], {})), \
                    patch.object(trial, "validated_scores", return_value=(scores, {"manipulation": {"toy": True}})), \
                    patch.object(trial, "attach_labels", return_value=labelled) as attach:
                result = trial.evaluate(out, destination)
            attach.assert_called_once()
            self.assertEqual(attach.call_args.args[2], "development")
            self.assertEqual(len(result["columns"]), 22)
            self.assertEqual(len(result["points"]), 15)
            arrays = {}
            for key, values in scores.items():
                saved = np.load(destination / result["points"]["_".join(key)]["file"]["path"], allow_pickle=False)
                expected = [average_precision_score(g.labels, row) for row in values]
                np.testing.assert_allclose(saved[:, 0], expected, rtol=0, atol=1e-15)
                self.assertEqual(saved.shape, (60, 22))
                arrays[key] = saved
            delta = trial.contrasts(arrays, 0)["original_minus_reassigned_first_domain_change_stage3"]
            actual = result["contrasts"]["average_precision"]["original_minus_reassigned_first_domain_change_stage3"]["mean"]
            self.assertAlmostEqual(actual, float(delta.mean()), places=15)
            self.assertEqual(result["label_parses"], {"development": 1, "train": 0, "heldout": 0, "owners": 0})

    def test_gate_failure_blocks_evaluator_before_destination_and_labels(self):
        with tempfile.TemporaryDirectory(prefix="seller_alias_cause_gate_") as folder:
            root = Path(folder)
            with patch.object(data, "ROOT", root), patch.object(probe, "original_manifest", return_value={}), \
                    patch.object(trial, "public_inputs", return_value=({}, [], {})), \
                    patch.object(trial, "validated_scores", side_effect=ValueError("missing point")), \
                    patch.object(trial, "attach_labels") as attach:
                with self.assertRaisesRegex(ValueError, "missing point"):
                    trial.evaluate(root / "run", root / "reports" / "new")
            attach.assert_not_called()
            self.assertFalse((root / "reports" / "new").exists())


if __name__ == "__main__":
    unittest.main()
