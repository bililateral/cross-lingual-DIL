"""Focused, hand-checkable contracts for the approved new generator."""
from __future__ import annotations

import copy
import csv
import io
import itertools
import json
import random
import sys
import tempfile
import unittest
from collections import Counter
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import step28_continual_expression_data as gen
import step28_continual_expression_data_verify as verify


class Draws:
    def __init__(self, values, integer=0):
        self.values = iter(values)
        self.integer = integer

    def random(self):
        return next(self.values)

    def randrange(self, size):
        if not 0 <= self.integer < size:
            raise AssertionError("Bad test integer")
        return self.integer


class DatasetContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = gen.load_json(gen.POLICY)
        cls.catalog = gen.load_json(gen.CATALOG)
        cls.checker = verify.TextChecker(cls.catalog)

    def example(self):
        group, items, owners = gen.make_group(self.policy, self.catalog, "A", "train", 0)
        # Exercise actual JSON/CSV serialization and independent re-reading.
        items = [json.loads(gen.json_line(row)) for row in items]
        text = io.StringIO(newline="")
        writer = csv.DictWriter(text, gen.PAIR_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(gen.pair_rows(owners))
        pairs = list(csv.DictReader(io.StringIO(text.getvalue())))
        return group, items, owners, pairs

    def test_catalog_contract_and_real_pmf_reference(self):
        bounds = gen.check_contract(self.policy, self.catalog)
        self.assertLessEqual(bounds["maximum_title_characters"], 48)
        self.assertLessEqual(bounds["maximum_description_characters"], 720)
        reference = gen.ROOT / self.policy["item_count_reference"]["path"]
        self.assertEqual(gen.digest(reference), self.policy["item_count_reference"]["sha256"])
        self.assertEqual(gen.load_json(reference)["item_count_pmf"], self.policy["item_count_pmf"])
        self.assertAlmostEqual(sum(int(n) * p for n, p in self.policy["item_count_pmf"].items()),
                               3.602071005915)

    def test_weighted_draw_hand_boundaries_and_zero_weight(self):
        weights = [0, 1, 3]
        self.assertEqual(gen.weighted_index(Draws([0]), weights), 1)
        self.assertEqual(gen.weighted_index(Draws([0.249]), weights), 1)
        self.assertEqual(gen.weighted_index(Draws([0.25]), weights), 2)
        self.assertEqual(gen.weighted_index(Draws([0.99999]), weights), 2)
        for weights in ([0, 0], [-1, 2], [float("nan"), 1]):
            with self.assertRaises(ValueError):
                gen.weighted_index(Draws([0]), weights)

    def test_weighted_without_replacement_renormalizes(self):
        # First picks index 1; remaining weights 1,0,1 place the boundary at 0.5.
        self.assertEqual(gen.preference(Draws([0.5, 0.49]), [1, 2, 1]), (0, 1))
        self.assertEqual(gen.preference(Draws([0.5, 0.50]), [1, 2, 1]), (1, 2))

    def test_redraw_may_equal_parent_without_retry(self):
        self.assertEqual(gen.inherit_preference(Draws([0.9, 0.1, 0.1]),
                                               (0, 1), [1, 1], 0.75), (0, 1))
        self.assertEqual(gen.inherit_style(Draws([0.9], integer=1), 1, 3, 0.75), 1)
        self.assertEqual(gen.inherit_style(Draws([0.749]), 2, 3, 0.75), 2)
        self.assertEqual(gen.inherit_style(Draws([0.75], integer=0), 2, 3, 0.75), 0)

    def test_exact_independent_population_marginals(self):
        weights = [Fraction(3, 20)] * 4 + [Fraction(1, 20)] * 8
        unordered = {}
        item = [Fraction(0)] * 12
        for first, second in itertools.permutations(range(12), 2):
            probability = weights[first] * weights[second] / (1 - weights[first])
            key = tuple(sorted((first, second)))
            unordered[key] = unordered.get(key, 0) + probability
            item[first] += probability / 2
            item[second] += probability / 2
        self.assertEqual(sum(unordered.values()), 1)
        final = [Fraction(7, 10) * q + Fraction(3, 10) * p for q, p in zip(item, weights)]
        self.assertEqual([sum(final[i:i+4]) for i in (0, 4, 8)],
                         [Fraction(4761, 8075), Fraction(1657, 8075), Fraction(1657, 8075)])
        collision = sum(p * p for p in unordered.values())
        self.assertAlmostEqual(float(collision), 0.02653624591436705)
        same = Fraction(9, 16) + Fraction(7, 16) * collision
        self.assertAlmostEqual(float(same), 0.5741096075875356)

    def test_all_subcategories_and_style_axes_render_from_common_pools(self):
        for sub_index in range(12):
            for style in itertools.product(range(12), range(12), range(12)):
                title, description = gen.render(random.Random(8), self.catalog, sub_index, style)
                row = dict(group_uid="a"*32, seller_uid="b"*32, item_uid="c"*32,
                           title=title, description=description)
                topic, observed_sub, observed_style = self.checker.check(row)
                self.assertEqual((topic, observed_sub, observed_style),
                                 (sub_index // 4, sub_index, style))

    def test_hand_group_structure_all_pairs_and_queries(self):
        group, items, owners, pairs = self.example()
        result = verify.check_group(items, owners, pairs, self.checker)
        self.assertEqual((len(owners), len(pairs), sum(int(p["label"]) for p in pairs)),
                         (28, 378, 20))
        self.assertEqual(len(result["controllers"]), 12)
        self.assertEqual(group["items"], len(items))
        for seller in result["accounts"]:
            candidates = [p for p in pairs if seller in (p["seller_uid_left"], p["seller_uid_right"])]
            self.assertEqual(len(candidates), 27)
            self.assertIn(sum(int(p["label"]) for p in candidates), (1, 2))
        self.assertEqual(Counter(Counter(o["controller_uid"] for o in owners).values()), {2: 8, 3: 4})

    def test_independent_verifier_rejects_label_and_edge_errors(self):
        _, items, owners, pairs = self.example()
        wrong = copy.deepcopy(pairs)
        wrong[0]["label"] = str(1 - int(wrong[0]["label"]))
        duplicate = copy.deepcopy(pairs)
        duplicate[-1] = duplicate[0].copy()
        for bad in (wrong, duplicate, pairs[:-1]):
            with self.assertRaises(ValueError):
                verify.check_group(items, owners, bad, self.checker)

    def test_independent_verifier_rejects_ownership_text_and_identity_injection(self):
        _, items, owners, pairs = self.example()
        bad_owner = copy.deepcopy(owners)
        bad_owner[-1]["seller_uid"] = bad_owner[0]["seller_uid"]
        with self.assertRaises(ValueError):
            verify.check_group(items, bad_owner, pairs, self.checker)
        for kind in ("extra_field", "contact_value", "wrong_group", "missing_account"):
            bad = copy.deepcopy(items)
            if kind == "extra_field":
                bad[0]["controller_uid"] = owners[0]["controller_uid"]
            elif kind == "contact_value":
                bad[0]["description"] += " 联系标识: synthetic_test"
            elif kind == "wrong_group":
                bad[0]["group_uid"] = "0" * 32
            else:
                seller = bad[0]["seller_uid"]
                bad = [row for row in bad if row["seller_uid"] != seller]
            with self.assertRaises(ValueError, msg=kind):
                verify.check_group(bad, owners, pairs, self.checker)

    def test_deterministic_groups_and_independent_namespaces(self):
        first = gen.make_group(self.policy, self.catalog, "B", "train", 1)
        # Consuming an unrelated stream cannot perturb any generation stream.
        unrelated = gen.stream(20260910, "unrelated")
        for _ in range(100):
            unrelated.random()
        self.assertEqual(first, gen.make_group(self.policy, self.catalog, "B", "train", 1))
        all_accounts, all_controllers, all_items = set(), set(), set()
        for domain, split in itertools.product(("A", "B", "C"), gen.SPLITS):
            _, items, owners = gen.make_group(self.policy, self.catalog, domain, split, 1)
            for seen, current in (
                (all_accounts, {row["seller_uid"] for row in owners}),
                (all_controllers, {row["controller_uid"] for row in owners}),
                (all_items, {row["item_uid"] for row in items}),
            ):
                self.assertFalse(seen & current)
                seen.update(current)

    def test_public_id_permutation_does_not_control_text_or_labels(self):
        _, original_items, original_owners = gen.make_group(self.policy, self.catalog, "A", "train", 0)
        original_stream = gen.stream

        class Reverse:
            def shuffle(self, values):
                values.reverse()

        def changed_stream(seed, *parts):
            return Reverse() if parts[-1] == "id_assignment" else original_stream(seed, *parts)

        with patch.object(gen, "stream", side_effect=changed_stream):
            _, changed_items, changed_owners = gen.make_group(self.policy, self.catalog, "A", "train", 0)

        def controller_text(items, owners):
            lookup = {r["seller_uid"]: r["controller_uid"] for r in owners}
            return Counter((lookup[r["seller_uid"]], r["item_uid"], r["title"], r["description"])
                           for r in items)
        self.assertNotEqual(original_items, changed_items)
        self.assertEqual(controller_text(original_items, original_owners),
                         controller_text(changed_items, changed_owners))

    def test_byte_budget_and_refuse_existing_output(self):
        row = dict(group_uid="a"*32, seller_uid="b"*32, item_uid="c"*32,
                   title="中"*48, description="中"*720)
        self.assertLessEqual(len(gen.json_line(row).encode("utf-8")) - 768*3, 256)
        bound = 80640*(768*3+256) + 10080*192 + 136080*128 + 360*256 + 4*1024**2
        self.assertEqual(bound, 230078464)
        self.assertLess(bound, self.policy["output_budget_bytes"])
        with tempfile.TemporaryDirectory(prefix="seller_alias_dataset_contract_") as tmp:
            with self.assertRaisesRegex(ValueError, "already exists"):
                gen.build_dataset(Path(tmp), gen.POLICY, gen.CATALOG,
                                  gen.ROOT / self.policy["item_count_reference"]["path"])
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_public_override_boundary_and_independent_redraw(self):
        self.assertEqual(gen.public_override(Draws([0.099], 2), 11, [0, 1, 2], .1), 2)
        self.assertEqual(gen.public_override(Draws([0.1]), 11, [0, 1, 2], .1), 11)
        self.assertEqual(gen.public_override(Draws([0.999], 1), 11, [0, 1, 2], 1), 1)
        self.assertEqual(gen.public_override(Draws([0]), 11, [0, 1, 2], 0), 11)
        # Public override may reproduce a personal style; it must not retry.
        self.assertEqual(gen.public_override(Draws([0], 1), 1, [0, 1, 2], .6), 1)

    def test_exact_enumeration_of_same_and_different_controller_agreement(self):
        for q in (Fraction(0), Fraction(1, 10), Fraction(3, 10), Fraction(3, 5), Fraction(1)):
            conditional = [[(1-q) * (Fraction(3, 4) * int(style == parent) + Fraction(1, 48))
                            + q * Fraction(int(style < 3), 3) for style in range(12)]
                           for parent in range(12)]
            self.assertTrue(all(sum(row) == 1 for row in conditional))
            positive = sum(sum(p*p for p in row) for row in conditional) / 12
            marginal = [sum(row[s] for row in conditional) / 12 for s in range(12)]
            negative = sum(p*p for p in marginal)
            result = verify.axis_agreement(float(q))
            self.assertAlmostEqual(float(positive), result['same_controller'])
            self.assertAlmostEqual(float(negative), result['different_controller'])
            self.assertEqual(sum(marginal[:3]), Fraction(1, 4) + Fraction(3, 4)*q)
        self.assertEqual(verify.axis_agreement(1)['difference'], 0)

    def test_public_change_preserves_ownership_counts_products_and_account_style(self):
        original = gen.make_group(self.policy, self.catalog, 'A', 'train', 0)
        changed_policy = copy.deepcopy(self.policy)
        changed_policy['public_style_probability']['A'] = [1, 1, 1]
        changed = gen.make_group(changed_policy, self.catalog, 'A', 'train', 0)
        self.assertEqual(original[0], changed[0])
        self.assertEqual(original[2], changed[2])
        self.assertEqual(list(gen.pair_rows(original[2])), list(gen.pair_rows(changed[2])))
        def product_rows(items):
            return [(r['seller_uid'], r['item_uid'], self.checker.titles[r['title']][0]) for r in items]
        self.assertEqual(product_rows(original[1]), product_rows(changed[1]))
        self.assertNotEqual(original[1], changed[1])
        result = verify.check_group(changed[1], changed[2], list(gen.pair_rows(changed[2])), self.checker)
        self.assertTrue(all(all(axis < 3 for axis in style) for style in result['styles'].values()))

    def test_contract_rejects_unconfirmed_generation_parameters(self):
        for key, value in (('root_seed', 1), ('style_axis_sizes', [3, 3, 4]),
                           ('public_style_indices', [[1, 2, 3]]*3),
                           ('public_style_probability', {'A': [.1, .3, .6]})):
            policy = copy.deepcopy(self.policy)
            policy[key] = value
            with self.assertRaises(ValueError, msg=key):
                gen.check_contract(policy, self.catalog)

    def test_standalone_verifier_rejects_theory_constant_mismatch(self):
        verify.check_theory_constants(self.policy)
        for key in ('preference_inheritance', 'preferred_item_probability'):
            policy = copy.deepcopy(self.policy)
            policy[key] = .5
            with self.assertRaisesRegex(ValueError, 'Theory constants differ'):
                verify.check_theory_constants(policy)


if __name__ == "__main__":
    unittest.main()
