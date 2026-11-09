"""Incremental CPU checks: exact graph recall and historical-use role isolation."""
import random
import re
import unittest

from step29_vendorlink_data import Ad
from step34_vendorlink_account_data import components, allocate, normalized_text


class AccountDataTest(unittest.TestCase):
    def test_exact_graph_matches_brute_force(self):
        rng = random.Random(41)
        ads = []
        for block in range(4):
            base = [f'w{block}_{i}' for i in range(90)]
            for n in range(12):
                values = base.copy()
                for _ in range(n % 5):
                    values[rng.randrange(len(values))] = 'changed'+str(n)
                ads.append(Ad(f'{block}_{n:02d}', 'header', ' '.join(values)))
        ads.extend([Ad('short1', 'one', 'two'), Ad('short2', 'ONE', 'two'),
                    Ad('short3', 'one', 'three'), Ad('unicode', '标题', '文字 示例')])
        links = {a.uid: {a.uid} for a in ads}
        for i, a in enumerate(ads):
            ta = re.findall(r'\w+', normalized_text(a))
            sa = {tuple(ta[k:k+5]) for k in range(len(ta)-4)}
            for b in ads[:i]:
                tb = re.findall(r'\w+', normalized_text(b))
                sb = {tuple(tb[k:k+5]) for k in range(len(tb)-4)}
                close = (10*len(sa & sb) >= 9*len(sa | sb) if len(ta) >= 5 and len(tb) >= 5
                         else normalized_text(a) == normalized_text(b))
                if close:
                    links[a.uid].add(b.uid)
                    links[b.uid].add(a.uid)
        expected, remaining = [], set(links)
        while remaining:
            seen, todo = set(), [min(remaining)]
            while todo:
                x = todo.pop()
                if x not in seen:
                    seen.add(x)
                    todo.extend(links[x]-seen)
            remaining -= seen
            expected.append(sorted(seen))
        actual, comparisons = components(ads)
        self.assertEqual(actual, sorted(expected, key=lambda x: x[0]))
        self.assertGreater(comparisons, 0)

    def test_contamination_and_fixed_roles(self):
        parts = [[f'{i:02d}'] for i in range(25)]
        # The minimum-UID representative is unused, but its component touches U.
        parts[0].append('old-used')
        used = {'old-used'}
        global_used = used | {'01', '02'}
        result = allocate(parts, used, global_used, 'Dream', 'account')
        self.assertEqual({k: len(v) for k, v in result.items()},
                         {'fit': 12, 'gallery': 4, 'dev': 2, 'confirm': 2})
        held = set(result['gallery']+result['dev']+result['confirm'])
        self.assertFalse(held & {'00', '01', '02', 'old-used'})
        self.assertEqual(result, allocate(list(reversed(parts)), used, global_used, 'Dream', 'account'))
        self.assertEqual(len(set(sum(result.values(), []))), 20)
        self.assertIsNone(allocate(parts[:19], used, global_used, 'Dream', 'account'))
        self.assertIsNone(allocate(parts, used, {p[0] for p in parts[:18]}, 'Dream', 'account'))


if __name__ == '__main__':
    unittest.main()
