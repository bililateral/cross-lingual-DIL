"""Only changed evaluation alignment/aggregation; synthetic CPU inputs."""
import csv
import io
import itertools
import unittest

import numpy as np

import step28_defaults_evaluate as e


def fixture():
    # Eight pairs and four triples: 28 accounts, 20 relevant pairs.
    sizes = [2]*8+[3]*4
    owners = [i for i, size in enumerate(sizes) for _ in range(size)]
    return np.array([owners[i] == owners[j] for i, j in itertools.combinations(range(28), 2)], np.uint8)


class Evaluation(unittest.TestCase):
    def test_all_22_and_undefined(self):
        y = fixture()[None, :]
        scores = np.where(y, -.5, -1.)
        values, counts, queries = e.evaluate_matrix(y, scores)
        # Perfect ranking can coexist with no positive threshold decisions.
        np.testing.assert_allclose(values[0, [0, 1, 2, 3]], 1.)
        np.testing.assert_allclose(queries[:, :, :2], 1.)
        self.assertAlmostEqual(queries[0, :, 2].mean(), 22/28)
        np.testing.assert_allclose(queries[:, :, 3:], 1.)
        self.assertTrue(np.isnan(values[0, 6]) and np.isnan(values[0, 11]))
        self.assertEqual(counts, [dict(tp=0, fp=0, fn=20, tn=358)])
        self.assertEqual(values[0, 7], 0.)
        self.assertEqual(values[0, 9], 1.)
        self.assertEqual(e.coverage(values)['precision']['undefined'], 1)

    def test_pooled_is_not_macro_or_zero_fill(self):
        counts = [dict(tp=0, fp=0, fn=20, tn=358), dict(tp=10, fp=2, fn=10, tn=356)]
        p = e.pooled(counts)
        self.assertAlmostEqual(p['metrics']['precision'], 10/12)
        self.assertAlmostEqual(p['metrics']['recall'], 10/40)
        self.assertAlmostEqual(p['metrics']['f1'], 20/52)
        self.assertAlmostEqual(p['metrics']['mcc'], (10*714-2*30)/np.sqrt(12*40*716*744))
        self.assertIsNone(e.pooled(counts[:1])['metrics']['precision'])
        self.assertIsNone(e.pooled(counts[:1])['metrics']['mcc'])

    def test_pair_alignment_rejects_duplicates_and_wrong_group(self):
        sellers = [f'a{i:02}' for i in range(28)]
        edges = list(itertools.combinations(sellers, 2)); truth = fixture()
        rows = [('g', a, b, str(int(y))) for (a, b), y in zip(edges, truth)][::-1]
        def payload(rs):
            f = io.StringIO(); w = csv.writer(f)
            w.writerow(['group_uid', 'seller_uid_left', 'seller_uid_right', 'label']); w.writerows(rs)
            return f.getvalue().encode()
        np.testing.assert_array_equal(e.labels_from_bytes(payload(rows), ['g'])[0], truth)
        for bad, ids in ((rows+rows[:1], ['g']), (rows, ['wrong']), (rows[:-1], ['g'])):
            with self.assertRaises(ValueError): e.labels_from_bytes(payload(bad), ids)

    def test_actual_domain_pairing_and_shared_bootstrap(self):
        cells = {(s, d): np.full((20, 22), s*10+i) for s in (1, 2, 3) for i, d in enumerate('ABC')}
        a = e.contribution('BCA', cells, 'A_acq'); o = e.contribution('BCA', cells, 'O')
        f = e.contribution('BCA', cells, 'F')
        np.testing.assert_allclose(f, a-o)
        self.assertEqual(o.mean(1).sum(0)[0], (31+32)/2)
        self.assertEqual(f.mean(1).sum(0)[0], ((11-31)+(22-32))/2)
        # Same paired observations must have exactly zero contrast interval,
        # even though individual group values have nonzero dispersion.
        noisy = np.arange(3*20*22).reshape(3,20,22).astype(float)
        draws = np.random.Generator(np.random.PCG64(2026110202)).integers(20,size=(5000,3,20))
        z = e.interval(noisy-noisy, draws)
        self.assertTrue(all(v == dict(mean=0., conditional_95pct_interval=[0.,0.]) for v in z.values()))
        noisy[0,0,6] = np.nan
        self.assertIsNone(e.interval(noisy,draws)['precision']['mean'])

    def test_full_summary_and_signed_pooled_fields(self):
        records, evaluated = {}, {}
        for name in e.METHODS:
            for order in e.ORDERS:
                for seed in e.SEEDS:
                    rid = f'{order}_s{seed}_{name}'
                    records[rid] = dict(spec=dict(name=name, order=order, seed=seed))
                    evaluated[rid] = {}
                    for stage in (1, 2, 3):
                        tp, fp = {1:(10,2), 2:(8,2), 3:(5,3)}[stage]
                        counts = [dict(tp=tp, fp=fp, fn=20-tp, tn=358-fp) for _ in range(60)]
                        values = np.full((60,22), float(stage))
                        if name == 'H_default' and stage == 3: values[0,6] = np.nan
                        evaluated[rid][stage] = {role:(values, counts, None) for role in e.ROLES}
        result = e.analyze(records, evaluated, lambda:None)
        self.assertEqual(result['absolute']['S_default']['raw']['O']['map']['mean'], 3.)
        self.assertEqual(result['absolute']['S_default']['raw']['F']['map']['mean'], -1.5)
        supplement = result['supplementary_classification']['S_default']['raw']['F']
        self.assertAlmostEqual(supplement['pooled_metrics']['precision'], 18/22-5/8)
        self.assertEqual(supplement['coverage']['precision']['total'], 720)
        cover = result['supplementary_classification']['H_default']['raw']['O']['coverage']['precision']
        self.assertEqual((cover['total'],cover['undefined']), (360,6))
        self.assertIsNone(result['absolute']['H_default']['raw']['O']['precision']['mean'])


if __name__ == '__main__':
    unittest.main()
