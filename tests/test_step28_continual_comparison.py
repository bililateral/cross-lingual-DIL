"""Handwritten CPU evidence only; run on authorized Linux py310, never Windows."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import step28_continual_comparison as m
import step28_continual_comparison_evaluate as ev
import step28_continual_comparison_run as run


class Tokenizer:
    def encode(self, text, **_):
        return [1+int(hashlib.sha256(word.encode()).hexdigest()[:8], 16) % 63 for word in text.split()]

    def __call__(self, texts, **_):
        rows = [[0, *self.encode(t)] for t in texts]
        length = max(map(len, rows))
        return {'input_ids': torch.tensor([r+[0]*(length-len(r)) for r in rows]),
                'attention_mask': torch.tensor([[1]*len(r)+[0]*(length-len(r)) for r in rows])}


class Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(64, 4)

    def forward(self, features):
        z = self.embedding(features['input_ids'])
        mask = features['attention_mask'][..., None]
        cls = (z*mask).sum(1)/mask.sum(1)
        return {'token_embeddings': cls[:, None, :]}


class Encoder(torch.nn.ModuleList):
    def __init__(self):
        super().__init__([Transformer()])
        self.tokenizer = Tokenizer()

    def forward(self, features):
        raw = self[0](features)['token_embeddings'][:, 0]
        return {'sentence_embedding': torch.nn.functional.normalize(raw, dim=1)}


def group(index=0):
    uid = f'handwritten{index:03d}'
    sellers = tuple(f'{uid}.a{i:02d}' for i in range(28))
    owners = [i//2 for i in range(16)]+[8+(i-16)//3 for i in range(16, 28)]
    items = tuple(tuple((f'{seller}.r{r}', f'title {index} {i} {r}', f'description {index} {i+r}')
                        for r in range(2)) for i, seller in enumerate(sellers))
    labels = tuple(int(owners[i] == owners[j]) for i in range(28) for j in range(i+1, 28))
    return m.data.Group(uid, sellers, items, labels)


def fixture(name):
    torch.manual_seed(7)
    spec = m.Spec(name, 'ABC', 4)
    method = m.Method(spec)
    model = m.core.build_model(Encoder(), 8, 5)
    if name == 'L2R_text_fixed':
        del model.head
    c = dict(input=dict(microbatch=16, token_budget=256), comparison_spec=spec.identity(),
             optimizer=dict(encoder_lr=1e-5, head_lr=.001, encoder_weight_decay=.01,
                            head_weight_decay=0., betas=[.9, .999], eps=1e-8, foreach=False))
    optimizer = (torch.optim.AdamW(model.encoder.parameters(), lr=1e-5, foreach=False)
                 if name == 'L2R_text_fixed' else m.core.make_optimizer(model, c))
    method.bind(model, method.metadata())
    return method, model, optimizer, c


class ComparisonTests(unittest.TestCase):
    def test_omp_ridge_zero_and_small_improvement(self):
        w, _ = m.gcr.nonnegative_omp(np.array([[2.]]), np.array([3.]), ['a'])
        self.assertAlmostEqual(w['a'], 6/4.5)
        d = np.array([[1., -1., 1., -1., 1., -1.]])
        w, log = m.gcr.nonnegative_omp(d, d.sum(1), list('abcdef'))
        self.assertEqual(w, {})
        self.assertEqual(log['objective'], 0.)
        d = np.full((1, 6), 1e-6)
        w, log = m.gcr.nonnegative_omp(d, d.sum(1), list('abcdef'))
        self.assertTrue(w)
        self.assertEqual(len(log['expansions']), 6)
        self.assertGreater(sum(w.values()), 0.)
        # One atom points away from the target, exercising an active zero bound.
        w, _ = m.gcr.nonnegative_omp(np.array([[1., -1.], [0., 1.]]), np.array([1., 0.]), ['a', 'b'])
        self.assertAlmostEqual(w['a'], 2/3)
        self.assertNotIn('b', w)

    def test_omp_keeps_tiny_first_step(self):
        d = np.array([[2., 1., 1., 2., 1., 1.], [1e6, -1e6, 0., 1e6, -1e6, 0.]])
        w, log = m.gcr.nonnegative_omp(d, d.sum(1), list('abcdef'))
        self.assertTrue(w)
        self.assertGreaterEqual(len(log['expansions']), 2)
        self.assertLess(log['objective'], 64/3)

    def test_pss_sign_and_fixed_reference_isd(self):
        q, p = np.array([1., 0.]), np.array([2., 0.])
        candidates = np.array([[1., 1.], [3., 1.], [2., -1.]])
        # Self is included: each first/second item has ISD 2/3, third has 4/3.
        np.testing.assert_allclose(m.l2r.priority(q, p, candidates), [.8+.2*2/3, -.8+.2*2/3, .2*4/3])
        with self.assertRaises(ValueError):
            m.l2r.priority(np.zeros(2), p, candidates)

    def test_gcr_full_head_gradient_and_absolute_replay_weight(self):
        _, model, _, c = fixture('ER_GCR_group_fixed')
        g = group()
        actual = m.gcr.head_gradient(model, g, c)
        z = m.five.record.record_vectors(model, g, c)
        table = m.five.record.pair_table(model, z)
        objective, _ = m.gcr.supervised(table, g)
        expected = torch.autograd.grad(.5*objective, list(model.head.parameters()))
        reference = torch.cat([x.flatten() for x in expected]).detach().numpy()
        np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=2e-7)
        model.zero_grad(set_to_none=True)
        m.gcr.backward(model, g, c, 17, .5)
        first = {n: p.grad.clone() for n, p in model.named_parameters()}
        self.assertGreater(float(first['encoder.0.embedding.weight'].norm()), 0.)
        model.zero_grad(set_to_none=True)
        m.gcr.backward(model, g, c, 17, 1.)
        for n, p in model.named_parameters():
            torch.testing.assert_close(p.grad, 2*first[n], rtol=2e-5, atol=2e-7)

    def test_gcr_real_update_admission_and_roundtrip(self):
        method, model, optimizer, c = fixture('ER_GCR_group_fixed')
        current = [group(i) for i in range(48)]
        memory = method.memory()
        stage = method.stage(1, current, memory, model)
        before = copy.deepcopy(model.state_dict())
        first = method.update(model, optimizer, stage, current[0], c, 1)
        self.assertNotIn('replay', first)
        self.assertEqual(stage.entered, {current[0].uid})
        self.assertTrue(any(not torch.equal(v, before[k]) for k, v in model.state_dict().items()))
        payload, transient = memory.to_bytes(), stage.transient()
        restored_memory = method.memory_from(payload)
        restored_stage = method.stage(1, current, restored_memory, model)
        restored_stage.restore_transient(transient)
        self.assertEqual(stage.draw(), restored_stage.draw())
        stage.after_update(current[0])
        self.assertEqual(len(stage.entered), 1)

    def test_l2r_multi_positive_real_update_and_false_negative_mask(self):
        method, model, optimizer, c = fixture('L2R_text_fixed')
        current = [group(i) for i in range(48)]
        memory = method.memory()
        stage = method.stage(1, current, memory, model)
        rows = stage.current[current[0].uid]
        queries = [(q, [p for p in rows if p.uid != q.uid and m.l2r.same_identity(q, p)]) for q in rows]
        loss, log = stage.block(model, c, queries, 1, 'current', lambda: None)
        self.assertEqual(log['positive_items'], 40)
        self.assertEqual(log['queries'], 28)
        by_id = stage.index.by_id
        for item in log['selections']:
            q = by_id[item['query']]
            self.assertTrue(all(uid == item['positive'] or not m.l2r.same_identity(q, by_id[uid])
                                for uid in item['denominator']))
        loss.backward()
        self.assertGreater(float(model.encoder[0].embedding.weight.grad.norm()), 0.)
        before = model.encoder[0].embedding.weight.detach().clone()
        result = method.update(model, optimizer, stage, current[0], c, 1)
        self.assertEqual(result['current']['positive_items'], 40)
        self.assertFalse(torch.equal(before, model.encoder[0].embedding.weight))
        self.assertEqual(method.score(model, current[:1], c).shape, (1, 378))
        # The scorer really uses raw CLS rather than normalized sentence output.
        z = m.l2r.encode(model, rows, c)
        i, j = torch.triu_indices(28, 28, 1)
        np.testing.assert_allclose(method.score(model, current[:1], c)[0],
                                   (z[i]*z[j]).sum(1).detach().numpy(), atol=1e-6)
        with self.assertRaisesRegex(ValueError, 'Not every current query'):
            stage.retain(model, c)

    def test_l2r_selected_text_changes_actual_ce_and_history(self):
        method, model, _, c = fixture('L2R_text_fixed')
        current = [group(i) for i in range(48)]
        stage = method.stage(1, current, method.memory(), model)
        q, p = stage.current[current[0].uid][:2]
        # Same candidate pool, a different selector, real text encoding and CE.
        pairs = [(q, [p])]
        loss, log = stage.block(model, c, pairs, 11, 'current', lambda: None)
        chosen = log['selections'][0]['new']
        with mock.patch.object(m.l2r, 'priority', side_effect=lambda *a: -m.l2r.isd(a[0], a[2], a[2])):
            changed_loss, changed = stage.block(model, c, pairs, 11, 'current', lambda: None)
        self.assertNotEqual(chosen, changed['selections'][0]['new'])
        first, = torch.autograd.grad(loss, [model.encoder[0].embedding.weight])
        second, = torch.autograd.grad(changed_loss, [model.encoder[0].embedding.weight])
        self.assertGreater(float((first-second).norm()), 1e-6)
        # A paid 14-package history exercises 0.1 historical CE through real input.
        old = [group(i) for i in range(100, 102)]
        rows = [a for g in old for a in m.l2r.accounts(g, 'A')]
        memory = method.memory()
        for q in rows[:14]:
            positive = next(a for a in rows if a.uid != q.uid and m.l2r.same_identity(a, q))
            negatives = [a for a in rows if not m.l2r.same_identity(a, q)][:10]
            memory.packages[q.uid] = dict(query=q, positive=positive, negatives=negatives)
        memory.stage, memory.seen = 1, 1344
        next_stage = method.stage(2, current, memory, model)
        queries = [(v['query'], [v['positive']]) for v in memory.packages.values()]
        model.zero_grad(set_to_none=True)
        reference, _ = next_stage.block(model, c, queries, 1, 'history', lambda: None)
        expected, = torch.autograd.grad(.1*reference, [model.encoder[0].embedding.weight])
        model.zero_grad(set_to_none=True)
        both = next_stage.backward(model, current[0], c, 1)
        combined = model.encoder[0].embedding.weight.grad.clone()
        model.zero_grad(set_to_none=True)
        current_rows = next_stage.current[current[0].uid]
        current_pairs = [(q, [p for p in current_rows if p.uid != q.uid and m.l2r.same_identity(q, p)]) for q in current_rows]
        current_loss, _ = next_stage.block(model, c, current_pairs, 1, 'current', lambda: None)
        current_loss.backward()
        torch.testing.assert_close(combined-model.encoder[0].embedding.weight.grad, expected, rtol=2e-5, atol=2e-6)
        self.assertEqual(both['history']['queries'], 14)
        self.assertGreater(float(expected.norm()), 1e-6)
        self.assertEqual(method.memory_from(memory.to_bytes()).to_bytes(), memory.to_bytes())
        self.assertEqual(memory.summary()['slots'], 168)

    def test_l2r_per_query_replacement_and_paid_memory(self):
        method, model, _, c = fixture('L2R_text_fixed')
        old, current = [group(i) for i in range(100, 148)], [group(i) for i in range(48)]
        memory = method.memory()
        stage1 = method.stage(1, old, memory, model)
        # Hand-specified per-query selected-negative tables, not inferred training evidence.
        for q in stage1.index.rows:
            stage1.mc[q.uid] = {a.uid for a in stage1.index.rows if not m.l2r.same_identity(a, q)}
        stage1.retain(model, c)
        self.assertEqual((memory.seen, len(memory.packages), memory.summary()['slots']), (1344, 14, 168))
        before = {q: {a.uid for a in p['negatives']} for q, p in memory.packages.items()}
        stage2 = method.stage(2, current, memory, model)
        for q in [*stage2.index.rows, *(p['query'] for p in memory.packages.values())]:
            stage2.mc[q.uid] = {a.uid for a in stage2.index.rows if not m.l2r.same_identity(a, q)}
        # Keep old directory slots to isolate the old-query ISD replacement operation.
        with mock.patch.object(memory.rng, 'randrange', side_effect=lambda n: n-1):
            evidence = stage2.retain(model, c)
        for q, package in memory.packages.items():
            after = {a.uid for a in package['negatives']}
            self.assertEqual(evidence[q]['replaced'], 3)
            self.assertEqual(len(after-before[q]), 3)
            self.assertTrue(after-before[q] <= stage2.mc[q])
        self.assertEqual(method.memory_from(memory.to_bytes()).to_bytes(), memory.to_bytes())

    def test_gcr_selection_weights_reach_later_replay(self):
        method, model, _, c = fixture('ER_GCR_group_fixed')
        current = [group(i) for i in range(48)]
        memory = method.memory()
        stage = method.stage(1, current, memory, model)
        for g in current:
            stage.after_update(g)  # Isolated admission test; actual update ordering tested separately.
        evidence = stage.retain(model, c)
        self.assertTrue(memory.weights)
        self.assertLessEqual(len(memory.weights), 6)
        self.assertEqual(evidence['gradient_parameters'], sum(p.numel() for p in model.head.parameters()))
        self.assertNotAlmostEqual(sum(memory.weights.values()), len(memory.weights), places=5)
        later = method.stage(2, [group(i) for i in range(100, 148)], method.memory_from(memory.to_bytes()), model)
        transient = later.transient()
        chosen, weight, pool = later.draw()
        later.restore_transient(transient)
        model.zero_grad(set_to_none=True)
        row = later.backward(model, later.current[sorted(later.current)[0]], c, 1)
        self.assertEqual(row['replay']['uid'], chosen.uid)
        self.assertEqual(pool, 'history_pool')
        self.assertEqual(row['replay']['coefficient'], .5*weight)
        self.assertGreater(float(model.encoder[0].embedding.weight.grad.norm()), 0.)

    def test_actual_checkpoint_resume_next_update(self):
        for name in m.NEW_METHODS:
            with self.subTest(method=name), tempfile.TemporaryDirectory() as folder:
                method, model, optimizer, c = fixture(name)
                current = [group(i) for i in range(48)]
                memory = method.memory()
                stage = method.stage(1, current, memory, model)
                method.update(model, optimizer, stage, current[0], c, 1)
                metadata = {**method.metadata(), 'rng': m.diag.rng_state(),
                            'modes': [v.training for v in model.modules()]}
                memory_bytes, transient = memory.to_bytes(), copy.deepcopy(stage.transient())
                path = Path(folder)/'full.pt'
                receipt = m.core.save_state(path, model, optimizer, metadata)
                method.update(model, optimizer, stage, current[1], c, 2)
                reference = m.core.state_digest(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                                     transient=stage.transient(), memory=memory.to_bytes().decode()))
                with torch.no_grad():
                    for parameter in model.parameters():
                        parameter.add_(1.)
                method.restore(path, receipt, model, optimizer)
                restored = method.stage(1, current, method.memory_from(memory_bytes), model)
                restored.restore_transient(transient)
                method.update(model, optimizer, restored, current[1], c, 2)
                actual = m.core.state_digest(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                                 transient=restored.transient(), memory=restored.memory.to_bytes().decode()))
                self.assertEqual(reference, actual)

    def test_evaluation_domain_pairing_order_guard_and_four_directions(self):
        spec = m.Spec('L2R_text_fixed', 'BCA', 4)
        cells = {f'{a}_{b}': np.full((40, 22), a*10+b, float) for a, b in ev.CELLS}
        old = ev.contributions(spec, cells, 'O')
        np.testing.assert_array_equal(old[:, 0, 0], [0., 15.5, 16.])
        np.testing.assert_allclose(ev.contributions(spec, cells, 'F'),
                                   ev.contributions(spec, cells, 'A_acq')-old)
        draws = np.zeros((7, 3, 40), dtype=np.int64)
        values = np.zeros((3, 40, 22))
        values[:, :, :] = np.arange(40)[None, :, None]
        values[1] *= 2
        values[2] *= 3
        self.assertEqual(ev.interval(values, draws, (.0125, .9875))['map']['conditional_interval'], [0., 0.])
        self.assertEqual(ev.interval(values, draws)['map']['mean'], 117.)
        raw = np.tile(np.linspace(-2, 2, 378, dtype=np.float32), (40, 1))
        roles, order = ev.calibrated_roles(raw, dict(stage=dict(a=2., b=1.), first=dict(a=1., b=0.)))
        self.assertTrue(order['stage-cal']['exact_order_and_ties_preserved'])
        self.assertEqual(roles['stage-cal'].shape, (40, 378))
        with mock.patch.object(m.parent.calibration, 'transform', return_value=np.zeros_like(raw)):
            with self.assertRaisesRegex(ValueError, 'order|ties'):
                ev.calibrated_roles(raw, dict(stage={}, first={}))
        comparisons = {a+'-'+b: {'raw': {f: {'map': dict(mean=1., conditional_interval=[.1, .2])}
                                        for f in ('O', 'F', 'N', 'Z')}} for a, b in ev.PRIMARY}
        for row in comparisons.values():
            row['raw']['F']['map']['conditional_interval'] = [-.2, -.1]
        self.assertTrue(ev.primary_decision(comparisons)['passed'])
        for a, b in ev.PRIMARY:
            for field, key, changed in (('O', 'conditional_interval', [0., 1.]),
                                        ('F', 'conditional_interval', [-1., 0.]), ('N', 'mean', -.01), ('Z', 'mean', -.01)):
                trial = copy.deepcopy(comparisons)
                trial[a+'-'+b]['raw'][field]['map'][key] = changed
                self.assertFalse(ev.primary_decision(trial)['passed'])

    def test_actual_analysis_entry_with_handwritten_saved_metrics(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            confirm, job = root/'confirmation', root/'analysis'
            confirm.mkdir()
            job.mkdir()
            ids = [f'handwritten.{d}.{i}' for d in 'ABC' for i in range(40)]
            result = dict(status='ALL_COMPARISON_METRICS_COMPLETE', sources=['handwritten'],
                          groups=ids, domains=[d for d in 'ABC' for _ in range(40)], forwards=8640, runs={})
            references = {}
            for method in m.METHODS:
                for stage, position in ev.CELLS:
                    level = (.7 if method == 'H_pos_tuned' else .6 if method == 'S_tuned' else
                             .5 if stage == 3 else .8)
                    matrix = np.repeat((level+np.arange(40)*.0001)[:, None], 22, axis=1)
                    path = confirm/f'{method}_{stage}_{position}.json'
                    m.data.write_json(path, dict(columns=list(m.parent.metrics.COLUMNS), values=matrix.tolist(), counts=[]))
                    references[method, stage, position] = m.data.record(path, confirm)
            for spec in m.specifications():
                cells = {}
                for stage, position in ev.CELLS:
                    domain = spec.order[position-1]
                    cells[f'{stage}_{position}'] = dict(domain=domain, ids=[uid for uid in ids if f'.{domain}.' in uid],
                        scores={r: {} for r in m.parent.ROLES}, metrics={r: {'matrix': references[spec.name, stage, position]}
                                                                      for r in m.parent.ROLES})
                result['runs'][spec.run_id] = cells
            m.data.write_json(confirm/'result.json', result)
            with mock.patch.object(run, 'sources', return_value=['handwritten']):
                ev.analyze(root, job, mock.Mock(check=lambda: None))
            output = m.data.read_json(job/'result.json')
            self.assertTrue(output['primary']['passed'])
            for a, b in ev.PRIMARY:
                row = output['comparisons'][a+'-'+b]['raw']
                for field, expected in (('O', .2), ('F', -.3), ('A_acq', -.1), ('N', .05), ('Z', .2)):
                    self.assertAlmostEqual(row[field]['map']['mean'], expected)
                    np.testing.assert_allclose(row[field]['map']['conditional_interval'], [expected]*2, atol=1e-12)
                    self.assertEqual(row[field]['map']['quantiles'], [.0125, .9875] if field in ('O', 'F') else [.025, .975])
            output['checks_are_handwritten_only'] = True
            print('Handwritten analysis evidence:', output['primary'],
                  output['comparisons']['H_pos_tuned-L2R_text_fixed']['raw']['O']['map'], flush=True)


if __name__ == '__main__':
    import scipy
    print('Environment:', sys.version, 'numpy', np.__version__, 'scipy', scipy.__version__,
          'torch', torch.__version__, flush=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    unittest.main()
