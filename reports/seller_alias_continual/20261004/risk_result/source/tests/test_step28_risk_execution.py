"""Incremental runner evidence on handwritten data, never formal datasets."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

import test_step28_risk_replay as unit
import test_step28_er_weight_contracts as inherited
import step28_risk_execution as adapter
import step28_er_weight_run as runner
import step28_er_weight_evaluate as evaluation
import step28_risk_study as study

method, fixtures = unit.method, unit.fixtures


class ExecutionTests(unittest.TestCase):
    def test_actual_runner_stage_and_mode_probe(self):
        p = study.contract()
        c, old, new, model, opt = unit.prepared()
        first = [fixtures.handmade_group(f'a{i:02}', i) for i in range(48)]
        current = [fixtures.handmade_group(f'b{i:02}', i + 70) for i in range(48)]
        parent = method.parent.Memory('ABC', method.parent.contract()['memory_seed'], False, {'a': 1., 'b': 0.})
        parent.retain(first, 1, None)
        memory = adapter.Memory.from_er(parent.to_bytes(), model, opt, c, lambda: None)
        refs = copy.deepcopy(memory.cache.auxiliary['risk_replay'])
        memory.auxiliary = {'new_metadata': True}
        self.assertEqual(memory.cache.auxiliary['risk_replay'], refs)
        restored = adapter.Memory.from_bytes(memory.to_bytes())
        self.assertEqual(restored.summary(), memory.summary())
        before = (runner.rng_state(), method.core.state_digest(model.state_dict()),
                  method.core.state_digest(opt.state_dict()), memory.to_bytes())
        diagnostic = adapter.mode_probe(model, memory, c, lambda: None)
        after = (runner.rng_state(), method.core.state_digest(model.state_dict()),
                 method.core.state_digest(opt.state_dict()), memory.to_bytes())
        self.assertEqual(before, after)
        self.assertEqual(diagnostic['eval_self'], [0., 0., 0.])
        parent.begin_stage(2)
        sequence, _ = method.parent.schedule(current, method.parent.contract(), 'ABC', 2)
        reference = {'current_ids': [g.uid for g in sequence],
                     'history_ids': [parent.draw()[0].uid for _ in range(288)],
                     'memory_after_training': parent.summary()}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root/'updates').mkdir()
            budget = fixtures.HandmadeBudget(); budget.state = lambda: {'handwritten': True}
            with mock.patch.object(adapter.risk, 'update', wraps=adapter.risk.update) as calls:
                row = runner.train_stage(model, opt, current, memory, c, 'ABC', 2, 'risk', root,
                                         reference, budget, p)
            self.assertEqual(calls.call_count, 288)
            for call in calls.call_args_list:
                self.assertEqual(call.kwargs['history_weight'], .1)
                self.assertEqual(call.kwargs['retention_weight'], .5)
                self.assertEqual(call.args[4], refs['groups'][call.args[3].uid]['values'])
            self.assertEqual(method.parent.adam_step(opt), 576)
            values = np.load(root/row['update_file']['path'])
            col = {name: values[:,i] for i,name in enumerate(row['update_columns'])}
            np.testing.assert_array_equal(col['total'], col['current_total'] + .1*col['history_total'] + .5*col['retention'])
            self.assertEqual(set(row['observations']), {'1','29','30','288'})
            self.assertIn('first_update_head_gradient_norms', row['risk_diagnostics'])

    def test_four_matched_comparisons(self):
        p = study.contract()
        old, domains = fixtures.synthetic_matrices()
        for roles in old.values():
            for values in roles.values(): values[:] = .3
        new = {}
        for order in method.parent.ORDERS:
            for arm, gain in [('risk', .08), ('tenth', .01), ('logit_tenth', .04), ('logit_quarter', .02)]:
                for stage in (2,3):
                    values = np.full((60,22), .3+gain, dtype=np.float64)
                    for key in ('brier','log_loss'):
                        values[:,runner.method.metrics.COLUMNS.index(key)] = .3-gain
                    (new if arm=='risk' else old)[f'{order}_{arm}_stage{stage}'] = {
                        role:values.copy() for role in method.parent.ROLES}
        out = evaluation.evaluate_matrices(new,old,domains,p)
        for ref, expected in [('tenth',.07),('logit_tenth',.04),('logit_quarter',.06),('seq',.08)]:
            comparison = out['comparisons']['risk_minus_'+ref]
            self.assertAlmostEqual(comparison['primary']['O']['map']['mean'],expected)
            self.assertEqual(len(comparison['interpretation']['checks']),23)
        self.assertNotIn('selection',out)

    def test_blind_gate_risk_coefficients_and_missing_endpoint(self):
        p = study.contract()
        with tempfile.TemporaryDirectory() as directory:
            root, manifest, reference = inherited.gate_fixture(Path(directory),p)
            # File/identity fixture only. Real memory and updates are checked above.
            for name, rec in manifest['training'].items():
                log = method.data.read_json(root/rec['path'])
                path = root/log['update_file']['path']
                values = np.load(path)
                cols = {k:values[:,i] for i,k in enumerate(log['update_columns'])}
                for key in ('rank','positive','negative'): cols['retention_'+key][:]=.2
                cols['retention'][:]=.2; cols['retention_weight'][:]=.5
                cols['total'][:]+=.1
                np.save(path,values)
                log['update_file']=method.data.record(path,root)
                method.data.write_json(root/rec['path'],log)
                manifest['training'][name]=method.data.record(root/rec['path'],root)
            with mock.patch.object(adapter,'verify_references'):
                self.assertEqual(len(runner.blind_gate(root,manifest,reference,p)),6)
                missing=copy.deepcopy(manifest); missing['points'].pop(next(iter(missing['points'])))
                with self.assertRaises(ValueError): runner.blind_gate(root,missing,reference,p)
                name=next(iter(manifest['training'])); rec=manifest['training'][name]
                log=method.data.read_json(root/rec['path']); path=root/log['update_file']['path']
                values=np.load(path); values[:,log['update_columns'].index('retention_weight')]=0
                np.save(path,values); log['update_file']=method.data.record(path,root)
                method.data.write_json(root/rec['path'],log); manifest['training'][name]=method.data.record(root/rec['path'],root)
                with self.assertRaises(ValueError): runner.blind_gate(root,manifest,reference,p)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
