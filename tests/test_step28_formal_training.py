"""Linux-only handwritten evidence for the two new formal training entries."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import step28_formal_queue as queue
import step28_forever_formal as forever
import step29_vendorlink_run as vendor_run
import step29_vendorlink_formal as vendor
import step28_forever_evaluate as forever_eval
import step29_vendorlink_evaluate as vendor_eval
from test_step28_forever import fixture as forever_fixture, dispatcher
from test_step29_vendorlink import fixture as vendor_fixture, model_for

EVIDENCE = {}


class FormalTrainingTests(unittest.TestCase):
    def test_vendor_query_metrics_and_cross_market_dependencies(self):
        ids = ['q0', 'q1', 'q2', 'q3']
        # Positive pairs (0,1) and (2,3); query0's positive ranks third.
        metrics = vendor_eval.query_metrics([0., 3., 2., 1., 2., 4.], [1, 0, 0, 0, 0, 1], ids)
        np.testing.assert_allclose(metrics[:, 0], [1/3, 1/3, 1., 1.], rtol=0, atol=0)
        self.assertEqual(metrics[0, 2], 0)
        self.assertEqual(metrics[0, 3], 1)
        self.assertAlmostEqual(metrics[0, 7], .5)
        partition = dict(split_seed=20261029, train_fraction=.1, dev_fraction=.1,
                         records_per_view=2, identities_per_group=4)
        handles = [f'hand{i}' for i in range(100)
                   if vendor_eval.split_for(f'hand{i}', 20261029, .1, .1) == 'heldout'][:7]
        rows = []
        for domain, names in (('Left', handles[:4]), ('Right', [handles[0].upper(), *handles[4:7]])):
            for style, name in enumerate(names):
                rows.extend(dict(market=domain, vendor=name, title=f'unique {domain} style{style} {name} product {i}',
                                 description=f'handwritten {domain} style{style} {name} details {i}') for i in range(20))
        dependency = vendor_eval.group_dependencies(rows, partition)
        self.assertEqual(dependency['components'], [['Left:heldout:0', 'Right:heldout:0']])
        self.assertEqual(dependency['query_counts'], {'Left:heldout:0': 8, 'Right:heldout:0': 8})
        self.assertTrue(all(name not in json.dumps(dependency) for name in handles))
        # Unequal group sizes: averaging group MAPs would give .5 instead of .375.
        dep = dict(components=[['a0', 'b0'], ['a1']], query_counts=dict(a0=2, a1=6, b0=4))
        totals = {'method': {'O': {'A': np.array([[1.], [0.]]), 'B': np.array([[1.]])}}}
        points = vendor_eval.aggregate(totals, {'A': ['a0', 'a1'], 'B': ['b0']}, dep)
        self.assertEqual(points['method']['O'][0], .375)
        # Even a large favorable point difference must never become a significance claim.
        absolute = {method: {field: np.full(10, 1. if method == 'H_full' else 0.)
                    for field in vendor_eval.FIELDS} for method in vendor.METHODS}
        summary = vendor_eval.comparison_summary(absolute)
        self.assertIsNone(summary['passed'])
        self.assertEqual(summary['primary_status'], 'DESCRIPTIVE_FIXED_DATA_ONLY')
        for fields in summary['comparisons'].values():
            for values in fields.values():
                for row in values.values():
                    self.assertEqual(row['mean'], 1.)
                    self.assertIsNone(row['conditional_95pct_interval'])
        EVIDENCE['vendor_statistics'] = dict(query_ap=[1/3, 1/3, 1., 1.],
            cross_market_casefold_component=True, no_handle_text_or_label_output=True,
            unequal_query_weight_reference=.375, favorable_points_cannot_pass_statistical_acceptance=True)

    def test_forever_metric_formulas_and_complete_blind_gate(self):
        spec = forever.m.Spec('ABC', 4)
        cells = {key: np.full((40, 22), value) for key, value in
                 {'1_1': .8, '2_1': .7, '2_2': .6, '3_1': .5, '3_2': .4, '3_3': .9}.items()}
        actual = {field: forever_eval.shared.contributions(spec, cells, field).mean(1).sum(0)[0]
                  for field in ('O', 'F', 'N', 'Z')}
        for key, expected in dict(O=.45, F=.25, N=.75, Z=.9).items():
            self.assertAlmostEqual(actual[key], expected)
        ids = [f'{domain}-{i}' for domain in 'ABC' for i in range(40)]
        result = dict(groups=ids, domains=[d for d in 'ABC' for _ in range(40)], forwards=4320, runs={})
        for name in forever_eval.NAMES:
            for s in forever.specs():
                result['runs'][forever_eval.run_id(name, s.order, s.seed)] = {
                    f'{stage}_{position}': dict(domain=s.order[position-1],
                        ids=[uid for uid in ids if uid.startswith(s.order[position-1] + '-')],
                        scores={role: {} for role in forever.m.parent.ROLES}) for stage, position in forever_eval.CELLS}
        forever_eval.validate_cells(result)
        del result['runs'][next(iter(result['runs']))]['3_3']
        with self.assertRaises(ValueError):
            forever_eval.validate_cells(result)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            queue.write(root/'old_gate.json', {'sources': []})
            queue.write(root/'completion.json', dict(status=forever.H_COMPLETE, source_files=[{'wrong': True}],
                spec=dict(name='H_pos_tuned', order='ABC', seed=4), updates=864,
                points={str(i): {} for i in (1, 2, 3)}))
            allowed = dict(h_reference_gate=queue.record(root/'old_gate.json'),
                           h_reference_runs=[dict(completion=queue.record(root/'completion.json'))] * 9)
            with self.assertRaisesRegex(ValueError, 'Pinned H reference'):
                forever.verify_reference(allowed)
        EVIDENCE['forever_statistics'] = dict(hand_reference=actual, missing_cell_rejected=True,
                                             no_generated_or_formal_data=True)

    def test_followup_waits_for_training_completion_and_uses_its_own_bucket(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            allowed = dict(campaign=str(root/'batch'), workspace=str(root), prior_used_seconds={},
                           budget_seconds=dict(training=90., analysis=90., control=30., preparation=30.),
                           limits=dict(rss_bytes=4*2**30, disk_bytes=64*2**20, free_reserve_bytes=1))
            rows = [dict(id='training', seconds=90)]
            followups = [dict(id='analysis', seconds=90, bucket='analysis', gpu=False)]
            def command(row, job, deadline):
                code = 'print("handwritten child")'
                if row['id'] == 'analysis':
                    code = 'import json,pathlib; assert json.loads(pathlib.Path(' + repr(str(root/'batch/training_complete.json')) + ').read_text())["status"] == "ALL_TRAINING_COMPLETE_EVALUATION_CLOSED"'
                return [sys.executable, '-c', code]
            with patch.object(queue, 'resources', return_value={'ready': True}):
                result = queue.dispatch(allowed, rows, command, lambda row, job: None, followups,
                                        preflight=lambda: time.sleep(.02))
            self.assertEqual(result['status'], 'ALL_FORMAL_ACTIONS_COMPLETE')
            self.assertGreater(result['used']['training'], 0)
            self.assertGreater(result['used']['analysis'], 0)
            self.assertGreaterEqual(result['used']['preparation'], .02)
            trained = json.loads((root/'batch/training_complete.json').read_text())
            self.assertEqual([r['id'] for r in trained['completed']], ['training'])
            EVIDENCE['followups'] = dict(training_gate_before_analysis=True, separate_actual_buckets=True)

    def test_compact_vendor_endpoints_preserve_actual_trajectory_and_scores(self):
        for name in ('VendorLink_SEQ', 'H_full'):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                initial = model_for(name, root)
                scores, states, updates = [], [], []
                for compact in (False, True):
                    model = copy.deepcopy(initial)
                    job = root/str(compact)
                    vendor_run.execute(model, lambda domain: [vendor_fixture(domain)], job, order='ABC', seed=4, epochs=2,
                        identity={'handwritten': True}, inference_endpoints=compact)
                    scores.append(vendor_run.evaluate(model, [vendor_fixture('C', 'heldout').blind()]))
                    updates.append((job/'updates.jsonl').read_bytes())
                    states.append([torch.load(job/f'stage{i}.pt', map_location='cpu', weights_only=False)
                                   for i in (1, 2, 3)])
                self.assertEqual(updates[0], updates[1])
                self.assertEqual(scores[0], scores[1])
                for old, compact in zip(states[0], states[1], strict=True):
                    self.assertNotIn('optimizer', compact)
                    self.assertNotIn('memory', compact)
                    self.assertEqual(old['model'].keys(), compact['model'].keys())
                    for key in old['model']:
                        torch.testing.assert_close(old['model'][key], compact['model'][key], rtol=0, atol=0)
                EVIDENCE[name] = dict(three_actual_stages=True, compact_vs_full_models_equal=True,
                                     real_update_logs_and_blind_scores_equal=True)

    def test_calibration_failure_preserves_actual_fit_before_raising(self):
        learner = forever_fixture()
        learner.phase = 'endpoint'
        supplied = dispatcher(learner.spec)
        _, calibration = supplied.current(1)
        with tempfile.TemporaryDirectory() as temp:
            store = forever.run.Store(Path(temp), {'handwritten': True})
            detail = dict(status='HANDWRITTEN_FIT_FAILURE', reason='known injected fit failure', a=2, b=3)
            with patch.object(forever.m.parent.calibration, 'fit', return_value=detail):
                with self.assertRaisesRegex(ValueError, 'Current calibration failed'):
                    forever.run.endpoint(learner, calibration, store)
            saved = list((Path(temp)/'maps').glob('*.json'))
            self.assertEqual(len(saved), 1)
            self.assertEqual(json.loads(saved[0].read_text()), detail)
            self.assertFalse(list((Path(temp)/'models').glob('*.pt')))
            EVIDENCE['calibration_failure'] = dict(real_score_path=True, failure_detail_saved=True,
                                                   no_success_endpoint=True)

    def test_single_queue_failure_charged_and_next_child_not_launched(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            allowed = dict(campaign=str(root/'batch'), workspace=str(root), prior_used_seconds=dict(training=7., control=0.),
                           budget_seconds=dict(training=200., control=30.),
                           limits=dict(rss_bytes=4*2**30, disk_bytes=64*2**20, free_reserve_bytes=1))
            rows = [dict(id='fail', seconds=90), dict(id='must_not_run', seconds=90)]
            def command(row, job, deadline):
                return [sys.executable, '-c', 'import sys;sys.exit(2)']
            with patch.object(queue, 'resources', return_value={'ready': True}):
                with self.assertRaisesRegex(RuntimeError, 'Worker failed'):
                    queue.dispatch(allowed, rows, command, lambda row, job: self.fail('Failed child cannot validate'))
            saved = json.loads((root/'batch/failure.json').read_text())
            self.assertEqual(len(saved['completed']), 1)
            self.assertFalse(saved['completed'][0]['success'])
            self.assertGreater(saved['used']['training'], 7)
            self.assertFalse((root/'batch/must_not_run').exists())
            EVIDENCE['serial_failure'] = dict(real_process_exit=2, next_not_dispatched=True,
                                             inherited_seconds=7, actual_seconds=saved['used']['training'])

    def test_serial_resource_failure_stops_only_its_child(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            allowed = dict(campaign=str(root/'batch'), workspace=str(root), prior_used_seconds={},
                budget_seconds=dict(training=90., control=30.),
                limits=dict(rss_bytes=1, disk_bytes=64*2**20, free_reserve_bytes=1))
            with patch.object(queue, 'resources', return_value={'ready': True}):
                with self.assertRaisesRegex(RuntimeError, 'resource cap'):
                    queue.dispatch(allowed, [dict(id='bounded', seconds=90)],
                        lambda row, job, deadline: [sys.executable, '-c', 'import time;time.sleep(20)'],
                        lambda row, job: self.fail('An over-limit worker cannot complete'))
            saved = json.loads((root/'batch/failure.json').read_text())
            self.assertFalse(saved['completed'][0]['success'])
            self.assertLess(saved['used']['training'], 10)
            EVIDENCE['resource_stop'] = dict(real_child_terminated=True, failed_fee_retained=True)

    def test_forever_real_formal_three_stage_flow_on_handwritten_tiny_model(self):
        learner = forever_fixture()
        spec = learner.spec
        supplied = dispatcher(spec)
        partition = supplied.supply.partition
        archive = dict(file_count=1, total_size_bytes=1, content_sha256='handwritten')
        c = dict(learner.c, models={'split_rank': archive})
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            job = root/spec.run_id
            job.mkdir()
            allowed = dict(campaign=str(root), sources=[dict(handwritten=True)], limits=dict(
                # Production reserve includes a fixed serialization cushion;
                # the outer CPU harness still enforces 64 MiB actual output.
                rss_bytes=4 * 2**30, disk_bytes=2 * 2**30, small_bytes=32 * 2**20,
                free_reserve_bytes=1, allocator_bytes=1))
            with patch.object(forever.m, 'config', return_value=c), \
                 patch.object(forever.m.five.base, 'model_config', return_value={}), \
                 patch.object(forever.m.core, 'model_files', return_value=archive), \
                 patch.object(forever.run, 'training_inputs', return_value=(supplied.supply, partition, {'handwritten': True})), \
                 patch.object(forever.m, 'fresh', return_value=(learner.model, learner.optimizer)), \
                 patch.object(torch.cuda, 'set_per_process_memory_fraction'), \
                 patch.object(torch.cuda, 'get_device_properties', return_value=type('P', (), {'total_memory': 1})()), \
                 patch.object(torch, 'set_num_interop_threads'):
                result = forever.execute(job, spec, time.monotonic() + 450, allowed)
            self.assertEqual(result['counts']['current'], 864)
            self.assertEqual(result['counts']['fixture'], 0)
            self.assertGreater(result['counts']['replay'], 0)
            self.assertEqual(set(result['points']), {'1', '2', '3'})
            self.assertFalse((job/'run/latest.json').exists())
            self.assertFalse(list((job/'run/work').glob('*.pt')))
            self.assertFalse(list((job/'run/anchors').glob('*.pt')))
            self.assertEqual(len(list((job/'run/models').glob('*.pt'))), 3)
            rows = [json.loads(line) for line in (job/'run/updates.jsonl').read_text().splitlines()]
            self.assertEqual(sum(r['role'] == 'current' for r in rows), 864)
            self.assertTrue(any(r['role'] == 'replay' and r['regularizer']['penalty'] > 0 for r in rows))
            # A missing actual endpoint must invalidate the batch even with the old success JSON.
            next((job/'run/models').glob('*.pt')).unlink()
            with self.assertRaises((FileNotFoundError, ValueError)):
                forever.validate_completion(spec, job/'run', allowed)
            EVIDENCE['forever_formal'] = dict(real_current_updates=864, counts=result['counts'],
                real_three_stage_calibration_and_restore=True, successful_cleanup=True,
                missing_endpoint_rejected=True, not_native_or_formal_data=True)


if __name__ == '__main__':
    if sys.platform != 'linux':
        raise SystemExit('Research tests run only in Linux py310')
    torch.set_num_threads(1)
    suite = (unittest.defaultTestLoader.loadTestsFromNames(sys.argv[1:], sys.modules[__name__])
             if len(sys.argv) > 1 else unittest.defaultTestLoader.loadTestsFromTestCase(FormalTrainingTests))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if os.environ.get('FORMAL_TEST_EVIDENCE'):
        queue.write(Path(os.environ['FORMAL_TEST_EVIDENCE']), dict(evidence=EVIDENCE,
            tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors)))
    raise SystemExit(not result.wasSuccessful())
