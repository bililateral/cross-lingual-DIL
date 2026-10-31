"""The one-run repair, exact donor merge and original H/S completion boundary."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import step28_comparison_finish as c


class FinishTest(unittest.TestCase):
    def fixture(self, root):
        old = root/'donor'
        old.mkdir()
        specs = c.m.specifications(new_only=True)
        original = root/'original.json'
        current = root/'current.json'
        c.shared.write(original, dict(sources=['original']))
        c.shared.write(current, dict(sources=['current']))
        pin = lambda p: {**c.data.record(p, root), 'path': str(p)}
        adoption = {}
        for index, spec in enumerate(s for s in specs if s.run_id != c.TARGET):
            gate = original if index < 14 else current
            completion = old/spec.run_id/'run/completion.json'
            completion.parent.mkdir(parents=True)
            c.shared.write(completion, dict(status=c.run.COMPLETE, source_files=c.data.read_json(gate)['sources']))
            adoption[spec.run_id] = dict(completion=str(completion), source_gate=pin(gate))
        budget = dict(limits=c.LIMITS, used={k: 0. for k in c.LIMITS}, active={},
                      failed=True, budget_exceeded=False,
                      completed=[dict(id=k, success=True) for k in adoption]+[dict(id=c.TARGET, success=False)])
        budget['used'].update(training=100., cpu=98.)
        failure = dict(status='STOP_NO_RETRY', budget=budget,
                       error=repr(RuntimeError('Worker failure retained after active siblings finished: '+c.TARGET)))
        c.shared.write(old/'failure.json', failure)
        c.shared.write(old/'budget.json', budget)
        campaign = root/'new'
        gate = dict(campaign=str(campaign), donor=dict(campaign=str(old)), adoption=adoption,
                    sources=['repaired'], additional_used_seconds={'cpu': 6}, launcher=str(root/'launcher'),
                    new_runs=[s.identity() for s in specs], workspace=str(root))
        Path(gate['launcher']).write_text('one use')
        return gate, failure

    def test_combined_budget_keeps_failed_fees_and_pending_reservations(self):
        old = dict(used={k: 0. for k in c.LIMITS}, active={}, failed=True)
        own = copy.deepcopy(old)
        old['used'].update(training=100., cpu=98.)
        own['used']['cpu'] = 6.
        own['active']['new'] = dict(bucket='training', allowance=259200.)
        row = c.check_combined_budget(old, own, True)
        self.assertEqual(row['used']['cpu'], 104.)
        self.assertEqual(row['used']['training'], 100.)
        self.assertEqual(row['reserved']['training'], 302400.)
        old['used']['training'] = c.LIMITS['training']-302399.
        with self.assertRaises(RuntimeError):
            c.check_combined_budget(old, own, True)

    def test_only_known_failure_and_all_17_successes_are_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            gate, failure = self.fixture(Path(directory))
            path = Path(gate['donor']['campaign'])/'failure.json'
            with patch.object(c, 'donor_processes', return_value=[{'pid': 1}]):
                self.assertIsNone(c.inspect_donor(gate))
            with patch.object(c, 'donor_processes', return_value=[]):
                self.assertEqual(c.inspect_donor(gate), failure)
                for change in ('missing', 'another_failure', 'over_budget'):
                    bad = copy.deepcopy(failure)
                    if change == 'missing':
                        bad['budget']['completed'].pop(0)
                    elif change == 'another_failure':
                        bad['budget']['completed'][0]['success'] = False
                    else:
                        bad['budget']['budget_exceeded'] = True
                    c.shared.write(path, bad)
                    with self.assertRaises(ValueError):
                        c.inspect_donor(gate)
                c.shared.write(path, failure)
                c.shared.write(path.parent/'completion.json', {})
                with self.assertRaises(ValueError):
                    c.inspect_donor(gate)

    def test_real_entry_dispatches_one_merge_then_posts_and_h_s_waits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gate, failure = self.fixture(root)
            campaign = Path(gate['campaign'])
            gate_path = root/'gate.json'
            c.shared.write(gate_path, gate)
            worker = root/'worker.py'
            worker.write_text(
                "import sys,json\nfrom pathlib import Path\n"
                "a=sys.argv[1:]\n"
                "if '--job' in a:\n"
                " p=Path(a[a.index('--job')+1]);(p/'run').mkdir(parents=True)\n"
                " (p/'run/completion.json').write_text(json.dumps({'source_files':['repaired']}))\n"
                " print(json.dumps({'event':'updates','run_id':p.name,'completed':1}),flush=True)\n"
                "else:\n"
                " p=Path(a[a.index('--campaign')+1]);assert (p/'training_complete.json').exists()\n"
                " with (p/'post_order.txt').open('a') as f:f.write(a[0]+'\\n')\n")
            before = {p: p.read_bytes() for p in Path(gate['donor']['campaign']).rglob('*.json')}
            real_dispatch = c.shared.dispatch
            def fast_dispatch(*args, **kwargs):
                return real_dispatch(*args, **kwargs, poll_seconds=.01)
            def endpoint_boundary(path, allowed):
                self.assertEqual(len(list(path.glob('*/run/completion.json'))), 18)
                for identity, item in gate['adoption'].items():
                    found = c.evaluation.trajectory_sources(allowed, identity, path/identity/'run/completion.json')
                    self.assertEqual(found, c.pinned(item['source_gate'])['sources'])
                return list(range(36))
            with patch.object(c, 'validate', return_value=gate), \
                    patch.object(c.shared, 'reference_check'), \
                    patch.object(c, 'combined_resources'), \
                    patch.object(c, 'donor_processes', return_value=[]), \
                    patch.object(c.shared, 'resources', return_value=True), \
                    patch.object(c.shared, 'dispatch', side_effect=fast_dispatch), \
                    patch.object(c.run, '__file__', str(worker)), \
                    patch.object(c.evaluation, '__file__', str(worker)), \
                    patch.object(c.evaluation, 'full_gate', side_effect=endpoint_boundary):
                c.execute(gate_path)
            self.assertEqual({p.name for p in (campaign/'dispatch').iterdir()},
                             {c.TARGET, 'generation', 'confirmation', 'analysis'})
            self.assertEqual((campaign/'post_order.txt').read_text(), 'generate\nconfirmation\nanalyze\n')
            self.assertFalse(Path(gate['launcher']).exists())
            self.assertEqual(before, {p: p.read_bytes() for p in before})
            result = c.data.read_json(campaign/'completion.json')
            self.assertEqual(result['budget']['used']['cpu'], 104.)
            self.assertGreater(result['budget']['used']['training'], 100.)
            self.assertEqual(len(result['budget']['completed']), 21)
            self.assertFalse(result['budget']['failed'])
            self.assertFalse(c.data.read_json(campaign/'inherited_failure.json')['failure']['budget']['completed'][-1]['success'])
            identity = next(iter(gate['adoption']))
            source = Path(gate['adoption'][identity]['completion'])
            target = campaign/identity/'run/completion.json'
            self.assertEqual(source.stat().st_ino, target.stat().st_ino)
            # Replace the adopted link, not the donor: exact bytes must still be checked.
            target.unlink()
            target.write_text('{}')
            with self.assertRaises(ValueError):
                c.evaluation.trajectory_sources(gate, identity, target)
            pin = lambda p: {**c.data.record(p, root), 'path': str(p)}
            launch = root/'launch.json'
            c.shared.write(launch, dict(cwd=gate['workspace'], formal_gate=pin(gate_path)))
            previous = dict(campaign=str(campaign), workspace=gate['workspace'], gate=pin(gate_path), launch=pin(launch))
            with patch.object(c.defaults, 'predecessor_processes', return_value=[]):
                self.assertTrue(c.defaults.predecessor_state(previous)['ready'])
                bad = copy.deepcopy(result)
                bad['budget']['completed'].pop()
                c.shared.write(campaign/'completion.json', bad)
                with self.assertRaises(ValueError):
                    c.defaults.predecessor_state(previous)
                c.shared.write(campaign/'completion.json', result)
                c.shared.write(campaign/'failure.json', {'status': 'STOP_NO_RETRY'})
                with self.assertRaises(RuntimeError):
                    c.defaults.predecessor_state(previous)


if __name__ == '__main__':
    unittest.main()
