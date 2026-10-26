"""Only new queue, resource deferral, directory ownership and budget behavior."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import step28_comparison_control as control


class ComparisonControlTest(unittest.TestCase):
    def test_long_reservation_retains_failures_and_whole_queue(self):
        with tempfile.TemporaryDirectory() as directory:
            clock = [0.]
            ledger = control.Ledger(Path(directory)/'budget.json',
                dict(training=800*3600, control=3600, cpu=1200, native=3000),
                dict(cpu=42, native=2479), clock=lambda: clock[0])
            ledger.reserve('l2r', 'training', 72*3600, remaining=684*3600)
            self.assertEqual(ledger.active['l2r']['deadline'], 72*3600)
            with self.assertRaises(ValueError):
                ledger.reserve('extra', 'training', 50*3600, remaining=684*3600)
            clock[0] = 19.
            with self.assertRaises(RuntimeError):
                ledger.settle('l2r', False)
            self.assertEqual(ledger.used['training'], 19.)
            self.assertEqual((ledger.used['cpu'], ledger.used['native']), (42., 2479.))
            self.assertFalse(ledger.completed[0]['success'])
            with self.assertRaises(ValueError):
                ledger.reserve('l2r', 'training', 72*3600)

    def test_real_three_children_own_dirs_and_low_slot_waits(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = Path(directory)
            script = campaign/'worker.py'
            script.write_text("from pathlib import Path\nimport sys,time,os,json\n"
                "p=Path(sys.argv[1]);p.mkdir();time.sleep(.2)\n"
                "(p/'receipt.json').write_text(json.dumps({'cores':list(os.sched_getaffinity(0)),"
                "'threads':os.environ['OMP_NUM_THREADS']}))\n")
            ledger = control.Ledger(campaign/'budget.json', dict(training=1000, control=30), {})
            observed = []
            def admit(active):
                observed.append(active)
                return len(observed) > 2
            def command(row, job, deadline, metadata):
                self.assertFalse(job.exists())
                return [sys.executable, '-B', str(script), str(job)]
            rows = [dict(id=f'worker{i}', allowance=120.) for i in range(4)]
            control.dispatch(rows, campaign, ledger, 'training', command,
                             admit=admit, poll_seconds=.03)
            self.assertEqual(len(ledger.completed), 4)
            self.assertTrue(all(row['success'] for row in ledger.completed))
            receipts = [json.loads((campaign/row['id']/'receipt.json').read_text()) for row in rows]
            self.assertTrue(all(len(row['cores']) == 1 and row['threads'] == '1' for row in receipts))
            self.assertEqual(len({row['cores'][0] for row in receipts[:3]}), 3)
            self.assertEqual(observed[:2], [0, 0])
            self.assertEqual(ledger.active, {})

    def test_real_failed_child_stops_owned_children_and_keeps_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = Path(directory)
            script = campaign/'worker.py'
            script.write_text("import sys,time\nfrom pathlib import Path\n"
                "Path(sys.argv[1]).mkdir()\n"
                "time.sleep(.1 if sys.argv[2]=='bad' else 20)\n"
                "raise SystemExit(3 if sys.argv[2]=='bad' else 0)\n")
            ledger = control.Ledger(campaign/'budget.json', dict(training=1000, control=30), {})
            rows = [dict(id=name, allowance=120.) for name in ('bad', 'other', 'never')]
            command = lambda row, job, deadline, metadata: [sys.executable, '-B', str(script), str(job), row['id']]
            with self.assertRaises(RuntimeError):
                control.dispatch(rows, campaign, ledger, 'training', command,
                                 max_workers=2, poll_seconds=.03)
            self.assertEqual({row['id'] for row in ledger.completed}, {'bad', 'other'})
            self.assertTrue(all(not row['success'] for row in ledger.completed))
            self.assertGreater(ledger.used['training'], .1)
            self.assertTrue(ledger.failed)
            self.assertEqual(ledger.active, {})
            self.assertFalse((campaign/'never').exists())

    def test_success_cost_only_once_and_active_reservations_are_not_spent(self):
        with tempfile.TemporaryDirectory() as directory:
            clock = [10.]
            ledger = control.Ledger(Path(directory)/'budget.json', dict(training=100, control=30),
                                    {}, clock=lambda: clock[0])
            ledger.reserve('a', 'training', 40, remaining=40)
            ledger.reserve('b', 'training', 40)
            self.assertEqual(ledger.used['training'], 0.)
            self.assertEqual(ledger.available('training'), 20.)
            clock[0] = 15.
            ledger.settle('a', True)
            clock[0] = 17.
            ledger.settle('b', True)
            self.assertEqual(ledger.used['training'], 12.)
            self.assertEqual(ledger.available('training'), 88.)
            with self.assertRaises(ValueError):
                ledger.reserve('a', 'training', 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
