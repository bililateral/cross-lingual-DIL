"""Incremental waiting conditions and actual CLI wiring; no data/model/GPU use."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import step28_defaults_control as control


class DefaultsControlTest(unittest.TestCase):
    def test_gpu_foreign_process_allowed_and_unrealized_own_slots_reserved(self):
        gib = control.GIB
        self.assertTrue(control.gpu_admission(11*gib, [(900, 20*gib)], set(), 0)['ready'])
        self.assertFalse(control.gpu_admission(10*gib, [], set(), 0)['ready'])
        # A just-spawned child has not allocated yet: the same free bytes cannot fund two slots.
        self.assertFalse(control.gpu_admission(20*gib, [(900, 10*gib)], {101}, 1)['ready'])
        self.assertTrue(control.gpu_admission(21*gib, [(900, 10*gib)], {101}, 1)['ready'])
        self.assertTrue(control.gpu_admission(13*gib, [(101, 8*gib), (900, 10*gib)], {101}, 1)['ready'])
        self.assertFalse(control.gpu_admission(12*gib, [(101, 8*gib), (900, 10*gib)], {101}, 1)['ready'])

    def predecessor(self, directory):
        root = Path(directory)
        campaign = root/'old'
        campaign.mkdir()
        gate = dict(campaign=str(campaign), workspace=str(root/'workspace'), sources=['pinned'],
                    new_runs=[dict(name='old', order='ABC', seed=4)])
        gate_path, launch_path = root/'gate.json', root/'launch.json'
        control.shared.write(gate_path, gate)
        record = lambda p: {**control.data.record(p, root), 'path': str(p)}
        launch = dict(pid=123, owner='owner', created=1., cwd=gate['workspace'], formal_gate=record(gate_path))
        control.shared.write(launch_path, launch)
        previous = dict(campaign=str(campaign), workspace=gate['workspace'], gate=record(gate_path),
                        launch=record(launch_path))
        result = dict(status=control.PREDECESSOR_COMPLETE, sources=gate['sources'],
            budget=dict(failed=False, active={}, limits={'training': 100}, used={'training': 10},
                completed=[dict(id=k, success=True) for k in
                           ('ABC_s4_old', 'generation', 'confirmation', 'analysis')]))
        return previous, result, campaign

    def test_predecessor_requires_final_success_and_no_live_refilling_controller(self):
        with tempfile.TemporaryDirectory() as directory:
            previous, result, campaign = self.predecessor(directory)
            control.shared.write(campaign/'training_complete.json', {'status': 'TRAINING_ONLY'})
            with patch.object(control, 'predecessor_processes', return_value=[{'pid': 123}]):
                self.assertFalse(control.predecessor_state(previous)['ready'])
                control.shared.write(campaign/'completion.json', result)
                self.assertFalse(control.predecessor_state(previous)['ready'])
            with patch.object(control, 'predecessor_processes', return_value=[]):
                self.assertTrue(control.predecessor_state(previous)['ready'])
                bad = copy.deepcopy(result)
                bad['budget']['completed'].pop()
                control.shared.write(campaign/'completion.json', bad)
                with self.assertRaises(ValueError):
                    control.predecessor_state(previous)
                (campaign/'completion.json').unlink()
                with self.assertRaises(RuntimeError):
                    control.predecessor_state(previous)
                control.shared.write(campaign/'completion.json', result)
                control.shared.write(campaign/'failure.json', {'failed': True})
                with self.assertRaises(RuntimeError):
                    control.predecessor_state(previous)

    def test_process_identity_does_not_confuse_foreign_cuda_or_reused_pid(self):
        class Process:
            def __init__(self, pid, created, command, cwd):
                self.info = dict(pid=pid, create_time=created, cmdline=command,
                                 username='owner', status='running')
                self.directory = cwd
            def cwd(self):
                return self.directory
        previous = dict(workspace='/p/workspace', campaign='/p/formal', gate={'path': '/p/gate'})
        launch = dict(owner='owner', pid=123, created=1.)
        reused = Process(123, 2., ['foreign_cuda.py'], '/foreign')
        real = Process(456, 3., ['python', 'step28_continual_comparison_run.py'], '/p/workspace')
        with patch.object(control.psutil, 'process_iter', return_value=[reused, real]):
            self.assertEqual([v['pid'] for v in control.predecessor_processes(previous, launch)], [456])

    def test_self_delete_requires_matching_actual_update(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            active = root/'active_defaults_wait.py'
            active.write_text('retired only after update')
            log = root/'dispatch/ABC_s4_S_default/console.txt'
            log.parent.mkdir(parents=True)
            event = dict(event='updates', run_id='ABC_s4_S_default', stage=1, completed=0)
            log.write_text(json.dumps(event)+'\n')
            self.assertIsNone(control.first_update(root, active, {'ABC_s4_S_default'}))
            self.assertTrue(active.exists())
            event['completed'] = 1
            log.write_text(json.dumps(event)+'\n')
            self.assertIsNone(control.first_update(root, active, {'different_run'}))
            self.assertTrue(control.first_update(root, active, {'ABC_s4_S_default'})['listener_deleted'])
            self.assertFalse(active.exists())
            self.assertEqual(control.data.read_json(root/'first_update.json')['event'], event)

    def test_actual_external_copy_cli_rejects_missing_scope_before_any_job(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            active, gate = root/'active_defaults_wait.py', root/'gate.json'
            active.write_bytes((control.data.ROOT/'scripts/step28_defaults_control.py').read_bytes())
            control.shared.write(gate, dict(campaign=str(root/'formal')))
            result = subprocess.run([sys.executable, '-B', str(active), '--workspace', str(control.data.ROOT),
                '--gate', str(gate)], capture_output=True, text=True, timeout=15,
                env=dict(os.environ, PYTHONPATH=str(control.data.ROOT/'scripts')))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Exact original-S/H blind-training scope or sources differ', result.stderr)
            self.assertFalse((root/'formal').exists())

    def test_wait_wiring_one_claim_fixed_queue_and_actual_training_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign = root/'formal'
            gate = dict(campaign=str(campaign), prior_used_seconds={'control': 222}, predecessor={},
                        sources=control.run.sources(), limits={'disk_bytes': 128*control.GIB, 'small_bytes': 2*control.GIB})
            dispatched = []
            def dispatch(queue, target, ledger, bucket, command, **options):
                self.assertEqual((target, bucket, len(queue)), (campaign, 'training', 18))
                self.assertEqual(sum(r['allowance'] for r in queue), 108*3600)
                metadata = campaign/'dispatch/example'
                metadata.mkdir(parents=True)
                line = command(queue[0], campaign/queue[0]['id'], 12345., metadata)
                self.assertEqual(line[2], control.run.__file__)
                self.assertEqual(line[-2:], ['--deadline', '12345.0'])
                self.assertEqual(control.data.read_json(metadata/'spec.json'), queue[0]['spec'])
                self.assertFalse((campaign/queue[0]['id']).exists())
                self.assertTrue(options['admit'](1))
                dispatched.append(queue)
            with patch.object(control, 'validate_gate', return_value=gate), \
                    patch.object(control, 'predecessor_state', side_effect=[{'ready': False}, {'ready': True}, {'ready': True}]) as previous, \
                    patch.object(control, 'resources', return_value={'ready': True}) as resources, \
                    patch.object(control.time, 'sleep') as sleep, \
                    patch.object(control.shared, 'dispatch', side_effect=dispatch), \
                    patch.object(control, 'complete_blind', return_value=[]):
                control.execute(root/'gate.json', root/'active_defaults_wait.py')
                self.assertEqual(previous.call_count, 3)
                sleep.assert_called_once_with(60)
                self.assertEqual(resources.call_args_list[0].kwargs, {'startup': True})
                self.assertEqual(len(dispatched), 1)
                before = (campaign/'completion.json').read_bytes()
                with self.assertRaises(FileExistsError):
                    control.execute(root/'gate.json', root/'active_defaults_wait.py')
                self.assertEqual((campaign/'completion.json').read_bytes(), before)


if __name__ == '__main__':
    unittest.main(verbosity=2)
