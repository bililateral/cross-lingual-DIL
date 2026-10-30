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
from types import SimpleNamespace

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
        self.assertFalse(control.gpu_admission(20*gib, [(101, int(11.25*gib))], {101, 102}, 2)['ready'])

    def test_growth_release_and_stable_foreign_cuda_at_actual_resource_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            gib = control.GIB
            cases = [
                # Old free=16 followed by own=8 must not pass against required=12.5.
                (['16384', '101, 8192', '8192', '101, 8192'], False),
                # Reverse ordering alone would miss own release; defer this changed observation too.
                (['8192', '101, 8192', '16384', ''], False),
                # Free held constant by other activity still cannot hide a change in own allocation.
                (['16384', '', '16384', '101, 8192'], False),
                (['13312', '101, 8192\n900, 10240', '13312', '101, 8192\n900, 10240'], True),
            ]
            with patch.object(control.shared.prior.run, 'owned_processes', return_value={101: None}), \
                    patch.object(control.psutil, 'virtual_memory', return_value=SimpleNamespace(available=256*gib)), \
                    patch('shutil.disk_usage', return_value=SimpleNamespace(free=256*gib)):
                for outputs, expected in cases:
                    with self.subTest(outputs=outputs), \
                            patch.object(control.subprocess, 'check_output', side_effect=outputs) as query:
                        row = control.resources(Path(directory), 1)
                        self.assertEqual(row['ready'], expected)
                        self.assertEqual(query.call_count, 4)
                        self.assertTrue(all(0 < call.kwargs['timeout'] <= 10 for call in query.call_args_list))
                with patch.object(control.time, 'monotonic', side_effect=[0., 1., 8., 16., 21.]), \
                        patch.object(control.subprocess, 'check_output', side_effect=['13312', '101, 8192', '13312']) as query:
                    with self.assertRaises(TimeoutError):
                        control.resources(Path(directory), 1)
                    self.assertEqual(query.call_count, 3)

    def test_prior_uses_latest_cpu_and_retains_native_and_preparation(self):
        gate = dict(prior_used_seconds={'control': 352}, prior_breakdown_seconds=dict(
            cpu_all_attempts=61, native_all_attempts=171, preparation_conservative_charge=120))
        control.validate_prior(gate, {'cumulative_seconds': 61})
        for wrong in (222, 347):
            gate['prior_used_seconds']['control'] = wrong
            with self.assertRaises(ValueError):
                control.validate_prior(gate, {'cumulative_seconds': 61})

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

    def test_actual_h_s_entry_keeps_healthy_slots_and_rejects_incomplete_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign = root/'formal'
            specs = control.m.specifications()[:5]
            identities = [s.run_id for s in specs]
            script = root/'worker.py'
            script.write_text("import sys,time,os,json\nfrom pathlib import Path\n"
                "p=Path(sys.argv[1]);p.mkdir()\n"
                "(p/'start.json').write_text(json.dumps({'time':time.monotonic(),'cpu':list(os.sched_getaffinity(0))}))\n"
                "if p.name==sys.argv[2]:\n"
                " time.sleep(.1);raise SystemExit(3)\n"
                "deadline=time.monotonic()+4\n"
                "while not (p.parent/'dispatch'/sys.argv[2]/'failure.json').exists():\n"
                " if time.monotonic()>deadline:raise SystemExit(9)\n"
                " time.sleep(.01)\n"
                "time.sleep(.1)\n"
                "(p/'end.json').write_text(json.dumps({'time':time.monotonic()}))\n")
            gate = dict(campaign=str(campaign), prior_used_seconds={'control': 347}, predecessor={},
                        sources=control.run.sources(), limits={'disk_bytes': 128*control.GIB, 'small_bytes': 2*control.GIB})
            original_dispatch = control.shared.dispatch
            def dispatch(*args, **kwargs):
                return original_dispatch(*args, **kwargs, poll_seconds=.02)
            command = lambda row, job, deadline, metadata, gate_path: [
                sys.executable, '-B', str(script), str(job), identities[0]]
            with patch.object(control, 'validate_gate', return_value=gate), \
                    patch.object(control.m, 'specifications', return_value=specs), \
                    patch.object(control, 'predecessor_state', return_value={'ready': True}), \
                    patch.object(control, 'resources', return_value={'ready': True}), \
                    patch.object(control, 'training_command', side_effect=command), \
                    patch.object(control.shared, 'dispatch', side_effect=dispatch), \
                    patch.object(control, 'complete_blind') as complete:
                with self.assertRaisesRegex(RuntimeError, 'after active siblings finished'):
                    control.execute(root/'gate.json', root/'active_defaults_wait.py')
            complete.assert_not_called()
            self.assertFalse((campaign/'completion.json').exists())
            failure = control.data.read_json(campaign/'failure.json')
            outcomes = {r['id']: r['success'] for r in failure['budget']['completed']}
            self.assertEqual(outcomes, {name: name != identities[0] for name in identities})
            self.assertGreater(failure['budget']['used']['training'], .1)
            self.assertGreaterEqual(failure['budget']['used']['control'], 347)
            self.assertEqual(failure['budget']['active'], {})
            starts = {name: control.data.read_json(campaign/name/'start.json') for name in identities}
            for name in identities[3:]:
                self.assertNotEqual(starts[name]['cpu'], starts[identities[0]]['cpu'])
                previous = next(k for k in identities[1:3] if starts[k]['cpu'] == starts[name]['cpu'])
                self.assertGreaterEqual(starts[name]['time'], control.data.read_json(campaign/previous/'end.json')['time'])
            print('Actual H/S entry evidence:', dict(outcomes=outcomes,
                  cpu_assignment={k: v['cpu'] for k, v in starts.items()},
                  failed_slot_not_reused=True, completion_published=False), flush=True)

    def test_wait_wiring_one_claim_fixed_queue_and_actual_training_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign = root/'formal'
            gate = dict(campaign=str(campaign), prior_used_seconds={'control': 347}, predecessor={},
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
                # Delayed first dispatch must keep startup thresholds, even after the precheck.
                self.assertTrue(options['admit'](0))
                self.assertTrue(resources.call_args.kwargs['startup'])
                with patch.object(control, 'resources', wraps=original_resources), \
                        patch.object(control, 'gpu_snapshot', return_value=(16*control.GIB, [])), \
                        patch.object(control.shared.prior.run, 'owned_processes', return_value={}), \
                        patch.object(control.psutil, 'virtual_memory', return_value=SimpleNamespace(available=80*control.GIB)), \
                        patch('shutil.disk_usage', return_value=SimpleNamespace(free=100*control.GIB)):
                    self.assertFalse(options['admit'](0))
                    ledger.completed.append({'id': queue[0]['id'], 'success': True})
                    self.assertTrue(options['admit'](0))
                self.assertTrue(options['admit'](1))
                self.assertFalse(resources.call_args.kwargs['startup'])
                dispatched.append(queue)
            original_resources = control.resources
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
