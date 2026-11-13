"""Bounded Linux CPU checks for this beta/three-slot/waiting increment."""
import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

if sys.platform != 'linux' or Path(sys.executable).resolve() != Path('/home/yongpeng/miniconda3/envs/py310/bin/python').resolve():
    raise RuntimeError('Existing Linux py310 only')

import psutil
import torch
from transformers import BertConfig, BertModel, BertTokenizer
from step29_vendorlink_data import Ad, View, Group
from step35_vendorlink_account_methods import AccountModel
from step35_vendorlink_account_policy import (AccountSettings, BETA, BETA_PROTOCOL, BETA_LIMITS,
    DETERMINISTIC_CAPS, beta_tasks, deterministic_tasks, summarize, beta_comparison, decide)
from step35_vendorlink_account_control import (make_jobs, validate_training, pipeline, beta_reference,
    deterministic_baseline, configure_compute)
from step35_vendorlink_account_run import execute
from step35_vendorlink_account_io import record, write
from step30_evolution_io import Ledger
from step30_evolution_control import Runner, kernel_identity, resource_observation
import step35_vendorlink_beta_wait as waiter

EVIDENCE = {}
SELECTED = dict(H='H_CE_RMS', V='V1')


class Checks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def model(self, setting):
        vocab = self.root/'vocab.txt'
        vocab.write_text('\n'.join(['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'one', 'two', 'three', 'four', 'five']))
        tokenizer = BertTokenizer(str(vocab), do_lower_case=True)
        torch.manual_seed(712)
        config = BertConfig(vocab_size=10, hidden_size=8, num_hidden_layers=4,
                            num_attention_heads=2, intermediate_size=16, max_position_embeddings=512)
        config._attn_implementation = 'sdpa'
        encoder = BertModel(config)
        encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        return AccountModel(encoder, tokenizer, replace(setting, channel_tokens=12))

    def test_beta_actual_gradient_and_execute(self):
        task = beta_tasks(SELECTED)[0]
        old = deterministic_tasks(SELECTED)[0]
        delta = {k for k in task['settings'] if task['settings'][k] != old['settings'][k]}
        self.assertEqual(delta, {'beta'})
        self.assertEqual(BETA*4, old['settings']['beta'])
        m = self.model(AccountSettings(**old['settings']))
        low = copy.deepcopy(m)
        low.settings = replace(m.settings, beta=BETA)
        group = Group('manual', 'AlphaBay', 'fit', tuple(View(f'v{i}_{v}', tuple(
            Ad(f'a{i}_{v}_{r}', 'one two', f'three four {i}') for r in range(2)), f'account{i:03d}')
            for i in range(4) for v in range(2)))
        teacher = m.teacher(group)+torch.linspace(-.2, .3, 120)
        m.train(); low.train()
        high_log, low_log = m.backward_birth(group, teacher), low.backward_birth(group, teacher)
        errors = [float((a.grad*.25-b.grad).abs().max()) for a,b in zip(m.parameters(), low.parameters()) if a.grad is not None]
        self.assertLess(max(errors), 1e-7)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in low.encoder.parameters() if p.grad is not None), 0)
        self.assertAlmostEqual(low_log['weighted'], high_log['weighted']*.25, places=8)
        task['epochs'] = 1
        task['settings']['channel_tokens'] = 12
        def fit(domain):
            return [dict(account_uid=f'account{i:03d}', records=[dict(uid=f'{domain}_{i}_{r}',
                title='one two', description='three four') for r in range(12)]) for i in range(4)]
        c = execute(self.model(AccountSettings(**task['settings'])), fit, self.root/'run', task,
                    {'handwritten': True}, {'handwritten': True})
        manifest = {'counts': {d: {'eligible_accounts': 4} for d in task['domains']}}
        validate_training(task, c, manifest, {'handwritten': True})
        self.assertEqual(c['updates'], 9)
        self.assertEqual(c['workload']['deterministic_history_groups'], 6)
        broken = copy.deepcopy(c); broken['workload']['deterministic_history_groups'] -= 1
        with self.assertRaises(ValueError): validate_training(task, broken, manifest, {'handwritten': True})
        EVIDENCE['beta'] = dict(gradient_quarter_max_abs=max(errors), weighted_ratio=low_log['weighted']/high_log['weighted'],
                               updates=c['updates'], history_groups=6, endpoint_beta=c['task']['settings']['beta'])

    def test_original_results_and_response_boundaries(self):
        g = json.loads(ARGS.gate.read_text())
        baseline, reference = deterministic_baseline(g), beta_reference(g)
        unchanged = decide(baseline, True)
        self.assertFalse(unchanged['checks']['meaningful_forgetting'])
        def rows(r22, r31, r32, r33):
            return [dict(seed=s, **summarize(dict(R01=.1, R11=.92, R12=.88, R21=.9, R22=r22,
                    R23=.84, R31=r31, R32=r32, R33=r33))) for s in (51,52,53)]
        synthetic = dict(H_SEQ=rows(.955,.91,.93,.93), H_ER=rows(.96,.92,.93,.94),
                         H_full=rows(.89,.89,.91,.935), V_CLS=rows(.93,.84,.86,.92), V_SET=rows(.93,.84,.86,.92))
        ref = {'H_full': copy.deepcopy(synthetic['H_full'])}
        candidate = {'H_full': rows(.96,.90,.92,.94)}
        good = beta_comparison(synthetic, ref, candidate)
        self.assertTrue(good['joint_response'])
        self.assertGreater(candidate['H_full'][0]['F'], ref['H_full'][0]['F'])
        for key, changed in [('dream_restore', ('GB', .039999)), ('agora_protect', ('GC', -.001)),
                             ('old_final_improve', ('R31', .889))]:
            bad = copy.deepcopy(candidate)
            for row in bad['H_full']: row[changed[0]] = changed[1]
            self.assertFalse(beta_comparison(synthetic, ref, bad)['response_checks'][key])
        missing = {'H_full': candidate['H_full'][:2]}
        with self.assertRaises(ValueError): beta_comparison(synthetic, ref, missing)
        phases, saved = [], {}
        def phase(name, selected): phases.append(name); return reference
        result = pipeline(phase, saved.__setitem__, baseline, reference=reference)
        self.assertEqual(phases, ['M'])
        self.assertFalse(result['joint_response'])
        self.assertFalse(result['confirmation'] or result['application'] or result['automatic_followup'])
        self.assertEqual(result['original_gates']['original_M'], unchanged)
        with self.assertRaisesRegex(RuntimeError, 'incomplete'):
            pipeline(lambda *a: (_ for _ in ()).throw(RuntimeError('incomplete')), saved.__setitem__, baseline, reference=reference)
        EVIDENCE['decision'] = dict(original_forgetting=False, synthetic_joint_pass=True,
            higher_F_does_not_fail_response=True, missing_seed_rejected=True, no_downstream=True)

    def dependency(self):
        root, state = self.root/'chinese', self.root/'state'
        root.mkdir(); state.mkdir()
        p = psutil.Process()
        ident = dict(pid=p.pid, owner=p.username(), cwd=p.cwd(), command=p.cmdline(), **kernel_identity(p.pid))
        gate = self.root/'chinese_gate.json'; write(gate, {'sources': []})
        request = self.root/'request.json'
        write(request, dict(campaign=str(root), workspace=str(root/'workspace'), original_gate=str(gate), updated_sources=[]))
        return dict(campaign=str(root), state=str(state), controller=ident, request=record(request), gate=record(gate))

    def test_chinese_completion_is_not_idle_or_pause(self):
        dep = self.dependency(); g = {'dependency': dep}
        root, state = Path(dep['campaign']), Path(dep['state'])
        (root/'pause_dispatch').touch()
        self.assertFalse(waiter.dependency_observation(g)['ready'])
        write(root/'completion.json', dict(status='COMPLETE_PENDING_RESULT_REVIEW_MODELS_RETAINED',
            new_trajectories=63, full_reused=9, sources=[], controller_update=dict(request=dep['request']['path'],updated_sources=[])))
        self.assertFalse(waiter.dependency_observation(g)['ready'])
        write(state/'completion.json', {'status':'CONTROLLED_CAMPAIGN_COMPLETE'})
        with patch.object(waiter.psutil, 'process_iter', return_value=[]):
            self.assertTrue(waiter.dependency_observation(g)['ready'])
        write(root/'failure.json', {'status':'STOP_NO_RETRY'})
        with self.assertRaises(RuntimeError): waiter.dependency_observation(g)
        EVIDENCE['dependency'] = dict(pause_not_completion=True, supervisor_closure_required=True, failure_blocks=True)

    def test_wait_joint_trigger_and_unique_attempt(self):
        root = self.root/'new'; root.mkdir()
        wrapper = root/'once_wait.py'; wrapper.write_text('# fixture')
        gatepath = root/'gate.json'; gatepath.write_text('{}')
        g = dict(protocol=BETA_PROTOCOL, experiment_root=str(root), campaign=str(root/'campaign'), waiter=record(wrapper),
                 limits=BETA_LIMITS, prior_seconds=dict(training=0,gpu=0,auxiliary=5), caps=DETERMINISTIC_CAPS)
        calls, sleeps = [], []
        with (patch.object(waiter, 'dependency_observation', side_effect=[{'ready':False,'status':'WAIT_CHINESE'},
                {'ready':True}, {'ready':True}]), patch.object(waiter, 'resource_observation', side_effect=[{'ready':False},{'ready':True}]) as res,
                patch.object(waiter.time, 'sleep', side_effect=sleeps.append), patch.object(waiter, 'campaign', side_effect=lambda *a: calls.append(a))):
            waiter.wait_then_run(g, gatepath, time.monotonic())
        self.assertEqual(sleeps, [120,120]); self.assertEqual(res.call_count, 2); self.assertEqual(len(calls), 1)
        with self.assertRaises(FileExistsError): waiter.wait_then_run(g, gatepath, time.monotonic())
        ready = json.loads((root/'wait/ready.json').read_text())
        self.assertLess(ready['charged_seconds'], 10)
        EVIDENCE['wait'] = dict(dependency_then_resources=True, campaign_calls=1, duplicate_rejected=True)

    def test_three_real_cpu_slots_and_resource_reservations(self):
        jobs = make_jobs(self.root, 'M', beta_tasks(SELECTED), 'train')
        self.assertEqual([t['seed'] for t in beta_tasks(SELECTED)], [51,52,53])
        self.assertEqual(sum(j['seconds'] for j in jobs), 140400)
        ledger = Ledger(self.root, dict(training=0,gpu=0,auxiliary=0), 0, caps=DETERMINISTIC_CAPS)
        script = self.root/'cpu_worker.py'
        script.write_text('import argparse,json,pathlib,psutil,time\np=argparse.ArgumentParser();p.add_argument("--target");a,x=p.parse_known_args();time.sleep(.3);pathlib.Path(a.target,"cores.json").write_text(json.dumps(psutil.Process().cpu_affinity()))\n')
        g = dict(experiment_root=str(self.root),workspace=str(self.root),other_gpu_commitments=[])
        runner = Runner(g, self.root/'gate.json', self.root, ledger, worker_entry=script,
                        caps=DETERMINISTIC_CAPS, limits=BETA_LIMITS, halt_dispatch_on_failure=True, continue_after_oom=True)
        class NV:
            def snapshot(self): return {'processes':{},'free':32*2**30}
            def close(self): pass
        completed, own_counts = [], []
        def admit(g, root, gpu, owned, nv, limits):
            own_counts.append(len(owned)); return {'ready':True}
        tiny = [dict(j,seconds=15) for j in jobs]
        with patch('step28_formal_queue.GpuMemory', NV), patch('step30_evolution_control.resource_observation', side_effect=admit):
            runner.run_many(tiny,3,lambda job: completed.append(json.loads((job['target']/'cores.json').read_text())))
        self.assertEqual(len(completed),3); self.assertEqual(len(set(sum(completed,[]))),6)
        self.assertEqual(max(own_counts),2)
        ident = dict(pid=psutil.Process().pid,owner=psutil.Process().username(),created=-100.,reserved_bytes=5*2**30,**kernel_identity(psutil.Process().pid))
        class TightNV(NV):
            def snapshot(self): return {'processes':{},'free':9*2**30}
        obs = resource_observation(g,self.root,True,[ident],TightNV(),limits=BETA_LIMITS)
        self.assertFalse(obs['ready']); self.assertEqual(obs['unreflected_commitments'],5*2**30)
        EVIDENCE['three_slots'] = dict(cpu_affinities=completed,reserved_seconds=140400,
                                      start_ticks_used_despite_created_time_difference=True, pending_gpu_deducted=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--gate',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    ARGS=p.parse_args(); configure_compute(); started=time.monotonic()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    write(ARGS.output,dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
          failures=len(result.failures),errors=len(result.errors),evidence=EVIDENCE,body_seconds=time.monotonic()-started,
          formal_data_read=False,gpu_used=False))
    sys.exit(0 if result.wasSuccessful() else 1)
