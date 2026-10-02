"""Actual tiny continuation/checkpoint continuity and low-specific gate regressions.
No native model or project data; the first Adam counter is a declared fixture.
"""
from __future__ import annotations
import copy,json,os,tempfile,traceback,unittest
from pathlib import Path
from unittest import mock
import numpy as np
import torch
import step28_er_weight as m
import step28_er_weight_run as run
import test_step28_bge_continual_contracts as f
import test_step28_er_weight_contracts as wf
E=Path(os.environ.get('ER_AUDIT_EVIDENCE','/mnt/data/er_low_audit/evidence'))/'paths'

def write(name,value):
    path=E/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

class IndependentPaths(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True);E.mkdir(parents=True,exist_ok=True)

    def test_actual_288_tenth_updates_checkpoint_rng_and_next_update(self):
        p=m.contract('low');model,opt,c,_,_=wf.prepared()
        root=E/'tiny_stage2'
        for folder in ('updates','work','models','scores','maps','points'): (root/folder).mkdir(parents=True,exist_ok=True)
        firstmap={'a':1.13,'b':-.41}
        current=[f.handmade_group(f'ind_fit_B_{i}',2) for i in range(48)]
        history=[f.handmade_group(f'ind_fit_A_{i}',1) for i in range(48)]
        memory=m.parent.Memory('ABC',m.parent.contract()['memory_seed'],False,firstmap);memory.retain(history,1,None)
        original_cache=memory.to_bytes();(root/'handmade_first_cache.json').write_bytes(original_cache)
        oracle=m.parent.Memory.from_bytes(original_cache);oracle.begin_stage(2)
        sequence,stream=m.parent.schedule(current,m.parent.contract(),'ABC',2)
        history_ids=[oracle.draw()[0].uid for _ in range(288)]
        expected={'current_ids':[g.uid for g in sequence],'history_ids':history_ids,'memory_after_training':oracle.summary()}
        write('paired_schedule_fixture.json',dict(note='Original unchanged schedule/memory algorithm applied to handwritten groups, not live project cache',**expected,current_dropout_stream=stream))
        budget=f.HandmadeBudget();budget.state=lambda:{'independent_tiny_fixture':True}
        with mock.patch.object(m,'update',wraps=m.update) as update:
            log=run.train_stage(model,opt,current,memory,c,'ABC',2,'tenth',root,expected,budget,p)
            self.assertEqual(update.call_count,288)
            self.assertTrue(all(call.args[5]==.1 for call in update.call_args_list))
        self.assertEqual(m.parent.adam_step(opt),576);self.assertEqual(log['history_ids'],history_ids)
        self.assertEqual(memory.summary()['members'],oracle.summary()['members']);self.assertEqual(log['current_ids'],expected['current_ids'])
        values=np.load(root/log['update_file']['path'],allow_pickle=False)
        v={key:values[:,i] for i,key in enumerate(m.STEP_COLUMNS)}
        self.assertTrue(np.array_equal(v['history_weight'],np.full(288,.1)))
        np.testing.assert_array_equal(v['weighted_history_total'],.1*v['history_total'])
        self.assertEqual(v['encoder_lr'][-1],0.);self.assertTrue(np.all(v['head_lr']==.001))
        before_model=m.core.state_digest(model.state_dict());before_opt=m.core.state_digest(opt.state_dict());before_rng=run.rng_state()
        # A tiny full-state reference is retained independently; production removes only its verified intermediate.
        fullpath=root/'independent_full_state.pt';metadata={'rng':before_rng,'scope':'handwritten tiny reference after 288 actual continuations','config':c}
        full=m.core.save_state(fullpath,model,opt,metadata)
        point=run.checkpoint(root,'ABC_tenth_stage2',model,opt,c,'ABC',2,[f.handmade_group(f'ind_cal_B_{i}') for i in range(12)],[f.handmade_group(f'ind_valid_{i}') for i in range(60)],firstmap,budget,.1,p)
        self.assertEqual(before_model,m.core.state_digest(model.state_dict()));self.assertEqual(before_opt,m.core.state_digest(opt.state_dict()));self.assertEqual(before_rng,run.rng_state())
        self.assertEqual(point['history_weight'],.1);self.assertEqual(point['policy_sha256'],m.LOW_POLICY_SHA256)
        self.assertEqual(point['first_map_parameters'],firstmap);self.assertFalse((root/point['full_checkpoint']['path']).exists())
        fresh=f.tiny_model();freshopt=m.core.make_optimizer(fresh,c)
        restored=m.core.restore_state(fullpath,fresh,freshopt,full['state_sha256']);run.restore_rng(restored['rng'])
        self.assertEqual(before_model,m.core.state_digest(fresh.state_dict()));self.assertEqual(before_opt,m.core.state_digest(freshopt.state_dict()))
        next_cur=f.handmade_group('ind_fit_C_next',3);next_old=memory.draw()[0]
        # Two actual stage-3 updates compare uninterrupted and file-restored states, with identical dropout seeds.
        a=m.update(model,opt,next_cur,next_old,c,.1,3,1,581,582)
        b=m.update(fresh,freshopt,next_cur,next_old,c,.1,3,1,581,582)
        self.assertEqual(m.core.state_digest(model.state_dict()),m.core.state_digest(fresh.state_dict()))
        self.assertEqual(m.core.state_digest(opt.state_dict()),m.core.state_digest(freshopt.state_dict()))
        self.assertEqual(a,b)
        evidence=dict(actual_continuation_updates=288,actual_next_stage_parity_updates=2,actual_initial_warm_updates=1,initial_Adam_counter_rebased_fixture=288,after_288_Adam_counter=576,next_Adam_counter=m.parent.adam_step(opt),gradient_presentations_in_continuation=576,full_record=full,checkpoint=point,model_before_checkpoint=before_model,optimizer_before_checkpoint=before_opt,RNG_before_equals_after=True,continuation_after_restore_exact=True,current_groups=48,retained_history_groups=len(memory.summary()['members']),history_unique_draws=len(set(history_ids)),first_cache_bytes=len(original_cache),next_update=a)
        write('actual_stage_checkpoint_evidence.json',evidence)

    def test_low_six_point_gate_and_rejected_targeted_mutations(self):
        p=m.contract('low');failures=[]
        with tempfile.TemporaryDirectory() as directory:
            root,manifest,reference=wf.gate_fixture(Path(directory),p)
            baseline=copy.deepcopy(manifest)
            self.assertEqual(len(run.blind_gate(root,manifest,reference,p)),6)
            write('gate_complete_manifest_fixture.json',manifest)
            # These are purposely fake model identity files, never native-checkpoint evidence.
            for name,change in [('wrong_weight',lambda x:x['restored_starts']['ABC_tenth'].update(adam_step=289)),('missing_endpoint',lambda x:x['points'].pop('CAB_tenth_stage3')),('wrong_policy_sha',lambda x:x.update(policy_sha256=m.POLICY_SHA256)),('extra_point',lambda x:x['points'].update(ABC_quarter_stage2=x['points']['ABC_tenth_stage2']))]:
                modified=copy.deepcopy(baseline);change(modified)
                try:run.blind_gate(root,modified,reference,p)
                except ValueError as error:failures.append(dict(case=name,expected_rejection=True,error=str(error),traceback=traceback.format_exc()))
                else:self.fail('Mutation should have been rejected: '+name)
            # Real weight-field mutation with size/hash record updated so it exercises the scientific check.
            name='ABC_tenth_stage2';path=root/baseline['training'][name]['path'];saved=path.read_bytes();log=json.loads(saved);log['history_weight']=.25;m.data.write_json(path,log)
            modified=copy.deepcopy(baseline);modified['training'][name]=m.data.record(path,root)
            try:run.blind_gate(root,modified,reference,p)
            except ValueError as error:failures.append(dict(case='quarter_training_weight_in_tenth_study',expected_rejection=True,error=str(error),traceback=traceback.format_exc()))
            else:self.fail('Quarter update in tenth study should be rejected')
            path.write_bytes(saved)
            self.assertEqual(len(run.blind_gate(root,baseline,reference,p)),6)
        write('gate_mutation_evidence.json',dict(scope='Six endpoint file fixtures; not six trained native models',complete_points=6,targeted_expected_rejections=failures))

if __name__=='__main__':unittest.main(verbosity=2)
