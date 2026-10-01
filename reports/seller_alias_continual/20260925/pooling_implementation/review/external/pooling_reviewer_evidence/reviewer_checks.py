"""Independent reviewer checks. ONLY handwritten data and tiny CPU models.
Receipt fixtures are explicitly synthetic; no actual BGE execution is claimed.
Original submitted files are never modified.
"""
from __future__ import annotations
import contextlib, copy, hashlib, itertools, json, math, os, pathlib, sys, tempfile, unittest
from unittest import mock
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint as activation_checkpoint
ROOT = pathlib.Path('/mnt/data/pooling_review_extracted')
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'tests')]
import step28_alias_pooling as m
import step28_alias_pooling_run as r
from test_step28_alias_pooling_contracts import handmade_group, tiny_model, TinyEncoder
from test_step28_chinese_base_contracts import public_fixture, truth28
BASE_SOURCES = m.base.sources()
SOURCES = r.sources()
POLICY = m.contract()
DETAILS = {}
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

class Budget:
    def check(self, *args): pass
    def state(self): return {'handmade_only': True}


def make_fixture(root: pathlib.Path):
    """Synthetic file receipts for exercising the real twelve-model validation."""
    p = copy.deepcopy(POLICY)
    groups, meta = public_fixture()
    selected, part = m.base.partition(groups, meta, m.reference_config(p, 's0'))
    y = np.tile(truth28(), (60,1))
    rng = np.random.default_rng(75)
    logits = (rng.normal(-1, .8, y.shape) + y*.9).astype(np.float32)
    matrix, counts = m.metrics.group_metrics(y, logits)
    history = root/'old_run'; ar = history/'split_rank'; er = root/'old_evaluation'
    (ar/'models').mkdir(parents=True); (ar/'scores').mkdir(); (er/'split_rank').mkdir(parents=True)
    old_arm = {'updates':864, 'preflight':{'initial_model_state_sha256':'handmade-common'}, 'points':{}}
    schedule,stream = m.base.schedule(selected['fit'],m.base.contract())
    old_arm.update(group_schedule_sha256=hashlib.sha256(m.data.json_bytes([g.uid for g in schedule])).hexdigest(),dropout_stream=stream)
    models=[]
    for epoch in ('3','6'):
        f=ar/'models'/f'epoch{epoch}.pt'; f.write_bytes(f'NOT A REAL MODEL old {epoch}'.encode()); models.append(f)
        sf=ar/'scores'/f'epoch{epoch}_development.npy'; np.save(sf,logits)
        old_arm['points'][epoch]={'model':{**m.data.record(f,ar),'actual_reload_verified':True},
            'model_state_sha256':f'handmade-old-{epoch}', 'scores':{'development':m.data.record(sf,ar)}}
        np.save(er/'split_rank'/f'epoch{epoch}_metrics.npy',matrix)
    cal_y = np.tile(truth28(),(36,1)); cal_s=np.tile(logits[0],(36,1))
    cal = m.base.calibrate(cal_y,cal_s,[x['domain'] for x in part['calibration']])
    old_arm['points']['6']['scores']['calibration']={'sha256':'handmade-calibration-score'}
    cal.update(epoch=6,model_state_sha256='handmade-old-6',score_sha256='handmade-calibration-score')
    m.data.write_json(ar/'calibration.json',cal);old_arm['calibration']=m.data.record(ar/'calibration.json',ar)
    m.data.write_json(ar/'manifest.json',old_arm)
    m.data.write_json(history/'partition.json',part)
    oldrun={'status':m.base.COMPLETE,'config':m.base.contract(),'source_files':BASE_SOURCES,
            'arms':{'split_rank':{'manifest':m.data.record(ar/'manifest.json',history)}}}
    m.data.write_json(history/'manifest.json',oldrun)
    m.data.write_json(history/'completion.json',{'manifest_sha256':m.data.sha256(history/'manifest.json')})
    oldcol={'group_ids':[x['group_uid'] for x in part['development']],
            'domains':[x['domain'] for x in part['development']],
            'arms':{'split_rank':{'points':{e:{'counts_at_logit_zero':counts} for e in ('3','6')}}}}
    m.data.write_json(er/'collected.json',oldcol)
    p['historical_d'].update(run_root='old_run',evaluation_root='old_evaluation',
        files=[m.data.record(f,root) for f in sorted(list(history.rglob('*'))+list(er.rglob('*'))) if f.is_file() and f.suffix!='.pt'],
        expected_initial_common_state_sha256='handmade-common',
        group_schedule_sha256=old_arm['group_schedule_sha256'],dropout_stream=stream,
        model_records_from_frozen_manifest={f'epoch{e}':old_arm['points'][e]['model'] for e in ('3','6')})
    out=root/'job'/'run';out.mkdir(parents=True)
    m.data.write_json(root/'audit.json',{'handmade_receipt':True})
    m.data.write_json(out/'partition.json',part)
    checked={'handmade_public_inputs':True}
    run={'status':r.COMPLETE,'source_files':SOURCES,'policy':p,'inputs':checked,
         'runs':{},'physical_updates':4320,'label_parses':{'train':1,'development':0,'heldout':0,'owners':0},
         'native_verification':m.data.record(root/'audit.json',root),'partition':m.data.record(out/'partition.json',out),
         'preflights':{}}
    for run_id in m.NEW_RUNS:
        seed,kind=run_id.split('_');c=m.reference_config(p,seed)
        folder=out/run_id;(folder/'models').mkdir(parents=True);(folder/'scores').mkdir()
        sch,st=m.base.schedule(selected['fit'],c)
        before={'handmade_only':True};run['preflights'][run_id]=before
        mods=('encoder','head','aggregation') if kind=='weighted' else ('encoder','head')
        observations={key:{'finite_nonzero_gradient':True,'parameters_changed':True} for key in mods}
        arm={'run_id':run_id,'reference_config':c,'updates':864,'label_parses':0,'dropout_stream':st,
             'group_schedule_sha256':hashlib.sha256(m.data.json_bytes([g.uid for g in sch])).hexdigest(),
             'fit_group_ids':[g.uid for g in selected['fit']],
             'calibration_group_ids':[g.uid for g in selected['calibration']], 'preflight':before,'points':{},
             'training':[{'start':start,'stop':start+432,'updates':432,
                 'observations':{k:observations for k in (('1','2') if start==0 else ('433',))},
                 'mean_losses_by_epoch':{'bce':[.5]*3,'rank':[2.]*3,'total':[2.5]*3}} for start in (0,432)]}
        for epoch in ('3','6'):
            pt={'run_id':run_id,'epoch':int(epoch),'full_model_and_adam_reloaded':True,
                'model_state_sha256':f'handmade-{run_id}-{epoch}','scores':{},'train_metrics':{}}
            for role in m.base.ROLES:
                n=m.base.ROLE_SIZES[role];ss=logits if role=='development' else np.tile(logits[0],(n,1))
                f=folder/'scores'/f'epoch{epoch}_{role}.npy';np.save(f,ss);pt['scores'][role]=m.data.record(f,folder)
                if role!='development':
                    f=folder/'scores'/f'epoch{epoch}_{role}_metrics.npy';np.save(f,np.zeros((n,22),np.float64))
                    pt['train_metrics'][role]={'file':m.data.record(f,folder)}
            f=folder/'models'/f'epoch{epoch}.pt';f.write_bytes(f'NOT A REAL MODEL {run_id} {epoch}'.encode());models.append(f)
            pt['model']={**m.data.record(f,folder),'actual_reload_verified':True}
            arm['points'][epoch]=pt
        ca=m.base.calibrate(cal_y,cal_s,[x['domain'] for x in part['calibration']])
        ca.update(epoch=6,model_state_sha256=arm['points']['6']['model_state_sha256'],score_sha256=arm['points']['6']['scores']['calibration']['sha256'])
        m.data.write_json(folder/'calibration.json',ca);arm['calibration']=m.data.record(folder/'calibration.json',folder)
        m.data.write_json(folder/'manifest.json',arm)
        run['runs'][run_id]={'updates':864,'manifest':m.data.record(folder/'manifest.json',out)}
    m.data.write_json(out/'manifest.json',run)
    m.data.write_json(out/'completion.json',{'status':r.COMPLETE,'manifest_sha256':m.data.sha256(out/'manifest.json')})
    return p,groups,meta,checked,out,models

@contextlib.contextmanager
def fixture_context(root,p,groups,meta,checked):
    with mock.patch.object(m.data,'ROOT',root),mock.patch.object(m,'contract',return_value=p),\
         mock.patch.object(m.base,'sources',return_value=BASE_SOURCES),mock.patch.object(r,'sources',return_value=SOURCES),\
         mock.patch.object(m.base.public,'public_inputs',return_value=(groups,meta,checked)):
        yield

class ReviewerChecks(unittest.TestCase):
    def test_heterogeneous_bootstrap_all_22(self):
        rng=np.random.default_rng(901)
        domains=np.array(list('ABC'*20))
        delta=rng.normal(.007,.018,(3,60,22))
        # Strongly correlated fixed seeds: catches treating 180 rows as independent.
        delta += rng.normal(0,.1,(1,60,22))
        result=m.paired_summary(delta,domains.tolist(),POLICY['evaluation'])
        draws=np.random.default_rng(20260925).integers(0,20,(5000,3,20))
        weights=np.zeros((5000,60))
        for b in range(5000):
            for di,d in enumerate('ABC'):
                rows=np.flatnonzero(domains==d)
                weights[b,rows]=np.bincount(draws[b,di],minlength=20)/60
        # Count-weight formulation and manual linear percentile interpolation.
        boot=weights @ (delta[0]+delta[1]+delta[2])/3
        errors=[]
        for j,key in enumerate(m.metrics.COLUMNS):
            ordered=np.sort(boot[:,j]);q=[]
            for prob in (.025,.975):
                at=(len(ordered)-1)*prob;lo=math.floor(at);hi=math.ceil(at)
                q.append(ordered[lo]+(at-lo)*(ordered[hi]-ordered[lo]))
            np.testing.assert_allclose(result['metrics'][key]['conditional_95pct_interval'],q,rtol=0,atol=1e-14)
            np.testing.assert_allclose(result['metrics'][key]['per_seed'],delta[:,:,j].mean(1),rtol=0,atol=1e-14)
            errors.append(max(abs(np.array(q)-result['metrics'][key]['conditional_95pct_interval'])))
        DETAILS['heterogeneous_bootstrap']={'metrics':22,'draws':5000,'max_interval_error':max(errors)}

    def test_new_adam_group_independent_first_step(self):
        model=tiny_model(POLICY,True);c=m.reference_config(POLICY,'s0');opt=m.make_optimizer(model,c,POLICY)
        before={name:p.detach().clone() for name,p in model.aggregation.named_parameters()}
        log=m.update(model,opt,handmade_group(),c,441,observe=True)
        pg=opt.param_groups[2];assert pg['lr']==.001 and pg['weight_decay']==.01 and pg['betas']==(.9,.999) and pg['eps']==1e-8 and pg['foreach'] is False
        maximum=0
        for name,p in model.aggregation.named_parameters():
            # Independent AdamW t=1 expression, using actual clipped gradient.
            g=p.grad.detach().double();expected=before[name].double()*(1-.001*.01)-.001*g/(g.abs()+1e-8)
            maximum=max(maximum,float((p.detach().double()-expected).abs().max()))
            torch.testing.assert_close(p.detach().double(),expected,rtol=0,atol=4e-8)
        assert log['modules']['aggregation']['parameter_gradient_norms_before_clip']['hidden.weight']==0
        DETAILS['aggregation_adam_first_step_max_error']=maximum

    def test_dropout_checkpoint_shared_rng(self):
        class Encoder(TinyEncoder):
            def __init__(self):
                super().__init__();self.drop=torch.nn.Dropout(.3);self.seen=[]
            def forward(self,features):
                def block(ids):return self.drop(self.embedding(ids)).mean(1)
                result=activation_checkpoint(block,features['input_ids'],use_reentrant=False,preserve_rng_state=True) if self.training else block(features['input_ids'])
                self.seen.append(result.detach().clone());return {'sentence_embedding':result}
        torch.manual_seed(111);d=m.core.build_model(Encoder(),16,7)
        w=copy.deepcopy(d);rng=torch.random.get_rng_state().clone();m.attach_aggregation(w,POLICY,61,dimension=4)
        assert torch.equal(rng,torch.random.get_rng_state())
        c=m.reference_config(POLICY,'s0');states=[];values=[]
        for model in (d,w):
            opt=m.make_optimizer(model,c,POLICY);m.update(model,opt,handmade_group(),c,912,observe=True)
            states.append(torch.random.get_rng_state().clone());values.append(model.encoder.seen)
        assert torch.equal(*states);assert len(values[0])==len(values[1])
        for a,b in zip(*values):torch.testing.assert_close(a,b,rtol=0,atol=0)
        DETAILS['dropout_cpu_checkpoint']={'identical_encoder_microbatches':len(values[0]),'post_backward_rng_equal':True,'real_BGE':False}

    def test_checkpoint_fresh_model_and_adam_continuation(self):
        c=m.reference_config(POLICY,'s0');one=tiny_model(POLICY,True);opt=m.make_optimizer(one,c,POLICY)
        m.update(one,opt,handmade_group(),c,88)
        with tempfile.TemporaryDirectory() as d:
            out=pathlib.Path(d)
            for name in ('scores','models','work'):(out/name).mkdir()
            report=r.checkpoint(one,opt,POLICY,c,'s0_weighted',3,{role:[handmade_group()] for role in m.base.ROLES},out,Budget())
            assert report['full_model_and_adam_reloaded'] and not list((out/'work').iterdir())
            fresh=tiny_model(POLICY,True)
            m.core.restore_state(out/'models/epoch3.pt',fresh,None,report['model']['state_sha256'])
            np.testing.assert_array_equal(m.score(one,[handmade_group()],c),m.score(fresh,[handmade_group()],c))
            full=m.core.save_state(out/'full.pt',one,opt,{'handmade':True})
            opt2=m.make_optimizer(fresh,c,POLICY)
            m.core.restore_state(out/'full.pt',fresh,opt2,full['state_sha256'])
            for model,optimizer in ((one,opt),(fresh,opt2)):m.update(model,optimizer,handmade_group(),c,89)
            assert m.core.state_digest(one.state_dict())==m.core.state_digest(fresh.state_dict())
            assert m.core.state_digest(opt.state_dict())==m.core.state_digest(opt2.state_dict())
        DETAILS['tiny_checkpoint']={'actual_runner_checkpoint':True,'fresh_model_inference_equal':True,'next_update_model_and_adam_equal':True,'formal_epochs_completed':0}

    def test_all_twelve_model_files_rejected_before_parser_on_damage(self):
        with tempfile.TemporaryDirectory() as d:
            root=pathlib.Path(d);p,groups,meta,checked,out,files=make_fixture(root)
            with fixture_context(root,p,groups,meta,checked):
                _,_,result=r.validate_run(out,p,groups,meta,checked)
                assert len(result['fresh_model_files_verified_before_valid'])==12
                for i,path in enumerate(files):
                    payload=path.read_bytes();path.write_bytes(bytes([payload[0]^1])+payload[1:])
                    try:
                        with mock.patch.object(m.base.public,'attach_labels',side_effect=AssertionError('Parser must not be called')) as parser:
                            with self.assertRaises(ValueError):r.evaluate(out,root/f'rejected_{i}',p,Budget())
                            parser.assert_not_called()
                    finally:path.write_bytes(payload)
        DETAILS['twelve_file_gate']={'intact_pass':True,'same_length_damage_rejected_before_parser':12,'real_model_files':False}

    def test_collection_failure_resume_and_historical_difference(self):
        with tempfile.TemporaryDirectory() as d:
            root=pathlib.Path(d);p,groups,meta,checked,out,files=make_fixture(root)
            labelled=[m.data.Group(g.uid,g.sellers,g.items,tuple(map(int,truth28()))) for g in groups['development']]
            scenarios=[('primary_summary',m,'paired_summary'),('automatic_interval',m.base,'automatic_report'),('metric_comparison',m.base,'metric_comparison')]
            with fixture_context(root,p,groups,meta,checked):
                with mock.patch.object(m.base.public,'attach_labels',return_value=labelled) as parser:
                    normal=r.evaluate(out,root/'normal',p,Budget());self.assertEqual(parser.call_count,1)
                self.assertFalse(normal['acceptance']['passed']) # Equal arms do NOT become an improvement.
                for name,module,function in scenarios:
                    dest=root/name
                    with mock.patch.object(m.base.public,'attach_labels',return_value=labelled) as parser,\
                         mock.patch.object(module,function,side_effect=RuntimeError('handmade injected failure')):
                        with self.assertRaisesRegex(RuntimeError,'handmade injected failure'):r.evaluate(out,dest,p,Budget())
                        self.assertEqual(parser.call_count,1)
                    self.assertEqual(len(list(dest.glob('*/epoch*_metrics.npy'))),12)
                    collected=m.data.read_json(dest/'collected.json')
                    self.assertTrue((dest/'historical_alignment.json').exists())
                    for run in collected['runs'].values():
                        self.assertEqual(len(run['automatic_classification']['counts_by_group']),60)
                        for pt in run['points'].values():self.assertEqual(len(pt['counts_at_logit_zero']),60)
                    with mock.patch.object(m.base.public,'attach_labels',side_effect=AssertionError('No reparse')),\
                         mock.patch.object(m.base.public,'public_inputs',side_effect=AssertionError('No input reload')):
                        resumed=r.finalize(dest)
                    for key in ('paired_primary','per_seed_comparisons','acceptance'):self.assertEqual(resumed[key],normal[key])
                # Changed saved metric with a matching local receipt models a real
                # history-alignment discrepancy, not a corrupt-file gate failure.
                dest=root/'historical_discrepancy';dest.mkdir()
                arrays,part,run=r.validate_run(out,p,groups,meta,checked)
                collected=r.collect(np.tile(truth28(),(60,1)),arrays,part,dest,p,{'handmade':True})
                f=dest/'s0_d/epoch3_metrics.npy';value=np.load(f);value[0,0]+=1e-6;np.save(f,value)
                collected['runs']['s0_d']['points']['3']['file']=m.data.record(f,dest);m.data.write_json(dest/'collected.json',collected)
                with mock.patch.object(m,'paired_summary',side_effect=AssertionError('History first')),\
                     mock.patch.object(m.base.public,'attach_labels',side_effect=AssertionError('No reparse')):
                    with self.assertRaisesRegex(ValueError,'Historical D evaluation differs'):r.finalize(dest)
                alignment=m.data.read_json(dest/'historical_alignment.json')
                self.assertGreater(alignment['points']['3']['max_absolute_difference'],1e-7)
                self.assertEqual(len(list(dest.glob('*/epoch*_metrics.npy'))),12)
        DETAILS['collection_failure_resume']={'failure_points':3,'matrices_preserved_each':12,'counts_complete':True,'parse_per_evaluation':1,'resume_formal_input_and_label_parses':0,'history_discrepancy_saved_before_rejection':True}

    def test_handwritten_retrieval_and_curve_semantics(self):
        y=truth28();pairs=list(itertools.combinations(range(28),2));rng=np.random.default_rng(12);maximum=0
        for s in (np.zeros(378),rng.integers(-2,3,378).astype(float),rng.normal(size=378),np.where(y,4.,-4.),np.where(y,-4.,4.)):
            actual=m.metrics.retrieval(y[None,:],s[None,:],28)[0];rows=[]
            for query in range(28):
                opts=[]
                for edge,(a,b) in enumerate(pairs):
                    if a==query or b==query:opts.append((float(s[edge]),b if a==query else a,int(y[edge])))
                opts.sort(key=lambda x:(-x[0],x[1]));rel=[x[2] for x in opts];n=sum(rel);hit=[i+1 for i,z in enumerate(rel) if z]
                row=[sum(sum(rel[:pos])/pos for pos in hit)/n,1/min(hit)]
                row.extend(sum(rel[:k])/n for k in (1,3,5,10))
                row.extend(sum(z/math.log2(i+2) for i,z in enumerate(rel[:k]))/sum(1/math.log2(i+2) for i in range(min(k,n))) for k in (1,3,5,10))
                rows.append(row)
            expected=np.array(rows).mean(0);np.testing.assert_allclose(actual,expected,atol=1e-14,rtol=0)
            maximum=max(maximum,float(np.max(abs(actual-expected))))
            # Independent pair-counting interpretation of AUC (including ties).
            pos=s[y==1];neg=s[y==0];auc=float(((pos[:,None]>neg).mean()+.5*(pos[:,None]==neg).mean()))
            self.assertAlmostEqual(m.metrics.curve_metrics(y,s)['roc_auc'],auc,places=14)
        DETAILS['handmade_metrics']={'score_patterns':5,'retrieval_columns':10,'max_error':maximum,'auc_independent_pair_counting':True}

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ReviewerChecks)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    DETAILS.update(environment={'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'cpu_threads':torch.get_num_threads(),'cuda':torch.cuda.is_available()},
        tests={'run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped)},
        scope={'formal_texts_read':0,'formal_labels_read':0,'real_BGE_loads':0,'formal_updates':0,'GPU_execution':False})
    pathlib.Path(__file__).with_name('reviewer_checks.json').write_text(json.dumps(DETAILS,ensure_ascii=False,indent=2))
    sys.exit(0 if result.wasSuccessful() else 1)
