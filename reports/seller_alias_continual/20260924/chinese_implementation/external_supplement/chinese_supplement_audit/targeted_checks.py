"""Independent R1/R2 regression checks on synthetic inputs only.
No BGE, CUDA, formal labels or trained checkpoint is loaded. The completed-run gate
is mocked here to isolate reporting; the submitted 12 contracts exercise its synthetic receipt path.
"""
from __future__ import annotations
from pathlib import Path
from contextlib import ExitStack
from unittest import mock
import copy, json, sys, tempfile, time, types, unittest, zipfile
import numpy as np
ROOT=Path(sys.argv[1] if len(sys.argv)>1 else '/mnt/data/chinese_supplement_review')
OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'tests')]
import step28_chinese_base as base
import test_step28_chinese_base_contracts as fixtures
with zipfile.ZipFile('/mnt/data/chinese_review.zip') as z:
    original_text=z.read('scripts/step28_chinese_base.py').decode()
old=types.ModuleType('original_report_only');old.__file__=str(ROOT/'scripts/step28_chinese_base.py')
exec(compile(original_text,'<original_attachment_runner>','exec'),old.__dict__)
RESULTS={'scope':'Handmade evaluation fixtures; no formal data/labels, no BGE, no CUDA, no formal updates.',
         'gate_scope':'validated_run and historical_reference are mocked for independent reporting checks; original submitted contracts separately exercise synthetic complete-run receipts.'}

def fixture():
    c=base.contract(); groups,metadata=fixtures.public_fixture()
    _,part=base.partition(groups,metadata,c)
    rng=np.random.default_rng(67019)
    owners=np.repeat(np.arange(12),[3]*4+[2]*8)
    pairs=np.array([(i,j) for i in range(28) for j in range(i+1,28)])
    truth=[]; labelled=[]
    for g in groups['development']:
        o=owners[rng.permutation(28)]
        t=(o[pairs[:,0]]==o[pairs[:,1]]).astype(np.uint8)
        truth.append(t);labelled.append(base.data.Group(g.uid,g.sellers,g.items,tuple(t)))
    truth=np.array(truth)
    arrays={}
    for ai,arm in enumerate(base.ARMS):
        points={}
        for epoch in ['3','6']:
            z=rng.normal(size=(60,378)) + truth*(.6+.1*ai) + np.repeat([-.3,.05,.2],20)[:,None]
            points[epoch]={'development':(np.round(z*8)/8).astype(np.float32)}
        arrays[arm]={'points':points,'threshold':float(ai)*.125+.25}
    reference={'scores':(np.round((rng.normal(size=(60,378))+.4*truth)*8)/8).astype(np.float32),
               'threshold':.625,'files':{'handmade_fixture':True}}
    r={'partition':{'path':'partition.json','sha256':'handmade_fixture'},'formal_training_seconds':0.0}
    return c,groups,metadata,part,labelled,arrays,reference,r

def invoke(module,root,tag,pack,fault=None):
    c,groups,metadata,part,labelled,arrays,reference,r=pack
    out=root/'mock_completed_run';out.mkdir(exist_ok=True)
    (out/'manifest.json').write_text('{"handmade_fixture":true}\n')
    dst=root/tag
    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(module.public,'public_inputs',return_value=(groups,metadata,{'handmade':True})))
        parse=stack.enter_context(mock.patch.object(module.public,'attach_labels',return_value=labelled))
        stack.enter_context(mock.patch.object(module,'validated_run',return_value=(arrays,part,r)))
        stack.enter_context(mock.patch.object(module,'historical_reference',return_value=reference))
        if fault=='rng':
            stack.enter_context(mock.patch.object(module.np.random,'default_rng',side_effect=RuntimeError('injected rng')))
        elif fault in ('automatic_first','automatic_fifth'):
            real=module.automatic_report; limit=1 if fault=='automatic_first' else 5; calls=[0]
            def injected(*args,**kwargs):
                calls[0]+=1
                if calls[0]==limit: raise RuntimeError('injected '+fault)
                return real(*args,**kwargs)
            stack.enter_context(mock.patch.object(module,'automatic_report',side_effect=injected))
        elif fault=='comparison':
            stack.enter_context(mock.patch.object(module,'metric_comparison',side_effect=RuntimeError('injected comparison')))
        try:
            result=module.evaluate(out,dst)
        except RuntimeError as e:
            if fault is None or not str(e).startswith('injected '): raise
            result=None
        assert parse.call_count==1
    return dst,result

def subtract_added(value):
    if isinstance(value,dict):
        return {k:subtract_added(v) for k,v in value.items() if k not in ('fixed_classification','domains','group_ids','partition')}
    if isinstance(value,list): return [subtract_added(x) for x in value]
    return value

class TargetedChecks(unittest.TestCase):
    def test_R1_success_unchanged_from_original(self):
        with tempfile.TemporaryDirectory() as td:
            pack=fixture(); root=Path(td)
            old_dst,previous=invoke(old,root,'old',pack)
            new_dst,current=invoke(base,root,'new',pack)
            self.assertEqual(subtract_added(previous),subtract_added(current))
            self.assertEqual(len(current['comparisons']),7)
            self.assertEqual(len(current['columns']),22)
            self.assertEqual(current['primary_comparison'],pack[0]['primary_comparison'])
            self.assertTrue((new_dst/'evaluation.json').exists())
            self.assertFalse((new_dst/'failure.json').exists())
            RESULTS['success_equivalence']={'existing_fields_exactly_equal':True,'comparisons':7,'metrics':22,
              'handmade_parser_calls_per_invocation':1,'extra_fields_only':['domains','group_ids','partition','fixed_classification']}

    def test_R1_faults_and_counts_only_reconstruction(self):
        checks=[]
        with tempfile.TemporaryDirectory() as td:
            pack=fixture(); root=Path(td)
            _,success=invoke(base,root,'success',pack)
            for fault in ['rng','automatic_first','automatic_fifth','comparison']:
                with self.subTest(fault=fault):
                    dst,result=invoke(base,root,fault,pack,fault)
                    self.assertIsNone(result)
                    saved=base.data.read_json(dst/'collected.json')
                    self.assertEqual(saved['status'],'CHINESE_BASE_METRICS_COLLECTED_BEFORE_UNCERTAINTY')
                    self.assertFalse((dst/'evaluation.json').exists())
                    self.assertEqual(base.data.read_json(dst/'failure.json')['status'],'EVALUATION_FAILED_NO_RETRY')
                    self.assertEqual(len(list(dst.rglob('*.npy'))),9)
                    self.assertEqual(saved['domains'],[r['domain'] for r in pack[3]['development']])
                    self.assertEqual(saved['group_ids'],[r['group_uid'] for r in pack[3]['development']])
                    matrices={};counts={};file_count=0;fixed_count=0;auto_count=0
                    for arm in base.ARMS:
                        for epoch in ['3','6']:
                            point=saved['arms'][arm]['points'][epoch]
                            self.assertEqual(point,success['arms'][arm]['points'][epoch])
                            p=base.data.verify(dst/point['file']['path'],point['file'])
                            arr=np.load(p,allow_pickle=False)
                            self.assertEqual(arr.shape,(60,22));file_count+=1;fixed_count+=1
                            if epoch=='6': matrices[arm]=arr
                        raw=saved['arms'][arm]['automatic_classification']
                        self.assertEqual(raw['counts_by_group'],success['arms'][arm]['automatic_classification']['counts_by_group'])
                        self.assertEqual(raw['threshold'],success['arms'][arm]['automatic_classification']['threshold'])
                        counts[arm]=np.asarray(raw['counts_by_group'],dtype=np.int64);auto_count+=1
                    hist=saved['historical_labse']
                    for k in ['counts_at_logit_zero','counts_by_group','fixed_classification','threshold']:
                        self.assertEqual(hist[k],success['historical_labse'][k])
                    matrices['historical_labse']=np.load(base.data.verify(dst/hist['file']['path'],hist['file']),allow_pickle=False)
                    counts['historical_labse']=np.asarray(hist['counts_by_group'],dtype=np.int64)
                    file_count+=1;fixed_count+=1;auto_count+=1
                    reconstructed=0
                    with mock.patch.object(base.public,'attach_labels',side_effect=AssertionError('No parsing during reconstruction')), \
                         mock.patch.object(base.public,'public_inputs',side_effect=AssertionError('No public data access during reconstruction')):
                        e=saved['config']['evaluation']
                        draws=np.random.default_rng(e['bootstrap_seed']).integers(0,20,size=(e['bootstrap_replicates'],3,20))
                        for arm in [*base.ARMS,'historical_labse']:
                            expected=success['arms'][arm]['automatic_classification'] if arm in base.ARMS else success['historical_labse']['automatic_classification']
                            got=base.automatic_report(counts[arm],saved['domains'],draws)
                            self.assertEqual(got,{k:v for k,v in expected.items() if k not in ('threshold','counts_by_group')})
                        for a,b in saved['config']['comparisons']:
                            got={'metrics':base.metric_comparison(matrices[a],matrices[b],saved['domains'],draws),
                                 'automatic':base.automatic_comparison(counts[a],counts[b],saved['domains'],draws)}
                            self.assertEqual(got,success['comparisons'][f'{a}_minus_{b}']);reconstructed+=1
                    checks.append({'fault':fault,'matrix_files':file_count,'fixed_zero_count_tables':fixed_count,
                      'calibrated_count_tables':auto_count,'group_ids':len(saved['group_ids']),
                      'handmade_parser_calls':1,'success_receipt':False,'reconstruction_new_parser_calls':0,
                      'all_five_automatic_reports_equal':True,'reconstructed_comparisons_equal':reconstructed})
                    if fault=='automatic_first':
                        (OUT/'first_interval_collection_handmade.json').write_text(json.dumps(saved,ensure_ascii=False,indent=2)+'\n')
        RESULTS['faults']=checks

    def test_R2_independent_aggregate_arithmetic(self):
        rng=np.random.default_rng(4407)
        domains=np.array(list('A'*20+'B'*20+'C'*20))
        tp=rng.integers(0,21,60);fp=rng.integers(0,70,60)
        rows=[{'tp':int(t),'fp':int(f),'fn':20-int(t),'tn':358-int(f)} for t,f in zip(tp,fp)]
        order=rng.permutation(60);rows=[rows[i] for i in order];domains=domains[order].tolist()
        got=base.fixed_classification(rows,domains)
        for scope in ['pooled','A','B','C']:
            selected=rows if scope=='pooled' else [r for r,d in zip(rows,domains) if d==scope]
            totals={k:sum(r[k] for r in selected) for k in ['tp','fp','fn','tn']}
            t,f,n,z=(totals[k] for k in ['tp','fp','fn','tn'])
            expected={**totals,'precision':t/(t+f) if t+f else 0.,'recall':t/(t+n) if t+n else 0.,
                       'f1':2*t/(2*t+f+n) if 2*t+f+n else 0.,'fpr':f/(f+z) if f+z else 0.}
            actual=got['pooled'] if scope=='pooled' else got['by_domain'][scope]
            self.assertEqual(actual,expected)
        witness=[{'tp':1,'fp':0,'fn':19,'tn':358},{'tp':9,'fp':9,'fn':11,'tn':349}]*3
        w=base.fixed_classification(witness,list('AABBCC'))
        self.assertEqual(w['pooled']['precision'],10/19)
        self.assertNotEqual(w['pooled']['precision'],.75)
        zero=base.fixed_classification([{'tp':0,'fp':0,'fn':20,'tn':358}]*3,list('ABC'))
        self.assertEqual(zero['pooled']['precision'],0.)
        self.assertEqual(zero['pooled']['f1'],0.)
        RESULTS['fixed_aggregation']={'independent_60_group_shuffled_domain_check':True,
          'witness_pooled_precision':w['pooled']['precision'],'witness_macro_precision':.75,
          'no_predicted_positive_case':True,'aggregation_values':got}

if __name__=='__main__':
    start=time.monotonic()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(TargetedChecks)
    run=unittest.TextTestRunner(verbosity=2).run(suite)
    RESULTS.update(status='PASS' if run.wasSuccessful() else 'FAIL',tests=run.testsRun,skipped=len(run.skipped),
                   failures=len(run.failures),errors=len(run.errors),seconds=time.monotonic()-start)
    (OUT/'targeted_results.json').write_text(json.dumps(RESULTS,ensure_ascii=False,indent=2)+'\n')
    sys.exit(0 if run.wasSuccessful() else 1)
