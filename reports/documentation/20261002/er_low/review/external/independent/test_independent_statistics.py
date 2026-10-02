"""Saved-only low-study audit with handwritten new labels/scores and frozen old metrics.
The new synthetic rows are NOT lambda=.1 scientific observations.
"""
from __future__ import annotations
import copy, hashlib, json, os, traceback, unittest
from pathlib import Path
from unittest import mock
import numpy as np
import step28_er_weight as m
import step28_er_weight_run as run
import step28_er_weight_evaluate as prod
import test_step28_bge_continual_contracts as fixture
import statistics_reference as ref
E=Path(os.environ.get('ER_AUDIT_EVIDENCE','/mnt/data/er_low_audit/evidence'))/'statistics'


def write(name,value):
    path=E/name;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def load_records(root,collection,names):
    arrays={};counts={};records=[]
    for name in names:
        arrays[name]={};counts[name]={}
        for role, info in collection['points'][name].items():
            for kind, record in info.items():
                path=root/record['path'];content=path.read_bytes()
                assert len(content)==record['bytes'] and hashlib.sha256(content).hexdigest()==record['sha256']
                records.append(dict(path=str(path),bytes=len(content),sha256=hashlib.sha256(content).hexdigest()))
            arrays[name][role]=np.load(root/info['matrix']['path'],allow_pickle=False)
            counts[name][role]=json.loads((root/info['counts']['path']).read_text())
    return arrays,counts,records


class IndependentStatistics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        E.mkdir(parents=True,exist_ok=True)
        cls.p=m.contract('low');cls.oldjob=m.data.ROOT/cls.p['baseline']['local_small_job'];cls.weightjob=m.data.ROOT/cls.p['weight_reference']['local_small_job']
        baseline=m.baseline(cls.p,cls.oldjob,cls.weightjob)
        cls.domains=baseline['collected']['domains']
        cls.groups=[fixture.handmade_group(uid,i%7) for i,uid in enumerate(baseline['collected']['group_ids'])]
        truth=np.asarray([g.labels for g in cls.groups],dtype=np.uint8)
        cls.scores={};saved_inputs={'handmade_truth_not_formal':truth};x=np.arange(378,dtype=np.float64)
        for oi,order in enumerate(('ABC','BCA','CAB')):
            for stage in (2,3):
                name=f'{order}_tenth_stage{stage}'
                raw=np.empty((60,378),dtype=np.float32)
                for i,d in enumerate(cls.domains):
                    signal=.35+.04*oi+.08*stage+.03*'ABC'.index(d)+.006*(i%20)
                    raw[i]=(np.sin((x+3*i+11*stage)*.173)+.31*np.cos((x+7*oi)*.093)+signal*truth[i]-.5+.009*i).astype(np.float32)
                roles={'raw':raw,'stage-cal':raw.astype(np.float64)*(.8+.04*stage)-1.2+.08*oi,'first-cal':raw.astype(np.float64)*.71-.95}
                cls.scores[name]=roles
                for role,a in roles.items():saved_inputs[f'{name}__{role}']=a
        np.savez_compressed(E/'handwritten_new_inputs.npz',**saved_inputs)
        cls.root=E/'collection';cls.collection=run.collect(cls.root,cls.scores,cls.groups,baseline['partition'],m.sources(cls.p),cls.p)
        assert len(list(cls.root.glob('*.npy')))==18 and len(list(cls.root.glob('*_counts.json')))==18
        assert not (cls.root/'evaluation.json').exists()
        # Force a known post-save failure; it must leave all 18 and all 81 saved sets recoverable.
        expected=None
        with mock.patch.object(prod,'evaluate_matrices',side_effect=RuntimeError('INJECTED_AFTER_ALL_NEW_AND_OLD_SETS_SAVED')):
            try:prod.finalize(cls.root,cls.oldjob/'evaluation',cls.p,cls.weightjob/'evaluation')
            except RuntimeError as e:
                expected=dict(kind='EXPECTED_FAULT_INJECTION_NOT_TEST_FAILURE',exception=str(e),traceback=traceback.format_exc())
        assert expected is not None
        reused=json.loads((cls.root/'reference/collected.json').read_text())
        assert reused['status']=='REUSED_81_FROZEN_METRIC_COUNT_SETS'
        expected.update(new_sets=len(cls.collection['points'])*3,reused_sets=len(reused['points'])*3,evaluation_exists=(cls.root/'evaluation.json').exists())
        write('expected_postsave_failure.json',expected)
        with mock.patch.object(m.base,'load_model',side_effect=AssertionError('No model loading in saved-only statistics')),mock.patch.object(run.prior,'parse_once',side_effect=AssertionError('No train/valid parsing in saved-only statistics')):
            cls.result=prod.finalize(cls.root,cls.oldjob/'evaluation',cls.p,cls.weightjob/'evaluation')
        cls.new,cls.newcounts,newrecords=load_records(cls.root,cls.collection,list(cls.collection['points']))
        cls.old,cls.oldcounts,oldrecords=load_records(cls.root/'reference',reused,list(reused['points']))
        cls.arrays={**cls.old,**cls.new};cls.counts={**cls.oldcounts,**cls.newcounts}
        write('matrix_count_hash_checks.json',dict(new_records=newrecords,reused_records=oldrecords,new_sets=18,old_sets=81))
        cls.independent,draws=ref.evaluate(cls.arrays,cls.domains)
        write('independent_endpoints_comparisons.json',cls.independent)
        np.save(E/'independent_bootstrap_draws.npy',draws,allow_pickle=False)

    def test_all_handwritten_new_metric_rows_and_old_count_ratios(self):
        errors=[];checks=0;countrows=0
        for name,roles in self.scores.items():
            for role,scores in roles.items():
                for i,group in enumerate(self.groups):
                    expected,c=ref.metrics_one(group.labels,scores[i]);observed=self.new[name][role][i]
                    errors.append(float(np.max(np.abs(observed-expected))));checks+=len(expected);countrows+=1
                    self.assertEqual(c,self.newcounts[name][role][i])
                    np.testing.assert_allclose(observed,expected,atol=2e-12,rtol=0)
        old_errors=[];old_checks=0
        for name,roles in self.oldcounts.items():
            for role,rows in roles.items():
                for i,row in enumerate(rows):
                    self.assertEqual(row['tp']+row['fn'],20);self.assertEqual(row['fp']+row['tn'],358)
                    for key,value in ref.confusion(row).items():
                        err=abs(self.old[name][role][i,ref.COLUMNS.index(key)]-value);old_errors.append(err);old_checks+=1
                        self.assertLessEqual(err,2e-12)
        write('metric_row_results.json',dict(new_metric_scalar_checks=checks,new_confusion_rows_exact=countrows,new_max_abs_error=max(errors),old_classification_scalar_checks=old_checks,old_max_abs_error=max(old_errors),no_formal_labels_read=True))

    def test_every_endpoint_interval_order_comparison_and_guard(self):
        scalar_count=0;max_error=0.
        def compare(rows,expected):
            nonlocal scalar_count,max_error
            self.assertEqual(set(rows),set(expected))
            for metric,a in rows.items():
                b=expected[metric]
                aa=[a['mean'],*a['conditional_95pct_interval'],*(a['per_order'][o] for o in ref.ORDERS)]
                bb=[b['mean'],*b['conditional_95pct_interval'],*(b['per_order'][o] for o in ref.ORDERS)]
                scalar_count+=len(aa);max_error=max(max_error,float(np.max(np.abs(np.asarray(aa)-bb))))
                np.testing.assert_allclose(aa,bb,atol=3e-12,rtol=0)
        for arm,roles in self.result['endpoints'].items():
            for role,endpoints in roles.items():
                for endpoint,rows in endpoints.items():compare(rows,self.independent['endpoints'][arm][role][endpoint])
        for name,comparison in self.result['comparisons'].items():
            expected=self.independent['comparisons'][name]
            for kind in ('primary','against_raw_reference'):
                for endpoint,rows in comparison[kind].items():compare(rows,expected[kind][endpoint])
            self.assertEqual(comparison['interpretation']['checks'],expected['checks'])
            self.assertEqual(comparison['interpretation']['pilot_observed_checks_pass'],expected['all_23'])
        self.assertEqual(set(self.result['comparisons']),{'tenth_minus_quarter','tenth_minus_seq'})
        self.assertEqual(self.result['selection']['selected'],self.independent['selected'])
        draws=np.load(self.root/'bootstrap_draws.npy',allow_pickle=False)
        self.assertTrue(np.array_equal(draws,np.load(E/'independent_bootstrap_draws.npy',allow_pickle=False)))
        write('full_statistics_results.json',dict(numeric_scalar_checks=scalar_count,max_abs_error=max_error,boolean_guard_checks=46,draws_shape=list(draws.shape),selected=self.result['selection']['selected'],comparisons={name:{'passed':sum(row['checks'].values()),'total':23,'O_MAP':row['primary']['O']['map']} for name,row in self.independent['comparisons'].items()},scope='Handwritten new scores versus frozen old metrics; NOT scientific lambda=.1 outcome'))

    def test_pooled_counts_and_saved_only_idempotent_recovery(self):
        error=0.;n=0
        for name,roles in self.counts.items():
            for role,rows in roles.items():
                output=self.result['absolute_stage_results'][name][role]['pooled_fixed_half_classification']
                for domain in ('all','A','B','C'):
                    selected=[r for i,r in enumerate(rows) if domain=='all' or self.domains[i]==domain]
                    pooled={key:sum(r[key] for r in selected) for key in ('tp','fp','fn','tn')}
                    actual=output['pooled'] if domain=='all' else output['by_domain'][domain]
                    for key,value in {**pooled,**ref.confusion(pooled),'fpr':pooled['fp']/(pooled['fp']+pooled['tn'])}.items():
                        n+=1;error=max(error,abs(actual[key]-value));self.assertAlmostEqual(actual[key],value,places=12)
        before=hashlib.sha256((self.root/'evaluation.json').read_bytes()).hexdigest()
        with mock.patch.object(m.base,'load_model',side_effect=AssertionError('No model reading')),mock.patch.object(run.prior,'parse_once',side_effect=AssertionError('No data parser')):
            prod.finalize(self.root,self.oldjob/'evaluation',self.p,self.weightjob/'evaluation')
        after=hashlib.sha256((self.root/'evaluation.json').read_bytes()).hexdigest();self.assertEqual(before,after)
        write('recovery_and_pooled_results.json',dict(pooled_scalar_checks=n,max_abs_error=error,evaluation_sha256_before=before,evaluation_sha256_after=after,identical=True,new_metric_count_sets=self.result['new_metric_count_sets'],reused_metric_count_sets=self.result['reused_metric_count_sets']))

    def test_selection_uses_quarter_and_all_23_not_seq(self):
        endpoints={'tenth':{'primary':{'O':{'map':{'mean':.4},'recall_at_5':{'mean':.6}}}}}
        cases=[]
        for quarter in (False,True):
            for seq in (False,True):
                comparisons={f'tenth_minus_{arm}':{'interpretation':{'pilot_observed_checks_pass':value}} for arm,value in [('quarter',quarter),('seq',seq)]}
                got=prod.select_configuration(endpoints,comparisons,self.p)
                self.assertEqual(got['selected'],'tenth' if quarter else 'quarter')
                cases.append(dict(quarter_pass=quarter,seq_pass=seq,selection=got))
        # Construct favorable nonzero differences; flip each of the 23 conditions individually.
        delta={ep:{k:dict(mean=(-.1 if k in ('brier','log_loss') else .1),conditional_95pct_interval=[.05,.15],per_order={o:.1 for o in ref.ORDERS}) for k in ref.COLUMNS} for ep in ref.ENDPOINTS}
        raw={ep:copy.deepcopy(delta[ep]) for ep in ('O','N')}
        self.assertTrue(all(ref.checks(delta,raw).values()))
        flips=[]
        for target in ref.checks(delta,raw):
            d=copy.deepcopy(delta);r=copy.deepcopy(raw)
            if target=='old_map_improves':d['O']['map']['mean']=0.
            elif target=='old_map_interval_above_zero':d['O']['map']['conditional_95pct_interval'][0]=0.
            elif target=='old_recall5_improves':d['O']['recall_at_5']['mean']=0.
            elif target=='new_map_non_decrease':d['N']['map']['mean']=-.001
            elif target=='new_recall5_non_decrease':d['N']['recall_at_5']['mean']=-.001
            else:
                ep,rest=target.split('_',1)
                if rest.endswith('_against_raw_reference'):
                    metric=rest.removesuffix('_against_raw_reference');r[ep][metric]['mean']=.001
                else:
                    metric=rest.removesuffix('_non_degradation');d[ep][metric]['mean']=.001 if metric in ('brier','log_loss') else -.001
            expected=ref.checks(d,r);actual=prod.previous.comparison_checks(d,r)
            self.assertEqual(actual['checks'],expected);self.assertEqual([key for key,value in expected.items() if not value],[target])
            self.assertFalse(actual['pilot_observed_checks_pass']);flips.append(target)
        write('selection_truth_table_and_23_individual_guards.json',dict(truth_table=cases,individually_flipped_guards=flips))

    def test_metrics_ties_and_handcalculated_endpoint_difference(self):
        labels=self.groups[0].labels
        for scores in (np.zeros(378),np.round(np.sin(np.arange(378)),1),np.linspace(-3,3,378)):
            expected,counts=ref.metrics_one(labels,scores);actual,c=m.metrics.group_metrics(np.array([labels],dtype=np.uint8),np.array([scores]))
            np.testing.assert_allclose(actual[0],expected,atol=2e-12,rtol=0);self.assertEqual(c[0],counts)
        # A deliberately nonconstant R(t,j) family, not computed using production endpoint terms.
        arrays={};domains=['A']*20+['B']*20+['C']*20
        for oi,order in enumerate(ref.ORDERS):
            for stage in (1,2,3):
                for arm in ('seq','er','half','quarter','tenth'):
                    name=order+'_shared' if stage==1 else f'{order}_{arm}_stage{stage}'
                    v=np.zeros((60,22));base=.1*stage+.01*oi+np.repeat([.003,.007,.011],20)+np.tile(np.arange(20)*.0001,3)
                    offset={'seq':.02,'er':.04,'half':.06,'quarter':.08,'tenth':.13}[arm] if stage>1 else 0.
                    v[:]=np.asarray(base+offset)[:,None]
                    arrays[name]={role:v.copy() for role in ('raw','stage-cal','first-cal')}
        _,weights=ref.draw_counts()
        a=ref.direct_endpoint_samples(arrays,domains,'tenth','primary',weights)
        b=ref.direct_endpoint_samples(arrays,domains,'quarter','primary',weights)
        manual={ep:(.05 if ep in ('O','N','Z','final_all') else -.05 if ep in ('F_first','F') else .025) for ep in ref.ENDPOINTS}
        evidence={}
        for ep,value in manual.items():
            reference=ref.summary(a[ep]-b[ep])['map'];self.assertAlmostEqual(reference['mean'],value,places=12)
            f=prod.previous.endpoint_fields({k.replace('_tenth_', '_er_'):v for k,v in arrays.items() if '_tenth_' in k or '_shared' in k},domains,'er','primary',ep)
            g=prod.previous.endpoint_fields({k.replace('_quarter_', '_er_'):v for k,v in arrays.items() if '_quarter_' in k or '_shared' in k},domains,'er','primary',ep)
            observed=prod.previous.summarize_field(f-g,prod.previous.bootstrap_draws())['map']
            self.assertAlmostEqual(observed['mean'],value,places=12);evidence[ep]=dict(hand_expected_map_delta=value,observed=observed)
        write('tie_and_handcomputed_endpoint_checks.json',dict(tie_cases=3,handwritten_R_cases=evidence))


if __name__=='__main__':unittest.main(verbosity=2)
