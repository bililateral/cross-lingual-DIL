"""External CPU audit. All labels are handmade; no formal CSV or model loading.
Existing fake_run is used ONLY to construct synthetic filesystem receipts.
It is not a model run and is not an oracle for numerical checks.
"""
from __future__ import annotations
import copy, hashlib, importlib.metadata, itertools, json, os, platform, sys, tempfile, time, unittest, zipfile
from pathlib import Path
from unittest import mock
import numpy as np
import torch
ROOT = Path('/mnt/data/chinese_review')
OUT = Path('/mnt/data/chinese_external_audit')
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'tests')]
import step28_chinese_base as b
import test_step28_chinese_base_contracts as fixture

torch.set_num_threads(1)
PAIRS = list(itertools.combinations(range(28), 2))
# Independently hand-specified controller components, used only for unit testing.
CONTROLLERS = np.repeat(np.arange(12), [2]*8+[3]*4)
Y = np.array([int(CONTROLLERS[i] == CONTROLLERS[j]) for i,j in PAIRS], dtype=np.uint8)
RESULT = {'scope':'handmade CPU only; no formal data/labels, native weights, or CUDA',
          'environment': {'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,
                          'numpy':np.__version__,'cuda_available':torch.cuda.is_available(),'threads':torch.get_num_threads()},
          'checks':{}}

def record(name, details):
    RESULT['checks'][name] = details
    print(name+': '+json.dumps(details, ensure_ascii=False), flush=True)

class TinyTextEncoder(torch.nn.Module):
    """Engineering fixture, NOT BGE or a substitute for SentenceTransformer."""
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(97, 8)
        self.proj = torch.nn.Linear(8, 8)
    def tokenizer(self, texts, **kwargs):
        ids = [[ord(c)%96+1 for c in text] for text in texts]
        length=max(map(len,ids))
        return {'input_ids':torch.tensor([v+[0]*(length-len(v)) for v in ids]),
                'attention_mask':torch.tensor([[1]*len(v)+[0]*(length-len(v)) for v in ids])}
    def forward(self, batch):
        mask=batch['attention_mask'].unsqueeze(-1)
        return {'sentence_embedding':self.proj((self.emb(batch['input_ids'])*mask).sum(1)/mask.sum(1))}

def group():
    sellers=tuple(f'hand_s{i:02}' for i in range(28))
    items=tuple(tuple((f'hand_item_{i:02}_{j}',f'手写标题{i}条目{j}',f'手写描述{i%9}异同{j%3}')
                      for j in range(2+i%3)) for i in range(28))
    g=b.data.Group('independent_handmade',sellers,items,tuple(map(int,Y)))
    g.validate()
    return g

class Checks(unittest.TestCase):
    def test_01_inventory_and_historical_blind_reference(self):
        with zipfile.ZipFile(ROOT.parent/'chinese_review.zip') as z:
            self.assertIsNone(z.testzip())
            self.assertEqual(len(z.namelist()),len(set(z.namelist())))
        inv=b.data.read_json(ROOT/'reports/seller_alias_continual/20260924/chinese_implementation/source_inventory.json')
        for row in inv['files']: b.data.verify(ROOT/row['path'],row)
        c=b.contract()
        download=b.data.read_json(ROOT/'reports/seller_alias_continual/20260924/chinese_model/download.json')
        self.assertEqual(len(download['files']),12)
        self.assertEqual(sum(r['size_bytes'] for r in download['files']),1302803319)
        digest=hashlib.sha256(b.data.json_bytes(download['files']).rstrip(b'\n')).hexdigest()
        self.assertEqual(digest,download['content_sha256'])
        self.assertEqual(digest,c['models']['mean_bce']['content_sha256'])
        checked=b.data.read_json(ROOT/'reports/seller_alias_continual/20260924/chinese_implementation/checked_inputs.json')['files']
        receipt=b.data.read_json(ROOT/'reports/seller_alias_continual/20260924/chinese_implementation/cpu_initial/checked_sources.json')['verified_files']
        self.assertEqual(checked,receipt)
        for row in checked: b.data.verify(ROOT/row['path'],row)
        part=b.data.read_json(ROOT/c['historical_reference']['files']['partition.json']['path'])
        with mock.patch.object(b.public,'attach_labels',side_effect=AssertionError('No label access')):
            hist=b.historical_reference(c,part)
            changed=copy.deepcopy(part); changed['development'][0],changed['development'][1]=changed['development'][1],changed['development'][0]
            with self.assertRaises(ValueError): b.historical_reference(c,changed)
        self.assertEqual(hist['scores'].shape,(60,378))
        record('inventory_and_blind_reference',{'payloads_verified':50,'zip_crc':'PASS','checked_sources_verified':11,
                 'historical_shape':list(hist['scores'].shape),'historical_dtype':str(hist['scores'].dtype),
                 'historical_threshold':hist['threshold'],'partition_reordering_rejected':True,
                 'native_weights_present':False,'formal_label_parses':0})

    def test_02_rank_scalar_gradient_and_invalid_graph(self):
        rng=np.random.default_rng(4421)
        errors=[]
        for scale in [0,1,50]:
            values=rng.normal(size=378)*scale
            z=torch.tensor(values,dtype=torch.float64,requires_grad=True)
            terms=b.objectives(z,torch.tensor(Y,dtype=torch.float64),1)
            loss=[]; grad=np.zeros(378)
            for i in range(28):
                ix=np.array([k for k,p in enumerate(PAIRS) if i in p])
                x=values[ix]; positive=Y[ix].astype(bool)
                maximum=x.max(); e=np.exp(x-maximum); probability=e/e.sum()
                loss.append(maximum+np.log(e.sum())-x[positive].mean())
                grad[ix]+=(probability-positive/positive.sum())/28
            actual=torch.autograd.grad(terms['rank'],z,retain_graph=True)[0].numpy()
            np.testing.assert_allclose(actual,grad,atol=2e-15,rtol=2e-12)
            self.assertAlmostEqual(float(terms['rank'].detach()),float(np.mean(loss)),places=11)
            shifted=b.objectives(z+37,torch.tensor(Y,dtype=torch.float64),1)
            self.assertAlmostEqual(float(terms['rank'].detach()),float(shifted['rank'].detach()),places=11)
            expected_bce=np.mean(np.logaddexp(0,values)-Y*values)
            self.assertAlmostEqual(float(terms['bce'].detach()),float(expected_bce),places=11)
            errors.append(float(np.max(np.abs(actual-grad))))
        # 8 disjoint edges + one 12-cycle: identical edge/degree counts, invalid equivalence relation.
        edges={tuple(sorted((i,i+1))) for i in range(0,16,2)}
        edges|={tuple(sorted((i,16+(i-16+1)%12))) for i in range(16,28)}
        bad=torch.tensor([int(p in edges) for p in PAIRS],dtype=torch.float64)
        self.assertEqual(int(bad.sum()),20)
        with self.assertRaisesRegex(ValueError,'disjoint controller cliques'):
            b.objectives(torch.zeros(378,dtype=torch.float64),bad,1)
        record('rank_numeric',{'cases':3,'maximum_gradient_absolute_error':max(errors),'invalid_degree_two_cycle_rejected':True,
                              'equal_logits_rank':float(np.log(27)),'equal_logits_bce':float(np.log(2)),
                              'rank_invariant_to_constant_logit_shift':True})

    def test_03_unequal_items_moments_and_order(self):
        rng=np.random.default_rng(893)
        counts=[2,3,8]; n=sum(counts)
        x=rng.normal(size=(2*n,1024)).astype(np.float32)
        actual=b.pool_channels(torch.tensor(x),counts,'separate_moments').numpy()
        norm=x.astype(np.float64); norm/=np.linalg.norm(norm,axis=1,keepdims=True)
        expected=[]; offset=0
        for count in counts:
            parts=[]
            for base in [0,n]:
                channel=norm[base+offset:base+offset+count]
                parts.extend([channel.mean(0),np.sqrt(np.mean((channel-channel.mean(0))**2,axis=0)+1e-8)])
            row=np.concatenate(parts); expected.append(row/np.linalg.norm(row)); offset+=count
        expected=np.array(expected)
        np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-8)
        self.assertEqual(actual.shape,(3,4096))
        offsets=np.cumsum([0]+counts)
        permutation=np.concatenate([np.arange(offsets[i],offsets[i+1])[::-1] for i in range(3)])
        within=np.r_[permutation,permutation+n]
        np.testing.assert_allclose(b.pool_channels(torch.tensor(x[within]),counts,'separate_moments').numpy(),actual,rtol=2e-5,atol=2e-8)
        # Move complete account blocks in both channels together, preserving title/description alignment.
        idx=np.concatenate([np.arange(offsets[i],offsets[i+1]) for i in [2,0,1]])
        reordered=b.pool_channels(torch.tensor(x[np.r_[idx,idx+n]]),[8,2,3],'separate_moments').numpy()
        np.testing.assert_allclose(reordered,actual[[2,0,1]],rtol=2e-5,atol=2e-8)
        modified=x.copy(); modified[0]*=-1
        self.assertGreater(float(np.max(np.abs(b.pool_channels(torch.tensor(modified),counts,'separate_moments').numpy()-actual))),0)
        record('moments',{'counts':counts,'embedding_dim':1024,'account_shape':list(actual.shape),
                          'maximum_absolute_error':float(np.max(np.abs(expected-actual))),
                          'within_account_and_account_permutation':'PASS','changed_record_affects_account':True})

    def test_04_exact_calibration_and_gate_boundaries(self):
        truth=np.tile(Y,(36,1)); scores=np.full((36,378),-10,dtype=np.float32)
        scores[truth==1]=20
        domains=['A']*12+['B']*12+['C']*12
        for k in range(3):
            mask=np.flatnonzero(truth[k*12]==0)
            scores[k*12,mask[:6]]=np.array([10,9,8,7,6,6])+k
        result=b.calibrate(truth,scores,domains)
        self.assertEqual(result['threshold'],np.nextafter(np.float64(8),np.inf))
        self.assertTrue(all(v[1]<=4 for v in result['counts_by_domain'].values()))
        self.assertTrue(all(v['negative_pairs']==4296 for v in result['bounds'].values()))
        domains=['A']*20+['B']*20+['C']*20
        draws=np.random.default_rng(20260918).integers(0,20,size=(5000,3,20))
        counts=np.tile([10,0,10,358],(60,1)); counts[0]=[10,7,10,351]
        self.assertTrue(b.automatic_report(counts,domains,draws)['all_domains_pass'])
        bad=counts.copy(); bad[0]=[10,8,10,350]
        report=b.automatic_report(bad,domains,draws)
        self.assertFalse(report['all_domains_pass']); self.assertLess(report['pooled']['fpr'],.001)
        bad=counts.copy(); bad[0]=[9,7,11,351]
        self.assertFalse(b.automatic_report(bad,domains,draws)['all_domains_pass'])
        record('threshold_and_gates',{'threshold':result['threshold'],'calibration_counts_by_domain':result['counts_by_domain'],
              'fp7_tp200':'PASS','fp8_tp200':'REJECT','fp7_tp199':'REJECT','pooled_pass_cannot_hide_domain_failure':True})

    def test_05_metrics_and_paired_bootstrap_independent(self):
        from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, auc
        rng=np.random.default_rng(906); z=np.round(rng.normal(size=378),1)
        actual=b.metrics.curve_metrics(Y,z)
        precision,recall,_=precision_recall_curve(Y,z)
        self.assertAlmostEqual(actual['average_precision'],average_precision_score(Y,z),places=13)
        self.assertAlmostEqual(actual['roc_auc'],roc_auc_score(Y,z),places=13)
        self.assertAlmostEqual(actual['trapezoidal_pr_auc'],auc(recall,precision),places=13)
        rows=[]
        for i in range(28):
            candidates=[(z[k],j if i==a else a,int(Y[k])) for k,(a,j) in enumerate(PAIRS) if i in (a,j)]
            candidates.sort(key=lambda row:(-row[0],row[1]))
            rel=np.array([x[2] for x in candidates]); pos=np.flatnonzero(rel)+1
            row=[np.mean(np.arange(1,len(pos)+1)/pos),1/pos[0]]
            row += [float(rel[:k].sum()/rel.sum()) for k in [1,3,5,10]]
            row += [float((rel[:k]/np.log2(np.arange(2,k+2))).sum()/np.sum(1/np.log2(np.arange(2,min(k,int(rel.sum()))+2)))) for k in [1,3,5,10]]
            rows.append(row)
        np.testing.assert_allclose(b.metrics.retrieval(Y[None],z[None],28)[0],np.mean(rows,axis=0),rtol=1e-13,atol=1e-14)
        domains=['A']*20+['B']*20+['C']*20
        draws=np.random.default_rng(20260918).integers(0,20,size=(5000,3,20))
        count=np.array([[i%21,i%5,20-i%21,358-i%5] for i in range(60)])
        ref=np.array([[(i*3)%21,(i*7)%9,20-(i*3)%21,358-(i*7)%9] for i in range(60)])
        report=b.automatic_report(count,domains,draws)
        pooled=np.array([sum((count[20*d+draws[k,d]].sum(0) for d in range(3)),start=np.zeros(4,dtype=int)) for k in range(5000)])
        for name,n,j in [('fpr',1,3),('recall',0,2),('precision',0,1)]:
            vals=pooled[:,n]/(pooled[:,n]+pooled[:,j])
            np.testing.assert_allclose(report['pooled']['conditional_95pct_intervals'][name],np.quantile(vals,[.025,.975]),rtol=0,atol=0)
        comparison=b.automatic_comparison(count,ref,domains,draws)
        pooled_ref=np.array([sum((ref[20*d+draws[k,d]].sum(0) for d in range(3)),start=np.zeros(4,dtype=int)) for k in range(5000)])
        for name,n,j in [('fpr',1,3),('recall',0,2),('precision',0,1)]:
            vals=pooled[:,n]/(pooled[:,n]+pooled[:,j])-pooled_ref[:,n]/(pooled_ref[:,n]+pooled_ref[:,j])
            np.testing.assert_allclose(comparison[name]['pooled']['conditional_95pct_interval'],np.quantile(vals,[.025,.975]),rtol=0,atol=0)
        a=rng.normal(size=(60,22)); r=rng.normal(size=(60,22)); cmp=b.metric_comparison(a,r,domains,draws)
        independent=np.array([np.mean(np.concatenate([(a-r)[20*d+draws[k,d]] for d in range(3)]),axis=0) for k in range(5000)])
        for i,col in enumerate(b.metrics.COLUMNS):
            np.testing.assert_allclose(cmp[col]['conditional_95pct_interval'],np.quantile(independent[:,i],[.025,.975]),rtol=1e-12,atol=1e-14)
        record('metrics_bootstrap',{'columns':len(b.metrics.COLUMNS),'AP_PR_ROC_reference':'installed sklearn; handmade tied logits',
                'retrieval_reference':'independent query-wise Python ranking','replicates':5000,
                'pooled_intervals_and_paired_differences':'PASS','all22_metric_difference_intervals':'PASS'})

    def test_06_four_arm_update_adam_and_eight_tiny_checkpoints(self):
        c=b.contract(); g=group(); reports={}
        for arm in b.ARMS:
            torch.manual_seed(913)
            model=b.core.build_model(TinyTextEncoder(),32 if arm.startswith('split_') else 8,12)
            optimizer=b.core.make_optimizer(model,c)
            old={id(p):p.detach().clone() for p in model.parameters()}
            log=b.update(model,optimizer,g,c,arm,20260918,observe=True)
            errors=[]
            for pg in optimizer.param_groups:
                for p in pg['params']:
                    grad=p.grad.detach().double(); prior=old[id(p)].double()
                    expected=prior*(1-pg['lr']*pg['weight_decay'])-pg['lr']*grad/(grad.abs()+pg['eps'])
                    errors.append(float((p.detach().double()-expected).abs().max()))
                    torch.testing.assert_close(p.detach().double(),expected,atol=4e-7,rtol=1e-6)
                    self.assertEqual(float(optimizer.state[p]['step']),1.)
            clipped=np.sqrt(sum(float(p.grad.double().square().sum()) for p in model.parameters()))
            self.assertLessEqual(clipped,1.000002)
            with tempfile.TemporaryDirectory() as td:
                out=Path(td)
                for name in ['models','work','scores']: (out/name).mkdir()
                budget=b.persistence.Budget(out,c)
                for epoch in [3,6]:
                    # Epoch names are API fixture arguments; NOT six epochs of training.
                    p=b.checkpoint_point(model,optimizer,c,arm,epoch,{r:[g] for r in b.ROLES},out,budget)
                    self.assertTrue(p['full_model_and_adam_reloaded'])
                    self.assertFalse(list((out/'work').iterdir()))
                    self.assertEqual(len(p['scores']),3)
                self.assertEqual(len(list((out/'models').iterdir())),2)
            reports[arm]={'adam_first_step_max_abs_error':max(errors),'post_clip_total_norm':clipped,
                          'losses':{k:log[k] for k in ['bce','rank','total']},'tiny_checkpoint_points':2}
        record('tiny_updates_and_checkpoints',reports)

    def test_07_all_eight_model_corruption_gates_before_parser(self):
        cases=[]
        with tempfile.TemporaryDirectory() as td:
            run=Path(td)/'job'/'run'
            c,groups,metadata,checked=fixture.fake_run(run)
            for arm in b.ARMS:
                for epoch in [3,6]:
                    path=run/arm/'models'/f'epoch{epoch}.pt'; saved=path.read_bytes()
                    path.write_bytes(bytes([saved[0]^1])+saved[1:])
                    with mock.patch.object(b.public,'public_inputs',return_value=(groups,metadata,checked)), \
                         mock.patch.object(b.public,'attach_labels') as parse:
                        with self.assertRaises(ValueError): b.evaluate(run,Path(td)/f'reject_{arm}_{epoch}')
                        parse.assert_not_called()
                    path.write_bytes(saved); cases.append(f'{arm}/epoch{epoch}')
        record('all_eight_model_gates',{'same_size_corruptions_rejected':cases,'label_parser_calls':0,'files_are':'synthetic receipt fixture bytes, not weights'})

    def test_08_failure_order_and_fixed_zero_report_witness(self):
        with tempfile.TemporaryDirectory() as td:
            run=Path(td)/'job'/'run'; c,groups,metadata,checked=fixture.fake_run(run)
            handmade=[b.data.Group(g.uid,g.sellers,g.items,tuple(map(int,fixture.truth28()))) for g in groups['development']]
            reference={'scores':np.tile(np.where(fixture.truth28(),2.,-2.).astype(np.float32),(60,1)),
                       'threshold':0.,'files':{'fixture':'not historical scores'}}
            dest=Path(td)/'failure'
            with mock.patch.object(b.public,'public_inputs',return_value=(groups,metadata,checked)), \
                 mock.patch.object(b.public,'attach_labels',return_value=handmade) as parse, \
                 mock.patch.object(b,'historical_reference',return_value=reference), \
                 mock.patch.object(b,'automatic_report',side_effect=RuntimeError('injected interval failure')):
                with self.assertRaisesRegex(RuntimeError,'injected interval failure'): b.evaluate(run,dest)
                self.assertEqual(parse.call_count,1)
            files=[p.relative_to(dest).as_posix() for p in dest.rglob('*.npy')]
            self.assertEqual(len(files),2); self.assertFalse((dest/'collected.json').exists())
            failure=json.loads((dest/'failure.json').read_text())
            normal=Path(td)/'normal'
            order=[]; orig_report=b.automatic_report
            def trace(*args,**kwargs):
                order.append(len(list(normal.rglob('*.npy'))))
                return orig_report(*args,**kwargs)
            with mock.patch.object(b.public,'public_inputs',return_value=(groups,metadata,checked)), \
                 mock.patch.object(b.public,'attach_labels',return_value=handmade), \
                 mock.patch.object(b,'historical_reference',return_value=reference), \
                 mock.patch.object(b,'automatic_report',side_effect=trace):
                result=b.evaluate(run,normal)
            self.assertEqual(order,[2,4,6,8,9])
            self.assertEqual(len(result['comparisons']),7)
            schema=result['arms']['mean_bce']['points']['6']
            self.assertEqual(set(schema),{'file','counts_at_logit_zero','mean','by_domain'})
            record('reproduced_failure_order',{'status':'DEFECT_REPRODUCED','injected_at':'first automatic_report',
                'handmade_parser_calls':1,'matrices_surviving':files,'collected_exists':False,'failure_receipt':failure,
                'normal_run_matrices_present_at_each_automatic_report':order,'required_total_matrices':9,
                'zero_threshold_point_keys':list(schema),'zero_threshold_counts_available':'per group only; no pooled/domain micro classification summary'})


    def test_09_active_global_gradient_clipping(self):
        c=b.contract(); torch.manual_seed(721)
        model=b.core.build_model(TinyTextEncoder(),8,12)
        # Amplify only the handmade fixture; frozen production policy remains unchanged.
        with torch.no_grad(): model.head[2].weight.mul_(100.)
        optimizer=b.core.make_optimizer(model,c)
        log=b.update(model,optimizer,group(),c,'mean_bce',20260918)
        after=np.sqrt(sum(float(p.grad.double().square().sum()) for p in model.parameters() if p.grad is not None))
        self.assertGreater(log['gradient_norm_before_clip'],1.)
        self.assertAlmostEqual(after,1.,places=5)
        record('active_global_clip',{'production_clip_norm':c['optimizer']['clip_norm'],
              'handmade_preclip_norm':log['gradient_norm_before_clip'],'handmade_postclip_norm':after,
              'actual_clip_triggered':True})

if __name__=='__main__':
    start=time.monotonic()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
    res=unittest.TextTestRunner(verbosity=2).run(suite)
    RESULT.update(tests_run=res.testsRun,failures=len(res.failures),errors=len(res.errors),skipped=len(res.skipped),seconds=time.monotonic()-start)
    (OUT/'independent_results.json').write_text(json.dumps(RESULT,ensure_ascii=False,indent=2)+'\n')
    sys.exit(0 if res.wasSuccessful() else 1)
