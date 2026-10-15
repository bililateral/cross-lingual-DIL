#!/usr/bin/env python3
"""External isolated audit of the ONE authorized saved S/G/F result archive.
No project imports, Torch, model/Memory load, label reconstruction, network or training.
Usage: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python independent_review.py \
 --archive 'review_input(20261015-121313).zip' --output out
Requires Python 3 and NumPy. The archive is read directly; it is not executed.
"""
from __future__ import annotations
import argparse, collections, csv, hashlib, io, json, math, os, platform, random, sys, time, traceback, zipfile
from pathlib import Path
import numpy as np
EXPECTED_SIZE=6123032
EXPECTED_SHA='07f941b2d689c4f9e2cd024364b57e3607bd9bb762868378a854b5be97ef83f6'
EXPECTED_COUNT=376

def sha(b): return hashlib.sha256(b).hexdigest()
def seed(*parts):
    b=(json.dumps(list(parts),ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    return int.from_bytes(hashlib.sha256(b).digest()[:8],'big')%(2**63-1)
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

class Audit:
    def __init__(self, archive:Path, out:Path):
        self.out=out; out.mkdir(parents=True,exist_ok=True); self.started=time.perf_counter()
        blob=archive.read_bytes(); self.identity=dict(file=archive.name,bytes=len(blob),sha256=sha(blob))
        self.require(len(blob)==EXPECTED_SIZE and sha(blob)==EXPECTED_SHA,'authorized archive identity')
        self.z=zipfile.ZipFile(io.BytesIO(blob));self.names=self.z.namelist()
        self.require(len(self.names)==EXPECTED_COUNT and len(set(self.names))==EXPECTED_COUNT,'376 unique members')
        self.used=collections.defaultdict(set);self.errors=collections.defaultdict(lambda:[0,0.]);self.checks=0
        self.results={};self.stats_rows=[];self.diag_rows=[];self.inventory=[]
    def require(self,v,msg):
        if not bool(v):raise AssertionError(msg)
        if hasattr(self,'checks'):self.checks+=1
    def b(self,n,purpose='structured evidence'):
        self.used[n].add(purpose); return self.z.read(n)
    def j(self,n):return json.loads(self.b(n))
    def arr(self,n):
        self.used[n].add('numeric array fully loaded, no pickle');return np.load(io.BytesIO(self.b(n)),allow_pickle=False)
    def ver(self,base,rec):
        n=base+rec['path'];b=self.b(n,'size and SHA reference')
        self.require(len(b)==rec['bytes'] and sha(b)==rec['sha256'],'reference identity '+n);return n
    def cmp(self,a,b,label,atol=1e-12):
        x,y=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
        self.require(x.shape==y.shape,label+' shape')
        self.require(not np.isinf(x).any() and not np.isinf(y).any(),label+' no infinities')
        self.require(np.array_equal(np.isnan(x),np.isnan(y)),label+' identical N/A mask')
        good=~np.isnan(x);err=float(np.max(np.abs(x[good]-y[good]))) if good.any() else 0.
        self.require(err<=atol,label+' numeric difference '+str(err))
        self.errors[label][0]+=int(x.size);self.errors[label][1]=max(self.errors[label][1],err)
    def identity_check(self):
        im=self.j('input_manifest.json');self.require(im['count']==375==len(im['files']),'input manifest count')
        self.require(set(r['path'] for r in im['files'])==set(self.names)-{'input_manifest.json'},'manifest exact coverage')
        for n in self.names:
            b=self.b(n,'full byte identity only'); self.inventory.append(dict(path=n,bytes=len(b),sha256=sha(b)))
        for r in im['files']:self.ver('',r)
        dump(self.out/'input_identity.json',{**self.identity,'members':len(self.names),'manifest_payloads':375})
        with (self.out/'input_members.tsv').open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['path','bytes','sha256'],delimiter='\t');w.writeheader();w.writerows(self.inventory)
        for n,cnt,total in [('result/receive_manifest.json',347,11094881),('result/analysis_manifest.json',7,491617)]:
            d=self.j(n);self.require(len(d['files'])==cnt,'transfer count')
            self.require(sum(x['bytes'] for x in d['files'])==total,'transfer bytes')
            for rec in d['files']:self.ver('result/',rec)
        rv=self.j('result/receive_verification.json')
        for field,n in [('observation_sha256','result/observation.json'),('manifest_sha256','result/receive_manifest.json'),('closeout_sha256','result/closeout.json')]:
            self.require(rv[field]==sha(self.b(n)),'receipt '+field)
        # All array containers, including each of the 36 blinded diagnostic score files.
        arrays=[]
        for n in self.names:
            if n.endswith('.npy'):
                x=self.arr(n);self.require(x.dtype.kind in 'fiu' and np.isfinite(x).all(),'finite numeric array '+n)
                arrays.append(dict(path=n,shape=list(x.shape),dtype=str(x.dtype),elements=int(x.size)))
        dump(self.out/'array_inventory.json',arrays);self.results['array_files']=len(arrays)
    def source_execution(self):
        base='result/job/';run=base+'run/'
        self.m=self.j(run+'manifest.json');self.c=self.j(base+'completion.json');self.col=self.j(base+'evaluation/collected.json')
        self.ev=self.j(base+'evaluation/evaluation.json');before=self.j(base+'before_valid.json');ex=self.j(base+'execution.json')
        self.gate=self.j('result/formal_gate.json');self.policy=self.j(run+'sources/schema/step28_balanced_pooling_policy.json')
        source=self.m['source_files'];self.require(len(source)==31,'31 sources')
        for obj in [self.c,self.col,self.ev,before,self.gate,self.j(run+'startup.json')]:self.require(obj['source_files']==source,'source-chain equality')
        self.require(ex['sources']==source,'execution sources');self.require(sha(self.b('result/formal_gate.json'))==ex['gate']['sha256'],'execution gate hash')
        for rec in source:self.ver(run+'sources/',rec)
        self.ver(base,before['manifest']); self.require(before['access']==dict(train=1,valid=0,heldout=0,owners=0),'blind access')
        self.require(self.j(base+'access.json')==self.c['access']==dict(train=1,valid=1,heldout=0,owners=0),'final access')
        self.require(base+'failure.json' not in self.names,'no current failure member')
        old=self.j('context/reports/seller_alias_continual/20261015/balanced_pooling_failure/evidence/reports/job/failure.json')
        self.require(self.c['budget']['prior_charged_seconds']==old['budget']['elapsed_seconds'],'old failed cost retained')
        self.cmp(self.c['budget']['elapsed_seconds']+self.c['budget']['prior_charged_seconds'],self.c['budget']['cumulative_seconds'],'cost',1e-9)
        self.require(self.c['budget']['limits']==self.policy['runtime']==self.gate['runtime'],'effective runtime')
        self.require(self.c['budget']['cumulative_seconds']<97200,'cumulative budget')
        for peak,lim in [('peak_cuda_reserved_bytes','maximum_cuda_reserved_bytes'),('peak_output_bytes','maximum_output_bytes'),('peak_rss_bytes','maximum_rss_bytes')]:
            self.require(self.c['budget'][peak]<=self.policy['runtime'][lim],peak)
        self.part=self.j(self.ver(run,self.m['partition']));self.points=[d+'_'+a for a in 'SGF' for d in 'AB'];self.roles=['raw','stage-cal','first-cal']
        self.require(set(self.m['points'])==set(self.points),'six endpoints');self.require(self.col['group_ids']==[r['group_uid'] for r in self.part['development']],'group order')
        self.require(self.col['domains']==[r['domain'] for r in self.part['development']],'domain order')
        ids=[r['group_uid'] for v in self.part.values() for r in v];self.require(len(ids)==160==len(set(ids)),'partition disjoint')
        for role,num in [('fit',48),('calibration',12),('development',20)]:
            for dom in 'AB':self.require(sum(x['domain']==dom for x in self.part[role])==num,'partition sizes')
        self.rows=[np.flatnonzero(np.array(self.col['domains'])==d) for d in 'AB']
        self.require([len(x) for x in self.rows]==[20,20],'20 groups per domain')
        init=self.m['initials'];self.require(all(init[a]==init['S'] for a in 'SGF'),'same independent initial identities')
        self.require(len({self.m['points']['A_'+a]['continuity']['model'] for a in 'SGF'})==3,'different learned A model identities')
        for a in 'SGF':self.require(self.m['starts'][a]==self.m['points']['A_'+a]['continuity'],'own A to B continuation')
        # Qualifications are read, not rerun. Match 26 unchanged sources and the four declared changes + one addition.
        native=self.j('context/reports/documentation/20261015/balanced_pooling_repair/qualification/gpu/attempt01/result/result.json')
        ns={r['path']:r for r in native['source_files']};cs={r['path']:r for r in source}
        changed=[n for n in ns if ns[n]!=cs[n]]; added=sorted(set(cs)-set(ns))
        self.require(len(ns)==30 and len(changed)==4 and len(added)==1,'source qualification delta sizes')
        self.require(set(changed)=={'scripts/step28_balanced_pooling_run.py','scripts/run_step28_balanced_pooling_linux_20261015.sh','tests/test_step28_balanced_pooling.py','schema/step28_balanced_pooling_policy.json'},'declared changed source set')
        self.results['qualification_inheritance']={'unchanged':26,'changed':changed,'added':added,'native_actual_updates':native['actual_updates'],'native_cumulative_updates':native['native']['cumulative_updates'],'native_inner_charged_seconds':native['cumulative_charged_seconds'],'outer_accounting_seconds_from_disposition':561}
        self.results['execution']={'source_count':31,'cost':self.c['budget'],'access_before':before['access'],'access_final':self.c['access'],'old_failure_budget':old['budget']}
    def stat(self,x,target,domain,kind,name,role):
        di='AB'.index(domain)
        for j,metric in enumerate(self.cols):
            v=x[:,j];t=target[metric];missing=int(np.isnan(v).sum())
            if missing:
                self.require(t['mean'] is None and t['conditional_95pct_interval'] is None,'N/A mean interval')
                self.require(t['undefined_groups']==missing and t['total_groups']==20,'N/A group count')
                vals=[None,None,None]
            else:
                # Direct gathered whole-group resampling, independent of the submitted audit's frequency multiplication.
                est=v[self.draws[:,di,:]].mean(axis=1);ci=np.quantile(est,[.025,.975],method='linear')
                vals=[float(v.mean()),*ci.tolist()];self.cmp(vals,[t['mean'],*t['conditional_95pct_interval']],'1980 statistics')
            self.stats_rows.append([kind,name,role,domain,metric,*vals])
    def metrics(self):
        run='result/job/run/';er='result/job/evaluation/';self.cols=self.col['columns'];self.require(len(self.cols)==22,'22 columns')
        self.draws=np.random.Generator(np.random.PCG64(20260930)).integers(20,size=(5000,2,20))
        self.results['bootstrap']={'shape':list(self.draws.shape),'dtype':str(self.draws.dtype),'raw_bytes_sha256':sha(self.draws.astype('<i8').tobytes()),'seed':20260930,'replicates':5000,'quantile':'linear'}
        self.M={};self.R={};summary=self.j('result/analysis/summary.json');class_summary={};no_pos={};cal={}
        rankcols=[0,1,2,3,*range(12,22)]
        for point in self.points:
            p=self.m['points'][point];standalone=self.j(run+'points/'+point+'.json');self.require(p==standalone,'point record identity')
            self.require(p['adam_step']==(288 if point[0]=='A' else 576),'endpoint Adam count')
            self.require(p['full_restore_verified'] is True and p['inference_restore_verified'] is True,'restore receipt booleans')
            mapping=self.j(self.ver(run,p['mapping']));self.require(mapping['a']>0 and mapping['group_count']==12 and mapping['pair_count']==4536 and mapping['positive_count']==240,'calibration scope and positive slope')
            self.require(mapping['optimizer_success'] is True and len(mapping['trajectory'])==mapping['optimizer_iterations'],'calibration trajectory evidence')
            self.require(mapping['a']==p['maps']['stage']['a'] and mapping['b']==p['maps']['stage']['b'],'mapping parameters')
            self.require(p['maps']['first']==self.m['points']['A_'+point[2]]['maps']['first'],'own fixed first calibration')
            self.require(p['calibration_ids']==[x['group_uid'] for x in self.part['calibration'] if x['domain']==point[0]],'calibration IDs')
            cal[point]={k:mapping[k] for k in ['a','b','initial_nll','final_nll','projected_gradient_max','optimizer_iterations']}
            self.M[point]={};self.R[point]={};no_pos[point]={};class_summary[point]={};qraw=None
            for role in self.roles:
                rec=self.col['points'][point][role];mat=self.j(self.ver(er,rec['matrix']));self.require(mat['columns']==self.cols,'matrix column identity')
                v=np.asarray(mat['values'],float);self.require(v.shape==(40,22),'matrix shape');self.M[point][role]=v
                counts=self.j(self.ver(er,rec['counts']));ct=np.asarray([[r[k] for k in ['tp','fp','fn','tn']] for r in counts]);self.require(ct.shape==(40,4) and np.issubdtype(ct.dtype,np.integer) and (ct>=0).all(),'counts integer nonnegative')
                tp,fp,fn,tn=ct.T;self.require(np.all(tp+fn==20) and np.all(fp+tn==358),'20/358 classes')
                mask=np.zeros(v.shape,bool);mask[:,6]=tp+fp==0;mask[:,11]=(tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)==0
                self.require(np.array_equal(np.isnan(v),mask),'exact null mask no other missing fields')
                rates=[]
                for a,b,c,d in ct:
                    den=(a+b)*(a+c)*(d+b)*(d+c)
                    rates.append([a/(a+b) if a+b else None,a/(a+c),2*a/(2*a+b+c),d/(d+b),(a/(a+c)+d/(d+b))/2,(a*d-b*c)/math.sqrt(den) if den else None])
                self.cmp(v[:,6:12],rates,'confusion-derived rates')
                q=self.arr(self.ver(er,rec['queries']));self.require(q.shape==(40,28,10) and np.isfinite(q).all() and (q>=0).all() and (q<=1).all(),'query metric shape and bounds')
                self.cmp(q.mean(1),v[:,12:],'query to group means')
                if role=='raw':qraw=q
                else:self.require(np.array_equal(q,qraw),'all query metrics equal across affine roles')
                s=self.arr(self.ver(run,p['scores'][role]));self.require(s.shape==(40,378),'scores dimensions');self.R[point][role]=s
                self.require(np.array_equal((s>=0).sum(1),tp+fp),'predicted positive counts from logits no labels')
                no_pos[point][role]=int((tp+fp==0).sum())
                if role=='raw':
                    for suffix in ['raw','calibration']:
                        orig=self.arr(self.ver(run,p['scores'][suffix]));self.require(orig.shape==((40,378) if suffix=='raw' else (12,378)),'restore score shapes')
                        for prefix in ['full','inference']:
                            restored=self.arr(self.ver(run,p['scores'][prefix+'-'+suffix]));self.require(np.array_equal(orig,restored),'exact two restore score equality')
                else:
                    raw=self.R[point]['raw'];a,b=(p['maps']['stage' if role=='stage-cal' else 'first'][k] for k in ('a','b'))
                    self.require(a>0,'positive affine slope');self.cmp(s,a*raw.astype(float)+b,'affine saved scores')
                    order=np.argsort(raw,axis=1,kind='stable');self.require(np.array_equal(order,np.argsort(s,axis=1,kind='stable')),'stable rank ordering')
                    self.require(np.array_equal(np.diff(np.take_along_axis(raw,order,1),axis=1)==0,np.diff(np.take_along_axis(s,order,1),axis=1)==0),'ties preserved')
                    self.cmp(v[:,rankcols],self.M[point]['raw'][:,rankcols],'ranking metrics role invariance')
                t=self.ev['absolute'][point][role]
                for i,d in enumerate('AB'):
                    self.stat(v[self.rows[i]],t['by_domain'][d],d,'absolute',point,role)
                    for j,k in enumerate(self.cols):self.cmp(t['by_domain'][d][k]['mean'],summary['absolute'][point][role][d][k],'Linux summary absolute')
                self.cmp(v.mean(0),[t['macro_all'][k] for k in self.cols],'macro all N/A propagation')
                self.require(t['macro_undefined_groups']==dict(zip(self.cols,mask.sum(0).tolist())),'macro null counts')
                pooled={}
                for d,idx in [('all',np.arange(40)),('A',self.rows[0]),('B',self.rows[1])]:
                    a,b,c,e=map(int,ct[idx].sum(0));pv=dict(tp=a,fp=b,fn=c,tn=e,fpr=b/(b+e),recall=a/(a+c),precision=a/(a+b) if a+b else None,f1=2*a/(2*a+b+c))
                    for k,val in pv.items():self.cmp(val,t['fixed_half_classification'][d][k],'pooled confusion rates');self.cmp(val,summary['classification'][point][role][d][k],'Linux pooled summary')
                    pooled[d]=pv
                class_summary[point][role]=pooled
            self.require(np.array_equal(self.R[point]['stage-cal'],self.R[point]['first-cal']) if point[0]=='A' else True,'A calibration aliases')
        for cand,other in [('F','S'),('G','S'),('F','G')]:
            name=cand+'-'+other
            for role in self.roles:
                for stage,key,kind in [('B','comparisons','final_contrast'),('A','initial_comparisons','initial_contrast')]:
                    diff=self.M[stage+'_'+cand][role]-self.M[stage+'_'+other][role]
                    for i,d in enumerate('AB'):self.stat(diff[self.rows[i]],self.ev[key][name][role][d],d,kind,name,role)
                diff=(self.M['A_'+cand][role]-self.M['B_'+cand][role])-(self.M['A_'+other][role]-self.M['B_'+other][role])
                self.stat(diff[self.rows[0]],self.ev['forgetting_comparisons'][name][role],'A','forgetting_contrast',name,role)
            for i,d in enumerate('AB'):
                v=(self.M['B_'+cand]['raw']-self.M['B_'+other]['raw'])[self.rows[i],12];t=summary['paired_map_groups'][name][d]
                self.require([int((v>0).sum()),int((v<0).sum()),int((v==0).sum())]==[t['positive'],t['negative'],t['zero']],'paired MAP group signs');self.cmp(v,t['values'],'paired group MAP values')
        for arm in 'SGF':
            for role in self.roles:
                diff=self.M['B_'+arm][role]-self.M['A_'+arm][role]
                self.stat(diff[self.rows[0]],self.ev['old_changes'][arm][role],'A','old_change',arm,role)
        self.require(len(self.stats_rows)==1980,'1980 rows regenerated')
        original=list(csv.reader(io.StringIO(self.b('result/analysis/all_metrics.csv').decode())));self.require(len(original)==1981,'CSV data rows 1980')
        for a,b in zip(self.stats_rows,original[1:]):
            self.require(a[:5]==b[:5],'CSV row identity');self.cmp(a[5:],[float(x) if x!='' else None for x in b[5:]],'CSV 1980 values and nulls')
        with (self.out/'recomputed_statistics.csv').open('w',encoding='utf-8',newline='') as f:
            w=csv.writer(f);w.writerow(original[0]);w.writerows(self.stats_rows)
        dump(self.out/'classification_and_calibration.json',{'no_predicted_positive_groups':no_pos,'pooled':class_summary,'calibration':cal})
        self.results['statistics']={'rows':1980,'null_rows':sum(r[5] is None for r in self.stats_rows),'scalar_matrix_entries':18*40*22,'query_metric_entries':18*40*28*10,'confusion_count_entries':18*40*4,'no_predicted_positive_groups':no_pos}
    def training(self):
        run='result/job/run/';self.T={};members=self.m['points']['A_S']['memory_summary']['members']
        rng=random.Random(seed(20260930,'ABC','retention'));retained=[]
        afit=sorted(x['group_uid'] for x in self.part['fit'] if x['domain']=='A')
        for seen,uid in enumerate(afit,1):
            k=seen-1 if seen<=6 else rng.randrange(seen)
            if k<6:
                if k==len(retained):retained.append(uid)
                else:retained[k]=uid
        self.require(retained==members,'independent Algorithm R membership')
        hrng=random.Random(seed(20260930,'ABC',2,'history_draws'));history=[hrng.choice(members) for _ in range(288)]
        min_grad=[];loss_error=0.;perpoint={}
        for point in self.points:
            p=self.m['points'][point];a=point[2];stage=1 if point[0]=='A' else 2
            t=self.j(self.ver(run,self.m['training'][point]));self.T[point]=t
            self.require(t['updates']==288 and len(t['rows'])==288 and t['adam_step']==stage*288,'stage call counts')
            fit=[x['group_uid'] for x in self.part['fit'] if x['domain']==point[0]]
            rng=random.Random(seed(20260918,'ABC',stage,'current'));schedule=[]
            for _ in range(6):v=sorted(fit);rng.shuffle(v);schedule.extend(v)
            self.require(t['current_ids']==schedule,'independent current schedule')
            self.require(t['history_ids']==([] if stage==1 else history),'independent history draw sequence')
            ms=p['memory_summary'];self.require(ms['members']==members and ms['seen']==48 and ms['bytes']<=1048576,'memory bound members')
            self.require(p['memory']['bytes']==ms['bytes'] and p['memory']['sha256']==ms['sha256']==p['continuity']['memory_sha256'],'memory identities')
            if stage==2:
                for k in ['members','seen','origins','tables']:
                    self.require(t['memory_before'][k]==t['memory_after'][k]==ms[k]==self.m['points']['A_'+a]['memory_summary'][k],'same own frozen historical '+k)
                self.require(t['memory_before']['count']==0 and t['memory_after']['count']==ms['count']==288,'history count')
            else:self.require(t['memory_before'] is None and t['memory_after'] is None and ms['count']==0,'no A history')
            dist=[]
            for k,row in enumerate(t['rows'],1):
                self.require(row['step']==k and row['current_uid']==schedule[k-1] and row['history_uid']==(None if stage==1 else history[k-1]),'row step UID')
                lr=1e-5*k/29 if k<=29 else 1e-5*(288-k)/259
                self.cmp(row['encoder_lr'],lr,'all encoder LR');self.require(row['head_lr']==.001 and row['pooling']==p['pooling'],'head LR and pooling identity')
                self.require(np.isfinite(row['gradient_norm']) and row['gradient_norm']>0,'finite nonzero combined gradient')
                self.require(row['current']['record_gradient_norm']>0,'nonzero current record gradient')
                self.require(row['current']['mse1']==(0.0 if a=='S' else None) and row['current']['distillation']==0 and row['current']['mse0']==0,'current objective contract')
                expected=row['current']['supervised'];self.cmp(expected,row['current']['weighted'],'current weighted',2e-6)
                if stage==2:
                    h=row['history'];self.require(h['record_gradient_norm']>0 and h['distillation']>0 and ((h['mse1'] is not None and np.isfinite(h['mse1']) and h['mse1']>=0) if a=='S' else h['mse1'] is None),'actual history gradient and D0 no D1')
                    self.cmp(h['distillation'],h['mse0'],'D0 equality');weighted=.1*h['supervised']+.5*h['mse0']
                    loss_error=max(loss_error,abs(h['weighted']-weighted));self.require(abs(h['weighted']-weighted)<=2e-6+2e-6*abs(weighted),'history objective weights')
                    expected+=weighted;dist.append(h['distillation'])
                self.require(abs(row['total']-expected)<=2e-6+2e-6*abs(expected),'total objective decomposition')
                loss_error=max(loss_error,abs(row['total']-expected));min_grad.append(row['gradient_norm'])
            perpoint[point]={'updates':len(t['rows']),'encoder_zero_lr_calls':sum(r['encoder_lr']==0 for r in t['rows']),'nonzero_history_D0':len(dist),'last_adam_step':t['adam_step']}
        for d in 'AB':
            ts=[self.T[d+'_'+a] for a in 'SGF']
            signatures=[(t['current_ids'],t['history_ids'],[(r['current']['assignment_sha256'],r['history']['assignment_sha256'] if d=='B' else None) for r in t['rows']]) for t in ts]
            self.require(signatures[0]==signatures[1]==signatures[2],'all cross-arm actual A1 signatures')
        for uid in members:self.require(len({self.m['points']['A_'+a]['memory_summary']['tables'][uid] for a in 'SGF'})==3,'own different birth T summaries')
        self.require(sum(x['updates'] for x in self.T.values())==self.c['physical_updates']==self.m['physical_updates']==1728,'1728 actual logged optimizer calls')
        self.require(sum(len(t['rows'])*(1 if p[0]=='A' else 2) for p,t in self.T.items())==self.c['gradient_group_presentations']==2592,'2592 gradient group presentations')
        self.results['training']={'points':perpoint,'calls':1728,'gradient_groups':2592,'encoder_zero_lr_calls':6,'combined_gradient_min':min(min_grad),'maximum_loss_decomposition_error':loss_error,'A1_limit':'same recorded assignments; raw record ownership unavailable, no A1 reconstruction','parameter_displacement_limit':'per-step separate encoder/head displacement not saved'}
        solver={};increments=[];previous_terminal=0
        for point in ['A_F','B_F']:
            s=self.m['points'][point]['solver_summary'];solver[point]=s
            prior=0 if point[0]=='A' else self.m['points']['A_F']['solver_summary']['blocks']
            per_update=756 if point[0]=='A' else 1890
            per_diag=33264 if point[0]=='A' else 44604
            for k,row in enumerate(self.T[point]['rows'],1):
                st=row['solver_summary'];expected=prior+k*per_update+((k-1)//48)*per_diag
                self.require(st['blocks']==expected,'solver cumulative block accounting per update')
                self.require(st['failures']==0 and st['feasible_error']<=1e-12 and st['gap_checked']<=5e-9 and st['ordinary_marginal_error']<=1e-10 and st['iterations']<=4096,'per update solver original limits')
                n=st['terminal_acceptances']
                if n:
                    self.require(st['terminal_min_iteration']==st['iterations']==4096 and st['marginal_error']>1e-10,'terminal branch recorded conditions')
                else:self.require(st['terminal_min_iteration'] is None and st['marginal_error']<=1e-10,'ordinary acceptance conditions')
                if n!=previous_terminal:increments.append(dict(point=point,step=k,new=n-previous_terminal,cumulative=n))
                previous_terminal=n
            self.require(s['blocks']==prior+288*per_update+6*per_diag+3*52*378,'full endpoint block accounting')
            self.require(s['failures']==0 and s['ordinary_marginal_error']<=1e-10 and s['feasible_error']<=1e-12 and s['gap_checked']<=5e-9,'endpoint numerical summaries')
        self.require(increments==[dict(point='B_F',step=248,new=3,cumulative=3)],'three first terminal blocks at B248')
        self.results['solver']={'endpoints':solver,'increments':increments,'B_incremental_blocks':solver['B_F']['blocks']-solver['A_F']['blocks'],'total_successes':solver['B_F']['blocks'],'ordinary_successes':solver['B_F']['blocks']-3,'accounting':{'A_updates':288*756,'A_diagnostics':6*33264,'A_endpoints':3*52*378,'B_updates':288*1890,'B_diagnostics':6*44604,'B_endpoints':3*52*378},'limits':'logged successful block summaries and exact call accounting; no per-block K/P/Q independent replay or rigorous universal floating point certificate'}
    def diagnostics(self):
        run='result/job/run/';over=self.j('result/job/diagnostics_evaluation/overfitting.json');ls=self.j('result/analysis/summary.json');summary={};decisions=collections.Counter();allcurves=[]
        for point in self.points:
            stage=1 if point[0]=='A' else 2
            captures=[self.j(self.ver(run+'diagnostics/',rec)) for rec in self.m['diagnostics'][point]]
            self.require(len(captures)==6 and [x['epoch'] for x in captures]==list(range(1,7)),'six actual epochs')
            fitids=[r['group_uid'] for r in self.part['fit'] if r['domain']==point[0]]
            cacheids=[] if stage==1 else self.m['points'][point]['memory_summary']['members']
            fits=[];caches=[];valids=[];res=[]
            for e,d in enumerate(captures,1):
                self.require(d['point']==point and d['arm']==point[2] and d['stage']==stage and d['stage_step']==48*e and d['adam_step']==(stage-1)*288+48*e,'diagnostic step')
                self.require(d['fit_ids']==fitids and d['cache_ids']==cacheids and d['valid_ids']==self.col['group_ids'],'diagnostic populations')
                self.require(d['valid_labels_used'] is False and d['mode']=='A0_eval_no_autograd','diagnostic blindness receipt')
                f=self.arr(self.ver(run+'diagnostics/',d['fit']));c=self.arr(self.ver(run+'diagnostics/',d['cache']));v=self.arr(self.ver('result/job/diagnostics_evaluation/',over['group_metrics'][f'{point}_epoch{e}']))
                self.require(f.shape==(48,3) and c.shape==(len(cacheids),3) and v.shape==(40,3),'diagnostic array dimensions')
                blind=self.arr(self.ver(run+'diagnostics/',d['valid_blind']));self.require(blind.shape==(40,378),'blind diagnostic score dimensions')
                if e==6:
                    self.require(np.array_equal(blind,self.R[point]['raw']),'last diagnostic raw endpoint exact')
                    self.cmp(v,self.M[point]['raw'][:,[12,16,5]],'last diagnostic metrics vs endpoint')
                fits.append(f);caches.append(c);valids.append(v)
                self.require([x['uid'] for x in d['teacher_errors']]==cacheids,'teacher error UIDs')
                if d['teacher_errors']:
                    vals={k:float(np.mean([r[k] for r in d['teacher_errors']])) for k in ['D0','D1','residual_cross_mean','residual_difference_mse']}
                    for k,x in vals.items():self.cmp(x,ls['residuals'][point][e-1][k],'teacher residual means')
                    res.append({'epoch':e,**vals})
                for i,dom in enumerate('AB'):
                    values=v[self.rows[i]].mean(0);self.cmp(values,ls['valid_curves'][point][dom][e-1],'all 36 valid curves')
                    allcurves.append([point,e,'valid',dom,*values.tolist()])
                allcurves.append([point,e,'fit',point[0],*f.mean(0).tolist()])
                if len(c):allcurves.append([point,e,'cache','A',*c.mean(0).tolist()])
            summary[point]={};v=np.stack(valids)
            for role,train,dom in [('fit',np.stack(fits),point[0]),('cache',np.stack(caches),'A')]:
                if train.shape[1]==0:
                    self.require(over['points'][point]['cache']['status']=='NOT_APPLICABLE_FIRST_STAGE','A no cache N/A');continue
                target=over['points'][point][role][dom];vv=v[:,self.rows['AB'.index(dom)]]
                rng=np.random.Generator(np.random.PCG64(seed(20260930,'ABC',stage,role,dom,'diagnostics')))
                td=rng.integers(train.shape[1],size=(5000,train.shape[1]));vd=rng.integers(20,size=(5000,20))
                # Frequency-weighted computation is independent of the submitted diagnostic audit's array gathering.
                tf=np.stack([np.bincount(row,minlength=train.shape[1]) for row in td])/train.shape[1]
                vf=np.stack([np.bincount(row,minlength=20) for row in vd])/20
                signs=np.array([1.,1.,-1.]);tm=train.mean(1);vm=vv.mean(1);gap=(tm-vm)*signs
                tb=(train[-1]-train[0])*signs;vb=(vv[-1]-vv[0])*signs
                tboot=tf@tb;vboot=vf@vb
                gboot=np.stack([(tf@train[e]-vf@vv[e])*signs for e in range(6)])
                tci=np.quantile(tboot,[.025,.975],axis=0,method='linear').T;vci=np.quantile(vboot,[.025,.975],axis=0,method='linear').T
                gci=np.quantile(gboot[-1]-gboot[0],[.025,.975],axis=0,method='linear').T
                self.cmp(tm,target['train_raw_curve'],'all train curves');self.cmp(vm,target['valid_raw_curve'],'all valid curves');self.cmp(gap,target['benefit_gap_curve'],'all gap curves')
                self.cmp(np.quantile(gboot,[.025,.975],axis=1,method='linear').transpose(1,2,0),target['benefit_gap_conditional_intervals'],'all epoch gap intervals')
                for mn,ci,key in [(tb.mean(0),tci,'epoch1_to_6_train_benefit'),(vb.mean(0),vci,'epoch1_to_6_valid_benefit'),(gap[-1]-gap[0],gci,'epoch1_to_6_gap_widening')]:
                    self.cmp(mn,target[key]['mean'],'diagnostic change means');self.cmp(ci,target[key]['conditional_95pct_interval'],'diagnostic change intervals')
                labels={}
                for j,k in enumerate(['map','recall_at_5','log_loss']):
                    point_signal=tb[:,j].mean()>0 and vb[:,j].mean()<0 and gap[-1,j]-gap[0,j]>0
                    supported=tci[j,0]>0 and vci[j,1]<0 and gci[j,0]>0
                    label='有相应迹象' if supported else '证据不足' if point_signal else '未见明确迹象'
                    self.require(label==target['interpretation'][k]==ls['overfit_interpretations'][point][role][dom][k],'diagnostic exact interpretation')
                    decisions[label]+=1;labels[k]=label
                summary[point][role]={'domain':dom,'train_benefit':tb.mean(0).tolist(),'train_interval':tci.tolist(),'valid_benefit':vb.mean(0).tolist(),'valid_interval':vci.tolist(),'gap_widening':(gap[-1]-gap[0]).tolist(),'gap_interval':gci.tolist(),'interpretation':labels}
            summary[point]['residuals']=res
        self.require(sum(decisions.values())==27 and decisions['证据不足']==2 and decisions['未见明确迹象']==25,'27 diagnostic labels')
        with (self.out/'all_diagnostic_curves.csv').open('w',encoding='utf-8',newline='') as f:
            w=csv.writer(f);w.writerow(['point','epoch','role','domain','map','recall_at_5','log_loss']);w.writerows(allcurves)
        dump(self.out/'diagnostic_recalculation.json',summary);self.results['diagnostics']={'points':36,'labels':dict(decisions),'curve_rows':len(allcurves)}
    def costs_and_addenda(self):
        console=self.b('result/job.console.txt','all raw console lines parsed').decode().splitlines();events=[]
        for line in console:
            if line.startswith('{'):events.append(json.loads(line))
        self.require(len(events)==78,'78 progress samples')
        self.require(all(events[i]['elapsed_seconds']<events[i+1]['elapsed_seconds'] for i in range(len(events)-1)),'monotonic progress times')
        windows={}
        for p in self.points:
            es=[x for x in events if x['point']==p]
            self.require([x['completed'] for x in es]==[1,*range(24,289,24)],'progress sequence '+p)
        for a in 'SGF':
            first=next(x for x in events if x['point']=='A_'+a and x['completed']==1)
            last=next(x for x in events if x['point']=='B_'+a and x['completed']==288)
            windows[a]=last['elapsed_seconds']-first['elapsed_seconds']
        self.results['cost_windows']={'seconds':windows,'F_vs_S_percent':100*(windows['F']/windows['S']-1),'G_vs_S_percent':100*(windows['G']/windows['S']-1),'scope':'first A logged update to last B logged update; includes intermediate diagnostics/save/restore, excludes margins; not pure training/inference benchmark'}
        obs=self.j('result/observation.json');self.require(obs['completion']==self.c,'observation completion equality')
        custody=collections.Counter()
        for rec in obs['weights_and_memory']:
            p=self.m['points'][rec['point']][rec['kind']]
            self.require(all(rec[k]==p[k] for k in ['path','bytes','sha256']),'custody identity to endpoint')
            custody[rec['kind']+'_bytes']+=rec['bytes'];custody[rec['kind']+'_count']+=1
        self.results['custody']=dict(custody)
        close=self.j('result/closeout.json');self.require(close['report_bytes']==len(self.b('result/report.zh.txt')) and close['report_sha256']==sha(self.b('result/report.zh.txt')),'closeout report identity')
        # Document the loss-forgetting sign without silently changing frozen results.
        rows=[]
        for r in self.stats_rows:
            if r[0]=='forgetting_contrast' and r[4] in ['brier','log_loss']:
                rows.append(r[:5]+r[5:]+[-r[5],-r[7],-r[6]])
        with (self.out/'loss_forgetting_direction_addendum.csv').open('w',encoding='utf-8',newline='') as f:
            w=csv.writer(f);w.writerow(['kind','contrast','role','domain','metric','saved_mean_AminusB','saved_low','saved_high','loss_worsening_mean_BminusA','loss_worsening_low','loss_worsening_high']);w.writerows(rows)
        self.results['loss_forgetting_semantics']={'affected_rows':len(rows),'frozen_definition':self.ev['forgetting_sign'],'operation':'For a loss, conventional positive-worsening forgetting contrast is the negative of the saved A-minus-B metric contrast. Transform interval [lo,hi] to [-hi,-lo]. Not a change of endpoint or new statistical test.'}
    def finish(self,status='COMPLETE',error=None):
        result={'status':status,'environment':{'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),'threads':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']}},'elapsed_seconds':time.perf_counter()-self.started,'input':self.identity,'assertions':self.checks,'numeric_checks':dict(self.errors),'results':self.results,'limits':'External isolated saved-evidence consistency audit only. No project Linux, Torch/model/Memory load, formal labels, AP/ROC/Brier/log-loss label reconstruction, qualification rerun, training or new scientific resampling choices.'}
        if error:result['error']=error
        dump(self.out/'audit_output.json',result)
        dump(self.out/'programmatic_read_scope.json',[{'path':n,'purposes':sorted(p)} for n,p in sorted(self.used.items())])
        print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))

def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    a=Audit(args.archive,args.output)
    try:
        for name in ['identity_check','source_execution','metrics','training','diagnostics','costs_and_addenda']:
            print('BEGIN',name,flush=True);getattr(a,name)();print('COMPLETE',name,flush=True)
        a.finish()
    except Exception:
        a.finish('REVIEW_SCRIPT_FAILED',traceback.format_exc());raise
if __name__=='__main__':main()
