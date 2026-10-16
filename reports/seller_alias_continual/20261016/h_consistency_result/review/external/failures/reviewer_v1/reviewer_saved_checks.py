"""Narrow independent reconciliation of H_cons saved evidence.
No imports from submitted audit or production; no labels, models or Memory loads.
All resampling uses only the already registered choices. This is NOT native qualification.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, platform, random, time
from collections import Counter
from datetime import datetime
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,required=True); ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args(); a.output.mkdir(parents=True,exist_ok=False); start=time.perf_counter()
    R=a.input; E=R/'evidence'; J=E/'reports/job'; run=J/'run'
    reads=set(); verified=[]; differences=[]; checks=Counter(); result={}
    def load(p):
        reads.add(p.relative_to(R).as_posix()); return json.loads(p.read_text(encoding='utf-8'))
    def arr(p):
        reads.add(p.relative_to(R).as_posix()); return np.load(p,allow_pickle=False)
    def verify(parent,rec):
        p=parent/rec['path']; b=p.read_bytes()
        assert len(b)==rec['bytes'] and hashlib.sha256(b).hexdigest()==rec['sha256'], str(p)
        verified.append(p.relative_to(R).as_posix()); return p
    def eq(x,y,label):
        x,y=np.asarray(x,float),np.asarray(y,float)
        assert x.shape==y.shape and np.array_equal(np.isnan(x),np.isnan(y)),label
        assert not np.isinf(x).any() and not np.isinf(y).any(),label
        ok=~np.isnan(x); d=float(np.max(np.abs(x[ok]-y[ok]))) if ok.any() else 0.
        assert d<=1e-12,(label,d); differences.append(d); checks[label.split('/')[0]]+=x.size
    def seed(n,*parts):
        b=(json.dumps([n,*parts],ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
        return int.from_bytes(hashlib.sha256(b).digest()[:8],'big')%(2**63-1)
    manifest=load(R/'input_manifest.json')
    for rec in manifest['files']: verify(R,rec)
    assert set(x['path'] for x in manifest['files'])=={p.relative_to(R).as_posix() for p in R.rglob('*') if p.is_file()}-{'input_manifest.json'}
    m=load(run/'manifest.json'); part=load(verify(run,m['partition'])); c=load(J/'evaluation/collected.json'); ev=load(J/'evaluation/evaluation.json')
    comp=load(J/'completion.json'); before=load(J/'before_valid.json'); execution=load(J/'execution.json'); pol=load(E/'schema/step28_h_consistency_policy.json')
    gate=load(verify(E,execution['gate'])); auth=load(verify(E,gate['authorization'])); verify(E,auth['contract'])
    assert auth['status']=='APPROVED_FULL_CONTRACT_GATED_SEQUENCE'
    assert gate['kind']=='FORMAL' and gate['job']=='reports/job' and gate['runtime']==pol['runtime'] and gate['supervision']==pol['supervision']
    for key in ('external_review','primary_disposition'): verify(R/'context',gate[key])
    verify(E,gate['requalification_authorization'])
    result['inherited_not_reexecuted']={key:gate[key] for key in ('cpu','gpu','gpu_wallclock')}
    assert all(not (E/gate[key]['path']).exists() for key in ('cpu','gpu','gpu_wallclock'))
    assert gate['source_files']==execution['sources']==m['source_files']==comp['source_files']==before['source_files']==c['source_files']==ev['source_files']
    for rec in comp['source_files']: verify(E,rec); verify(run/'sources',rec)
    verify(J,before['manifest']); verify(E,m['baseline_identity'])
    assert load(J/'access.json')==comp['access']==dict(train=1,valid=1,heldout=0,owners=0)
    assert before['access']==dict(train=1,valid=0,heldout=0,owners=0) and before['status']=='ALL_BLIND_GATES_PASS'
    assert m['physical_updates']==comp['physical_updates']==576 and m['gradient_group_presentations']==comp['gradient_group_presentations']==864
    assert not (J/'failure.json').exists()
    for fn,parent in [('inventory.json',E/'reports'),('analysis_inventory.json',R)]:
        inv=load(R/fn); assert inv['count']==len(inv['files']) and inv['bytes']==sum(x['bytes'] for x in inv['files'])
        for rec in inv['files']: verify(parent,rec)
    base=load(E/'reports/documentation/20261016/h_consistency/baseline_sources.json')
    roots={k:E/v['root'] for k,v in base['baselines'].items()}
    assert len(set(p.resolve() for p in roots.values()))==2
    for rec in base['files']+base['unchanged_direct_sources']: verify(E,rec)
    result['baseline_roots']={k:str(p.relative_to(E)) for k,p in roots.items()}
    result['baseline_counts']={k:sum(str(rec['path']).startswith(str(p.relative_to(E))+'/') for rec in base['files']) for k,p in roots.items()}
    old={}; origins={p:J/'evaluation' for p in m['points']}; point_meta=dict(m['points']); collections={p:c for p in m['points']}
    for k,info in base['baselines'].items():
        root=roots[k]; oc=load(root/'evaluation/collected.json'); om=load(root/'run/manifest.json'); old[info['arm']]=(root,oc,om)
        assert om['public_inputs']==m['public_inputs'] and om['pretrained_archive']==m['pretrained_archive']
        assert load(root/'run/partition.json')==part
        assert om['initials'][info['arm']]==m['initials']['H_cons']==info['initial_identity']
        assert all(oc[x]==c[x] for x in ('group_ids','domains','columns'))
        assert load(root/'access.json')==dict(train=1,valid=1,heldout=0,owners=0)
        assert load(root/'before_valid.json')['access']==dict(train=1,valid=0,heldout=0,owners=0)
        for p in info['points']: origins[p]=root/'evaluation'; point_meta[p]=om['points'][p]; collections[p]=oc
    assert [x['group_uid'] for x in part['development']]==c['group_ids']
    assert [x['domain'] for x in part['development']]==c['domains'] and len(set(c['group_ids']))==40
    uidsets={k:{x['group_uid'] for x in v} for k,v in part.items()}
    assert all(not uidsets[x]&uidsets[y] for x in uidsets for y in uidsets if x!=y)
    for domain in 'AB':
        for k,n in [('fit',48),('calibration',12),('development',20)]: assert sum(x['domain']==domain for x in part[k])==n
    assert all(x['domain'] in 'AB' for v in part.values() for x in v)
    # Independent replay of already fixed UID schedules, not records or labels.
    ma=m['points']['A_H_cons']['memory_summary']; mb=m['points']['B_H_cons']['memory_summary']
    fitA=sorted(x['group_uid'] for x in part['fit'] if x['domain']=='A'); rng=random.Random(seed(20260930,'ABC','retention')); retained=[]
    for seen,u in enumerate(fitA,1):
        k=seen-1 if seen<=6 else rng.randrange(seen)
        if k<6:
            if k==len(retained): retained.append(u)
            else: retained[k]=u
    assert retained==ma['members']==mb['members']
    assert m['starts']['H_cons']==m['points']['A_H_cons']['continuity']
    for p in m['points'].values():
        assert p['continuity']['memory_sha256']==p['memory_summary']['sha256']==p['memory']['sha256']
        assert p['memory_summary']['bytes']==p['memory']['bytes']<=1048576
    assert all(ma[k]==mb[k] for k in ('members','origins','seen','tables'))
    result['own_birth_tables_different_from_baselines']={}
    for arm,(_,_,om) in old.items():
        oldt=om['points']['A_'+arm]['memory_summary']['tables']
        result['own_birth_tables_different_from_baselines'][arm]=sum(ma['tables'][u]!=oldt[u] for u in retained)
    training_summary={}; all_training={}
    for stage,d in enumerate('AB',1):
        point=d+'_H_cons'; t=load(verify(run,m['training'][point])); all_training[point]=t
        fit=[r['group_uid'] for r in part['fit'] if r['domain']==d]
        rng=random.Random(seed(20260918,'ABC',stage,'current')); schedule=[]
        for epoch in range(6):
            v=sorted(fit); rng.shuffle(v); schedule.extend(v)
        hr=random.Random(seed(20260930,'ABC',stage,'history_draws')); hist=[] if stage==1 else [hr.choice(retained) for _ in range(288)]
        assert t['current_ids']==schedule and t['history_ids']==hist and t['adam_step']==288*stage
        assert [r['current_uid'] for r in t['rows']]==schedule
        assert [r['history_uid'] for r in t['rows']]==([None]*288 if stage==1 else hist)
        assert len(t['rows'])==t['updates']==288
        component_rows={r:[] for r in (('current',) if stage==1 else ('current','history'))}
        max64=0.
        for n,row in enumerate(t['rows'],1):
            expected_lr=1e-5*n/29 if n<=29 else 1e-5*(288-n)/259
            assert row['step']==n and row['encoder_lr']==expected_lr and row['head_lr']==.001
            assert row['training']==pol['training'] and row['pooling']==pol['pooling']
            assert np.isfinite(row['gradient_norm']) and row['gradient_norm']>0
            total=0.
            for role in component_rows:
                r=row[role]; w=np.float32(1 if role=='current' else .1)
                assert r['qualification_without_c'] is False and r['mse1'] is None
                assert all(np.isfinite(r[k]) and float(np.float32(r[k]))==r[k] for k in ['supervised','consistency','mse0'])
                assert 0<=r['consistency']<=math.log(2) and r['mse0']>=0 and r['distillation']==r['mse0']
                assert np.isfinite(r['record_gradient_norm']) and r['record_gradient_norm']>0
                if role=='current': assert r['mse0']==0
                terms=[w*np.float32(r['supervised']),w*np.float32(r['consistency']),np.float32(.5)*np.float32(r['mse0'])]
                assert [r[k] for k in ['supervised_contribution','consistency_contribution','teacher_contribution']]==[float(x) for x in terms]
                weighted=float(np.float32(np.float32(terms[0]+terms[1])+terms[2])); assert weighted==r['weighted']; total+=weighted
                ref=float(w)*r['supervised']+float(w)*r['consistency']+.5*r['mse0']; max64=max(max64,abs(ref-weighted))
                component_rows[role].append(r)
            assert row['total']==total
        for arm,(root,_,_) in old.items():
            ot=load(root/f'run/updates/{d}_{arm}.json')
            assert ot['current_ids']==schedule and ot['history_ids']==hist
            for r,s in zip(t['rows'],ot['rows'],strict=True):
                for k in ('step','current_uid','history_uid','encoder_lr','head_lr'): assert r[k]==s[k]
                for role in component_rows: assert r[role]['assignment_sha256']==s[role]['assignment_sha256']
        if stage==1: assert t['memory_before'] is None and t['memory_after'] is None
        else:
            for k in ('members','origins','seen','tables'): assert t['memory_before'][k]==t['memory_after'][k]==ma[k]
            assert t['memory_before']['count']==0 and t['memory_after']['count']==288
        training_summary[point]={'exact_fp32_rows':288,'last_encoder_lr':0,'maximum_branch_difference_from_fp64_sum_using_FP32_weight':max64,
            'components':{role:{'nonzero_JS':sum(x['consistency']!=0 for x in rr),'mean_JS':float(np.mean([x['consistency'] for x in rr])),
                'mean_weighted_JS':float(np.mean([x['consistency_contribution'] for x in rr])),
                'mean_teacher_contribution':float(np.mean([x['teacher_contribution'] for x in rr]))} for role,rr in component_rows.items()}}
    result['training']=training_summary
    # Eighteen matrix/query/count bundles: six endpoints, three probability roles.
    columns=c['columns']; roles=['raw','stage-cal','first-cal']; matrices={}; classification={}; rows=[np.flatnonzero(np.array(c['domains'])==d) for d in 'AB']
    for point,origin in origins.items():
        matrices[point]={}; classification[point]={}
        for role in roles:
            rec=collections[point]['points'][point][role]; mm=load(verify(origin,rec['matrix'])); v=np.array(mm['values'],float); q=arr(verify(origin,rec['queries'])); cnt=load(verify(origin,rec['counts']))
            assert mm['columns']==columns and v.shape==(40,22) and q.shape==(40,28,10) and np.isfinite(q).all()
            assert np.all((q>=0)&(q<=1)); eq(q.mean(1),v[:,12:],'query_group/'+point+'/'+role)
            if role!='raw':
                oq=arr(origin/collections[point]['points'][point]['raw']['queries']['path']); assert np.array_equal(q,oq)
            counts=np.array([[x[k] for k in ('tp','fp','fn','tn')] for x in cnt],dtype=np.int64)
            assert len(cnt)==40 and all(isinstance(x[k],int) and x[k]>=0 for x in cnt for k in ('tp','fp','fn','tn'))
            tp,fp,fn,tn=counts.T; assert np.all(tp+fn==20) and np.all(fp+tn==358)
            null=np.zeros((40,22),bool); null[:,6]=tp+fp==0; denom=(tp+fp)*(tp+fn)*(tn+fp)*(tn+fn); null[:,11]=denom==0
            assert np.array_equal(np.isnan(v),null)
            rates=[]
            for t,p,f,n in counts:
                den=(t+p)*(t+f)*(n+p)*(n+f)
                rates.append([t/(t+p) if t+p else np.nan,t/(t+f),2*t/(2*t+p+f),n/(n+p),.5*(t/(t+f)+n/(n+p)),(t*n-p*f)/math.sqrt(den) if den else np.nan])
            eq(v[:,6:12],rates,'count_rates/'+point+'/'+role)
            eq(np.isnan(v).sum(0),list(ev['absolute'][point][role]['macro_undefined_groups'][k] for k in columns),'NA_counts/'+point+'/'+role)
            eq(v.mean(0),[ev['absolute'][point][role]['macro_all'][k] for k in columns],'macro/'+point+'/'+role)
            classification[point][role]={}
            for d,sel in [('all',np.arange(40)),('A',rows[0]),('B',rows[1])]:
                t,p,f,n=map(int,counts[sel].sum(0)); rates=dict(tp=t,fp=p,fn=f,tn=n,precision=t/(t+p) if t+p else None,recall=t/(t+f),f1=2*t/(2*t+p+f),fpr=p/(p+n))
                assert rates==ev['absolute'][point][role]['fixed_half_classification'][d]
                classification[point][role][d]=rates
            matrices[point][role]=v
            if point in m['points']:
                scores=arr(verify(run,m['points'][point]['scores'][role])); assert scores.shape==(40,378) and np.isfinite(scores).all()
                assert np.array_equal((scores>=0).sum(1),tp+fp)
        if point in m['points']:
            p=m['points'][point]; mapping=load(verify(run,p['mapping'])); d=point[0]; stage=1 if d=='A' else 2
            assert p['calibration_ids']==[r['group_uid'] for r in part['calibration'] if r['domain']==d]
            assert mapping['role']=='calibration' and mapping['group_count']==12 and mapping['pair_count']==4536 and mapping['positive_count']==240
            assert mapping['optimizer_success'] and mapping['status']=='PASS_CALIBRATION_FIT' and mapping['projected_gradient_max']<=1e-6
            assert mapping['final_nll']<=mapping['initial_nll']+1e-12
            assert mapping['optimizer_iterations']<=200 and mapping['objective_calls']<=2000
            assert p['maps']['stage']=={k:mapping[k] for k in ('a','b')}
            assert p['maps']['first']==m['points']['A_H_cons']['maps']['stage']
            raw=arr(run/p['scores']['raw']['path'])
            for role in ('stage-cal','first-cal'):
                mp=p['maps']['stage' if role=='stage-cal' else 'first']; assert .001<=mp['a']<=100 and -100<=mp['b']<=100
                sc=arr(run/p['scores'][role]['path']); assert np.array_equal(sc,mp['a']*raw.astype(float)+mp['b'])
                order=np.argsort(raw,axis=1,kind='stable'); assert np.array_equal(order,np.argsort(sc,axis=1,kind='stable'))
                a0=np.take_along_axis(raw,order,1); b0=np.take_along_axis(sc,order,1)
                assert np.array_equal(np.diff(a0,axis=1)==0,np.diff(b0,axis=1)==0)
            for role,shape in [('raw',(40,378)),('calibration',(12,378))]:
                x=arr(verify(run,p['scores'][role])); assert x.shape==shape and x.dtype==np.float32
                for prefix in ('full','inference'): assert np.array_equal(x,arr(verify(run,p['scores'][prefix+'-'+role])))
    result['classification']=classification
    # All registered statistics, direct group indexing rather than audit.py frequency multiplication.
    draws=np.random.Generator(np.random.PCG64(20260930)).integers(20,size=(5000,2,20)); stats=[]; stat_key={}
    def stats_for(v, target, i, kind, name, role, domain):
        for j,k in enumerate(columns):
            x=v[:,j]; t=target[k]; missing=int(np.isnan(x).sum())
            if missing:
                assert t['mean'] is None and t['conditional_95pct_interval'] is None and t['undefined_groups']==missing and t['total_groups']==20
                vals=[None,None,None]
            else:
                bs=x[draws[:,i]].mean(1); vals=[float(x.mean()),*np.quantile(bs,[.025,.975]).tolist()]
                eq(vals,[t['mean'],*t['conditional_95pct_interval']],'registered_statistics/'+kind)
            key=(kind,name,role,domain,k); stat_key[key]=vals; stats.append([*key,*vals])
    for point,vv in matrices.items():
        for role,v in vv.items():
            for i,d in enumerate('AB'): stats_for(v[rows[i]],ev['absolute'][point][role]['by_domain'][d],i,'absolute',point,role,d)
    for other in ('S','G_half'):
        name='H_cons-'+other
        for role in roles:
            for stage,section,kind in [('A','initial_comparisons','initial_contrast'),('B','comparisons','final_contrast')]:
                v=matrices[stage+'_H_cons'][role]-matrices[stage+'_'+other][role]
                for i,d in enumerate('AB'): stats_for(v[rows[i]],ev[section][name][role][d],i,kind,name,role,d)
            # Direct bad-oriented forgetting, separately for each metric.
            fc=[]
            for j,k in enumerate(columns):
                ca,cb=matrices['A_H_cons'][role][:,j],matrices['B_H_cons'][role][:,j]
                oa,ob=matrices['A_'+other][role][:,j],matrices['B_'+other][role][:,j]
                fc.append((cb-ca)-(ob-oa) if k in ('brier','log_loss') else (ca-cb)-(oa-ob))
            stats_for(np.stack(fc,1)[rows[0]],ev['forgetting_comparisons'][name][role],0,'forgetting_contrast',name,role,'A')
    for arm in ('H_cons','S','G_half'):
        for role in roles: stats_for((matrices['B_'+arm][role]-matrices['A_'+arm][role])[rows[0]],ev['old_changes'][arm][role],0,'old_change',arm,role,'A')
    assert len(stats)==1650
    for file in [R/'analysis/all_metrics.csv',a.output.parent/'reviewer_analysis/all_metrics.csv']:
        rr=list(csv.reader(file.open())); assert len(rr)==1651
        for r in rr[1:]: eq(stat_key[tuple(r[:5])],[float(x) if x else None for x in r[5:]],'CSV_reconciliation')
    with (a.output/'independent_all_metrics.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['kind','point_or_contrast','role','domain','metric','mean','lower_95','upper_95']); w.writerows(stats)
    result['statistics']={'rows':len(stats),'NA_rows':sum(r[-1] is None for r in stats),'bootstrap_shape':list(draws.shape),'bootstrap_indices_sha256':hashlib.sha256(draws.astype('<i8').tobytes()).hexdigest()}
    result['key_MAP']={other:{d:stat_key[('final_contrast','H_cons-'+other,'raw',d,'map')] for d in 'AB'} for other in ('S','G_half')}
    result['success']={'passed':all(result['key_MAP']['S'][d][1]>0 for d in 'AB'),'direct_H_new_B_positive':result['key_MAP']['G_half']['B'][1]>0}
    assert all(ev['success'][k]==v for k,v in result['success'].items())
    # All twelve diagnostics: identity/scope/hash plus the registered nine decisions.
    over=load(J/'diagnostics_evaluation/overfitting.json'); diagnostic=[]
    for stage,d in enumerate('AB',1):
        point=d+'_H_cons'; fs=[]; cs=[]; vs=[]
        fit=[r['group_uid'] for r in part['fit'] if r['domain']==d]
        for epoch,rec in enumerate(m['diagnostics'][point],1):
            cap=load(verify(run/'diagnostics',rec)); cache=[] if stage==1 else retained
            assert cap['point']==point and cap['arm']=='H_cons' and cap['seed']=='s0' and cap['order']=='ABC'
            assert cap['epoch']==epoch and cap['stage']==stage and cap['stage_step']==48*epoch and cap['adam_step']==288*(stage-1)+48*epoch
            assert cap['fit_ids']==fit and cap['cache_ids']==cache and cap['valid_ids']==c['group_ids']
            assert cap['fit_domains']==[d]*48 and cap['cache_domains']==['A']*len(cache)
            assert cap['training']==pol['training'] and cap['pooling']==pol['pooling'] and cap['valid_labels_used'] is False and cap['mode']=='A0_eval_no_autograd'
            assert cap['metric_columns']==['map','recall_at_5','log_loss']
            ff=arr(verify(run/'diagnostics',cap['fit'])); cc=arr(verify(run/'diagnostics',cap['cache'])); ss=arr(verify(run/'diagnostics',cap['valid_blind']))
            vv=arr(verify(J/'diagnostics_evaluation',over['group_metrics'][point+f'_epoch{epoch}']))
            assert ff.shape==(48,3) and cc.shape==(len(cache),3) and ss.shape==(40,378) and vv.shape==(40,3)
            assert all(np.isfinite(x).all() for x in (ff,cc,ss,vv))
            assert [r['uid'] for r in cap['teacher_errors']]==cache
            for x in cap['teacher_errors']:
                eq(x['residual_difference_mse'],x['D0']+x['D1']-2*x['residual_cross_mean'],'residual_identity')
            if epoch==6:
                assert np.array_equal(ss,arr(run/m['points'][point]['scores']['raw']['path']))
                eq(vv,matrices[point]['raw'][:,[columns.index(k) for k in cap['metric_columns']]],'diagnostic_endpoint')
            fs.append(ff);cs.append(cc);vs.append(vv)
        for role,tt,domain in [('fit',np.stack(fs),d)]+([] if stage==1 else [('cache',np.stack(cs),'A')]):
            vv=np.stack(vs)[:,rows[0 if domain=='A' else 1]]; signs=np.array([1,1,-1])
            rng=np.random.Generator(np.random.PCG64(seed(20260930,'ABC',stage,role,domain,'diagnostics')))
            td=rng.integers(tt.shape[1],size=(5000,tt.shape[1])); vd=rng.integers(20,size=(5000,20))
            tc=(tt[-1]-tt[0])*signs; vc=(vv[-1]-vv[0])*signs
            tm,vm=tc.mean(0),vc.mean(0); ti=np.quantile(tc[td].mean(1),[.025,.975],axis=0).T; vi=np.quantile(vc[vd].mean(1),[.025,.975],axis=0).T
            gap=(tt.mean(1)-vv.mean(1))*signs; boot=(tt[:,td].mean(2)-vv[:,vd].mean(2))*signs; gm=gap[-1]-gap[0]; gi=np.quantile(boot[-1]-boot[0],[.025,.975],axis=0).T
            target=over['points'][point][role][domain]
            for key,value in [('train_raw_curve',tt.mean(1)),('valid_raw_curve',vv.mean(1)),('benefit_gap_curve',gap),('benefit_gap_conditional_intervals',np.quantile(boot,[.025,.975],axis=1).transpose(1,2,0))]: eq(value,target[key],'diagnostic_curves')
            for key,mm,ci in [('epoch1_to_6_train_benefit',tm,ti),('epoch1_to_6_valid_benefit',vm,vi),('epoch1_to_6_gap_widening',gm,gi)]: eq(mm,target[key]['mean'],'diagnostic_deltas');eq(ci,target[key]['conditional_95pct_interval'],'diagnostic_intervals')
            for k,metric in enumerate(('map','recall_at_5','log_loss')):
                ps=tm[k]>0 and vm[k]<0 and gm[k]>0; support=ti[k,0]>0 and vi[k,1]<0 and gi[k,0]>0
                decision='有相应迹象' if support else '证据不足' if ps else '未见明确迹象'
                assert decision==target['interpretation'][metric]
                diagnostic.append({'point':point,'role':role,'domain':domain,'metric':metric,'train_benefit':float(tm[k]),'valid_benefit':float(vm[k]),'gap_widening':float(gm[k]),'gap_CI':gi[k].tolist(),'interpretation':decision})
    assert len(diagnostic)==9
    result['diagnostics']=diagnostic
    wrapper=(E/'reports/job.wrapper.txt').read_text().splitlines(); assert wrapper[1]=='exit_code=0'
    seconds=(datetime.fromisoformat(wrapper[2])-datetime.fromisoformat(wrapper[0])).total_seconds(); assert seconds==6043
    clock=load(R/'analysis_wallclock.json'); assert clock['charged_seconds']==math.ceil(clock['elapsed_seconds'])==2
    assert clock['formal_training_charged_seconds']==seconds and clock['formal_cumulative_seconds']==6045
    assert clock['script_sha256']==hashlib.sha256((R/'audit.py').read_bytes()).hexdigest()
    budget=comp['budget']; assert budget['elapsed_seconds']<=seconds and budget['limits']==pol['runtime']
    for k in ('cuda_reserved','rss','output'): assert budget['peak_'+k+'_bytes']<=budget['limits']['maximum_'+k+'_bytes']
    console=[json.loads(x) for x in (E/'reports/job.console.txt').read_text().splitlines() if x.startswith('{')]
    for d in 'AB': assert [x['completed'] for x in console if x['point']==d+'_H_cons']==[1,*range(24,289,24)]
    assert all(x['elapsed_seconds']<y['elapsed_seconds'] for x,y in zip(console,console[1:]))
    result['budget']={'formal_wrapper_seconds':seconds,'original_local_analysis_charged_seconds':2,'project_total_seconds':6045,'cpu_seconds_inherited':51,'gpu_seconds_inherited':gate['gpu_cumulative_charged_seconds'],'peak_bytes':{k:budget[k] for k in budget if k.startswith('peak_')}}
    # End-of-review identity recheck: no source or input modification.
    for rec in manifest['files']: verify(R,rec)
    result.update(status='PASS_INDEPENDENT_SAVED_RECONCILIATION',python=platform.python_version(),numpy=np.__version__,elapsed_seconds=time.perf_counter()-start,
        maximum_absolute_difference=max(differences),numeric_comparisons=sum(checks.values()),checks_by_category=dict(checks),verified_unique_files=len(set(verified)),
        limits=['No labels/models/Memory/torch imports; no new training.','AP/ROC/Brier/log-loss group values and original query rankings not regenerated from labels.','A1 allocation bytes not regenerated without record/ownership payload; compared 864 saved allocation hashes with each fixed baseline.','CPU/GPU raw records absent from this results package; inherit narrowly through supplied closure reports and bound identities, not requalification.'])
    (a.output/'reviewer_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (a.output/'programmatic_read_scope.json').write_text(json.dumps({'json_numpy_read':sorted(reads),'hash_verified':sorted(set(verified))},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['status','elapsed_seconds','maximum_absolute_difference','numeric_comparisons','statistics','success','key_MAP','training']},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
