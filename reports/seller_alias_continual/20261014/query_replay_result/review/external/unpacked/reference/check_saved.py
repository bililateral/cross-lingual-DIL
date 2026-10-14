"""Independent, label-free review of C_q saved results. Never imports project code.
Usage: python check_saved.py --input original_input.zip --out NEW_DIRECTORY
Only numpy and Python standard library. All original ZIP members are left unchanged.
This is a web-workspace reference check, NOT project Linux/native qualification.
"""
from __future__ import annotations
import argparse, collections, csv, hashlib, io, json, math, os, platform, random
import sys, tarfile, time, traceback, zipfile
from pathlib import Path
import numpy as np
EXPECTED_SHA='9013c992a925bac6e69ba603438d5ca62692fcc38d5306a89f8a942f05973c96'
POINTS=['A_shared','B_S','B_Q','B_C_q','B_C_b']
ROLES=['raw','stage-cal','first-cal']
CONTRASTS=[('C_q','S'),('Q','S'),('C_q','Q'),('C_q','C_b')]
def sha(b): return hashlib.sha256(b).hexdigest()
def canon(o): return (json.dumps(o,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def seed(*parts): return int.from_bytes(hashlib.sha256(canon(list(parts))).digest()[:8],'big')%(2**63-1)
def put(p,o): p.write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def table(p,headers,rows):
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f,delimiter='\t');w.writerow(headers);w.writerows(rows)
class Check:
    def __init__(self): self.tests=collections.Counter();self.numbers=collections.Counter();self.errors=collections.defaultdict(float)
    def yes(self, condition, category, detail=''):
        self.tests[category]+=1
        if not bool(condition): raise AssertionError(category+': '+detail)
    def close(self,a,b,category,tol=1e-12):
        a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
        self.yes(a.shape==b.shape,category,'shape')
        self.yes(np.isfinite(a).all() and np.isfinite(b).all(),category,'finite')
        e=float(np.max(np.abs(a-b))) if a.size else 0.
        self.errors[category]=max(self.errors[category],e);self.numbers[category]+=int(a.size)
        self.yes(e<=tol,category,f'error {e} > {tol}')
def main(args):
    started=time.perf_counter();o=args.out;o.mkdir(parents=True,exist_ok=False);c=Check();binary=args.input.read_bytes()
    c.yes(sha(binary)==EXPECTED_SHA,'zip_identity'); z=zipfile.ZipFile(io.BytesIO(binary)); names=z.namelist()
    c.yes(len(names)==len(set(names))==357,'zip_identity');c.yes(z.testzip() is None,'zip_crc')
    def raw(n): return z.read(n)
    used=set()
    def js(n): used.add(n);return json.loads(raw(n))
    def arr(n):
        used.add(n);a=np.load(io.BytesIO(raw(n)),allow_pickle=False);c.yes(np.isfinite(a).all(),'arrays_finite',n);return a
    members=[dict(path=i.filename,bytes=i.file_size,compressed_bytes=i.compress_size,crc32=f'{i.CRC:08x}',sha256=sha(raw(i.filename))) for i in z.infolist()]
    put(o/'input_identity.json',dict(filename=args.input.name,bytes=len(binary),sha256=sha(binary),members=members))
    table(o/'input_members.tsv',['path','bytes','sha256','crc32','compressed_bytes'],[[v[k] for k in ['path','bytes','sha256','crc32','compressed_bytes']] for v in members])
    def verify(rec,prefix=''):
        n=prefix+rec['path'];b=raw(n)
        c.yes(len(b)==rec['bytes'] and sha(b)==rec['sha256'],'referenced_identity',n);return n
    identities={}
    for f in ['inventory.json','source_manifest.json','analysis_manifest.json']:
        r=js(f);records=r['files']
        for v in records:verify(v)
        identities[f]=dict(files=len(records),bytes=sum(v['bytes'] for v in records))
        if 'file_count' in r:c.yes(r['file_count']==len(records),'inventory_count')
        if 'total_bytes' in r:c.yes(r['total_bytes']==sum(v['bytes'] for v in records),'inventory_count')
    comp=js('job/completion.json');man=js('job/run/manifest.json');before=js('job/before_valid.json');access=js('job/access.json');execution=js('job/execution.json');gate=js('qualification/formal_gate.json')
    col=js('job/evaluation/collected.json');ev=js('job/evaluation/evaluation.json');of=js('job/diagnostics_evaluation/overfitting.json');analysis=js('analysis/summary.json')
    sources=comp['source_files']
    for obj in [man,before,gate,col,ev,js('qualification/gpu/attempt02/result/result.json'),js('qualification/cpu/attempt03/result/result.json')]:c.yes(obj['source_files']==sources,'source_chain')
    c.yes(execution['sources']==sources,'source_chain')
    c.yes(len(sources)==29,'source_chain')
    for rec in sources:verify(rec,'source/')
    tarinfo={}
    for attempt in ('01','02','03'):
        path=f'qualification/cpu/attempt{attempt}/source.tar.gz';t=tarfile.open(fileobj=io.BytesIO(raw(path)),mode='r:gz'); entries={m.name:t.extractfile(m).read() for m in t.getmembers() if m.isfile()}; recs=js(f'qualification/cpu/attempt{attempt}/result/result.json')['source_files']; checked=0
        for rec in recs:
            matches=[(n,b) for n,b in entries.items() if n==rec['path'] or n.endswith('/'+rec['path'])]
            c.yes(len(matches)==1,'cpu_frozen_tar',rec['path']);b=matches[0][1]
            c.yes(len(b)==rec['bytes'] and sha(b)==rec['sha256'],'cpu_frozen_tar',rec['path']);checked+=1
        diffs=[r['path'] for r in recs if sha(raw('source/'+r['path']))!=r['sha256']]
        tarinfo[path]=dict(verified_sources=checked,different_from_current=diffs)
    verify(before['manifest'],'job/')
    c.yes(before['status']=='ALL_BLIND_GATES_PASS' and before['access']==dict(train=1,valid=0,heldout=0,owners=0),'blind_gate')
    c.yes(access==comp['access']==dict(train=1,valid=1,heldout=0,owners=0),'access_record')
    c.yes(comp['physical_updates']==man['physical_updates']==1440 and comp['gradient_group_presentations']==man['gradient_group_presentations']==2592,'counts')
    c.yes('job/failure.json' not in names,'formal_failure_record')
    c.yes(comp['status']=='COMPLETE_QUERY_REPLAY_FIXED_AB','completion')
    st=js('qualification/state.json');nat=js('qualification/gpu/attempt02/result/result.json');fail=js('qualification/gpu/attempt01/result/result.json');auth=js('qualification/native_authorization.json')
    c.yes(sum(v['actual_updates'] for v in st['gpu_attempts'])==st['gpu_used_actual_updates']==11,'qualification_ledger')
    c.yes(sum(v['charged_seconds'] for v in st['gpu_attempts'])==st['gpu_used_conservative_seconds']==469,'qualification_ledger')
    c.yes(sum(v['charged_seconds'] for v in st['cpu_attempts'])==st['cpu_used_conservative_seconds']==46,'qualification_ledger')
    c.yes(auth['cumulative_actual_updates_maximum']==11 and auth['cumulative_seconds_maximum']==600,'qualification_authorization')
    c.yes(nat['actual_updates']==6 and len(fail['completed_updates'])==5,'qualification_updates')
    nt=nat['native']; mm=nt['measurements']; projection=dict(train_seconds=288*sum(mm[k] for k in ['t_A','t_S','t_Q','t_q','t_b']),forward_seconds=3454*mm['f']+144*mm['h'],io_seconds=mm['L']+5*mm['s_F']+9*mm['r_F']+5*mm['s_I']+5*mm['r_I'])
    projection['projected_total_seconds']=1.25*(sum(projection.values())+3600);projection['projected_peak_output_bytes']=math.ceil(1.10*(2*mm['W_F']+5*mm['W_I']+2*2**30))
    for k,v in projection.items():c.close(v,nt['projection'][k],'projection')
    for k in ['new_term_encoder_head_effect','query_encoder_head_effect','next_update_bitwise_equal']:c.yes(nt[k] is True,'native_record')
    for effect in [nt['effects'],nt['Q_minus_S']['effects']]:
        for module in ['encoder','head']:
            v=effect[module];c.yes(v['parameter_max_delta']>0 and v['post_clip_gradient_max_delta']>0,'native_record')
    for key in ['content_sha256','file_count','total_size_bytes']:c.yes(nt['pretrained_archive'][key]==man['pretrained_archive'][key],'pretrained_identity_record')
    c.yes(execution['numerics']==nat['numerics'],'numerics_record')
    lim=comp['budget']['limits']; b=comp['budget']
    for a,k in [('elapsed_seconds','maximum_gpu_stage_seconds'),('peak_cuda_reserved_bytes','maximum_cuda_reserved_bytes'),('peak_output_bytes','maximum_output_bytes'),('peak_rss_bytes','maximum_rss_bytes')]:c.yes(b[a]<=lim[k],'resource_record',a)
    part=js('job/run/partition.json');verify(man['partition'],'job/run/')
    c.yes([r['group_uid'] for r in part['development']]==col['group_ids'],'row_alignment')
    c.yes(len(set(col['group_ids']))==40,'row_alignment')
    for role,n in [('fit',48),('calibration',12),('development',20)]:
        for d in 'AB':c.yes(sum(r['domain']==d for r in part[role])==n,'partition')
    c.yes(len({r['group_uid'] for v in part.values() for r in v})==160,'partition')
    members0=man['starts']['S']['members']
    for a in ['Q','C_q','C_b']:c.yes(man['starts'][a]==man['starts']['S'],'common_full_start_record')
    # Reconstruct UID-only scheduling/reservoir/draws without reading any Memory payload.
    fitids={d:[r['group_uid'] for r in part['fit'] if r['domain']==d] for d in 'AB'}
    ret=[];rng=random.Random(seed(20260930,'ABC','retention'))
    for seen,u in enumerate(sorted(fitids['A']),1):
        i=seen-1 if seen<=6 else rng.randrange(seen)
        if i<6:
            if i==len(ret):ret.append(u)
            else:ret[i]=u
    c.yes(ret==members0,'uid_reservoir')
    dr=random.Random(seed(20260930,'ABC',2,'history_draws'));history_ids=[dr.choice(ret) for _ in range(288)]
    logs={};training_summary={};presentations=0
    for p in POINTS:
        stage=1 if p=='A_shared' else 2;arm='S' if stage==1 else p[2:];point=man['points'][p]
        c.yes(js('job/run/points/'+p+'.json')==point,'endpoint_records')
        c.yes(point['full_restore_verified'] and point['inference_restore_verified'] and point['adam_step']==stage*288,'restore_records')
        for k in ['members','seen','tables','origins']:c.yes(point['memory_summary'][k]==man['points']['A_shared']['memory_summary'][k],'memory_record')
        c.yes(point['memory_summary']['bytes']<=1048576,'memory_budget_record')
        log=js(verify(man['training'][p],'job/run/'));logs[p]=log
        c.yes(len(log['rows'])==log['updates']==288 and log['adam_step']==stage*288,'updates')
        rng=random.Random(seed(20260918,'ABC',stage,'current'));sched=[]
        for epoch in range(6):q=sorted(fitids['AB'[stage-1]]);rng.shuffle(q);sched+=q
        c.yes(log['current_ids']==sched==[r['current_uid'] for r in log['rows']],'uid_schedule')
        c.yes(log['history_ids']==([] if stage==1 else history_ids),'uid_history')
        if stage==2:
            for key in ['members','seen','tables','origins']:c.yes(log['memory_before'][key]==log['memory_after'][key],'memory_record')
            c.yes(log['memory_before']['count']==0 and log['memory_after']['count']==288,'memory_draw_count')
        for step,row in enumerate(log['rows'],1):
            c.yes(row['step']==step and np.isfinite(row['gradient_norm']) and row['gradient_norm']>0,'per_step_record')
            lr=1e-5*step/29 if step<=29 else 1e-5*(288-step)/259
            c.close(row['encoder_lr'],lr,'learning_rate');c.close(row['head_lr'],.001,'learning_rate')
            cc=row['current'];c.close(cc['weighted'],cc['supervised'],'current_S');presentations+=1
            total=cc['supervised']
            if stage==2:
                h=row['history'];presentations+=1
                c.yes(row['history_uid']==history_ids[step-1],'uid_history')
                weight=.25 if arm in ['C_q','C_b'] else 0.
                expected=.1*h['supervised']+.5*h['mse0']+weight*h.get('response',0.)
                c.close(h['weighted'],expected,'fp32_history_decomposition',tol=2e-6+2e-6*abs(expected));total+=expected
                if arm!='S':
                    c.yes(h['response_weight']==weight,'R_weights')
                    c.close(h['weighted_response'],weight*h['response'],'R_weighted')
                    c.close(h['supervised'],.5*(h['natural_supervised']+h['augmented_supervised']),'fp32_supervised',tol=1e-6)
                    c.yes(h['augmentation_rule']==('within_account_bootstrap' if arm=='C_b' else 'query_replay'),'augmentation_rule')
            c.close(row['total'],total,'fp32_total_decomposition',tol=2e-6+2e-6*abs(total))
        if stage==2:
            hs=[r['history'] for r in log['rows']]
            training_summary[p]=dict(mean_R=float(np.mean([h.get('response',0.) for h in hs])),mean_weighted_R=float(np.mean([h.get('weighted_response',0.) for h in hs])),nonzero_weighted_R_steps=sum(h.get('weighted_response',0.)>0 for h in hs),last_encoder_lr=log['rows'][-1]['encoder_lr'],memory_max_recorded_bytes=max(log['memory_before']['bytes'],log['memory_after']['bytes'],point['memory_summary']['bytes']))
    c.yes(presentations==2592,'gradient_presentations')
    c.yes([r['history']['assignment_sha256'] for r in logs['B_Q']['rows']]==[r['history']['assignment_sha256'] for r in logs['B_C_q']['rows']],'288_assignment_hash_pairs')
    cols=col['columns'];domains=np.array(col['domains']);rr=[np.where(domains==d)[0] for d in 'AB']; draw=np.random.Generator(np.random.PCG64(20260930)).integers(20,size=(5000,2,20))
    c.yes(list(map(len,rr))==[20,20] and len(cols)==22,'metric_scope')
    def stats(v,i):return v.mean(0),np.quantile(v[draw[:,i]].mean(1),[.025,.975],axis=0,method='linear')
    statrows=[]
    def checkstats(v,r,i,kind,name,role,d):
        mean,ci=stats(v,i)
        expected=np.array([[r[k]['mean'],*r[k]['conditional_95pct_interval']] for k in cols])
        c.close(np.column_stack((mean,ci.T)),expected,'fixed_statistics')
        statrows.extend([[kind,name,role,d,k,float(mean[j]),float(ci[0,j]),float(ci[1,j])] for j,k in enumerate(cols)])
    matrices={};classification_rows=[];mapping_rows=[]
    def rates(ct):
        tp,fp,fn,tn=map(float,ct)
        precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/(tp+fn);sp=tn/(tn+fp);den=math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        return dict(precision=precision,recall=recall,f1=2*tp/(2*tp+fp+fn),specificity=sp,balanced_accuracy=(recall+sp)/2,mcc=(tp*tn-fp*fn)/den if den else 0.)
    for p in POINTS:
        point=man['points'][p];mapping=js(verify(point['mapping'],'job/run/'))
        c.yes(mapping['status']=='PASS_CALIBRATION_FIT' and mapping['optimizer_success'] and mapping['a']>0,'calibration_fit_record')
        c.yes(mapping['group_count']==12 and mapping['pair_count']==4536 and mapping['positive_count']==240,'calibration_fit_record')
        c.yes(point['maps']['first']==man['points']['A_shared']['maps']['first'],'first_mapping')
        mapping_rows.append([p,mapping['a'],mapping['b'],mapping['initial_nll'],mapping['final_nll'],mapping['projected_gradient_max']])
        rawscores=arr(verify(point['scores']['raw'],'job/run/'));cal=arr(verify(point['scores']['calibration'],'job/run/'))
        c.yes(rawscores.shape==(40,378) and rawscores.dtype==np.float32 and cal.shape==(12,378),'score_scope')
        matrices[p]={}
        for role in ROLES:
            rec=col['points'][p][role];m=arr(verify(rec['matrix'],'job/evaluation/'));q=arr(verify(rec['queries'],'job/evaluation/'));ct=js(verify(rec['counts'],'job/evaluation/'));sc=arr(verify(point['scores'][role],'job/run/'))
            c.yes(m.shape==(40,22) and m.dtype==np.float64 and q.shape==(40,28,10),'metric_shape')
            matrices[p][role]=m;c.close(q.mean(1),m[:,12:],'query_group_aggregation')
            c.yes(len(ct)==40,'counts_shape')
            counts=np.array([[row[k] for k in ['tp','fp','fn','tn']] for row in ct],dtype=np.int64)
            c.yes((counts>=0).all() and np.all(counts[:,0]+counts[:,2]==20) and np.all(counts[:,1]+counts[:,3]==358),'counts_denominators')
            # No labels: only number predicted positive can independently be checked against logits.
            c.close((sc>=0).sum(1),counts[:,0]+counts[:,1],'predicted_positive_count')
            group_rates=[rates(row) for row in counts]
            for k in group_rates[0]:c.close([v[k] for v in group_rates],m[:,cols.index(k)],'per_group_rates_from_counts')
            target=ev['absolute'][p][role]
            c.close(m.mean(0),[target['macro_all'][k] for k in cols],'macro_averages')
            for i,d in enumerate('AB'):checkstats(m[rr[i]],target['by_domain'][d],i,'absolute',p,role,d)
            for d,selected in [('all',np.arange(40)),('A',rr[0]),('B',rr[1])]:
                s=counts[selected].sum(0);r=rates(s);tr=target['fixed_half_classification'][d]
                for k,v in zip(['tp','fp','fn','tn'],s):c.close(v,tr[k],'pooled_counts')
                for k in ['precision','recall','f1']:c.close(r[k],tr[k],'pooled_rates')
                c.close(s[1]/(s[1]+s[3]),tr['fpr'],'pooled_rates')
                classification_rows.append([p,role,d,*map(int,s),r['precision'],r['recall'],r['f1'],s[1]/(s[1]+s[3]),*[float(m[selected,cols.index(k)].mean()) for k in ['precision','recall','f1','mcc']]])
            if role!='raw':
                af=point['maps']['stage' if role=='stage-cal' else 'first'];c.close(sc,af['a']*rawscores.astype(float)+af['b'],'affine_scores',tol=0.)
                order=np.argsort(rawscores,axis=1,kind='stable');after=np.argsort(sc,axis=1,kind='stable');c.yes(np.array_equal(order,after),'calibration_order')
                x=np.take_along_axis(rawscores,order,1);y=np.take_along_axis(sc,order,1)
                c.yes(np.array_equal(x[:,1:]==x[:,:-1],y[:,1:]==y[:,:-1]),'calibration_exact_ties')
                inds=[0,1,2,3,*range(12,22)];c.close(m[:,inds],matrices[p]['raw'][:,inds],'calibration_rank_invariance')
    map_groups={}
    for a,b in CONTRASTS:
        name=a+'-'+b;map_groups[name]={}
        for role in ROLES:
            delta=matrices['B_'+a][role]-matrices['B_'+b][role]
            for i,d in enumerate('AB'):
                checkstats(delta[rr[i]],ev['comparisons'][name][role][d],i,'contrast',name,role,d)
                if role=='raw':
                    v=delta[rr[i],cols.index('map')];map_groups[name][d]=dict(positive=int((v>0).sum()),negative=int((v<0).sum()),zero=int((v==0).sum()),values=v.tolist())
    for arm in ['S','Q','C_q','C_b']:
        for role in ROLES:checkstats((matrices['B_'+arm][role]-matrices['A_shared'][role])[rr[0]],ev['old_changes'][arm][role],0,'old_change',arm,role,'A')
    c.yes(len(statrows)==1452,'statistics_rows')
    oldcsv=list(csv.DictReader(io.StringIO(raw('analysis/all_metrics.csv').decode())))
    d0={tuple(r[:5]):r[5:] for r in statrows};d1={(r['kind'],r['point_or_contrast'],r['role'],r['domain'],r['metric']):[float(r[k]) for k in ['mean','lower_95','upper_95']] for r in oldcsv}
    c.yes(set(d0)==set(d1),'analysis_csv_keys')
    for k in d0:c.close(d0[k],d1[k],'analysis_csv_values')
    table(o/'all_statistics.tsv',['kind','point_or_contrast','role','domain','metric','mean','lower95','upper95'],statrows)
    table(o/'classification.tsv',['point','role','domain','TP','FP','FN','TN','pooled_precision','pooled_recall','pooled_F1','pooled_FPR','macro_precision','macro_recall','macro_F1','macro_MCC'],classification_rows)
    table(o/'calibration_records.tsv',['point','a','b','initial_nll','final_nll','projected_gradient_max'],mapping_rows)
    # Entire six-point diagnostic reconstruction, including raw gaps and all 27 rules.
    diagrows=[];curve_rows=[];diagnostics={};residuals={};nlabels=0
    for p in POINTS:
        stage=1 if p=='A_shared' else 2;arm='S' if stage==1 else p[2:];fit=[];cache=[];valid=[];residuals[p]=[]
        captures=[js(verify(rec,'job/run/diagnostics/')) for rec in man['diagnostics'][p]]
        c.yes([v['epoch'] for v in captures]==list(range(1,7)),'diagnostic_count')
        for epoch,d in enumerate(captures,1):
            c.yes(d['valid_labels_used'] is False and d['valid_ids']==col['group_ids'] and d['adam_step']==(stage-1)*288+epoch*48,'diagnostic_identity')
            c.yes(d['fit_ids']==fitids['AB'[stage-1]] and d['cache_ids']==([] if stage==1 else ret),'diagnostic_populations')
            f=arr(verify(d['fit'],'job/run/diagnostics/'));ca=arr(verify(d['cache'],'job/run/diagnostics/'));bl=arr(verify(d['valid_blind'],'job/run/diagnostics/'))
            vm=arr(verify(of['group_metrics'][p+'_epoch'+str(epoch)],'job/diagnostics_evaluation/'))
            c.yes(f.shape==(48,3) and ca.shape==((0 if stage==1 else 6),3) and vm.shape==(40,3) and bl.shape==(40,378),'diagnostic_shapes')
            fit.append(f);cache.append(ca);valid.append(vm)
            if epoch==6:
                c.close(vm,matrices[p]['raw'][:,[cols.index(k) for k in ['map','recall_at_5','log_loss']]],'epoch6_metric_identity')
                c.yes(raw('job/run/diagnostics/'+d['valid_blind']['path'])==raw('job/run/'+man['points'][p]['scores']['raw']['path']),'epoch6_score_bytes')
            es=d['teacher_errors'];c.yes([e['uid'] for e in es]==d['cache_ids'],'residual_population')
            for e in es:
                c.close(e['weighted_R_q'],.25*e['R_q'],'R_diagnostic_weight')
                c.close(e['D1']+e['D0']-2*e['residual_cross_mean'],e['residual_difference_mse'],'residual_algebra')
                c.yes('violations' not in e,'no_defined_violations_input')
            if es:
                keys=['D0','D1','residual_cross_mean','residual_difference_mse','R_q','weighted_R_q'];residuals[p].append(dict(epoch=epoch,**{k:float(np.mean([e[k] for e in es])) for k in keys}))
        fit,cache,valid=np.stack(fit),np.stack(cache),np.stack(valid);diagnostics[p]={}
        for i,d in enumerate('AB'):
            c.close(valid[:,rr[i]].mean(1),analysis['valid_curves'][p][d],'analysis_valid_curves')
        for role,t,d in [('fit',fit,'A' if stage==1 else 'B'),('cache',cache,'A')]:
            if not t.shape[1]:continue
            v=valid[:,rr['AB'.index(d)]];rng=np.random.Generator(np.random.PCG64(seed(20260930,'ABC',stage,role,d,'diagnostics')));td=rng.integers(t.shape[1],size=(5000,t.shape[1]));vd=rng.integers(20,size=(5000,20));sign=np.array([1,1,-1])
            tc=(t[-1]-t[0])*sign;vc=(v[-1]-v[0])*sign
            tb=tc[td].mean(1);vb=vc[vd].mean(1);tc_ci=np.quantile(tb,[.025,.975],axis=0).T;vc_ci=np.quantile(vb,[.025,.975],axis=0).T
            g=(t.mean(1)-v.mean(1))*sign;bg=(t[:,td].mean(2)-v[:,vd].mean(2))*sign;gc=g[-1]-g[0];gci=np.quantile(bg[-1]-bg[0],[.025,.975],axis=0).T
            target=of['points'][p][role][d]
            for a,b in [(t.mean(1),target['train_raw_curve']),(v.mean(1),target['valid_raw_curve']),(g,target['benefit_gap_curve']),(np.quantile(bg,[.025,.975],axis=1).transpose(1,2,0),target['benefit_gap_conditional_intervals']),(tc.mean(0),target['epoch1_to_6_train_benefit']['mean']),(vc.mean(0),target['epoch1_to_6_valid_benefit']['mean']),(tc_ci,target['epoch1_to_6_train_benefit']['conditional_95pct_interval']),(vc_ci,target['epoch1_to_6_valid_benefit']['conditional_95pct_interval']),(gc,target['epoch1_to_6_gap_widening']['mean']),(gci,target['epoch1_to_6_gap_widening']['conditional_95pct_interval'])]:c.close(a,b,'diagnostic_reconstruction')
            diagnostics[p][role]={};tcmean=tc.mean(0);vcmean=vc.mean(0)
            for j,k in enumerate(['map','recall_at_5','log_loss']):
                signal=tcmean[j]>0 and vcmean[j]<0 and gc[j]>0;strong=tc_ci[j,0]>0 and vc_ci[j,1]<0 and gci[j,0]>0
                decision='有相应迹象' if strong else ('证据不足' if signal else '未见明确迹象')
                c.yes(decision==target['interpretation'][k]==analysis['overfit_interpretations'][p][role][d][k],'diagnostic_27_labels');nlabels+=1
                diagrows.append([p,role,d,k,tcmean[j],*tc_ci[j],vcmean[j],*vc_ci[j],gc[j],*gci[j],decision])
                diagnostics[p][role][k]=dict(domain=d,train_benefit=float(tcmean[j]),train_interval=tc_ci[j].tolist(),valid_benefit=float(vcmean[j]),valid_interval=vc_ci[j].tolist(),gap_widening=float(gc[j]),gap_interval=gci[j].tolist(),decision=decision)
                for ep in range(6):curve_rows.append([p,role,d,k,ep+1,float(t[ep,:,j].mean()),float(v[ep,:,j].mean()),float(g[ep,j])])
    c.yes(nlabels==27,'27_labels')
    c.yes(all(residuals['B_C_q'][i]['R_q']<residuals['B_Q'][i]['R_q'] for i in range(6)),'Cq_Q_cache_R_six_points')
    table(o/'diagnostic_decisions.tsv',['point','role','domain','metric','train_benefit','train_lo','train_hi','valid_benefit','valid_lo','valid_hi','gap_widening','gap_lo','gap_hi','interpretation'],diagrows)
    table(o/'diagnostic_curves.tsv',['point','role','domain','metric','epoch','train_raw','valid_raw','benefit_gap'],curve_rows)
    put(o/'diagnostics.json',dict(decisions=diagnostics,residuals=residuals,diagnostic_labels_count=nlabels))
    # The stale violations_total field is explicitly NOT a scientific check.
    obsolete=[dict(point=p,epoch=r['epoch'],value=r['violations_total'],disposition='NO_DEFINED_CONSTRAINT; ignored') for p,rs in analysis['residuals'].items() for r in rs if 'violations_total' in r]
    put(o/'obsolete_field.json',obsolete)
    receive=js('receive_verification.json');oldids=receive['supplemental_verification']['shared_control_scores']
    for v in oldids:c.yes(v['identical'] and sha(raw(v['current_path']))==v['sha256'],'current_side_Ccf_identity')
    events=[]
    for line in raw('job.console.txt').decode().splitlines():
        if line.startswith('{'):
            v=json.loads(line)
            if v.get('event')=='updates':events.append(v)
    timing={}
    for p in POINTS:
        vv=[v for v in events if v.get('point')==p];c.yes([v['completed'] for v in vv]==[1,*range(24,289,24)],'console_progress')
        timing[p]=dict(after_step1_seconds=vv[0]['elapsed_seconds'],after_step288_seconds=vv[-1]['elapsed_seconds'],step2_to288_plus_6_diagnostics_seconds=vv[-1]['elapsed_seconds']-vv[0]['elapsed_seconds'])
    totals=dict(weights_bytes=sum(man['points'][p]['model']['bytes'] for p in POINTS),memory_snapshots_bytes=sum(man['points'][p]['memory']['bytes'] for p in POINTS))
    put(o/'execution_evidence.json',dict(identity_checks=identities,cpu_source_snapshots=tarinfo,qualification_cpu_seconds=46,qualification_gpu_conservative_seconds=469,qualification_gpu_actual_updates=11,formal_updates=1440,gradient_group_presentations=presentations,restored_endpoints=5,diagnostics=30,common_start_model_sha=man['starts']['S']['model'],common_start_adam_sha=man['starts']['S']['adam'],uid_reservoir=ret,candidate_assignment_hash_pairs=288,training=training_summary,formal_budget=comp['budget'],formal_console_intervals=timing,native_projection=projection,native_effects={k:{q:v for q,v in nt['effects'][k].items() if not isinstance(v,list)} for k in ['encoder','head']},custody_identity_totals=totals,source_model_identity='Source-bound records only; no model files loaded',Ccf_comparison='Verified current six score hashes against supplied receipt; prior C_cf bytes are not included and were not re-read'))
    put(o/'main_results.json',dict(map_absolute={p:{d:ev['absolute'][p]['raw']['by_domain'][d]['map'] for d in 'AB'} for p in POINTS},comparisons={a+'-'+b:{d:ev['comparisons'][a+'-'+b]['raw'][d]['map'] for d in 'AB'} for a,b in CONTRASTS},old_changes={a:ev['old_changes'][a]['raw']['map'] for a in ['S','Q','C_q','C_b']},paired_map_groups=map_groups))
    put(o/'data_read_scope.json',dict(programmatically_parsed=sorted(used),all_members_hashed=len(names),formal_label_reads=0,model_loads=0,memory_deserializations=0,training_updates=0,project_connections=0,external_research=0))
    put(o/'environment.json',dict(python=sys.version,numpy=np.__version__,platform=platform.platform(),cpu_threads_env={k:os.getenv(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']},scope='Web-workspace reference computation; not project-native qualification'))
    result=dict(status='PASS_SAVED_ARTIFACT_REFERENCE_CHECKS',elapsed_seconds=time.perf_counter()-started,assertions_by_category=dict(c.tests),scalar_comparisons_by_category=dict(c.numbers),maximum_absolute_errors_by_category=dict(c.errors),scalar_comparisons_total=sum(c.numbers.values()),statistic_rows=1452,diagnostic_labels=27,input_unchanged=(sha(args.input.read_bytes())==EXPECTED_SHA),limits='Not a scientific acceptance gate. No labels, training, model recovery, Memory parsing, new random seeds or methods. Fixed pre-existing bootstrap draws only.')
    put(o/'check_result.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    try:main(args)
    except BaseException:
        if args.out.exists():put(args.out/'reference_failure.json',dict(status='REFERENCE_CHECK_FAILED',traceback=traceback.format_exc(),project_failure=False))
        raise
