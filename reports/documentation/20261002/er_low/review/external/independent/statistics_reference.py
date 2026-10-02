"""Pure numerical references: explicit metrics, direct R(t,j), count-weight bootstrap.
No project metric, endpoint, summary, comparison or selection functions are used.
"""
from __future__ import annotations
import itertools,math
import numpy as np
COLUMNS=('average_precision','trapezoidal_pr_auc','roc_auc','recall_at_fpr_1pct','brier','log_loss','precision','recall','f1','specificity','balanced_accuracy','mcc','map','mrr','recall_at_1','recall_at_3','recall_at_5','recall_at_10','ndcg_at_1','ndcg_at_3','ndcg_at_5','ndcg_at_10')
ORDERS=('ABC','BCA','CAB');ROLES=('raw','stage-cal','first-cal','primary');ENDPOINTS=('O','N','Z','F_first','F','G','final_all')
RANK_INDEX=[0,1,2,3,*range(12,22)]
LOSS_INDEX=[4,5]
PAIRS=list(itertools.combinations(range(28),2))
INCIDENTS=[[(k,v if u==q else u) for k,(u,v) in enumerate(PAIRS) if q in (u,v)] for q in range(28)]


def confusion(counts):
    t,p,n,u=(counts[k] for k in ('tp','fp','fn','tn'))
    precision=t/(t+p) if t+p else 0.;recall=t/(t+n) if t+n else 0.;specificity=u/(u+p) if u+p else 0.
    den=math.sqrt((t+p)*(t+n)*(u+p)*(u+n))
    return dict(precision=precision,recall=recall,f1=2*t/(2*t+p+n) if 2*t+p+n else 0.,specificity=specificity,balanced_accuracy=(recall+specificity)/2,mcc=(t*u-p*n)/den if den else 0.)


def metrics_one(labels,scores):
    y=np.asarray(labels,dtype=np.uint8);z=np.asarray(scores,dtype=np.float64)
    assert y.shape==z.shape==(378,)
    tp=int(sum(int(a==1 and b>=0) for a,b in zip(y,z)));fp=int(sum(int(a==0 and b>=0) for a,b in zip(y,z)))
    counts=dict(tp=tp,fp=fp,fn=int(y.sum())-tp,tn=378-int(y.sum())-fp)
    result=confusion(counts);prob=np.exp(-np.logaddexp(0.,-z));clipped=np.clip(prob,1e-15,1-1e-15)
    result.update(brier=float(sum((float(p)-int(t))**2 for t,p in zip(y,prob))/378),log_loss=float(-sum(int(t)*math.log(float(p))+(1-int(t))*math.log1p(-float(p)) for t,p in zip(y,clipped))/378))
    ordered=sorted(range(378),key=lambda k:(-z[k],k));pos=int(y.sum());neg=378-pos
    tp=fp=0;prev_r=prev_f=0.;prev_p=1.;ap=trap=roc=low=0.;i=0
    while i<378:
        j=i
        while j<378 and z[ordered[j]]==z[ordered[i]]:
            tp+=int(y[ordered[j]]);fp+=1-int(y[ordered[j]]);j+=1
        r=tp/pos;f=fp/neg;p=tp/(tp+fp)
        ap+=(r-prev_r)*p;trap+=(r-prev_r)*(p+prev_p)/2;roc+=(f-prev_f)*(r+prev_r)/2
        if f<=.01:low=max(low,r)
        prev_r,prev_f,prev_p=r,f,p;i=j
    result.update(average_precision=ap,trapezoidal_pr_auc=trap,roc_auc=roc,recall_at_fpr_1pct=low)
    retrieval={key:0. for key in COLUMNS[12:]}
    for edges in INCIDENTS:
        ordered=sorted(edges,key=lambda t:(-z[t[0]],t[1]));rel=[int(y[e]) for e,_ in ordered];total=sum(rel);found=0;query_ap=0.;first=None
        for rank,r in enumerate(rel,1):
            found+=r
            if r:
                query_ap+=found/rank
                if first is None:first=rank
        retrieval['map']+=query_ap/total/28;retrieval['mrr']+=1/first/28
        for k in (1,3,5,10):
            retrieval[f'recall_at_{k}']+=sum(rel[:k])/total/28
            dcg=sum(r/math.log2(i+2) for i,r in enumerate(rel[:k]));ideal=sum(1/math.log2(i+2) for i in range(min(k,total)))
            retrieval[f'ndcg_at_{k}']+=dcg/ideal/28
    result.update(retrieval)
    return np.asarray([result[n] for n in COLUMNS]),counts


def draw_counts():
    draws=np.random.Generator(np.random.PCG64(20260930)).integers(0,20,size=(5000,3,20))
    weights=[]
    for d in range(3):
        w=np.stack([np.bincount(row,minlength=20) for row in draws[:,d]],axis=0).astype(np.float64)/20
        weights.append(np.vstack([np.full((1,20),1/20),w]))
    return draws,weights


def linear_quantile(values,q):
    v=np.sort(values,axis=0);at=(len(v)-1)*q;i=int(math.floor(at));f=at-i
    return v[i]*(1-f)+v[min(i+1,len(v)-1)]*f


def summary(ordered_samples):
    """ordered_samples: 3 orders x (1 point + 5000 resamples) x 22 metrics."""
    values=np.mean(ordered_samples,axis=0);lo=linear_quantile(values[1:],.025);hi=linear_quantile(values[1:],.975)
    return {k:dict(mean=float(values[0,j]),per_order={o:float(ordered_samples[i,0,j]) for i,o in enumerate(ORDERS)},conditional_95pct_interval=[float(lo[j]),float(hi[j])]) for j,k in enumerate(COLUMNS)}


def direct_endpoint_samples(arrays,domains,arm,role,weights):
    output={k:[] for k in ENDPOINTS};indices={d:[i for i,x in enumerate(domains) if x==d] for d in 'ABC'}
    assert all(len(v)==20 for v in indices.values())
    for order in ORDERS:
        s={}
        for t in (1,2,3):
            name=order+'_shared' if t==1 else f'{order}_{arm}_stage{t}'
            if role=='primary':
                a=arrays[name]['stage-cal'].copy();a[:,RANK_INDEX]=arrays[name]['raw'][:,RANK_INDEX]
            else:a=arrays[name][role]
            for d in 'ABC':s[t,d]=weights['ABC'.index(d)]@a[indices[d]]
        first,second,third=order
        values=dict(O=(s[3,first]+s[3,second])/2,N=(s[2,second]+s[3,third])/2,Z=s[3,third].copy(),F_first=s[1,first]-s[3,first],F=(s[1,first]-s[3,first]+s[2,second]-s[3,second])/2,G=(s[2,second]-s[1,second]+s[3,third]-s[2,third])/2,final_all=(s[3,first]+s[3,second]+s[3,third])/3)
        for key,value in values.items():
            if key in ('F_first','F','G'):value[:,LOSS_INDEX]*=-1
            output[key].append(value)
    return {k:np.stack(v,axis=0) for k,v in output.items()}


def checks(delta,raw):
    def mean(endpoint,key):return delta[endpoint][key]['mean']
    result=dict(old_map_improves=mean('O','map')>0,old_map_interval_above_zero=delta['O']['map']['conditional_95pct_interval'][0]>0,old_recall5_improves=mean('O','recall_at_5')>0,new_map_non_decrease=mean('N','map')>=0,new_recall5_non_decrease=mean('N','recall_at_5')>=0)
    for endpoint in ('O','N'):
        for key in ('average_precision','roc_auc','brier','log_loss'):
            value=mean(endpoint,key);result[f'{endpoint}_{key}_non_degradation']=value<=0 if key in ('brier','log_loss') else value>=0
        for key in ('brier','log_loss'):result[f'{endpoint}_{key}_against_raw_reference']=raw[endpoint][key]['mean']<=0
    for key in ('map','recall_at_5','average_precision','roc_auc','brier','log_loss'):
        value=mean('Z',key);result[f'Z_{key}_non_degradation']=value<=0 if key in ('brier','log_loss') else value>=0
    assert len(result)==23
    return {k:bool(v) for k,v in result.items()}


def evaluate(arrays,domains):
    draws,weights=draw_counts();all_samples={};endpoints={}
    for arm in ('seq','er','half','quarter','tenth'):
        endpoints[arm]={};all_samples[arm]={}
        for role in ROLES:
            samples=direct_endpoint_samples(arrays,domains,arm,role,weights)
            endpoints[arm][role]={key:summary(value) for key,value in samples.items()}
            if role in ('raw','primary'):all_samples[arm][role]=samples
    comparisons={}
    for arm in ('quarter','seq'):
        delta={key:summary(all_samples['tenth']['primary'][key]-all_samples[arm]['primary'][key]) for key in ENDPOINTS}
        raw={key:summary(all_samples['tenth']['primary'][key]-all_samples[arm]['raw'][key]) for key in ('O','N')}
        check=checks(delta,raw)
        comparisons['tenth_minus_'+arm]=dict(primary=delta,against_raw_reference=raw,checks=check,all_23=all(check.values()))
    return dict(endpoints=endpoints,comparisons=comparisons,selected='tenth' if comparisons['tenth_minus_quarter']['all_23'] else 'quarter'),draws
