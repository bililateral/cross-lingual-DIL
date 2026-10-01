#!/usr/bin/env python3
"""Independent saved-evidence audit. Standard library + NumPy only.

Does not import any project modules, reconstruct labels, load model weights,
train, or modify submitted evidence. Run with a NEW --out directory.
"""
from __future__ import annotations
import argparse, csv, datetime, hashlib, json, math, os, platform, random, sys
from pathlib import Path
import numpy as np

COLUMNS = ('average_precision','trapezoidal_pr_auc','roc_auc','recall_at_fpr_1pct',
           'brier','log_loss','precision','recall','f1','specificity','balanced_accuracy','mcc',
           'map','mrr','recall_at_1','recall_at_3','recall_at_5','recall_at_10',
           'ndcg_at_1','ndcg_at_3','ndcg_at_5','ndcg_at_10')
SEEDS=('s0','s1','s2')
RUNS=tuple(s+'_'+k for s in SEEDS for k in ('d','weighted'))
NEW=tuple(x for x in RUNS if x!='s0_d')
GUARDS={'recall_at_5':1,'average_precision':1,'roc_auc':1,'brier':-1,'log_loss':-1}

def jb(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def seed_for(seed,*parts):return int.from_bytes(hashlib.sha256(jb([seed,*parts])).digest()[:8],'big')%(2**63-1)

def require(condition, message):
    if not condition:raise AssertionError(message)

class Evidence:
    def __init__(self, root: Path):
        self.root=root;self.read_files={};self.checks={};self.assertions=0
    def identity(self,p,record=None):
        p=p.resolve();require(p.is_relative_to(self.root),'not within extracted submission')
        # Scope assertion for THIS REVIEWER, not a proposed production security change.
        require(p.suffix not in ('.pt','.pth','.bin','.safetensors') and p.name not in ('pairs.csv','items.jsonl','owners.csv'), 'outside saved-evidence review')
        actual={'bytes':p.stat().st_size,'sha256':sha(p)}
        if record is not None:require(actual=={k:record[k] for k in actual},'file identity: '+str(p))
        self.read_files[p.relative_to(self.root).as_posix()]=actual
        return p
    def j(self,p):return json.loads(self.identity(p).read_text(encoding='utf-8'))
    def a(self,base,record,rows,columns,dtype):
        p=self.identity(base/record['path'],record);a=np.load(p,allow_pickle=False)
        require(a.shape==(rows,columns) and a.dtype==np.dtype(dtype) and np.isfinite(a).all(),'array shape/type/finiteness: '+str(p))
        return a
    def equal(self,a,b,name,category='metrics',tol=1e-12):
        a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
        require(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),'shape/finiteness: '+name)
        d=float(np.max(np.abs(a-b))) if a.size else 0.
        require(d<=tol, f'{name}: difference {d} > {tol}')
        g=self.checks.setdefault(category,{'elements':0,'comparisons':0,'maximum_absolute_difference':0.,'tolerance':tol,'largest_difference_location':None})
        g['elements']+=int(a.size);g['comparisons']+=1
        if d>g['maximum_absolute_difference']:g['maximum_absolute_difference']=d;g['largest_difference_location']=name
    def mapping(self,expected,actual,name,category='metrics'):
        for k,v in expected.items():
            if isinstance(v,dict):self.mapping(v,actual[k],name+'/'+k,category)
            else:self.equal(v,actual[k],name+'/'+k,category)

def counts(raw,n):
    a=np.asarray([[r[k] for k in ('tp','fp','fn','tn')] for r in raw] if isinstance(raw[0],dict) else raw)
    require(a.shape==(n,4) and a.dtype.kind in 'iu' and (a>=0).all(),'counts shape')
    require((a[:,0]+a[:,2]==20).all() and (a[:,1]+a[:,3]==358).all(),'complete group counts')
    return a

def rates(t):
    """One or many already-summed confusion tables; never average precision ratios."""
    t=np.asarray(t,dtype=np.float64);tp,fp,fn,tn=np.moveaxis(t,-1,0)
    def div(a,b):return np.divide(a,b,out=np.zeros_like(a,dtype=float),where=b!=0)
    p=div(tp,tp+fp);r=div(tp,tp+fn);s=div(tn,tn+fp)
    return {'tp':tp,'fp':fp,'fn':fn,'tn':tn,'precision':p,'recall':r,
            'fpr':div(fp,fp+tn),'f1':div(2*tp,2*tp+fp+fn),'specificity':s,
            'balanced_accuracy':(r+s)/2,'mcc':div(tp*tn-fp*fn,np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))}

def quantile(a):
    return np.quantile(a,[.025,.975],axis=0,method='linear')

def csv_write(p,rows):
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main(root:Path,out:Path):
    out.mkdir(parents=True,exist_ok=False)
    if hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    E=Evidence(root)
    job=root/'reports/seller_alias_continual/20260925/pooling_execution/20260925_125203/job'
    resroot=root/'reports/seller_alias_continual/20260926/pooling_result'
    inv=E.j(resroot/'review/source_inventory.json')
    for rec in inv['files']:E.identity(root/rec['path'],rec)
    require(len(inv['files'])==165 and sum(r['bytes'] for r in inv['files'])==6376611,'submission inventory totals')
    manifest=E.j(job/'run/manifest.json');preparse=E.j(job/'evaluation/preparse_verification.json')
    collected=E.j(job/'evaluation/collected.json');evaluation=E.j(job/'evaluation/evaluation.json')
    complete=E.j(job/'completion.json');policy=E.j(root/'schema/step28_alias_pooling_policy.json')
    base_policy=E.j(root/'schema/step28_chinese_base_policy.json');partition=E.j(job/'run/partition.json')
    sync=E.j(resroot/'sync_inventory.json');receipt=E.j(resroot/'sync.json')
    require(sha(resroot/'sync_inventory.json')==receipt['inventory_sha256'],'sync inventory binding')
    require(len(sync['files'])==receipt['files']==95 and sum(r['bytes'] for r in sync['files'])==receipt['bytes']==4680563,'sync totals')
    for rec in sync['files']+manifest['source_files']:E.identity(root/rec['path'],rec)
    require(sync['frozen_sources_fresh_verified']==manifest['source_files']==preparse['sources'],'frozen source records')
    require(manifest['policy']==evaluation['policy']==collected['policy']==policy,'policy identity')
    require(manifest['physical_updates']==complete['formal_updates']==4320 and set(manifest['runs'])==set(NEW),'updates/run identities')
    require(complete['label_parses']=={'train':1,'development':1,'heldout':0,'owners':0},'reported parse scope')
    require(E.j(job/'run/access.json')=={'train_parse_attempts':1,'development':0,'heldout':0,'owners':0},'train access')
    require(E.j(job/'evaluation/access.json')=={'development_parse_attempts':1,'train':0,'heldout':0,'owners':0},'valid access')
    require((job/'exit_status.txt').read_text().strip()=='0','formal exit')
    require(E.j(job/'run/completion.json')['manifest_sha256']==sha(job/'run/manifest.json')==preparse['manifest_sha256'],'manifest binding')
    require(tuple(evaluation['columns'])==COLUMNS and evaluation['columns']==collected['columns'],'metric columns')
    require(evaluation['group_ids']==collected['group_ids']==[r['group_uid'] for r in partition['development']],'valid row IDs')
    require(evaluation['domains']==collected['domains']==[r['domain'] for r in partition['development']],'valid row domains')
    domains=np.asarray(evaluation['domains']);ix={d:np.flatnonzero(domains==d) for d in 'ABC'}
    require(len(set(evaluation['group_ids']))==60 and all(len(i)==20 for i in ix.values()),'valid domain group numbers')
    role_ids={k:[r['group_uid'] for r in partition[k]] for k in partition}
    require(len(set(sum(role_ids.values(),[])))==240,'role disjointness')
    for role,n in [('fit',48),('calibration',12),('development',20)]:
        require(all(sum(r['domain']==d for r in partition[role])==n for d in 'ABC'),'per-domain role partition')
    # Reconstruct the partition from the public group IDs alone.
    train_rows=partition['fit']+partition['calibration'];expected_fit=set()
    for d in 'ABC':
        ids=[r['group_uid'] for r in train_rows if r['domain']==d]
        ids.sort(key=lambda uid:hashlib.sha256(jb([base_policy['partition']['seed'],d,uid])).hexdigest())
        expected_fit.update(ids[:48])
    require(expected_fit==set(role_ids['fit']),'ID-only partition reconstruction')
    for rec in policy['historical_d']['files']:E.identity(root/rec['path'],rec)
    oldroot=root/policy['historical_d']['run_root'];old_eval=root/policy['historical_d']['evaluation_root']
    require(E.j(oldroot/'partition.json')==partition,'old D partition')
    oldparent=E.j(oldroot/'manifest.json')
    require(oldparent['config']==base_policy,'old D frozen base config')
    for k in ('file_count','total_size_bytes','content_sha256'):
        require(manifest['pretrained_archive'][k]==base_policy['models']['split_rank'][k],'pretrained receipt '+k)
    # No model payload is read: bind recorded hashes across training, pre-valid and deletion receipts.
    expected_models={};matrices={};summaries={};autocounts={};training={};trajectory=[];metric_rows=[];count_rows=[];auto_rows=[]
    blind_rows=0
    for run in RUNS:
        ar=oldroot/'split_rank' if run=='s0_d' else job/'run'/run
        a=E.j(ar/'manifest.json');seed=run.split('_')[0]
        if run!='s0_d':E.identity(job/'run'/manifest['runs'][run]['manifest']['path'],manifest['runs'][run]['manifest'])
        require(a['updates']==864 and a['parameter_count']==a['trainable_parameter_count'],'arm update/trainability')
        require(a['fit_group_ids']==role_ids['fit'] and a['calibration_group_ids']==role_ids['calibration'],'arm role identity')
        common=a['preflight'].get('common_state_sha256',a['preflight'].get('initial_model_state_sha256'))
        stream=seed_for(policy['seeds'][seed]['schedule_seed'],'BASE_JOINT','current')
        rng=random.Random(stream);schedule=[]
        for epoch in range(6):
            ordered=sorted(role_ids['fit']);rng.shuffle(ordered);schedule+=ordered
        require(stream==a['dropout_stream'] and hashlib.sha256(jb(schedule)).hexdigest()==a['group_schedule_sha256'],'reconstructed schedule '+run)
        losses={k:[] for k in ('bce','rank','total')};observations={}
        for seg,start in zip(a['training'],(0,432),strict=True):
            require((seg['start'],seg['stop'],seg['updates'])==(start,start+432,432),'segment lengths')
            key='losses_by_epoch' if run=='s0_d' else 'mean_losses_by_epoch'
            for k in losses:losses[k]+=seg[key][k]
            obs={str(start+1):seg['first_update_modules']} if run=='s0_d' else seg['observations']
            if run!='s0_d':require(set(obs)==({'1','2'} if start==0 else {'433'}),'observation step coverage')
            expected={'encoder','head','aggregation'} if run.endswith('weighted') else {'encoder','head'}
            for step,item in obs.items():
                require(set(item)==expected and all(v['finite_nonzero_gradient'] and v['parameters_changed'] for v in item.values()),'actual module observations')
                if run.endswith('weighted'):
                    norms=item['aggregation']['parameter_gradient_norms_before_clip']
                    require(set(norms)=={'hidden.weight','hidden.bias','output.weight'},'new parameter gradient names')
                    require(all(v is not None and math.isfinite(v) for v in norms.values()),'gradient finiteness')
                    if step=='1':require(norms['hidden.weight']==norms['hidden.bias']==0 and norms['output.weight']>0,'zero initialization gradient')
                    else:require(all(v>0 for v in norms.values()),'hidden task gradients after first step')
            observations.update(obs)
        E.equal(np.asarray(losses['bce'])+losses['rank'],losses['total'],run+'/saved loss sums','loss_roundoff',1e-6)
        training[run]={'losses':losses,'loss_strictly_decreasing':{k:bool((np.diff(v)<0).all()) for k,v in losses.items()},
            'common_initial_state':common,'schedule_sha256':a['group_schedule_sha256'],'dropout_stream':stream,
            'parameter_count':a['parameter_count'],'updates':a['updates'],'resources':a['resources'],
            'formal_training_seconds':a['formal_training_seconds'],'module_observations':observations}
        for epoch in ('3','6'):
            point=evaluation['runs'][run]['points'][epoch]
            require(point==collected['runs'][run]['points'][epoch],'collection preserved '+run+epoch)
            ap=a['points'][epoch];require(ap['full_model_and_adam_reloaded'] and ap['model']['actual_reload_verified'],'replay receipt')
            require(ap['model']['path']==f'models/epoch{epoch}.pt','model point role')
            p=(ar/ap['model']['path']).relative_to(root).as_posix()
            expected_models[p]={k:ap['model'][k] for k in ('bytes','sha256')}
            for role,n in [('fit',144),('calibration',36),('development',60)]:
                sr=ap['scores'][role];scores=E.a(ar,sr,n,378,'float32')
                mr=point['file'] if role=='development' else ap['train_metrics'][role]['file']
                mat=E.a(job/'evaluation' if role=='development' else ar,mr,n,22,'float64')
                c=counts(point['counts_at_logit_zero'] if role=='development' else ap['train_metrics'][role]['counts'],n)
                E.equal((scores>=0).sum(1),c[:,0]+c[:,1],run+epoch+role+'/blind predicted positives','blind_counts',0.)
                blind_rows+=n
                rr=rates(c)
                for metric in ('precision','recall','f1','specificity','balanced_accuracy','mcc'):
                    E.equal(mat[:,COLUMNS.index(metric)],rr[metric],run+epoch+role+'/'+metric,'group_confusion')
                role_domains=np.asarray([r['domain'] for r in partition[role]])
                for scope,index in {'all':np.arange(n),**{d:np.flatnonzero(role_domains==d) for d in 'ABC'}}.items():
                    means=np.array([math.fsum(mat[index,j])/len(index) for j in range(22)])
                    for j,metric in enumerate(COLUMNS):metric_rows.append({'run':run,'epoch':epoch,'split':role,'scope':scope,'metric':metric,'value':float(means[j])})
                    if role=='development':E.equal(means,[point['mean'][m] if scope=='all' else point['by_domain'][scope][m] for m in COLUMNS],run+epoch+scope+'/means','means')
                    totals=c[index].sum(0);rates_dict=rates(totals)
                    row={'run':run,'epoch':epoch,'split':role,'scope':scope,**{k:float(v) for k,v in rates_dict.items()}}
                    row['macro_precision']=float(rr['precision'][index].mean());row['macro_f1']=float(rr['f1'][index].mean());count_rows.append(row)
                    if role=='development':
                        target=point['fixed_classification']['pooled'] if scope=='all' else point['fixed_classification']['by_domain'][scope]
                        E.mapping({k:rates_dict[k] for k in ('tp','fp','fn','tn','fpr','recall','precision','f1')},target,run+epoch+scope+'/fixed','pooled_confusion')
                summary={m:float(math.fsum(mat[:,j])/n) for j,m in enumerate(COLUMNS)}
                summaries[run,epoch,role]=summary
                trajectory.append({'run_id':run,'epoch':epoch,'split':'valid' if role=='development' else role,**summary})
                if role=='development':matrices[run,epoch]=mat
        cal=E.j(E.identity(ar/a['calibration']['path'],a['calibration']))
        require(cal['epoch']==6 and cal['source_role']=='train_calibration_only','calibration source')
        require(cal['model_state_sha256']==a['points']['6']['model_state_sha256'] and cal['score_sha256']==a['points']['6']['scores']['calibration']['sha256'],'calibration binding')
        require(cal['threshold']==max(v['threshold'] for v in cal['bounds'].values()),'global threshold maximum')
        cc=counts(cal['counts_by_group'],36)
        for d,b in cal['bounds'].items():
            require(b['negative_pairs']==4296 and b['allowed_false_positives']==4,'calibration negative budget')
            require(b['threshold']==float(np.nextafter(np.float64(b['next_negative_logit']),np.inf)),'float64 threshold boundary')
            index=np.array([x['domain']==d for x in partition['calibration']]);tot=cc[index].sum(0)
            E.equal(tot,cal['counts_by_domain'][d],run+d+'/calibration totals','calibration',0.)
            require(tot[1]<=4,'calibration per-domain FP cap')
        cs=E.a(ar,a['points']['6']['scores']['calibration'],36,378,'float32')
        E.equal((cs.astype('float64')>=cal['threshold']).sum(1),cc[:,0]+cc[:,1],run+'/calibration positive totals','blind_counts',0.);blind_rows+=36
        auto=evaluation['runs'][run]['automatic_classification'];ac=counts(auto['counts_by_group'],60);autocounts[run]=ac
        require(auto['counts_by_group']==collected['runs'][run]['automatic_classification']['counts_by_group'],'automatic collection preserved')
        E.equal(auto['threshold'],cal['threshold'],run+'/auto threshold','calibration',0.)
        ds=E.a(ar,a['points']['6']['scores']['development'],60,378,'float32')
        E.equal((ds.astype('float64')>=auto['threshold']).sum(1),ac[:,0]+ac[:,1],run+'/valid auto positive totals','blind_counts',0.);blind_rows+=60
    for s in SEEDS:
        d=training[s+'_d'];w=training[s+'_weighted']
        require(all(d[k]==w[k] for k in ('common_initial_state','schedule_sha256','dropout_stream')),'paired sources '+s)
        require(w['parameter_count']-d['parameter_count']==131200,'added trainable parameter number')
    expected_preparse={r['path']:{k:r[k] for k in ('bytes','sha256')} for r in preparse['files']}
    require(len(expected_preparse)==len(preparse['files'])==12 and expected_preparse==expected_models,'all twelve pre-valid weight records')
    oldcoll=E.j(old_eval/'collected.json');alignment=E.j(job/'evaluation/historical_alignment.json')
    require(oldcoll['group_ids']==evaluation['group_ids'] and oldcoll['domains']==evaluation['domains'],'historical row identity')
    for epoch in ('3','6'):
        op=oldcoll['arms']['split_rank']['points'][epoch]
        om=E.a(old_eval,op['file'],60,22,'float64');diff=matrices['s0_d',epoch]-om
        E.equal(diff,np.zeros_like(diff),'old D '+epoch,'historical',0.)
        E.equal(diff,alignment['points'][epoch]['difference_by_group_metric'],'alignment '+epoch,'historical',0.)
        require(op['counts_at_logit_zero']==collected['runs']['s0_d']['points'][epoch]['counts_at_logit_zero'],'old D confusion identity')
    # Third implementation: generate each replicate in its own iteration, keeping ALL metrics and paired seeds together.
    delta=np.stack([matrices[s+'_weighted','6']-matrices[s+'_d','6'] for s in SEEDS])
    group_delta=delta.mean(0);draws=np.empty((5000,3,20),dtype=np.int64)
    boot=np.empty((5000,22));seed_boot=np.empty((5000,3,22));generator=np.random.default_rng(20260925)
    for b in range(5000):
        draws[b]=generator.integers(0,20,size=(3,20))
        chosen=np.concatenate([ix[d][draws[b,j]] for j,d in enumerate('ABC')])
        boot[b]=group_delta[chosen].mean(0)
        seed_boot[b]=delta[:,chosen,:].mean(1)
    # Verify the independent sequential draws equal the prospective generator stream, not a new bootstrap rule.
    E.equal(draws,np.random.default_rng(20260925).integers(0,20,size=(5000,3,20)),'sequential draw equivalence','bootstrap_draws',0.)
    ci=quantile(boot);seed_ci=quantile(seed_boot);paired={};primary_rows=[]
    for j,metric in enumerate(COLUMNS):
        dmean=float(np.stack([matrices[s+'_d','6'][:,j] for s in SEEDS]).mean())
        wmean=float(np.stack([matrices[s+'_weighted','6'][:,j] for s in SEEDS]).mean())
        expected={'mean':float(math.fsum(group_delta[:,j])/60),'per_seed':delta[:,:,j].mean(1).tolist(),
            'by_domain':{d:float(group_delta[v,j].mean()) for d,v in ix.items()},'conditional_95pct_interval':ci[:,j].tolist()}
        E.mapping(expected,evaluation['paired_primary']['metrics'][metric],metric+'/primary','primary_bootstrap')
        paired[metric]={'d_mean':dmean,'weighted_mean':wmean,**expected}
        for i,s in enumerate(SEEDS):E.mapping({'mean':float(delta[i,:,j].mean()),'conditional_95pct_interval':seed_ci[:,i,j]},evaluation['per_seed_comparisons'][s]['metrics'][metric],s+'/'+metric,'per_seed_bootstrap')
        primary_rows.append({'metric':metric,'d_mean':dmean,'weighted_mean':wmean,'delta':expected['mean'],'ci_low':float(ci[0,j]),'ci_high':float(ci[1,j]),
            **{s+'_delta':expected['per_seed'][i] for i,s in enumerate(SEEDS)},**{d+'_delta':expected['by_domain'][d] for d in 'ABC'}})
    # Full pooled and domain automatic reports AND paired comparisons (the submitted independent entry omits the latter).
    auto_boot={};auto_points={};auto_gates={}
    for run,ac in autocounts.items():
        auto=evaluation['runs'][run]['automatic_classification'];domain_totals={d:ac[ix[d]][draws[:,i]].sum(1) for i,d in enumerate('ABC')}
        domain_totals['pooled']=sum(domain_totals.values());auto_boot[run]={};auto_points[run]={};auto_gates[run]={}
        for scope in ('A','B','C','pooled'):
            selected=ac if scope=='pooled' else ac[ix[scope]];r=rates(selected.sum(0));br=rates(domain_totals[scope])
            dest=auto['pooled'] if scope=='pooled' else auto['by_domain'][scope]
            expected={k:r[k] for k in ('tp','fp','fn','tn','fpr','recall','precision','f1')}
            expected['conditional_95pct_intervals']={k:quantile(br[k]) for k in ('fpr','recall','precision')}
            E.mapping(expected,dest,run+scope+'/automatic','automatic_bootstrap')
            auto_boot[run][scope]=br;auto_points[run][scope]=r
            auto_rows.append({'run':run,'scope':scope,'threshold':auto['threshold'],**{k:float(v) for k,v in r.items()}})
            if scope!='pooled':
                passed=bool(r['fp']*1000<=r['fp']+r['tn'] and 2*r['tp']>=r['tp']+r['fn'])
                require(passed==dest['passes_point_gates'],'diagnostic absolute domain gate');auto_gates[run][scope]=passed
        require(all(auto_gates[run].values())==auto['all_domains_pass'],'diagnostic all domains gate')
    for s in SEEDS:
        w=s+'_weighted';d=s+'_d'
        for metric in ('fpr','recall','precision'):
            for scope in ('A','B','C','pooled'):
                expected={'difference':float(auto_points[w][scope][metric]-auto_points[d][scope][metric]),
                    'conditional_95pct_interval':quantile(auto_boot[w][scope][metric]-auto_boot[d][scope][metric])}
                E.mapping(expected,evaluation['per_seed_comparisons'][s]['automatic'][metric][scope],s+scope+metric+'/paired automatic','paired_automatic_bootstrap')
    checks={'map_minimum_observed_gain':paired['map']['mean']>=.01,'map_interval_above_zero':paired['map']['conditional_95pct_interval'][0]>0,
        'map_improves_each_paired_seed':all(v>0 for v in paired['map']['per_seed'])}
    for m,sgn in GUARDS.items():
        checks[m+'_mean_non_degradation']=sgn*paired[m]['mean']>=0
        checks[m+'_fixed_s0_non_degradation']=sgn*paired[m]['per_seed'][0]>=0
    require(checks==evaluation['acceptance']['checks']==complete['acceptance']['checks'],'all 13 acceptance flags')
    require(all(checks.values())==evaluation['acceptance']['passed']==False,'valid acceptance conclusion')
    require([k for k,v in checks.items() if not v]==evaluation['acceptance']['failed'],'failed-list exact coverage')
    # Compare all submitted analysis artifacts, not just selected narrative table values.
    oldanalysis=E.j(resroot/'analysis/analysis.json');newanalysis=E.j(resroot/'reviewer_analysis/analysis.json')
    for report,label in [(oldanalysis,'Linux submitted'),(newanalysis,'web entry')]:
        for m in COLUMNS:E.mapping(paired[m],report['paired'][m],label+'/'+m,'analysis_outputs')
        lookup={(t['run_id'],str(t['epoch']),t['split']):t for t in report['trajectories']}
        for t in trajectory:E.mapping({m:t[m] for m in COLUMNS},lookup[t['run_id'],t['epoch'],t['split']],label+'/trajectory','trajectory_outputs')
    for file in ('metrics.csv','trajectory.csv'):
        E.identity(resroot/'analysis'/file);E.identity(resroot/'reviewer_analysis'/file)
        require((resroot/'analysis'/file).read_bytes()==(resroot/'reviewer_analysis'/file).read_bytes(),'CSV cross-environment identical '+file)
    # All 22 rounded values, differences and intervals in the Chinese report.
    lines=(root/'docs/SELLER_ALIAS_POOLING_RESULT.zh.md').read_text().splitlines();narrative_count=0
    for line in lines:
        parts=[p.strip() for p in line.strip().strip('|').split('|')]
        if len(parts)==5 and parts[0] in COLUMNS:
            name=parts[0];values=[float(v) for v in parts[1:4]]+[float(v.strip()) for v in parts[4].strip('[]').split(',')]
            E.equal(values,[paired[name][k] for k in ('d_mean','weighted_mean','mean')]+paired[name]['conditional_95pct_interval'],name+'/report rounding','report_six_decimals',.500001e-6);narrative_count+=1
    require(narrative_count==22,'22 report rows')
    # Full observed progress, not a completion constant alone.
    log=(job/'train.log').read_text().splitlines();updates={x:[] for x in NEW}
    for line in log:
        try:r=json.loads(line)
        except json.JSONDecodeError:continue
        if r.get('event')=='updates':updates[r['run_id']].append(r['completed'])
    require(all(v==list(range(24,865,24)) for v in updates.values()),'all 180 progress milestones without duplicate retries')
    # Deletion receipts are records, not current remote filesystem verification.
    first=E.j(resroot/'weight_cleanup.json');first_inv=E.j(resroot/'weight_cleanup_inventory.json')
    second=E.j(root/'reports/maintenance/20260926/trained_weights/cleanup.json')
    require(first['inventory_sha256']==sha(resroot/'weight_cleanup_inventory.json'),'first cleanup inventory binding')
    require(len(first['removed'])==10 and sum(x['bytes'] for x in first['removed'])==first['logical_bytes_removed']==13067720890,'first deletion totals')
    require(len(second['removed'])==105 and sum(x['bytes'] for x in second['removed'])==second['logical_bytes_removed']==111917493162,'second deletion totals')
    deleted={x['path']:{k:x[k] for k in ('bytes','sha256')} for x in first['removed']+second['removed']}
    require(len(deleted)==115,'distinct deleted paths')
    require(all(deleted[p]==v for p,v in expected_models.items()),'12 evaluated model identities match their later deletion records')
    require(first['remaining_target_weights']==0 and second['remaining_trained_weights']==0,'reported deletion completion')
    deletion={'first_count':10,'first_bytes':first['logical_bytes_removed'],'second_count':105,'second_bytes':second['logical_bytes_removed'],
        'total_count':115,'total_bytes':sum(x['bytes'] for x in deleted.values()),'twelve_evaluated_models_match_deletion_records':True,
        'nonweight_metadata_unchanged_reported':second['preserved_nonweight_files_metadata_unchanged'],
        'scope':'receipt cross-check only; no access to remote payloads or full maintenance inventory'}
    starts=datetime.datetime.fromisoformat((job/'started.txt').read_text().strip());ends=datetime.datetime.fromisoformat((job/'finished.txt').read_text().strip())
    result={'status':'PASS_INDEPENDENT_SAVED_EVIDENCE_AUDIT; VALID_ACCEPTANCE_FAILED',
        'reviewed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'environment':{'python':sys.version,'numpy':np.__version__,
            'platform':platform.platform(),'cpu_affinity':sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
            'threads_status':[x for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('Threads:')],
            'thread_environment':{k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS')}},
        'numerical_checks':E.checks,'acceptance_checks':checks,'passed_checks':sum(checks.values()),'failed_checks':sum(not v for v in checks.values()),
        'paired':paired,'training':training,'trajectory':trajectory,'all_automatic_domain_gates':auto_gates,
        'model_record_count_bound':len(expected_models),'observed_training_milestones':{k:len(v) for k,v in updates.items()},'wall_seconds':(ends-starts).total_seconds(),
        'deletion':deletion,'blind_threshold_count_rows':blind_rows,'saved_matrices_read':36,'valid_matrices_read':12,
        'formal_label_reads':0,'formal_text_reads':0,'model_loads':0,'training_updates':0,'project_modules_imported':[],
        'limits':['No recomputation of label-dependent AP/AUC/retrieval/Brier/log-loss from formal ground truth.',
            'Runtime gradients/checkpoint replays/file hashes are audited receipts, not independently rerun GPU operations.',
            'No remote filesystem or deletion verification; no test or owner access; no new model conclusions.'],
        'read_files':E.read_files}
    (out/'independent_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    csv_write(out/'primary_22_metrics.csv',primary_rows);csv_write(out/'all_36_matrices_summaries.csv',metric_rows)
    csv_write(out/'all_fixed_confusion_summaries.csv',count_rows);csv_write(out/'all_automatic_confusion_summaries.csv',auto_rows)
    csv_write(out/'trajectories.csv',trajectory)
    np.save(out/'bootstrap_primary_5000x22.npy',boot,allow_pickle=False)
    print(json.dumps({k:v for k,v in result.items() if k in ('status','environment','numerical_checks','passed_checks','failed_checks','model_record_count_bound','observed_training_milestones','wall_seconds','deletion','blind_threshold_count_rows','saved_matrices_read','formal_label_reads','formal_text_reads','model_loads','training_updates')},ensure_ascii=False,indent=2))
    print('PRIMARY MAP',json.dumps(paired['map'],ensure_ascii=False))
    print('ALL TRAINING LOSS MONOTONICITY',json.dumps({r:training[r]['loss_strictly_decreasing'] for r in RUNS}))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    main(a.root.resolve(),a.out.resolve())
