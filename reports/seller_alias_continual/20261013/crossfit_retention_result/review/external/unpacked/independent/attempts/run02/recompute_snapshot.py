#!/usr/bin/env python3
"""Independent read-only review of the supplied C_cf saved result package.

No project modules, torch, labels, model state, Memory, or F/O arrays are loaded.
The authoritative statistic contract is the supplied document sections 4/6.
Existing diagnostic decision semantics are transcribed from the narrow
curve_comparison excerpt, not imported or executed from the project.
"""
from __future__ import annotations
import argparse, collections, csv, datetime, hashlib, json, math, platform
import sys, tarfile, time, zipfile
from pathlib import Path
import numpy as np

EXPECTED_COLUMNS = ['average_precision','trapezoidal_pr_auc','roc_auc',
 'recall_at_fpr_1pct','brier','log_loss','precision','recall','f1','specificity',
 'balanced_accuracy','mcc','map','mrr','recall_at_1','recall_at_3','recall_at_5',
 'recall_at_10','ndcg_at_1','ndcg_at_3','ndcg_at_5','ndcg_at_10']
POINTS = ['A_shared', 'B_S', 'B_C_cf', 'B_C_perm']
ROLES = ['raw', 'stage-cal', 'first-cal']
DIAG_COLUMNS = ['map', 'recall_at_5', 'log_loss']

def sha(data):
    return hashlib.sha256(data).hexdigest()

class Review:
    def __init__(self, root, out):
        self.root, self.out = root, out
        self.reads, self.references, self.checks, self.results = {}, [], [], {}
        self.num = collections.defaultdict(lambda: {'scalars':0,'max_abs_error':0.0})

    def bytes(self, path, mode):
        name = str(path)
        assert not any(x in name for x in ('/memory/', '/models/', '/work/'))
        assert not name.endswith(('F_scores.npy','U0_scores.npy','U1_scores.npy'))
        data = (self.root / name).read_bytes()
        rec = self.reads.setdefault(name, {'path':name, 'bytes':len(data), 'sha256':sha(data), 'modes':[]})
        if mode not in rec['modes']: rec['modes'].append(mode)
        return data

    def js(self, path):
        return json.loads(self.bytes(path, 'json_semantic_check').decode('utf-8-sig'))

    def arr(self, path):
        import io
        a = np.load(io.BytesIO(self.bytes(path, 'array_numeric_check')), allow_pickle=False)
        self.require(np.isfinite(a).all(), f'finite array {path}')
        return a

    def require(self, condition, name):
        if not bool(condition): raise AssertionError(name)
        self.checks.append(name)

    def eq(self, a, b, name):
        self.require(a == b, name)

    def close(self, a, b, category, name):
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        self.require(a.shape == b.shape, 'shape '+name)
        self.require(np.isfinite(a).all() and np.isfinite(b).all(), 'finite '+name)
        err = float(np.max(np.abs(a-b))) if a.size else 0.0
        n = self.num[category]; n['scalars'] += int(a.size)
        n['max_abs_error'] = max(n['max_abs_error'], err)
        self.require(err <= 1e-12, f'{name}: max error {err:.17g}')

    def ref(self, rec, prefix='', required=True):
        name = str(Path(prefix) / rec['path'])
        target = self.root / name
        if not target.exists():
            self.references.append({'path':name,'status':'PAYLOAD_NOT_SUPPLIED_RECEIPT_ONLY','reference':rec})
            self.require(not required, 'required reference exists '+name)
            return name
        data = self.bytes(name, 'reference_size_sha_check')
        self.eq(len(data), rec.get('bytes',rec.get('size_bytes')), 'reference size '+name)
        self.eq(sha(data),rec['sha256'],'reference hash '+name)
        self.references.append({'path':name,'status':'BYTES_MATCH','reference':rec})
        return name

    def write(self, name, data):
        path = self.out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

    def resolve(self, rec):
        path = rec['path']
        if path.startswith('reports/qualification/'):
            path = 'qualification/evidence/' + path.removeprefix('reports/qualification/')
        elif path.startswith('reports/documentation/20261012/crossfit_retention/qualification/'):
            path = 'qualification/' + path.split('/qualification/',1)[1]
        elif path.startswith('reports/documentation/'):
            path = 'review/support/' + path
        elif path.startswith(('scripts/','schema/','docs/','tests/')):
            path = 'source/' + path
        return self.ref({**rec,'path':path})

    def provenance(self):
        manifest = self.js('input_manifest.json')
        self.eq(len(manifest['files']),302,'302 payloads')
        self.eq({r['path'] for r in manifest['files']},
                {str(p.relative_to(self.root)) for p in self.root.rglob('*') if p.is_file()}-{'input_manifest.json'},
                'payload set matches exact extracted input')
        for rec in manifest['files']: self.ref(rec)
        # Tar archives are byte evidence only; do not extract or execute sources.
        qualification=[]
        for kind, attempts in [('cpu',range(1,5)),('gpu',range(1,3))]:
            for attempt in attempts:
                prefix=f'qualification/evidence/{kind}/attempt{attempt:02}'
                receipt=self.js(prefix+'/result/result.json')
                import io
                archive=self.bytes(prefix+'/source.tar.gz','tar_source_identity_check')
                with tarfile.open(fileobj=io.BytesIO(archive),mode='r:gz') as tf:
                    files={m.name.removeprefix('./'):tf.extractfile(m).read() for m in tf.getmembers() if m.isfile()}
                refs=receipt['source_files']
                self.eq(set(files),{r['path'] for r in refs},prefix+' source path set')
                for rec in refs:
                    self.eq(len(files[rec['path']]),rec['bytes'],prefix+' tar size '+rec['path'])
                    self.eq(sha(files[rec['path']]),rec['sha256'],prefix+' tar hash '+rec['path'])
                current=[]
                for rec in refs:
                    data=self.bytes('source/'+rec['path'],'current_source_hash_comparison')
                    if sha(data)!=rec['sha256']: current.append(rec['path'])
                wrapper=self.bytes(prefix+'/wrapper.txt','execution_log_read').decode().splitlines()
                external=float(wrapper[3])-float(wrapper[1])
                qualification.append({'kind':kind,'attempt':attempt,'status':receipt['status'],
                  'outer_seconds':external,'inner_seconds':receipt['elapsed_seconds'],
                  'outer_conservative_seconds':math.ceil(external),'tests_run':receipt.get('tests_run'),
                  'updates':receipt.get('actual_updates'), 'source_count':len(refs),
                  'different_current_paths':current, 'tar_sha256':sha(archive),
                  'torch':receipt.get('torch'), 'stage':receipt.get('stage')})
                if (kind,attempt) in [('cpu',4),('gpu',2)]:
                    self.eq(current,[],prefix+' matches current 29 sources')
                if kind=='cpu':
                    self.eq([receipt[k] for k in ['failures','errors','skipped']],[0,0,0],prefix+' clean CPU receipt')
        gate=self.js('qualification/formal_gate.json')
        retry=self.js('qualification/native_retry_gate.json')
        self.eq(gate['cpu']['path'],'reports/qualification/cpu/attempt04/result/result.json','formal current CPU')
        self.eq(gate['gpu']['path'],'reports/qualification/gpu/attempt02/result/result.json','formal current native')
        for obj in [gate,retry]:
            for key in ['authorization','native_supplement_authorization','review','implementation_review','cpu','gpu']:
                if key in obj: self.resolve(obj[key])
        auth=self.js('qualification/native_retry_authorization.json')
        self.resolve(auth['original_authorization'])
        self.eq(auth['scope']['cumulative_actual_updates_maximum'],12,'approved cumulative 12 native updates')
        self.eq(auth['scope']['cumulative_seconds_maximum'],600,'approved 600 seconds cumulative')
        self.eq(retry['native_prior_seconds'],122,'prior native consumption retained')
        self.eq(retry['maximum_updates_this_attempt'],7,'seven supplemental native updates')
        self.eq(gate['cumulative_qualification'],{'cpu_seconds_conservative':70,'gpu_seconds_conservative':341,'native_actual_updates':12},'qualification cumulative receipt')
        for kind,limit in [('cpu',300),('gpu',600)]:
            selected=[r for r in qualification if r['kind']==kind]
            total=sum(r['outer_seconds'] for r in selected)
            ceiling=sum(r['outer_conservative_seconds'] for r in selected)
            self.require(ceiling<=limit,kind+' cumulative budget')
            self.results[kind+'_qualification_seconds']={'actual_outer_sum':total,'sum_per_attempt_ceiling':ceiling}
        g1=self.js('qualification/evidence/gpu/attempt01/result/result.json')
        g2=self.js('qualification/evidence/gpu/attempt02/result/result.json')
        self.eq(g1['status'],'FAIL','first native failure retained')
        self.eq(g1['stage'],'paired_off','first native failure stage')
        self.eq(g2['actual_updates'],7,'second native receipt seven steps')
        self.eq(len(g2['completed_updates']),7,'second native all completed update rows')
        self.eq(g2['native']['next_update_bitwise_equal'],True,'native saved next-update equality')
        self.eq(g2['native']['new_term_encoder_head_effect'],True,'native saved new term effects')
        m=g2['measurements']
        train=288*(m['t_A']+m['t_U']+m['t_S']+m['t_cf']+m['t_perm'])
        forward=2684*m['f']+6*m['h']
        io=3*m['L']+4*m['s_F']+7*m['r_F']+4*m['s_I']+4*m['r_I']
        time_projection=1.25*(train+forward+io+3600)
        disk_projection=math.ceil(1.10*(2*m['W_F']+4*m['W_I']+2*1024**3))
        # Time components allow only floating reduction-rounding error.
        for key,actual in [('train_seconds',train),('forward_seconds',forward),('io_seconds',io),('projected_total_seconds',time_projection)]:
            expected=g2['native']['projection'][key]
            self.require(abs(actual-expected)<1e-8,'native projection '+key)
        self.eq(disk_projection,g2['native']['projection']['projected_peak_output_bytes'],'native disk projection')
        self.eq(gate['projection'],g2['native']['projection'],'formal binds native projection')
        self.write('qualification_chain.json',{'attempts':qualification,'seconds':self.results,
          'recomputed_projection':{'train_seconds':train,'forward_seconds':forward,'io_seconds':io,
          'total_seconds':time_projection,'peak_output_bytes':disk_projection},
          'boundary':'All GPU/CPU behavior above is supplied project receipt evidence, not performed in this web review.'})
        self.execution=self.js('job/execution.json')
        self.completion=self.js('job/completion.json')
        self.before=self.js('job/before_valid.json')
        self.manifest=self.js('job/run/manifest.json')
        self.collected=self.js('job/evaluation/collected.json')
        self.evaluation=self.js('job/evaluation/evaluation.json')
        sources=self.completion['source_files']
        for name,obj,key in [('execution',self.execution,'sources'),('before_valid',self.before,'source_files'),
          ('manifest',self.manifest,'source_files'),('collected',self.collected,'source_files'),
          ('evaluation',self.evaluation,'source_files'),('formal gate',gate,'source_files'),
          ('retry gate',retry,'source_files'),('CPU04',self.js('qualification/evidence/cpu/attempt04/result/result.json'),'source_files'),
          ('GPU02',g2,'source_files')]:
            self.eq(obj[key],sources,'current sources '+name)
        for rec in sources: self.ref(rec,'source')
        self.resolve(self.execution['gate'])
        self.ref(self.before['manifest'],'job')
        self.eq(self.before['access'],dict(train=1,valid=0,heldout=0,owners=0),'before-valid access')
        self.eq(self.before['status'],'ALL_BLIND_GATES_PASS','before-valid status')
        final=dict(train=1,valid=1,heldout=0,owners=0)
        self.eq(self.js('job/access.json'),final,'final access')
        self.eq(self.completion['access'],final,'completion access')
        self.eq(self.completion['status'],'COMPLETE_CROSSFIT_FIXED_AB','completion status')
        inventory=self.js('inventory.json')
        for rec in inventory['files']: self.ref(rec)
        self.eq(len(inventory['files']),280,'original transfer file count')
        receive=self.js('receive_verification.json')
        self.eq(sum(r['bytes'] for r in inventory['files']),receive['bytes'],'original transfer bytes')
        analysis_refs=self.js('analysis_manifest.json')
        for rec in analysis_refs: self.ref(rec)
        self.eq(len(analysis_refs),10,'analysis transfer count')
        self.results['transfer']={'original_files':280,'original_bytes':receive['bytes'],
           'analysis_files':len(analysis_refs),'analysis_bytes':sum(r['bytes'] for r in analysis_refs)}
        self.results['excluded_receipts']=inventory['excluded']
        for r in inventory['excluded']:
            self.require(not (self.root/r['path']).exists(),'excluded payload absent '+r['path'])
        self.eq(inventory['failure_exists'],False,'saved final failure-absence receipt')
        self.require(not any(inventory['old_training_pids_present'].values()),'saved process-ended receipt')
        budget=self.completion['budget']
        for value,limit in [('elapsed_seconds','maximum_gpu_stage_seconds'),('peak_output_bytes','maximum_output_bytes'),
           ('peak_cuda_reserved_bytes','maximum_cuda_reserved_bytes'),('peak_rss_bytes','maximum_rss_bytes')]:
            self.require(budget[value]<=budget['limits'][limit],'formal budget '+value)
        self.results['formal_budget']=budget

    def execution_check(self):
        m=self.manifest
        partition=self.js(self.ref(m['partition'],'job/run'))
        grouped={role:{d:[r['group_uid'] for r in rows if r['domain']==d] for d in 'AB'} for role,rows in partition.items()}
        self.eq({k:[len(v[d]) for d in 'AB'] for k,v in grouped.items()},
          {'fit':[48,48],'calibration':[12,12],'development':[20,20]},'partition cardinalities')
        allids=[r['group_uid'] for rows in partition.values() for r in rows]
        self.eq(len(set(allids)),160,'fit calibration valid group UID disjoint')
        self.eq(self.collected['group_ids'],[r['group_uid'] for r in partition['development']],'metric group order')
        self.eq(self.collected['domains'],[r['domain'] for r in partition['development']],'metric domain order')
        self.eq(m['starts']['S'],m['starts']['C_cf'],'S C_cf common start receipt')
        self.eq(m['starts']['S'],m['starts']['C_perm'],'S C_perm common start receipt')
        members=m['starts']['S']['members']
        self.eq(len(members),6,'six common memory groups')
        self.require(set(members)<=set(grouped['fit']['A']),'memory from A-fit')
        self.require(all(m['starts']['S']['modes']),'all start modes train in receipt')
        fold_order=sorted(grouped['fit']['A'],key=lambda uid:(sha(('C_cf/fold/s0/'+uid).encode()),uid.encode()))
        self.eq(m['folds'],[fold_order[:24],fold_order[24:]],'fixed hash folds independent recreation')
        training={point:self.js(self.ref(m['training'][point],'job/run')) for point in POINTS}
        btraining=[training[p] for p in POINTS[1:]]
        self.eq(btraining[0]['current_ids'],btraining[1]['current_ids'],'S cf current sequence')
        self.eq(btraining[0]['current_ids'],btraining[2]['current_ids'],'S perm current sequence')
        self.eq(btraining[0]['history_ids'],btraining[1]['history_ids'],'S cf history sequence')
        self.eq(btraining[0]['history_ids'],btraining[2]['history_ids'],'S perm history sequence')
        train_summary={}
        for point,t in training.items():
            self.eq(len(t['rows']),288,point+' update rows')
            self.eq(t['updates'],288,point+' updates count')
            self.eq([r['step'] for r in t['rows']],list(range(1,289)),point+' step sequence')
            self.eq(t['adam_step'],288 if point=='A_shared' else 576,point+' terminal Adam receipt')
            self.eq([r['current_uid'] for r in t['rows']],t['current_ids'],point+' rows align current IDs')
            if point=='A_shared':
                self.eq(t['history_ids'],[],point+' no history draws')
                self.require(all(r['history_uid'] is None for r in t['rows']),point+' every row has no history')
            else:
                self.eq([r['history_uid'] for r in t['rows']],t['history_ids'],point+' rows align history IDs')
            domain='A' if point=='A_shared' else 'B'
            for e in range(6):
                self.eq(set(t['current_ids'][e*48:(e+1)*48]),set(grouped['fit'][domain]),point+f' epoch {e+1} current fit coverage')
            for r in t['rows']:
                step=r['step']
                lr=1e-5*step/29 if step<=29 else 1e-5*(288-step)/259
                self.close(r['encoder_lr'],lr,'logged_learning_rate',point+f' lr {step}')
                self.close(r['head_lr'],1e-3,'logged_learning_rate',point+f' head lr {step}')
                if point!='A_shared': self.require(r['history_uid'] in members,point+' history from six groups')
            if point!='A_shared':
                for key in ['members','seen','tables','origins']:
                    self.eq(t['memory_before'][key],t['memory_after'][key],point+' frozen history '+key)
                hist=[r['history'] for r in t['rows']]
                train_summary[point]={'nonzero_distillation_steps':sum(r['distillation']>0 for r in hist),
                  'mean_distillation':float(np.mean([r['distillation'] for r in hist])),
                  'mean_history_supervised':float(np.mean([r['supervised'] for r in hist])),
                  'mean_current_supervised':float(np.mean([r['current']['supervised'] for r in t['rows']])),
                  'mean_record_gradient_norm':float(np.mean([r['record_gradient_norm'] for r in hist])),
                  'mean_history_violations':float(np.mean([r['violations'] for r in hist])) if point!='B_S' else None}
        source_summary={}
        for i,role in enumerate(['U0','U1']):
            s=self.js(self.ref(m['sources'][role],'job/run'))
            self.eq(s['initial'],m['initial'],role+' original shared pre-A identity')
            self.eq([s['updates'],s['adam_step'],len(s['rows'])],[144,144,144],role+' update/Adam cardinalities')
            self.eq([r['step'] for r in s['rows']],list(range(1,145)),role+' real update indexes')
            self.eq(set(s['evaluated_ids']),set(members)&set(m['folds'][i]),role+' exact target fold survivor set')
            self.require(not set(s['evaluated_ids'])&set(m['folds'][1-i]),role+' training and target disjoint')
            for epoch in range(6):
                self.eq({r['group_uid'] for r in s['rows'][epoch*24:(epoch+1)*24]},set(m['folds'][1-i]),role+f' training epoch {epoch+1}')
            for r in s['rows']:
                self.eq(r['lr_step'],2*r['step'],role+' actual versus LR index')
                t=r['lr_step']; lr=1e-5*t/29 if t<=29 else 1e-5*(288-t)/259
                self.close(r['encoder_lr'],lr,'logged_learning_rate',role+f' lr {t}')
            source_summary[role]={'updates':144,'training_groups':24,'evaluated_groups':len(s['evaluated_ids'])}
        updates=sum(t['updates'] for t in training.values())+sum(s['updates'] for s in source_summary.values())
        presentations=training['A_shared']['updates']+2*sum(t['updates'] for t in btraining)+288
        for obj in [m,self.completion]:
            self.eq(obj['physical_updates'],updates,'1440 updates from row ledgers')
            self.eq(obj['gradient_group_presentations'],presentations,'2304 presentations from row ledgers')
        birth=self.js(self.ref(m['birth'],'job/run'))
        self.eq(set(birth['groups']),set(members),'birth groups align memory')
        self.eq(sum(r['active'] for r in birth['groups'].values()),birth['active_total'],'birth active sum')
        self.eq(sum(r['changed_entries'] for r in birth['groups'].values()),birth['changed_total'],'birth changed sum')
        total=sum(r['comparisons'] for r in birth['groups'].values())
        self.eq(total,6096,'six canonical 1016-comparison groups')
        self.close(birth['changed_fraction'],birth['changed_total']/total,'birth_summary','changed fraction')
        for r in birth['groups'].values():
            self.require(0<r['active']<=1016 and r['minimum_positive']>0 and r['maximum']>=r['minimum_positive'],'nondegenerate birth receipt')
            self.eq([r['violations'],r['distillation']],[0,0.0],'birth zero violations receipt')
        sizes={}
        for point in POINTS:
            p=m['points'][point]
            self.eq(p,self.js('job/run/points/'+point+'.json'),point+' point receipt matches manifest')
            self.require(p['full_restore_verified'] and p['inference_restore_verified'],point+' saved restore receipts')
            self.eq(p['adam_step'],288 if point=='A_shared' else 576,point+' endpoint Adam')
            self.eq(p['memory_summary']['members'],members,point+' endpoint memory members')
            self.eq(p['memory_summary']['seen'],48,point+' no B retention')
            self.require(p['memory']['bytes']<=2**20,point+' memory byte receipt within cap')
            self.eq(p['calibration_ids'],grouped['calibration']['A' if point=='A_shared' else 'B'],point+' current-domain calibration IDs')
            sizes[point]=p['memory']['bytes']
            for key in ['memory','full_state','model']:
                self.ref(p[key],'job/run',required=False)
        self.results['training']=train_summary
        self.results['sources']=source_summary
        self.results['birth']=birth
        self.results['memory_bytes']=sizes
        self.results['candidate_memory_saving_fraction']=1-sizes['B_C_cf']/sizes['B_S']
        self.results['independent_path_cost']={'S':{'updates':576,'presentations':864},'C_cf':{'updates':864,'presentations':1152},'C_perm':{'updates':864,'presentations':1152}}
        self.partition=grouped

    def statistics(self):
        col,reported=self.collected,self.evaluation
        self.eq(col['columns'],EXPECTED_COLUMNS,'all 22 columns exact ordered identity')
        self.eq(reported['columns'],EXPECTED_COLUMNS,'reported 22 columns')
        self.eq(reported['primary_metric'],'map','MAP primary')
        self.eq(reported['primary_role'],'raw','raw primary')
        self.indices={d:np.flatnonzero(np.array(col['domains'])==d) for d in 'AB'}
        rng=np.random.Generator(np.random.PCG64(20260930))
        draws=rng.integers(20,size=(5000,2,20))
        np.save(self.out/'fixed_main_draws.npy',draws,allow_pickle=False)
        self.results['main_draw']={'generator':'PCG64','seed':20260930,'shape':list(draws.shape),
          'dtype':str(draws.dtype),'data_sha256':sha(draws.tobytes()),'quantile':'linear 0.025, 0.975',
          'scope':'exact reconstruction of existing registered draws; not a new bootstrap seed or experiment'}
        self.matrices={}; self.scores={}; self.allstats=[]; self.classification={}; self.mapping={}
        rankcols=[i for i,c in enumerate(EXPECTED_COLUMNS) if c not in ['brier','log_loss','precision','recall','f1','specificity','balanced_accuracy','mcc']]
        def stat(values,domain,target,kind,name,role):
            i='AB'.index(domain)
            # Direct indexed paired values; independent of supplied frequency-matrix implementation.
            boots=values[draws[:,i,:]].mean(axis=1)
            means=values.mean(axis=0)
            bounds=np.quantile(boots,[.025,.975],axis=0,method='linear')
            for j,c in enumerate(EXPECTED_COLUMNS):
                actual=[means[j],bounds[0,j],bounds[1,j]]
                self.close(actual,[target[c]['mean'],*target[c]['conditional_95pct_interval']],
                  'endpoint_statistic',f'{kind}/{name}/{role}/{domain}/{c}')
                self.allstats.append({'kind':kind,'point_or_contrast':name,'role':role,'domain':domain,
                  'metric':c,'mean':float(actual[0]),'lower_95':float(actual[1]),'upper_95':float(actual[2])})
        for point in POINTS:
            self.matrices[point]={}; self.scores[point]={}; self.classification[point]={}
            p=self.manifest['points'][point]
            mapping=self.js(self.ref(p['mapping'],'job/run'))
            self.eq(mapping['status'],'PASS_CALIBRATION_FIT',point+' map receipt PASS')
            self.require(mapping['a']>0,point+' positive calibration slope')
            self.eq([mapping['a'],mapping['b']],[p['maps']['stage']['a'],p['maps']['stage']['b']],point+' map pointer parameters')
            self.eq([mapping['group_count'],mapping['pair_count'],mapping['positive_count']],[12,4536,240],point+' calibration receipt cardinality')
            self.eq(p['maps']['first'],self.manifest['points']['A_shared']['maps']['stage'],point+' fixed A map')
            self.mapping[point]={k:mapping[k] for k in ['a','b','group_count','pair_count','positive_count','objective','initial_nll','final_nll']}
            cal=self.arr(self.ref(p['scores']['calibration'],'job/run'))
            self.eq(cal.shape,(12,378),point+' saved calibration score shape')
            for role in ROLES:
                rec=col['points'][point][role]
                matrix=self.arr(self.ref(rec['matrix'],'job/evaluation'))
                self.eq(matrix.shape,(40,22),point+'/'+role+' matrix shape')
                self.eq(matrix.dtype,np.dtype('float64'),point+'/'+role+' matrix dtype')
                self.matrices[point][role]=matrix
                target=reported['absolute'][point][role]
                for d in 'AB': stat(matrix[self.indices[d]],d,target['by_domain'][d],'absolute',point,role)
                self.close(matrix.mean(axis=0),[target['macro_all'][c] for c in EXPECTED_COLUMNS],
                  'macro_all',point+'/'+role+' 40-group mean')
                counts=self.js(self.ref(rec['counts'],'job/evaluation'))
                self.eq(len(counts),40,'40 count records '+point+'/'+role)
                c=np.array([[r[k] for k in ['tp','fp','fn','tn']] for r in counts],dtype=np.int64)
                self.require(np.all(c>=0),'nonnegative confusion counts '+point+'/'+role)
                self.require(np.all(c[:,0]+c[:,2]==20) and np.all(c[:,1]+c[:,3]==358),'20 positive 358 negative each group '+point+'/'+role)
                tp,fp,fn,tn=c.T
                precision=np.divide(tp,tp+fp,out=np.zeros(40),where=tp+fp>0)
                recall=tp/(tp+fn); specificity=tn/(tn+fp)
                f1=np.divide(2*tp,2*tp+fp+fn,out=np.zeros(40),where=2*tp+fp+fn>0)
                denom=np.sqrt((tp+fp).astype(float)*(tp+fn)*(tn+fp)*(tn+fn))
                mcc=np.divide(tp*tn-fp*fn,denom,out=np.zeros(40),where=denom>0)
                self.close(matrix[:,6:12],np.column_stack([precision,recall,f1,specificity,(recall+specificity)/2,mcc]),
                  'classification_from_counts',point+'/'+role+' six group metrics from counts')
                self.classification[point][role]={}
                for d,ix in [('all',np.arange(40)),*self.indices.items()]:
                    tp,fp,fn,tn=map(int,c[ix].sum(axis=0))
                    values=dict(tp=tp,fp=fp,fn=fn,tn=tn,fpr=fp/(fp+tn),recall=tp/(tp+fn),
                      precision=tp/(tp+fp) if tp+fp else 0.0,f1=2*tp/(2*tp+fp+fn))
                    for key,value in values.items():
                        self.close(value,target['fixed_half_classification'][d][key],'pooled_counts_rates',point+'/'+role+'/'+d+'/'+key)
                    self.classification[point][role][d]=values
                scores=self.arr(self.ref(p['scores'][role],'job/run'))
                self.eq(scores.shape,(40,378),point+'/'+role+' saved endpoint score shape')
                self.scores[point][role]=scores
                self.close((scores>=0).sum(axis=1),c[:,0]+c[:,1],'threshold_positive_counts',point+'/'+role+' predicted positive totals without labels')
                if role!='raw':
                    raw=self.scores[point]['raw']; affine=p['maps']['stage' if role=='stage-cal' else 'first']
                    self.close(scores,raw.astype(np.float64)*affine['a']+affine['b'],'affine_scores',point+'/'+role+' exact affine')
                    ir=np.argsort(raw,axis=1,kind='stable'); ic=np.argsort(scores,axis=1,kind='stable')
                    self.require(np.array_equal(ir,ic),point+'/'+role+' full ordering equal')
                    sr=np.take_along_axis(raw,ir,axis=1); sc=np.take_along_axis(scores,ir,axis=1)
                    self.require(np.array_equal(sr[:,1:]==sr[:,:-1],sc[:,1:]==sc[:,:-1]),point+'/'+role+' exact ties equal')
                    self.close(matrix[:,rankcols],self.matrices[point]['raw'][:,rankcols],'rank_role_invariance',point+'/'+role+' 14 ranking columns invariant')
            if point=='A_shared':
                self.require(np.array_equal(self.scores[point]['stage-cal'],self.scores[point]['first-cal']),'shared A calibrated roles identical scores')
        pairs={}
        for other,name in [('B_S','C_cf-S'),('B_C_perm','C_cf-C_perm')]:
            pairs[name]={}
            for role in ROLES:
                delta=self.matrices['B_C_cf'][role]-self.matrices[other][role]
                for d,ix in self.indices.items():
                    stat(delta[ix],d,reported['comparisons'][name][role][d],'contrast',name,role)
                    if role=='raw':
                        values=delta[ix,12]
                        pairs[name][d]={'positive':int((values>0).sum()),'negative':int((values<0).sum()),
                          'zero':int((values==0).sum()),'values':values.tolist(),'minimum':float(values.min()),'maximum':float(values.max())}
        for arm in ['S','C_cf','C_perm']:
            for role in ROLES:
                delta=self.matrices['B_'+arm][role]-self.matrices['A_shared'][role]
                stat(delta[self.indices['A']],'A',reported['old_changes'][arm][role],'old_change',arm,role)
        self.eq(len(self.allstats),990,'990 complete 22-metric statistic rows')
        csv_ref=list(csv.DictReader(self.bytes('analysis/all_metrics.csv','csv_numeric_check').decode().splitlines()))
        key=lambda r:tuple(r[k] for k in ['kind','point_or_contrast','role','domain','metric'])
        old={key(r):r for r in csv_ref}
        self.eq(set(old),{key(r) for r in self.allstats},'submitted CSV complete statistic identity')
        for r in self.allstats:
            self.close([r[k] for k in ['mean','lower_95','upper_95']],
              [float(old[key(r)][k]) for k in ['mean','lower_95','upper_95']], 'linux_reference_csv',str(key(r)))
        with (self.out/'all_metrics_recomputed.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(self.allstats[0]));w.writeheader();w.writerows(self.allstats)
        self.write('classification_recomputed.json',self.classification)
        self.write('calibration_roles.json',self.mapping)
        self.write('paired_map_groups.json',pairs)
        self.results['primary']=[r for r in self.allstats if r['metric']=='map' and r['role']=='raw']
        self.results['classification']=self.classification
        self.results['paired_map_groups']=pairs
        self.results['absolute']={p:{role:{d:{name:float(self.matrices[p][role][ix,j].mean()) for j,name in enumerate(EXPECTED_COLUMNS)} for d,ix in self.indices.items()} for role in ROLES} for p in POINTS}

    def diagnostics(self):
        overfit=self.js('job/diagnostics_evaluation/overfitting.json')
        self.eq(overfit['columns'],DIAG_COLUMNS,'three diagnostic metrics')
        fixed={}; outputs={}; residuals={}; curves={}; interpretations={}
        for point in POINTS:
            captures=[self.js(self.ref(r,'job/run/diagnostics')) for r in self.manifest['diagnostics'][point]]
            self.eq([r['epoch'] for r in captures],list(range(1,7)),point+' all six epochs')
            stage=1 if point=='A_shared' else 2
            domain='A' if stage==1 else 'B'
            fits=[]; caches=[]; valids=[]; residuals[point]=[]
            for d in captures:
                self.eq(d['valid_labels_used'],False,point+' diagnostic labels blind receipt')
                self.eq(d['valid_ids'],self.collected['group_ids'],point+' diagnostic valid row identity')
                self.eq(d['metric_columns'],DIAG_COLUMNS,point+' diagnostic columns')
                self.eq(d['fit_ids'],self.partition['fit'][domain],point+' diagnostic fit row identity')
                self.eq(d['cache_ids'],[] if stage==1 else self.manifest['starts']['S']['members'],point+' diagnostic cache row identity')
                self.eq(d['stage_step'],48*d['epoch'],point+' diagnostic exact update point')
                self.eq(d['adam_step'],(stage-1)*288+48*d['epoch'],point+' diagnostic Adam receipt')
                fits.append(self.arr(self.ref(d['fit'],'job/run/diagnostics')))
                caches.append(self.arr(self.ref(d['cache'],'job/run/diagnostics')))
                tag=point+'_epoch'+str(d['epoch'])
                valids.append(self.arr(self.ref(overfit['group_metrics'][tag],'job/diagnostics_evaluation')))
                scores=self.arr(self.ref(d['valid_blind'],'job/run/diagnostics'))
                self.eq(scores.shape,(40,378),tag+' full blind score shape')
                self.eq(fits[-1].shape,(48,3),tag+' fit shape')
                self.eq(caches[-1].shape,(0 if stage==1 else 6,3),tag+' cache shape')
                self.eq(valids[-1].shape,(40,3),tag+' valid metric shape')
                if d['epoch']==6:
                    self.require(np.array_equal(scores,self.scores[point]['raw']),point+' epoch6 endpoint scores exact')
                    self.close(valids[-1],self.matrices[point]['raw'][:,[12,16,5]],'diagnostic_endpoint',point+' epoch6 endpoint metrics')
                if d['teacher_errors']:
                    es=d['teacher_errors']; keys=[k for k,v in es[0].items() if k!='uid' and isinstance(v,(int,float))]
                    row={'epoch':d['epoch'],**{k:float(np.mean([r[k] for r in es])) for k in keys},'violations_total':sum(r.get('violations',0) for r in es)}
                    residuals[point].append(row)
            v=np.stack(valids)
            curves[point]={dom:v[:,ix,:].mean(axis=1).tolist() for dom,ix in self.indices.items()}
            outputs[point]={}; interpretations[point]={}
            for role,data,dom in [('fit',np.stack(fits),domain),('cache',np.stack(caches),'A')]:
                outputs[point][role]={}; interpretations[point][role]={}
                if data.shape[1]==0:
                    self.eq(overfit['points'][point][role]['status'],'NOT_APPLICABLE_FIRST_STAGE',point+' empty cache status')
                    continue
                key=(stage,role,dom)
                if key not in fixed:
                    encoded=(json.dumps([20260930,'ABC',stage,role,dom,'diagnostics'],ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
                    seed=int.from_bytes(hashlib.sha256(encoded).digest()[:8],'big')%(2**63-1)
                    rng=np.random.Generator(np.random.PCG64(seed))
                    td=rng.integers(data.shape[1],size=(5000,data.shape[1])); vd=rng.integers(20,size=(5000,20))
                    fixed[key]=(seed,td,vd)
                    np.savez_compressed(self.out/f'existing_diagnostic_draws_stage{stage}_{role}_{dom}.npz',train=td,valid=vd)
                seed,td,vd=fixed[key]
                valid=v[:,self.indices[dom],:]; sign=np.array([1.,1.,-1.])
                tc=(data[-1]-data[0])*sign; vc=(valid[-1]-valid[0])*sign
                tb=tc[td].mean(axis=1); vb=vc[vd].mean(axis=1)
                # Epoch within each group remains paired; train and valid group draws are separate.
                gap=(data.mean(axis=1)-valid.mean(axis=1))*sign
                gb=(data[:,td,:].mean(axis=2)-valid[:,vd,:].mean(axis=2))*sign
                ti=np.quantile(tb,[.025,.975],axis=0,method='linear').T
                vi=np.quantile(vb,[.025,.975],axis=0,method='linear').T
                wi=np.quantile(gb[-1]-gb[0],[.025,.975],axis=0,method='linear').T
                decisions={}
                for j,name in enumerate(DIAG_COLUMNS):
                    point_signal=tc[:,j].mean()>0 and vc[:,j].mean()<0 and gap[-1,j]>gap[0,j]
                    supported=ti[j,0]>0 and vi[j,1]<0 and wi[j,0]>0
                    decisions[name]='有相应迹象' if supported else '证据不足' if point_signal else '未见明确迹象'
                value={'train_raw_curve':data.mean(axis=1).tolist(),'valid_raw_curve':valid.mean(axis=1).tolist(),
                  'benefit_gap_curve':gap.tolist(),'benefit_gap_conditional_intervals':np.quantile(gb,[.025,.975],axis=1,method='linear').transpose(1,2,0).tolist(),
                  'epoch1_to_6_train_benefit':{'mean':tc.mean(axis=0).tolist(),'conditional_95pct_interval':ti.tolist()},
                  'epoch1_to_6_valid_benefit':{'mean':vc.mean(axis=0).tolist(),'conditional_95pct_interval':vi.tolist()},
                  'epoch1_to_6_gap_widening':{'mean':(gap[-1]-gap[0]).tolist(),'conditional_95pct_interval':wi.tolist()},
                  'interpretation':decisions,'existing_seed':seed,'fit_or_cache_groups':data.shape[1],'valid_groups':20}
                expected=overfit['points'][point][role][dom]
                for field in ['train_raw_curve','valid_raw_curve','benefit_gap_curve','benefit_gap_conditional_intervals']:
                    self.close(value[field],expected[field],'diagnostic_curve',point+'/'+role+'/'+dom+'/'+field)
                for field in ['epoch1_to_6_train_benefit','epoch1_to_6_valid_benefit','epoch1_to_6_gap_widening']:
                    for sub in ['mean','conditional_95pct_interval']:
                        self.close(value[field][sub],expected[field][sub],'diagnostic_epoch1_6',point+'/'+role+'/'+dom+'/'+field+'/'+sub)
                self.eq(decisions,expected['interpretation'],point+'/'+role+'/'+dom+' independently reconstructed interpretation')
                outputs[point][role][dom]=value; interpretations[point][role][dom]=decisions
        self.results['valid_curves']=curves;self.results['residuals']=residuals;self.results['overfit_interpretations']=interpretations
        self.write('diagnostics_recomputed.json',outputs)
        self.write('residuals_and_curves.json',{'residuals':residuals,'valid_curves':curves})
        self.results['diagnostic_existing_draws']=[{'stage':key[0],'role':key[1],'domain':key[2],'existing_seed':seed,
          'train_shape':list(td.shape),'valid_shape':list(vd.shape),'train_sha256':sha(td.tobytes()),'valid_sha256':sha(vd.tobytes())} for key,(seed,td,vd) in fixed.items()]
        saved=self.js('analysis/summary.json')
        def recursive(actual,expected,path):
            if isinstance(expected,dict):
                self.eq(set(actual),set(expected),'summary keys '+path)
                for k in expected: recursive(actual[k],expected[k],path+'/'+k)
            elif expected is None or isinstance(expected,str): self.eq(actual,expected,'summary literal '+path)
            else: self.close(actual,expected,'linux_reference_summary',path)
        for key in ['absolute','classification','paired_map_groups','training','sources','birth','valid_curves','residuals','overfit_interpretations']:
            recursive(self.results[key],saved[key],key)

    def finish(self,start):
        self.write('result_summary.json',self.results)
        self.write('input_reads.json',sorted(self.reads.values(),key=lambda r:r['path']))
        self.write('reference_checks.json',self.references)
        self.write('checks.json',self.checks)
        status={'status':'PASS_SAVED_RESULT_RECOMPUTATION','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),'elapsed_seconds':time.perf_counter()-start,
          'source_sha256':sha(Path(__file__).read_bytes()),'assertion_checks':len(self.checks),'numeric_comparisons_by_category':dict(self.num),
          'numeric_scalar_comparisons':sum(r['scalars'] for r in self.num.values()),
          'maximum_absolute_error':max(r['max_abs_error'] for r in self.num.values()),
          'semantic_or_numeric_input_files':sum(any(m in r['modes'] for m in ['json_semantic_check','array_numeric_check','csv_numeric_check']) for r in self.reads.values()),
          'array_files':sum('array_numeric_check' in r['modes'] for r in self.reads.values()),
          'project_model_loads':0,'formal_label_parses':0,'memory_deserializations':0,'source_F_O_array_reads':0,
          'project_linux_connections':0,'training_updates':0,
          'limits':'Numerical replay of supplied saved matrices/counts/scores and comparison of supplied receipts. No ground-truth metric reconstruction or model-update replay. Counts are correlated checks, not independent scientific tests.'}
        self.write('audit_status.json',status)
        print(json.dumps(status,ensure_ascii=False,indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();review=Review(args.input.resolve(),args.output.resolve())
    review.provenance();review.execution_check();review.statistics();review.diagnostics();review.finish(start)

if __name__=='__main__': main()
