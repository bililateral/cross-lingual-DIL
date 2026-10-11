"""One-off, read-only verification of the uploaded 2026-10-11 result package.

No project imports, model loading, labels, Memory deserialization, network,
Torch, GPU, or new experimental data. NumPy is used on already saved results.
Usage: python verify_saved_results.py --root EXTRACTED_PACKAGE --out OUTPUT
       [--archive ORIGINAL_ZIP]
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import random
import resource
import sys
import time
import zipfile

import numpy as np

ARMS = ('LOGIT0.1', 'R0', 'S')
ORDERS = ('ABC', 'BCA', 'CAB')
ROLES = ('raw', 'stage-cal', 'first-cal', 'primary')
ENDPOINTS = ('O', 'N', 'Z', 'F_first', 'F', 'G', 'final_all', 'A2')
PROB = ('brier', 'log_loss', 'precision', 'recall', 'f1', 'specificity', 'balanced_accuracy', 'mcc')
DCOLS = ('map', 'recall_at_5', 'log_loss')


def derived_seed(*parts):
    raw = (json.dumps(list(parts), ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False) + '\n').encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big') % (2**63 - 1)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


class Review:
    def __init__(self, root, out):
        self.root, self.out = root.resolve(), out.resolve()
        if self.out.is_relative_to(self.root):
            raise ValueError('Output must not modify or be placed inside the uploaded input')
        self.out.mkdir(parents=True, exist_ok=True)
        self.reads = defaultdict(set)
        self.failures = []
        self.check_counts = Counter()
        self.numeric = {}
        self.hashes = {}

    def check(self, good, name, detail=None):
        self.check_counts[name] += 1
        if not bool(good):
            self.failures.append({'check': name, 'detail': detail})

    def read(self, name, mode='json'):
        p = self.root / name
        self.reads[name].add(mode)
        if mode == 'json':
            return json.loads(p.read_text(encoding='utf-8'))
        if mode == 'text':
            return p.read_text(encoding='utf-8')
        if mode == 'array':
            return np.load(p, allow_pickle=False)
        return p.read_bytes()

    def verify(self, prefix, record):
        name = (Path(prefix) / record['path']).as_posix()
        raw = self.read(name, 'bytes')
        actual = (len(raw), hashlib.sha256(raw).hexdigest())
        self.check(actual == (record['bytes'], record['sha256']), 'linked_file_identity', name)
        return name

    def numbers(self, expected, observed, category, location, tol=1e-12):
        a, b = np.asarray(expected, dtype=np.float64), np.asarray(observed, dtype=np.float64)
        self.check(a.shape == b.shape, category + '_shape', location)
        if a.shape != b.shape:
            return
        error = float(np.max(np.abs(a-b), initial=0))
        state = self.numeric.setdefault(category, {'scalars': 0, 'max_absolute_error': 0., 'worst_location': None})
        state['scalars'] += int(a.size)
        if error > state['max_absolute_error']:
            state.update(max_absolute_error=error, worst_location=location)
        self.check(np.isfinite(a).all() and np.isfinite(b).all() and error <= tol,
                   category + '_numeric', {'location': location, 'max_absolute_error': error})

    def array(self, prefix, record, shape=None, dtype=None):
        name = self.verify(prefix, record)
        a = self.read(name, 'array')
        self.check((shape is None or a.shape == shape) and
                   (dtype is None or a.dtype == np.dtype(dtype)) and np.isfinite(a).all(),
                   'array_schema', name)
        return a

    def recursive_numbers(self, expected, observed, category, location):
        if isinstance(expected, dict):
            for k,v in expected.items():
                self.recursive_numbers(v, observed[k], category, location + '/' + k)
        elif isinstance(expected, str):
            self.check(expected == observed, category + '_text', location)
        else:
            self.numbers(expected, observed, category, location)


def verify_integrity(q, archive):
    inventory = []
    for p in sorted(q.root.rglob('*')):
        if p.is_file():
            name = p.relative_to(q.root).as_posix()
            raw = q.read(name, 'hash_only')
            q.hashes[name] = {'path': name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
            inventory.append(q.hashes[name])
    package = q.read('review/package_manifest.json')
    q.check(len(inventory) == 1266 and len(package['files']) == 1265, 'package_member_count')
    q.check(set(v['path'] for v in package['files']) == set(q.hashes)-{'review/package_manifest.json'}, 'package_member_coverage')
    for mf in ('review/package_manifest.json', 'sync_manifest.json', 'analysis_manifest.json'):
        obj = q.read(mf)
        for rec in obj['files']:
            q.check(q.hashes.get(rec['path']) == rec, 'manifest_identity', {'manifest': mf, 'path': rec['path']})
        if obj.get('total_bytes') is not None:
            q.check(sum(r['bytes'] for r in obj['files']) == obj['total_bytes'], 'manifest_total_bytes', mf)
    identity = {'uncompressed_files': len(inventory), 'uncompressed_bytes': sum(v['bytes'] for v in inventory),
                'declared_payload_records': len(package['files'])}
    if archive:
        raw = archive.read_bytes()
        identity.update(zip_bytes=len(raw), zip_sha256=hashlib.sha256(raw).hexdigest())
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            identity['zip_members'] = len(z.namelist())
            q.check(z.testzip() is None, 'archive_crc')
        q.check(identity['zip_bytes'] == 28532122 and identity['zip_members'] == 1266 and
                identity['zip_sha256'] == '9aaf0f2a82de854741313ee1b948432d335132085f5bc92e21e3ae9f9a2ff29a', 'archive_identity')
    return identity


def verify_sources(q, m, col, ev):
    gate, exe = q.read('qualification/formal_gate.json'), q.read('job/execution.json')
    sources = gate['source_files']
    q.check(len(sources) == 29, 'source_count')
    for table in (m['source_files'], col['source_files'], ev['source_files'], exe['sources']):
        q.check(table == sources, 'active_source_tables_equal')
    for rec in sources:
        q.verify('source', rec)
    q.verify('', {**exe['gate'], 'path': 'qualification/formal_gate.json'})
    p = q.read('source/schema/step28_record_attribution_policy.json')
    q.check(gate['runtime'] == p['runtime'] and gate['supervision'] == p['supervision'], 'gate_policy_equal')
    authorization = q.read('qualification/authorization_72h.json')
    q.verify('', {**gate['budget_authorization'], 'path': 'qualification/authorization_72h.json'})
    q.check(authorization['maximum_seconds'] == 259200 and authorization['previous_maximum_seconds'] == 172800,
            'authorization_72h_only')
    old = gate['verification_source_files']
    before, after = ({r['path']:r for r in rows} for rows in (old, sources))
    changed = sorted(k for k in before if before[k] != after.get(k))
    expected = sorted(('schema/step28_record_attribution_policy.json',
                       'scripts/run_step28_record_attribution_linux_20261009.sh', 'scripts/step28_record_attribution_run.py'))
    q.check(changed == expected and set(before) == set(after), 'budget_changed_source_set')
    q.verify('', {**gate['verification_archive'], 'path': 'qualification/source_snapshot.zip'})
    with zipfile.ZipFile(q.root/'qualification/source_snapshot.zip') as z:
        for rec in old:
            raw = z.read(rec['path'])
            q.check(len(raw)==rec['bytes'] and hashlib.sha256(raw).hexdigest()==rec['sha256'], 'native_snapshot_identity', rec['path'])
        for n, oldval, newval in ((expected[0], b'"maximum_gpu_stage_seconds": 172800', b'"maximum_gpu_stage_seconds": 259200'),
                                  (expected[1], b'172795', b'259195')):
            raw = z.read(n)
            q.check(raw.count(oldval)==1 and raw.replace(oldval,newval)==q.read('source/'+n,'bytes'), 'budget_exact_replacement', n)
        runner = 'scripts/step28_record_attribution_run.py'
        def stripped(raw):
            tree = ast.parse(raw)
            tree.body = [n for n in tree.body if not (isinstance(n,ast.FunctionDef) and n.name=='validate_gate')]
            return ast.dump(tree, include_attributes=False)
        q.check(stripped(z.read(runner)) == stripped(q.read('source/'+runner, 'bytes')), 'unchanged_scientific_ast')
    with zipfile.ZipFile(q.root/'qualification/source_snapshot_72h.zip') as z:
        for rec in sources:
            q.check(z.read(rec['path'])==q.read('source/'+rec['path'],'bytes'), 'formal_snapshot_identity', rec['path'])
        q.check(z.read('reports/qualification/authorization_72h.json')==q.read('qualification/authorization_72h.json','bytes'), 'formal_snapshot_authorization')
        q.check(z.read('scripts/run_step28_record_attribution_auto_linux_20261009.sh')==q.read('qualification/formal_listener/source.sh','bytes'), 'formal_snapshot_listener')
    desired = dict(cublas_workspace_config=':4096:8', cuda_matmul_allow_tf32=False,
                   cudnn_allow_tf32=False, cudnn_benchmark=False,
                   deterministic_algorithms=True, deterministic_warn_only=False)
    q.check(exe['numerics']==desired, 'formal_effective_numerics')
    native_gate = q.read('qualification/native_gate.json')
    q.check(native_gate['source_files']==old, 'native_source_tables_equal')
    for mode, path in (('cpu','qualification/cpu/attempt02/result/result.json'),
                       ('gpu','qualification/gpu/attempt01/result/result.json')):
        d = q.read(path)
        q.verify('', {**gate[mode], 'path':path})
        q.check(d['source_files']==old and d['numerics']==desired and d['mode']==mode and
                d['status']=='PASS_HANDWRITTEN_ONLY' and not d['formal_inputs'] and not d['formal_labels'], 'qualification_binding', mode)
        q.check(d['cumulative_seconds']<=p['cpu_verification' if mode=='cpu' else 'native_verification']['maximum_seconds'], 'qualification_budget', mode)
    q.verify('',{**gate['primary_review'],'path':'review/background/implementation_primary.zh.txt'})
    native=q.read('qualification/gpu/attempt01/result/result.json')
    for arm,row in native['native']['architectures'].items():
        q.check(row['maximum_records']==224 and row['channel_tokens']==256 and
                row['next_model_adam_grad_bitwise_equal'] and row['native_state_unchanged'] and
                all(row['first_changed'].values()) and all(row['paired_changed'].values()) and
                all(np.isfinite(v) and v>0 for v in row['paired_gradient_norms'].values()) and
                row['loss']==row['repeat_loss'] and row['numerics']==desired, 'native_claim_internal_consistency', arm)
    q.check(172800 < native['native']['projected_total_seconds'] <= 259200, 'projection_requires_72h')
    return {'active_sources':29, 'native_qualified_sources':29, 'changed_active_sources':changed,
            'scientific_ast_unchanged_outside_validate_gate':True,
            'native_elapsed_seconds':native['elapsed_seconds'],
            'native_projected_seconds':native['native']['projected_total_seconds'],
            'native_environment':{k:native['native'][k] for k in ('torch','cuda','gpu')},
            'effective_numerics':exe['numerics'], 'weight_and_memory_native_replay_performed_by_this_review':False}


def confusion_metrics(c):
    tp,fp,fn,tn = (np.asarray(c[k],np.float64) for k in ('tp','fp','fn','tn'))
    def div(a,b):
        return np.divide(a,b,out=np.zeros_like(np.asarray(a+b,dtype=float)),where=b!=0)
    recall, specificity = div(tp,tp+fn), div(tn,tn+fp)
    return {'precision':div(tp,tp+fp), 'recall':recall, 'f1':div(2*tp,2*tp+fp+fn),
            'specificity':specificity, 'balanced_accuracy':(recall+specificity)/2,
            'mcc':div(tp*tn-fp*fn,np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))}


def verify_metrics(q, m, col, ev):
    cols = col['metric_columns']
    domains=np.asarray(col['domains'])
    rows=[np.flatnonzero(domains==d) for d in 'ABC']
    q.check(len(cols)==22 and len(set(cols))==22 and len(set(col['group_ids']))==60 and all(len(x)==20 for x in rows), 'metric_dimensions')
    rank=[i for i,n in enumerate(cols) if n not in PROB]
    matrices={}
    countsets={}
    for point,roles in col['points'].items():
        matrices[point]={}
        for role,rec in roles.items():
            a=q.array('job/evaluation',rec['matrix'],(60,22),np.float64)
            counts=q.read(q.verify('job/evaluation',rec['counts']))
            q.check(len(counts)==60 and all(v['tp']+v['fn']==20 and v['fp']+v['tn']==358 for v in counts), 'confusion_class_totals', point+'/'+role)
            calculated=confusion_metrics({k:[r[k] for r in counts] for k in ('tp','fp','fn','tn')})
            for n,v in calculated.items():q.numbers(v,a[:,cols.index(n)],'count_derived_group_metrics',point+'/'+role+'/'+n)
            for d,ix in zip('ABC',rows):
                q.numbers(a[ix].mean(0),[ev['absolute_stage_results'][point][role]['macro_by_domain'][d][n] for n in cols], 'stage_domain_means',point+'/'+role+'/'+d)
            q.numbers(a.mean(0),[ev['absolute_stage_results'][point][role]['macro_all'][n] for n in cols], 'stage_all_means',point+'/'+role)
            matrices[point][role]=a
            countsets[point,role]=counts
        for role in ('stage-cal','first-cal'):
            q.check(np.array_equal(matrices[point]['raw'][:,rank],matrices[point][role][:,rank]), 'ranking_metric_invariance',point+'/'+role)
        matrices[point]['primary']=matrices[point]['stage-cal'].copy()
        matrices[point]['primary'][:,rank]=matrices[point]['raw'][:,rank]
    initial={}
    for family,rec in col['initial'].items():
        a=q.array('job/evaluation',rec['matrix'],(60,22),np.float64)
        counts=q.read(q.verify('job/evaluation',rec['counts']))
        q.check(len(counts)==60 and all(v['tp']+v['fn']==20 and v['fp']+v['tn']==358 for v in counts),'initial_confusion_class_totals',family)
        for n,v in confusion_metrics({k:[r[k] for r in counts] for k in ('tp','fp','fn','tn')}).items():
            q.numbers(v,a[:,cols.index(n)],'count_derived_group_metrics','initial/'+family+'/'+n)
        for d,ix in zip('ABC',rows):
            q.numbers(a[ix].mean(0),[ev['initial_raw'][family]['macro_by_domain'][d][n] for n in cols], 'initial_domain_means',family+'/'+d)
        initial[family]=a
        countsets['initial_'+family,'raw']=counts
    pooled_rows=[]
    for (point,role),counts in countsets.items():
        saved=(ev['initial_raw'][point[8:]] if point.startswith('initial_') else ev['absolute_stage_results'][point][role])['fixed_half_classification']
        for d,ix in [*zip('ABC',rows),('ALL',np.arange(60))]:
            totals={k:sum(counts[i][k] for i in ix) for k in ('tp','fp','fn','tn')}
            rates={k:float(v) for k,v in confusion_metrics(totals).items()}
            rates['fpr']=1-rates['specificity']
            ref=saved['pooled'] if d=='ALL' else saved['by_domain'][d]
            q.numbers([totals[k] for k in ('tp','fp','fn','tn')]+[rates[k] for k in ('precision','recall','f1','fpr')],
                      [ref[k] for k in ('tp','fp','fn','tn','precision','recall','f1','fpr')], 'pooled_classification',point+'/'+role+'/'+d)
            pooled_rows.append({'point':point,'role':role,'domain':d,**totals,**rates})
    write_csv(q.out/'pooled_classification.csv',pooled_rows)
    return matrices, initial, rows, countsets


def write_csv(path, rows):
    if not rows:return
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def verify_execution(q,m,col,matrices,countsets):
    part=q.read(q.verify('job/run',m['partition']))
    q.check([r['group_uid'] for r in part['development']]==col['group_ids'] and
            [r['domain'] for r in part['development']]==col['domains'],'development_alignment')
    partitions={k:{r['group_uid'] for r in v} for k,v in part.items()}
    q.check(all(not partitions[a]&partitions[b] for a,b in (('fit','calibration'),('fit','development'),('calibration','development'))),'partition_disjoint')
    q.check(all(Counter(r['domain'] for r in part[k])==dict.fromkeys('ABC',n) for k,n in (('fit',48),('calibration',12),('development',20))), 'partition_sizes')
    domain_of={r['group_uid']:r['domain'] for r in part['fit']}
    specs={f'{o}_{a}_stage{s}' for o in ORDERS for a in ARMS for s in (1,2,3)}
    q.check(all(set(m[k])==specs for k in ('points','training','diagnostics')), 'physical_stage_coverage')
    q.check(len(m['starts'])==9 and set(m['initial'])=={'LOGIT0.1','record'}, 'start_count')
    for o in ORDERS:
        q.check(m['starts'][o+'_R0']==m['starts'][o+'_S'], 'paired_record_initial_state',o)
        for a in ARMS:
            row=m['starts'][o+'_'+a]
            q.check(row['memory'] is None and row['adam_step']==0 and row['model_sha256']==m['initial_model_digests'][row['family']], 'untrained_start',o+'/'+a)
            q.check(row==m['starts']['ABC_'+a], 'same_family_start_across_orders',o+'/'+a)
    raw_scores={}
    for family,row in m['initial'].items():
        q.check(not row['valid_labels_used'] and row['model_sha256']==m['initial_model_digests'][family],'blind_initial_identity',family)
        q.array('job/run',row['raw'],(60,378),np.float32)
    totals=Counter();durations=Counter(); stage_rows=[]; losses=[]; retention={};logs={}
    excluded={r['path']:r for r in q.read('inventory.json')['excluded_payloads']}
    for point in sorted(specs):
        p=m['points'][point]; o,a,s=p['order'],p['arm'],p['stage']
        log=q.read(q.verify('job/run',m['training'][point]));logs[point]=log
        q.check(q.read('job/run/points/'+point+'.json')==p,'point_manifest_equal',point)
        q.check(point==f'{o}_{a}_stage{s}' and p['full_restore_verified'] and p['adam_step']==s*288 and
                log['adam_step']==s*288 and log['updates']==288 and len(log['rows'])==288,'stage_identity_restore_update',point)
        fit_ids=sorted(u for u,d in domain_of.items() if d==o[s-1])
        rng=random.Random(derived_seed(20260918,o,s,'current'));schedule=[]
        for _ in range(6):
            epoch=fit_ids.copy();rng.shuffle(epoch);schedule+=epoch
        q.check(log['current_ids']==schedule and [r['current_uid'] for r in log['rows']]==schedule,'current_schedule',point)
        if s==1:
            q.check(log['memory_before'] is None and log['memory_after'] is None and not log['history_ids'] and
                    all(r['history_uid'] is None for r in log['rows']), 'empty_first_stage_history',point)
        else:
            before,after=log['memory_before'],log['memory_after']
            rng=random.Random(derived_seed(20260930,o,s,'history_draws'))
            draws=[before['members'][rng.randrange(6)] for _ in range(288)]
            q.check(draws==log['history_ids']==[r['history_uid'] for r in log['rows']],'history_schedule',point)
            key='references' if a=='LOGIT0.1' else 'tables'
            origin='reference_origins' if a=='LOGIT0.1' else 'origins'
            q.check(all(before[k]==after[k] for k in ('members','seen',key,origin)), 'within_stage_memory_stable',point)
            prev=m['points'][f'{o}_{a}_stage{s-1}']['memory_summary']
            q.check(all(prev[k]==before[k] for k in ('members','seen',key,origin)), 'prior_memory_summary_links',point)
            saved=p['memory_summary']
            q.check(all(after[key][u]==saved[key][u] for u in set(after[key])&set(saved[key])), 'surviving_target_hashes',point)
            count_key='draw_count' if a=='LOGIT0.1' else 'count'
            q.check(before[count_key]==0 and after[count_key]==288,'history_draw_count',point)
        members=[];seen=0;rng=random.Random(derived_seed(20260930,o,'retention'))
        for d in o[:min(s,2)]:
            for uid in sorted(u for u,domain in domain_of.items() if domain==d):
                seen+=1;slot=len(members) if len(members)<6 else rng.randrange(seen)
                if slot<6:
                    if slot==len(members):members.append(uid)
                    else:members[slot]=uid
        mem=p['memory_summary'];bkey='serialized_bytes' if a=='LOGIT0.1' else 'bytes';okey='reference_origins' if a=='LOGIT0.1' else 'origins'
        q.check(mem['members']==members and mem['seen']==seen and mem[bkey]==p['memory']['bytes']<=1048576 and mem['sha256']==p['memory']['sha256'], 'memory_summary_budget_and_retention',point)
        q.check(all(domain_of[u]==o[b-1] for u,b in mem[okey].items()), 'memory_birth_domain',point)
        key=(o,s)
        if key in retention:q.check(retention[key]==members,'three_arm_retention_equal',point)
        retention[key]=members
        for kind in ('memory','model'):
            rec=p[kind];declared=excluded['reports/job/run/'+rec['path']]
            q.check(all(declared[k]==rec[k] for k in ('bytes','sha256')), 'excluded_payload_declared_identity',point+'/'+kind)
        q.check(not p['full_retained'] and not (q.root/'job/run'/p['full_state']['path']).exists(), 'temporary_checkpoint_removed',point)
        worst=0.
        for step,row in enumerate(log['rows'],1):
            lr=1e-5*(step/29 if step<=29 else (288-step)/259)
            q.check(row['step']==step and row['encoder_lr']==lr and row['head_lr']==.001 and
                    math.isfinite(row['gradient_norm']) and row['gradient_norm']>0,'actual_update_row',point+'/'+str(step))
            if a=='LOGIT0.1':
                total=row['current_total']+(.1*row['history_total']+.5*row['logit_mse'] if s>1 else 0.)
                q.check(row['adam_step']==(s-1)*288+step,'logit_row_adam',point+'/'+str(step))
                if s>1:q.check(row['history_weight']==.1 and row['logit_weight']==.5,'logit_loss_weights',point)
            else:
                total=row['current']['supervised']
                if s>1:total+=.1*row['history']['supervised']+.5*row['history']['mse0']
                for role in ('current','history') if s>1 else ('current',):
                    q.check((row[role]['mse1'] is None) if a=='R0' else row[role]['mse1'] is not None, 'regroup_loss_branch',point+'/'+str(step))
                    q.check(row[role]['distillation']==row[role]['mse0'],'D0_only_teacher',point+'/'+str(step))
            error=abs(total-row['total']);worst=max(worst,error)
            q.check(np.isclose(total,row['total'],rtol=2e-6,atol=2e-6),'actual_loss_decomposition',point+'/'+str(step))
        losses.append({'point':point,'max_absolute_loss_rounding':worst})
        totals['updates']+=len(log['rows']);totals['gradient_groups']+=288 if s==1 else 576
        durations[a]+=log['elapsed_seconds']
        cal=[r['group_uid'] for r in part['calibration'] if r['domain']==o[s-1]]
        q.check(p['calibration_ids']==cal,'calibration_partition',point)
        mapping=q.read(q.verify('job/run',p['mapping']))
        q.check(mapping['status']=='PASS_CALIBRATION_FIT','saved_calibration_fit_status',point)
        scores={r:q.array('job/run',rec,(12 if r=='calibration' else 60,378),np.float32 if r in ('raw','calibration') else np.float64) for r,rec in p['scores'].items()}
        raw_scores[point]=scores['raw']
        for role in ('raw','stage-cal','first-cal'):
            q.check(np.array_equal((scores[role]>=0).sum(axis=1),
                                    np.array([v['tp']+v['fp'] for v in countsets[point,role]])),
                    'scores_predicted_positive_match_counts',point+'/'+role)
        for role,key in (('stage-cal','stage'),('first-cal','first')):
            coef=p['maps'][key]
            z=coef['a']*scores['raw'].astype(np.float64)+coef['b']
            q.check(coef['a']>0 and np.array_equal(z,scores[role]),'affine_score_exact',point+'/'+role)
            q.check(np.array_equal(np.argsort(scores['raw'],axis=1,kind='stable'),np.argsort(scores[role],axis=1,kind='stable')),'affine_rank_order_exact',point+'/'+role)
        q.check(p['maps']['first']==m['points'][f'{o}_{a}_stage1']['maps']['stage'],'first_map_constant',point)
        stage_rows.append({'point':point,'order':o,'arm':a,'stage':s,'updates':288,'gradient_groups':288 if s==1 else 576,
                           'loop_plus_diagnostics_seconds':log['elapsed_seconds'],'memory_bytes':mem[bkey],
                           'memory_domains':dict(Counter(domain_of[u] for u in members)),
                           'calibration_a':p['maps']['stage']['a'],'calibration_b':p['maps']['stage']['b']})
    for o in ORDERS:
        for s in (1,2,3):
            base=logs[f'{o}_LOGIT0.1_stage{s}']
            for a in ('R0','S'):
                log=logs[f'{o}_{a}_stage{s}'];q.check(all(base[k]==log[k] for k in ('current_ids','history_ids')),'three_arm_supply_equal',f'{o}/{s}/{a}')
    console=q.read('job.console.txt','text');events=[];other=[]
    for line in console.splitlines():
        if not line.strip():continue
        try:events.append(json.loads(line))
        except ValueError:other.append(line)
    expected_events=[(f'{o}_{a}_stage{s}',step) for o in ORDERS for a in ARMS for s in (1,2,3) for step in (1,*range(24,289,24))]
    q.check([(r['point'],r['completed']) for r in events]==expected_events,'console_sequence_351_events')
    q.check(q.read('qualification/formal_listener/first_update.json')==events[0],'first_update_listener_matches_console')
    q.check(all(x['elapsed_seconds']<y['elapsed_seconds'] for x,y in zip(events,events[1:])),'console_elapsed_monotonic')
    complete,blind=q.read('job/completion.json'),q.read('job/before_valid.json')
    for k in ('evaluation','overfitting'):q.verify('job',complete[k])
    q.verify('job',blind['manifest'])
    q.check(complete['access']==q.read('job/access.json')==dict(train=1,valid=1,heldout=0,owners=0),'final_access_account')
    q.check(blind['access']==dict(train=1,valid=0,heldout=0,owners=0) and blind['states']==27 and blind['epochs']==162 and blind['initials']==2,'pre_valid_account')
    q.check(totals['updates']==m['physical_updates']==complete['physical_updates']==7776 and totals['gradient_groups']==m['gradient_group_presentations']==12960,'actual_total_work')
    q.check(complete['overall_pass_gate'] is False,'no_new_overall_gate')
    res=q.read('job/resource.json')
    for k,limit in (('peak_cuda_reserved_bytes','maximum_cuda_reserved_bytes'),('peak_rss_bytes','maximum_rss_bytes'),('peak_output_bytes','maximum_output_bytes')):
        q.check(complete['budget'][k]==res[k]<=res['limits'][limit] and all(e[k]<=res[k] for e in events),'resource_bound_record',k)
    wrapper=q.read('job.wrapper.txt','text').splitlines()
    wall=(datetime.fromisoformat(wrapper[-1])-datetime.fromisoformat(wrapper[0])).total_seconds()
    q.check(wrapper[1]=='exit_code=0' and wall<259200 and abs(wall-res['elapsed_seconds'])<2,'wrapper_budget_timing')
    q.check(not (q.root/'job/failure.json').exists(),'no_packaged_failure')
    info={'physical_updates':totals['updates'],'gradient_group_presentations':totals['gradient_groups'],
          'stages':len(stage_rows),'console_update_events':len(events),'non_json_console_lines':len(other),
          'wrapper_wall_seconds':wall,'budget_record':res,'loop_plus_diagnostics_seconds_by_arm':dict(durations),
          'model_bytes_inventory':sum(x['bytes'] for x in excluded.values() if x['kind']=='models'),
          'memory_bytes_inventory':sum(x['bytes'] for x in excluded.values() if x['kind']=='memory'),
          'max_memory_bytes':max(r['memory_bytes'] for r in stage_rows),
          'restore_verification_basis':'Source and recorded full_restore_verified plus saved-score consistency; no weights/Memory available here.'}
    write_json(q.out/'execution_details.json',{'summary':info,'stages':stage_rows,'loss_rounding':losses})
    return part,logs,raw_scores,info


def independent_endpoints(q,col,ev,matrices,rows):
    cols=col['metric_columns']
    # Design coefficients indexed by endpoint, order, chronological stage, actual domain.
    c=np.zeros((8,3,3,3),dtype=np.float64)
    for oi,o in enumerate(ORDERS):
        f,n,z=map('ABC'.index,o)
        c[0,oi,2,[f,n]]=.5                       # O
        c[1,oi,1,n]=.5;c[1,oi,2,z]=.5           # N
        c[2,oi,2,z]=1.                          # Z
        c[3,oi,0,f]=1.;c[3,oi,2,f]=-1.          # F_first
        c[4,oi,0,f]=.5;c[4,oi,2,f]=-.5
        c[4,oi,1,n]=.5;c[4,oi,2,n]=-.5         # F
        c[5,oi,1,n]=.5;c[5,oi,0,n]=-.5
        c[5,oi,2,z]=.5;c[5,oi,1,z]=-.5         # G
        c[6,oi,2,:]=1/3                        # final_all
        c[7,oi,1,n]=1.                         # A2
    draws=np.random.Generator(np.random.PCG64(20260930)).integers(0,20,size=(5000,3,20))
    freq=np.zeros((5000,3,20),dtype=np.float64)
    for d in range(3):
        for g in range(20):freq[:,d,g]=(draws[:,d]==g).sum(axis=1)/20
    weights=freq.reshape(5000,60)
    buffers={}
    for arm in ARMS:
        buffers[arm]={}
        for role in ROLES:
            x=np.empty((3,3,3,20,22))
            for oi,o in enumerate(ORDERS):
                for si in range(3):
                    a=matrices[f'{o}_{arm}_stage{si+1}'][role]
                    for di,ix in enumerate(rows):x[oi,si,di]=a[ix]
            b=np.einsum('eosd,osdgk->eodgk',c,x)
            for ei in (3,4,5):
                for metric in ('brier','log_loss'):b[ei,...,cols.index(metric)]*=-1
            buffers[arm][role]=b
    records=[];output={'absolute':{},'difference':{}}
    def summarize(table):
        per_order=table.sum(axis=1).mean(axis=1)
        group_table=table.mean(axis=0).reshape(60,22)
        means=per_order.mean(axis=0)
        boot=weights@group_table
        ci=np.quantile(boot,[.025,.975],axis=0,method='linear')
        return means,per_order,ci
    def compare(kind,name,role,tables,saved):
        out=output[kind].setdefault(name,{}).setdefault(role,{})
        for ei,ep in enumerate(ENDPOINTS):
            mean,orders,ci=summarize(tables[ei]);out[ep]={}
            for k,metric in enumerate(cols):
                v=saved[ep][metric]
                q.numbers(np.r_[mean[k],orders[:,k],ci[:,k]],
                          np.r_[v['mean'],[v['per_order'][o] for o in ORDERS],v['conditional_95pct_interval']],
                          'endpoints_25344',f'{name}/{role}/{ep}/{metric}')
                out[ep][metric]={'mean':float(mean[k]),'per_order':dict(zip(ORDERS,orders[:,k].tolist())),
                                 'conditional_95pct_interval':ci[:,k].tolist()}
                records.append(dict(type=kind,method_or_comparison=name,role=role,endpoint=ep,metric=metric,
                                    mean=float(mean[k]),lo95=float(ci[0,k]),hi95=float(ci[1,k]),
                                    **dict(zip(ORDERS,orders[:,k].tolist()))))
    for arm in ARMS:
        for role in ROLES:compare('absolute',arm,role,buffers[arm][role],ev['endpoints'][arm][role])
    for a,b in (('R0','LOGIT0.1'),('S','R0'),('S','LOGIT0.1')):
        for role in ROLES:compare('difference',a+'-'+b,role,buffers[a][role]-buffers[b][role],ev['comparisons'][a+'-'+b]['delta'][role])
    q.check(q.numeric['endpoints_25344']['scalars']==25344,'endpoints_scalar_count')
    write_json(q.out/'recomputed_endpoints.json',output)
    write_csv(q.out/'recomputed_endpoints.csv',records)
    saved=list(csv.DictReader(io.StringIO(q.read('analysis/all_endpoints.csv','text'))))
    key=lambda r:tuple(r[k] for k in ('type','method_or_comparison','role','endpoint','metric'))
    smap={key(r):r for r in saved}
    q.check(len(smap)==len(saved)==len(records)==4224,'reported_endpoint_csv_coverage')
    for r in records:
        q.numbers([r[k] for k in ('mean','lo95','hi95',*ORDERS)],
                  [float(smap[key(r)][k]) for k in ('mean','lo95','hi95',*ORDERS)],'reported_endpoint_csv',str(key(r)))
    stage_rows=[]
    for point,roles in matrices.items():
        for role,a in roles.items():
            if role=='primary':continue
            for d,ix in zip('ABC',rows):stage_rows.append(dict(point=point,role=role,actual_domain=d,**dict(zip(cols,a[ix].mean(axis=0).tolist()))))
    write_csv(q.out/'recomputed_stage_domain.csv',stage_rows)
    stage_saved=list(csv.DictReader(io.StringIO(q.read('analysis/all_stages.csv','text'))))
    sk=lambda r:(r['point'],r['role'],r['actual_domain'])
    ss={sk(r):r for r in stage_saved}
    q.check(len(ss)==len(stage_saved)==len(stage_rows)==243,'reported_stage_csv_coverage')
    for r in stage_rows:q.numbers([r[k] for k in cols],[float(ss[sk(r)][k]) for k in cols],'reported_stage_csv',str(sk(r)))
    return output


def diagnostic_reference(train,valid,seed):
    nt,nv=train.shape[1],valid.shape[1]
    if not nt:return {'status':'NOT_APPLICABLE_NO_SURVIVING_GROUPS'}
    rng=np.random.Generator(np.random.PCG64(seed))
    td=rng.integers(nt,size=(5000,nt));vd=rng.integers(nv,size=(5000,nv))
    # Frequency weights keep each group together across all six epochs.
    tw=np.stack([(td==i).sum(axis=1)/nt for i in range(nt)],axis=1)
    vw=np.stack([(vd==i).sum(axis=1)/nv for i in range(nv)],axis=1)
    tb=np.einsum('bg,egk->ebk',tw,train);vb=np.einsum('bg,egk->ebk',vw,valid)
    signs=np.array([1.,1.,-1.]);tm,vm=train.mean(1),valid.mean(1)
    gaps=(tm-vm)*signs;gb=(tb-vb)*signs
    # The frozen point rule reduces within-group changes before testing the sign.
    # Preserve that reduction: subtracting epoch means can flip rounding at zero.
    tchange=(train[-1]-train[0])*signs;vchange=(valid[-1]-valid[0])*signs
    tmean=tchange.mean(0);vmean=vchange.mean(0)
    tc=np.quantile(tw@tchange,[.025,.975],axis=0).T
    vc=np.quantile(vw@vchange,[.025,.975],axis=0).T
    gc=np.quantile(gb[-1]-gb[0],[.025,.975],axis=0).T
    decisions={}
    for k,metric in enumerate(DCOLS):
        supported=tc[k,0]>0 and vc[k,1]<0 and gc[k,0]>0
        point=tmean[k]>0 and vmean[k]<0 and gaps[-1,k]>gaps[0,k]
        decisions[metric]='有相应迹象' if supported else '证据不足' if point else '未见明确迹象'
    return {'fit_or_cache_groups':nt,'valid_groups':nv,'train_raw_curve':tm.tolist(),'valid_raw_curve':vm.tolist(),
            'benefit_gap_curve':gaps.tolist(),'benefit_gap_conditional_intervals':np.quantile(gb,[.025,.975],axis=1).transpose(1,2,0).tolist(),
            'epoch1_to_6_train_benefit':{'mean':tmean.tolist(),'conditional_95pct_interval':tc.tolist()},
            'epoch1_to_6_valid_benefit':{'mean':vmean.tolist(),'conditional_95pct_interval':vc.tolist()},
            'epoch1_to_6_gap_widening':{'mean':(gaps[-1]-gaps[0]).tolist(),'conditional_95pct_interval':gc.tolist()},
            'interpretation':decisions}


def verify_diagnostics(q,m,col,part,logs,raw_scores,matrices):
    report=q.read('job/evaluation/diagnostics/overfitting.json')
    counts={a:Counter() for a in ARMS};out={};curve_rows=[];entries_count=0;na=[]
    source_curves=q.read('analysis/curves.json')
    curve_map={(r['point'],r['role'],r['domain'],r['metric']):r for r in source_curves}
    for point,refs in m['diagnostics'].items():
        entries=[q.read(q.verify('job/run/diagnostics',r)) for r in refs]
        p=m['points'][point];o,a,s=p['order'],p['arm'],p['stage'];log=logs[point]
        q.check([e['epoch'] for e in entries]==list(range(1,7)),'diagnostic_epoch_coverage',point)
        fit=[r['group_uid'] for r in part['fit'] if r['domain']==o[s-1]]
        cache=[] if s==1 else log['memory_before']['members']
        curves={'fit':[],'cache':[],'valid':[]}
        for epoch,e in enumerate(entries,1):
            q.check(e['arm']==a and e['order']==o and e['stage']==s and e['stage_step']==48*epoch and
                    e['adam_step']==(s-1)*288+48*epoch and e['seed']=='s0' and not e['valid_labels_used'] and
                    e['mode']=='A0_eval_no_autograd','diagnostic_identity_step',point+'/'+str(epoch))
            q.check(e['fit_ids']==fit and e['cache_ids']==cache and e['valid_ids']==col['group_ids'] and
                    all(e[k]==entries[0][k] for k in ('fit_ids','fit_domains','cache_ids','cache_domains','valid_ids')), 'diagnostic_population_fixed',point+'/'+str(epoch))
            for role in ('fit','cache'):
                curves[role].append(q.array('job/run/diagnostics',e[role],(len(e[role+'_ids']),3),np.float64))
            raw=q.array('job/run/diagnostics',e['valid_blind'],(60,378),np.float32)
            values=q.array('job/evaluation/diagnostics',report['group_metrics'][f'{point}_epoch{epoch}'],(60,3),np.float64)
            curves['valid'].append(values)
            if epoch==6:
                q.check(np.array_equal(raw,raw_scores[point]),'epoch6_stage_score_identity',point)
                q.check(np.array_equal(values,matrices[point]['raw'][:,[col['metric_columns'].index(k) for k in DCOLS]]),'epoch6_stage_metric_identity',point)
            q.check([x['uid'] for x in e['teacher_errors']]==cache,'diagnostic_teacher_members',point)
            for er in e['teacher_errors']:
                if a=='LOGIT0.1':q.check(er['D1'] is None and er['D0']>=0,'logit_D1_not_applicable',point)
                else:
                    q.numbers(er['D0']+er['D1']-2*er['residual_cross_mean'],er['residual_difference_mse'],'residual_algebra',point)
            entries_count+=1
        curves={k:np.stack(v) for k,v in curves.items()}
        out[point]={}
        for role in ('fit','cache'):
            ds=[o[s-1]] if role=='fit' else list(o[:s-1])
            if not ds:
                q.check(report['points'][point][role]=={'status':'NOT_APPLICABLE_FIRST_STAGE'},'first_stage_cache_NA',point)
                na.append({'point':point,'role':role,'reason':'first_stage'});continue
            out[point][role]={}
            for d in ds:
                ti=np.flatnonzero(np.asarray(entries[0][role+'_domains'])==d)
                vi=np.flatnonzero(np.asarray(col['domains'])==d)
                ref=diagnostic_reference(curves[role][:,ti],curves['valid'][:,vi],derived_seed(20260930,o,s,role,d,'diagnostics'))
                saved=report['points'][point][role][d]
                q.recursive_numbers(ref,saved,'diagnostic_curves_and_intervals',point+'/'+role+'/'+d)
                out[point][role][d]=ref
                if not len(ti):
                    na.append({'point':point,'role':role,'domain':d,'reason':'no_surviving_groups'});continue
                for k,metric in enumerate(DCOLS):
                    label=ref['interpretation'][metric];counts[a][role+'/'+metric+'/'+label]+=1
                    old=curve_map[point,role,d,metric]
                    q.numbers([v[k] for v in ref['train_raw_curve']],old['train_curve'],'reported_curve_export',point+'/'+role+'/'+d+'/'+metric+'/train')
                    q.numbers([v[k] for v in ref['valid_raw_curve']],old['valid_curve'],'reported_curve_export',point+'/'+role+'/'+d+'/'+metric+'/valid')
                    row={'point':point,'arm':a,'role':role,'domain':d,'metric':metric,'train_groups':len(ti),'valid_groups':len(vi),'interpretation':label}
                    for i in range(6):row['train_epoch'+str(i+1)]=ref['train_raw_curve'][i][k]
                    for i in range(6):row['valid_epoch'+str(i+1)]=ref['valid_raw_curve'][i][k]
                    for key in ('epoch1_to_6_train_benefit','epoch1_to_6_valid_benefit','epoch1_to_6_gap_widening'):
                        mean,ci=ref[key]['mean'][k],ref[key]['conditional_95pct_interval'][k]
                        q.numbers([mean,*ci],[old[key]['mean'],*old[key]['conditional_95pct_interval']],'reported_curve_export',point+'/'+role+'/'+d+'/'+metric+'/'+key)
                        row[key+'_mean']=mean;row[key+'_lo95']=ci[0];row[key+'_hi95']=ci[1]
                    q.check(old['interpretation']==label,'reported_curve_interpretation',point+'/'+role+'/'+d+'/'+metric)
                    curve_rows.append(row)
    q.check(entries_count==162 and len(curve_rows)==153 and len(curve_map)==153,'diagnostic_result_coverage')
    q.check({a:dict(c) for a,c in counts.items()}==q.read('analysis/summary.json')['overfit_counts'],'diagnostic_summary_counts')
    result={'epoch_records':entries_count,'eligible_panels':len(curve_rows)//3,'metric_curves':len(curve_rows),
            'not_applicable':na,'interpretation_counts':{a:dict(c) for a,c in counts.items()}}
    write_json(q.out/'recomputed_diagnostics.json',{'summary':result,'points':out})
    write_csv(q.out/'recomputed_curves.csv',curve_rows)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--archive',type=Path)
    args=parser.parse_args();started=time.monotonic();q=Review(args.root,args.out)
    identity=verify_integrity(q,args.archive)
    m=q.read('job/run/manifest.json');col=q.read('job/evaluation/collected.json');ev=q.read('job/evaluation/evaluation.json')
    sources=verify_sources(q,m,col,ev)
    matrices,initial,rows,countsets=verify_metrics(q,m,col,ev)
    part,logs,raw,execution=verify_execution(q,m,col,matrices,countsets)
    endpoints=independent_endpoints(q,col,ev,matrices,rows)
    diagnostic=verify_diagnostics(q,m,col,part,logs,raw,matrices)
    summary={'status':'CONSISTENT_SAVED_EVIDENCE' if not q.failures else 'MISMATCHES_TO_REVIEW',
             'scope':'This upload only. Recorded outputs, no project imports, models, labels, Memory payloads, network, Torch or GPU.',
             'package':identity,'sources':sources,'execution':execution,
             'matrices':{'sets':len(countsets),'shape':[60,22],'matrix_values':len(countsets)*60*22,
                         'count_rows':len(countsets)*60,'rank_columns':14,'probability_and_threshold_columns':8},
             'numerical_comparisons':q.numeric,'diagnostics':diagnostic,
             'check_counts':dict(q.check_counts),'failure_count':len(q.failures),'failures':q.failures,
             'environment':{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,
                            'project_linux_connected':False,'torch_imported':'torch' in sys.modules,'gpu_used':False,
                            'thread_env':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},
                            'elapsed_seconds':time.monotonic()-started,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}}
    write_json(q.out/'verification_summary.json',summary)
    rows=[{**v,'programmatic_read_modes':';'.join(sorted(q.reads[name]))} for name,v in sorted(q.hashes.items())]
    write_csv(q.out/'input_file_inventory.csv',rows)
    print(json.dumps({'status':summary['status'],'failure_count':len(q.failures),
                      'numeric':q.numeric,'environment':summary['environment']},ensure_ascii=False,indent=2))
    if q.failures:raise SystemExit(1)


if __name__=='__main__':main()
