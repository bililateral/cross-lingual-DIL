"""Read saved Registered results; no model, raw text, qrels or metric regeneration."""
from pathlib import Path
from collections import Counter
import datetime as dt
import hashlib
import json
import math
import statistics as st
import sys
import time
import zipfile

START = time.monotonic()
assert sys.platform == 'linux'
assert Path(sys.executable).resolve() == Path('/home/yongpeng/miniconda3/envs/py310/bin/python').resolve()
ROOT = Path('/home/yongpeng/cross-lingual/reports/documentation/20261108/evolution_registered')
EX = ROOT/'execution'
C = EX/'campaign'
OUT = ROOT/'result'
OUT.mkdir(exist_ok=True)
M = ['map', 'mrr', 'recall@1', 'recall@3', 'recall@5', 'recall@10',
     'ndcg@1', 'ndcg@3', 'ndcg@5', 'ndcg@10']
CELLS = [(0,1,1),(1,1,1),(1,2,2),(2,1,1),(2,2,2),(2,3,3),(3,1,1),(3,2,2),(3,3,3),
         (2,1,2),(3,1,3),(3,2,3),(1,1,3),(1,2,3),(1,3,3)]

def read(p):
    return json.loads(p.read_text())

def write(p, v):
    p.write_text(json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False)+'\n')

def record(p):
    data = p.read_bytes()
    return dict(path=str(p), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())

def pinned(row):
    p = Path(row['path'])
    assert record(p) == row, str(p)
    return read(p)

def stamp(t):
    return dt.datetime.fromtimestamp(t, dt.timezone(dt.timedelta(hours=8))).isoformat()

complete = read(C/'completion.json')
result = pinned(complete['result'])
batch = read(C/'batch.json')
gate = pinned(batch['gate'])
pinned(batch['manifest'])
for s in gate['sources']:
    assert record(Path(s['path'])) == s
for key in ('scope', 'authorization', 'review_disposition'):
    assert record(Path(gate[key]['path'])) == gate[key]
assert len(gate['sources']) == 14
assert not (C/'failure.json').exists()
dispatch = read(C/'dispatch.json')
assert dispatch['status'] == 'COMPLETE' and not dispatch['active'] and not dispatch['failures']
assert not complete['automatic_next_phase'] and complete['training_trajectories'] == 6
assert complete['scoring_trajectories'] == 6 and len(batch['completed']) == 6
assert result['matrices'] == 90 and result['logical_scores'] == 1824768
rows = {r['task']['id']: r for r in result['rows']}
assert set(rows) == {f'P_{system}_{recipe}_s{s}' for system,recipe in [('H_SEQ','H1'),('V_SEQ','V1')] for s in (31,32,33)}
score_records = {v['task_id']: v['completion'] for v in read(C/'scoring_complete.json')}
assert set(score_records) == set(rows)
raw = [C/n for n in ('completion.json','dispatch.json','budget.json','batch.json','scoring_complete.json')]
raw += [C/'analysis/result.json', C/'analysis/dispatch_receipt.json',
        C/'analysis/access/complete_P_labels.json']
runtime, sources, frozen, growth = [], [], [], []
source_counts = Counter()
for entry in batch['completed']:
    task_id = entry['task_id']
    train = pinned(entry['completion'])
    assert train['identity'] == entry['identity'] and train['task'] == rows[task_id]['task']
    assert train['updates'] == 1440 and train['workload']['current_updates'] == 1440
    assert train['workload']['stage_model_saves'] == 3
    assert all(train['workload'][k] == 0 for k in ('history_groups','history_records','teacher_groups','teacher_records'))
    assert set(train['endpoints']) == {'0','1','2','3'}
    for endpoint in train['endpoints'].values():
        p=Path(endpoint['path'])
        assert p.is_file() and p.stat().st_size == endpoint['bytes']
    updates = C/'training'/task_id/'run/updates.jsonl'
    counts = Counter()
    with updates.open() as f:
        for i, line in enumerate(f, 1):
            u = json.loads(line)
            assert u['total_updates'] == i and u['optimizer_updates'] == 1
            assert u['history_group'] is None and math.isfinite(u['gradient_norm'])
            counts[(u['stage'], u['epoch'])] += 1
    assert i == 1440 and counts == Counter({(t,e):12 for t in (1,2,3) for e in range(1,41)})
    sr = pinned(score_records[task_id])
    assert sr['task'] == train['task'] and sr['endpoint_identity'] == train['identity']
    assert sr['manifest_identity'] == batch['manifest']
    assert set(sr['cells']) == {'_'.join(map(str,c)) for c in CELLS}
    for cell in sr['cells'].values():
        assert record(Path(cell['path'])) == cell
    metrics_path = C/'analysis'/(task_id+'_metrics.json')
    metrics = read(metrics_path)
    assert metrics['task'] == train['task'] and set(metrics['cells']) == set(sr['cells'])
    for key, value in metrics['cells'].items():
        assert len(value['query_ids']) == 96 and len(set(value['query_ids'])) == 96
        assert len(value['ranks']) == 96 and set(value['mean']) == set(M)
        assert value['mean'] == rows[task_id]['cells'][key]['mean']
        assert value['per_query']['map'] == value['per_query']['mrr']
        for name in M:
            vals = value['per_query'][name]
            assert len(vals)==96 and all(math.isfinite(x) and 0 <= x <= 1 for x in vals)
            assert abs(st.mean(vals)-value['mean'][name]) < 1e-12
    sources.extend([record(updates),record(metrics_path)])
    source_counts.update(updates=1440, matrices=15, logical_scores=304128, query_metric_vectors=1440)
    receipts = {}
    for phase in ('training','scoring'):
        path=C/phase/task_id/'dispatch_receipt.json'
        receipt=read(path)
        assert receipt['success'] and receipt['released'] and receipt['error'] is None
        assert receipt['peak_gpu_bytes'] <= 14*1024**3
        receipts[phase]=receipt
        raw.append(path)
    runtime.append(dict(task=task_id, **receipts))
    raw += [Path(entry['completion']['path']),Path(score_records[task_id]['path'])]

systems={}
for system in ('H_SEQ','V_SEQ'):
    rr=sorted([r for r in rows.values() if r['task']['system']==system], key=lambda r:r['task']['seed'])
    def metric(r,t,d,p,m='map'):
        return r['cells'][f'{t}_{d}_{p}']['mean'][m]
    q=[[metric(r,d,d,d) for d in (1,2,3)] for r in rr]
    r1=[[metric(r,d,d,d,'recall@1') for d in (1,2,3)] for r in rr]
    g=[[metric(r,d,d,d)-metric(r,d-1,d,d) for d in (1,2,3)] for r in rr]
    o=[[metric(r,3,d,d) for d in (1,2)] for r in rr]
    f=[[q[i][d]-o[i][d] for d in (0,1)] for i in range(3)]
    avg=lambda a:[st.mean(c) for c in zip(*a)]
    gates=dict(initial_mean_map=all(x>=.60 for x in avg(q)),
        initial_mean_r1=all(x>=.45 for x in avg(r1)), initial_each_map=all(x>=.55 for a in q for x in a),
        learning_mean=all(x>=.01 for x in avg(g)),learning_each=all(x>=0 for a in g for x in a),
        forgetting_mean=st.mean(x for a in f for x in a)>=.05,
        forgetting_each_batch=all(x>=.03 for x in avg(f)),
        forgetting_each_seed=all(st.mean(a)>0 for a in f),
        forgetting_two_seeds=sum(st.mean(a)>=.04 for a in f)>=2)
    saved=result['decision']['systems'][system]
    assert gates == saved['gates']
    for key,value in [('Q',q),('O',o),('G',g),('F',f)]:
        assert all(abs(a-b)<1e-12 for aa,bb in zip(value,saved[key]) for a,b in zip(aa,bb))
    systems[system]=dict(Q_mean=avg(q),R1_mean=avg(r1),G_mean=avg(g),O_mean=avg(o),F_mean=avg(f),
        F_overall=st.mean(x for a in f for x in a),F_seed=[st.mean(a) for a in f],gates=gates,
        final_fixed_map=[st.mean(metric(r,3,d,d) for r in rr) for d in (1,2,3)],
        final_growing_map=[st.mean(metric(r,3,d,3) for r in rr) for d in (1,2,3)],
        final_frozen_map=[st.mean(metric(r,1,d,3) for r in rr) for d in (1,2,3)])
    for r in rr:
        for kind,d,t,p in [('birth',2,2,2),('birth',3,3,3),('final',1,3,3),('final',2,3,3),('final',3,3,3)]:
            frozen.append(dict(system=system,seed=r['task']['seed'],kind=kind,batch=d,
                delta={m:metric(r,t,d,p,m)-metric(r,1,d,p,m) for m in M}))
        for t,d,p in [(2,1,2),(3,1,3),(3,2,3)]:
            growth.append(dict(system=system,seed=r['task']['seed'],t=t,batch=d,pool=p,
                delta={m:metric(r,t,d,p,m)-metric(r,t,d,d,m) for m in M}))
for collection, fields in [(frozen,('kind','batch')),(growth,('t','batch','pool'))]:
    for system in ('H_SEQ','V_SEQ'):
        keys=sorted({tuple(r[f] for f in fields) for r in collection if r['system']==system})
        for key in keys:
            selected=[r for r in collection if r['system']==system and tuple(r[f] for f in fields)==key and r['seed']!='mean']
            collection.append(dict(system=system,seed='mean',**dict(zip(fields,key)),
                                   delta={m:st.mean(r['delta'][m] for r in selected) for m in M}))

assert not all(all(v for k,v in s['gates'].items() if not k.startswith('forgetting')) for s in systems.values())
assert result['decision']['status']==complete['status']=='STOP_LEARNING'
label=read(C/'analysis/access/complete_P_labels.json')
assert len(label['sources'])==3
assert max(Path(r['path']).stat().st_mtime for r in score_records.values()) <= label['attempted_at']
budget=read(C/'budget.json')
launch=read(EX/'launch.json')
end=(C/'completion.json').stat().st_mtime
files=[p for p in C.rglob('*') if p.is_file()]
weights=[p for p in files if p.suffix=='.pt']
summary=dict(status='STOP_LEARNING',systems=systems,frozen=frozen,gallery_growth=growth,
    workload=dict(source_counts),runtime=runtime,started=launch['at'],completed=stamp(end),
    wall_seconds=end-launch['created'],campaign_budget=budget['used_seconds'],
    campaign_bytes=sum(p.stat().st_size for p in files),weight_files=len(weights),weight_bytes=sum(p.stat().st_size for p in weights),
    label_attempt_at=stamp(label['attempted_at']),sources_verified=14,verified_score_files=90,
    saved_per_query_mean_checks=900,endpoint_check='Saved restore/SHA chain reused; endpoint identity and current file sizes checked; no load or rehash of 7+ GiB weights',
    raw_text_or_qrels_read=False,new_inference_or_bootstrap=False,external_review='Explicitly waived by user for this result analysis')
write(OUT/'summary.json',summary)
write(OUT/'saved_sources.json',sources)
lines=['# 已保存结果的完整十指标表','',
       '所有值均来自已保存结果。MAP=MRR来自单正例定义，不能算作两份独立证据。无重新评分、标签解析或bootstrap。',
       '表中绝对值为百分数，差值为百分点；均值为三个固定seed等权。t为模型阶段、d为query批、p为累计gallery批数，候选数=96p。','',
       '## 全部90格及三seed均值','', '| 方法 | seed | t,d,p | '+' | '.join(M)+' |',
       '|---|---|---|'+'---:|'*10]
for system in ('H_SEQ','V_SEQ'):
    rr=sorted([r for r in rows.values() if r['task']['system']==system],key=lambda r:r['task']['seed'])
    for seed in (31,32,33,'mean'):
        selected=rr if seed=='mean' else [r for r in rr if r['task']['seed']==seed]
        for cell in CELLS:
            key='_'.join(map(str,cell))
            vals=[st.mean(r['cells'][key]['mean'][m] for r in selected) for m in M]
            lines.append(f'| {system} | {seed} | {cell[0]},{cell[1]},{cell[2]} | '+' | '.join(f'{100*v:.4f}' for v in vals)+' |')
for title,collection,fields in [('更新相对冻结θ1',frozen,('kind','batch')),('同端点扩库影响',growth,('t','batch','pool'))]:
    lines += ['',f'## {title}','', '| 方法 | seed | '+ ' | '.join(fields)+' | '+' | '.join(M)+' |',
              '|---|---|'+'---|'*len(fields)+'---:|'*10]
    for row in collection:
        lines.append(f"| {row['system']} | {row['seed']} | "+' | '.join(str(row[k]) for k in fields)+' | '+
                     ' | '.join(f"{100*row['delta'][m]:+.4f}" for m in M)+' |')
(OUT/'tables.zh.md').write_text('\n'.join(lines)+'\n')
raw += [OUT/'summary.json',OUT/'saved_sources.json',OUT/'tables.zh.md']
elapsed=time.monotonic()-START
charge=math.ceil(elapsed)
assert budget['used_seconds']['auxiliary']+charge <= budget['caps']['auxiliary']
write(OUT/'analysis_receipt.json',dict(source=record(Path(__file__).resolve()),input=complete['result'],
    elapsed_seconds=elapsed,charged_auxiliary_seconds=charge,prior_campaign_budget=budget['used_seconds'],
    cumulative_after_saved_analysis={**budget['used_seconds'],'auxiliary':budget['used_seconds']['auxiliary']+charge},
    cpu_test_seconds=20,raw_qrels_reads=0,model_loads=0,new_metric_generation=0,
    frozen_raw_budget_unchanged=True,external_review_waived=True))
raw.append(OUT/'analysis_receipt.json')
members=[]
for p in raw:
    name=str(p.relative_to(ROOT))
    members.append(dict(member=name,**record(p)))
write(OUT/'manifest.json',dict(files=members,excluded='Real text/identities/qrels/per-query metrics/score matrices/models stay on Linux'))
with zipfile.ZipFile(OUT/'transfer.zip','x',zipfile.ZIP_DEFLATED) as z:
    for p in raw:
        z.write(p,str(p.relative_to(ROOT)))
    z.write(OUT/'manifest.json','result/manifest.json')
print(json.dumps({k:v for k,v in summary.items() if k not in ('frozen','gallery_growth','runtime')},ensure_ascii=False,indent=2))
print('transfer_bytes',(OUT/'transfer.zip').stat().st_size)
print(hashlib.sha256((OUT/'transfer.zip').read_bytes()).hexdigest())
