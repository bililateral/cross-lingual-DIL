"""Mechanical checks and arithmetic of SAVED metrics; no model/label/prediction metric computation."""
import json, math, hashlib, time
from pathlib import Path
R=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
mean=lambda xs:math.fsum(xs)/len(xs)
def avg(rows):return [mean([r[i] for r in rows]) for i in range(10)]
def close(a,b):assert len(a)==len(b) and all(abs(x-y)<1e-10 for x,y in zip(a,b)),(a,b)
def write(name,value):(R/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
start=time.monotonic()
result=read(R/'formal/analysis/result.json')
gate=read(R/'inputs/chain/vendorlink/formal_gate.json')
blind=read(R/'formal/scoring/all_blind_complete.json')
dep=read(R/'inputs/dependencies.json')
done=read(R/'formal/completion.json')
manifest=read(R/'receive_manifest.json')
receipt=read(R/'receive.json')
assert receipt['status']=='ALL_REQUESTED_FILES_SIZE_SHA_MATCH'
assert result['status']=='VENDORLINK_FIXED_ANALYSIS_COMPLETE' and result['passed'] is None
assert result['primary_status']=='DESCRIPTIVE_FIXED_DATA_ONLY'
metrics=result['columns']; methods=['VendorLink_SEQ','H_SEQ','H_ER','H_full']
domains=dict(A='AlphaBay',B='Dream',C='Agora')
cells={}; per_run={}; work={m:{} for m in methods}; update_count=0; log_history={}; score_groups=0; query_rows=0
for rec in manifest['files']:
 p=R/rec['local'];raw=p.read_bytes()
 assert len(raw)==rec['bytes'] and hashlib.sha256(raw).hexdigest()==rec['sha256'],rec['local']
for spec in gate['runs']:
 rid=spec['id'];cells[rid]={}
 for key,cell in blind['runs'][rid].items():
  scores_path=R/'formal/scoring'/Path(cell['scores']['path']).name
  raw=scores_path.read_bytes()
  assert len(raw)==cell['scores']['bytes'] and hashlib.sha256(raw).hexdigest()==cell['scores']['sha256']
  scores=json.loads(raw); q=read(R/'formal/analysis'/f'{rid}_{key}_queries.json')
  assert q['columns']==metrics and set(scores)==set(q['queries'])
  rows=[]
  for uid,vals in q['queries'].items():
   n=dep['query_counts'][uid];sv=scores[uid]
   assert len(vals)==len(sv['view_ids'])==n and len(sv['scores'])==n*(n-1)//2
   assert len(set(sv['view_ids']))==n
   assert all(type(x) in (int,float) and math.isfinite(x) for x in sv['scores'])
   for v in vals:
    assert len(v)==10 and all(math.isfinite(x) and 0<=x<=1 for x in v)
    assert v[0]==v[1] and v[2]==v[6]
   rows.extend(vals);score_groups+=1;query_rows+=len(vals)
  cells[rid][key]=dict(domain=cell['domain'],queries=len(rows),metrics=avg(rows))
 v=lambda key:cells[rid][key]['metrics']
 fields=dict(Q=avg([v('1_1'),v('2_2')]),O=avg([v('3_1'),v('3_2')]),N=v('3_3'))
 fields['F']=[x-y for x,y in zip(fields['Q'],fields['O'])]
 for f in ('Q','O','F','N'):close(fields[f],result['per_run'][rid][f])
 fields['final']=avg([v('3_1'),v('3_2'),v('3_3')]);per_run[rid]=fields
 w=read(R/'formal'/rid/'run/workload.json')
 c=read(R/'formal'/rid/'run/completion.json')
 assert w==c['workload'] and w['complete_from_start'] and c['updates']==10280
 counts={k:0 for k in ('current_groups','current_ads','history_groups','history_ads')};n=0;hist=0;distill_nonzero=0
 for line in (R/'formal'/rid/'run/updates.jsonl').read_text().splitlines():
  row=json.loads(line);n+=1
  assert row['total_updates']==n and row['optimizer_updates']==1
  assert math.isfinite(row['gradient_norm'])
  for k in counts:counts[k]+=row['workload'][k]
  if 'history' in row:
   hist+=1
   distill_nonzero+=row['history']['distillation']>0
 for k,x in counts.items():assert x==w['totals'][k],(rid,k)
 assert n==10280
 if spec['method'] in ('VendorLink_SEQ','H_SEQ'):assert hist==0
 if spec['method']=='H_ER':assert hist==w['totals']['history_groups'] and distill_nonzero==0
 if spec['method']=='H_full':assert hist==w['totals']['history_groups'] and distill_nonzero>0
 log_history[rid]=dict(updates=n,history_updates=hist,nonzero_distillation_updates=distill_nonzero)
 update_count+=n
 for k,vv in w['totals'].items():work[spec['method']][k]=work[spec['method']].get(k,0)+vv
absolute={}
for m in methods:
 ids=[s['id'] for s in gate['runs'] if s['method']==m]
 absolute[m]={f:avg([per_run[rid][f] for rid in ids]) for f in ('Q','O','F','N','final')}
 for f in ('Q','O','F','N'):close(absolute[m][f],result['absolute'][m][f])
for other in methods[:-1]:
 for f in ('Q','O','F','N'):
  saved=result['comparisons']['H_full-'+other][f]
  for i,k in enumerate(metrics):
   assert saved[k]['conditional_95pct_interval'] is None
   assert abs(saved[k]['mean']-(absolute['H_full'][f][i]-absolute[other][f][i]))<1e-10
by_order={};by_seed={};by_market={}
for label,values,target in [('order',('ABC','BCA','CAB'),by_order),('seed',(4,5,6),by_seed)]:
 for value in values:
  target[str(value)]={m:{f:avg([per_run[s['id']][f] for s in gate['runs'] if s['method']==m and s[label]==value]) for f in ('Q','O','F','N','final')} for m in methods}
for domain in domains.values():
 by_market[domain]={}
 for m in methods:
  groups={f:[] for f in ('final','Q','O','F','N')}
  for s in gate['runs']:
   if s['method']!=m:continue
   pos=[domains[x] for x in s['order']].index(domain)+1
   final=cells[s['id']][f'3_{pos}']['metrics'];groups['final'].append(final)
   if pos<3:
    initial=cells[s['id']][f'{pos}_{pos}']['metrics']
    groups['Q'].append(initial);groups['O'].append(final);groups['F'].append([a-b for a,b in zip(initial,final)])
   else:groups['N'].append(final)
  by_market[domain][m]={f:avg(v) for f,v in groups.items()}
paired={}
for other in methods[:-1]:
 vals=[]
 for order in ('ABC','BCA','CAB'):
  for seed in (4,5,6):
   a=per_run[f'{order}_s{seed}_H_full'];b=per_run[f'{order}_s{seed}_{other}']
   vals.append(dict(order=order,seed=seed,**{f:a[f][0]-b[f][0] for f in ('Q','O','F','N','final')}))
 paired[other]=vals
cost={m:math.fsum(x['actual_seconds'] for x in done['completed'] if x['bucket']=='training' and next(s['method'] for s in gate['runs'] if s['id']==x['id'])==m) for m in methods}
summary=dict(columns=metrics,absolute=absolute,by_order=by_order,by_seed=by_seed,by_market=by_market,per_run=per_run,cells=cells,paired_map=paired,workload=work,training_seconds=cost)
write('saved_summary.json',summary)
check=dict(status='SAVED_FIELDS_AND_ARITHMETIC_MATCH',files=receipt['count'],bytes=receipt['bytes'],runs=36,cells=len(cells)*6,group_metric_blocks=score_groups,query_metric_rows=query_rows,total_optimizer_updates=update_count,source_count=65,history_log_checks=log_history,wall_seconds=time.monotonic()-start,scope='Windows standard-library metadata/SHA/count/finite checks and arithmetic of already-saved metrics; no labels, no metric recomputation from scores, no model loading or resampling')
write('saved_check.json',check)
lines=['VendorLink完整排序指标与分层表','2026-11-04；直接来自已保存query指标及result字段的简单算术汇总。','每市场query等权，九条轨迹等权；final=(2O+N)/3。Q初学，O旧市场最终，F=Q-O（越低越好，负值为后续改善），N最后市场。','ABC=AlphaBay→Dream→Agora；BCA=Dream→Agora→AlphaBay；CAB=Agora→AlphaBay→Dream。','MAP=MRR，因为每个query仅一个正例。本任务不报告分类/概率指标。所有区间和统计验收均未计算。','']
def table(title,data):
 lines.append(title);lines.append('指标\t'+'\t'.join(methods))
 for i,k in enumerate(metrics):lines.append(k+'\t'+'\t'.join(f'{data[m][i]:.9f}' for m in methods))
 lines.append('')
for f in ('final','Q','O','F','N'):table(f'全局 {f}',{m:absolute[m][f] for m in methods})
for kind,target in [('市场',by_market),('顺序',by_order),('种子',by_seed)]:
 for key,value in target.items():
  for f in ('final','Q','O','F','N'):table(f'{kind} {key} / {f}',{m:value[m][f] for m in methods})
lines.append('逐条Q/O/F/N/final（全部十指标）');lines.append('run\t角色\t'+'\t'.join(metrics))
for rid,fields in per_run.items():
 for f,v in fields.items():lines.append(rid+'\t'+f+'\t'+'\t'.join(f'{x:.9f}' for x in v))
lines+=['','全部216评分格（全部十指标）','run\t格\t市场\tquery数\t'+'\t'.join(metrics)]
for rid,rows in cells.items():
 for key,cell in rows.items():lines.append(f'{rid}\t{key}\t{cell["domain"]}\t{cell["queries"]}\t'+'\t'.join(f'{x:.9f}' for x in cell['metrics']))
(R/'tables.zh.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(dict(absolute=absolute,by_order_map={o:{m:{f:v[0] for f,v in fs.items()} for m,fs in rows.items()} for o,rows in by_order.items()},by_market_map={o:{m:{f:v[0] for f,v in fs.items()} for m,fs in rows.items()} for o,rows in by_market.items()},paired_map=paired,workload=work,training_seconds=cost,check={k:v for k,v in check.items() if k!='history_log_checks'}),ensure_ascii=False,indent=2))
