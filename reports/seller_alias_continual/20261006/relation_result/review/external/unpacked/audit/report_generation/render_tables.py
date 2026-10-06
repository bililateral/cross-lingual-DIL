#!/usr/bin/env python3
"""Render audit tables/CSV from already independently recomputed statistics."""
from pathlib import Path
import csv
import json
import shutil

WORK = Path(__file__).resolve().parents[1]
STATS = json.loads((WORK/'result_metrics_audit/recomputed_endpoints.json').read_text())
INTERPRETATION = json.loads((WORK/'result_interpretation_audit/results.json').read_text())
REPORT = WORK/'SELLER_ALIAS_RELATION_MEMORY.result_external_review.zh.md'
OUT = Path(__file__).parent
ORDERS = ('ABC','BCA','CAB')
EPS = ('O','N','Z','F_first','F','G','final_all')
COLS = list(STATS['endpoints']['relation']['primary']['O'])
assert len(COLS) == 22


def f(value, signed=False):
    return (f'{value:+.6f}' if signed else f'{value:.6f}').replace('-', '−')


def ci(record):
    lo, hi = record['conditional_95pct_interval']
    return f'[{f(lo)}, {f(hi)}]'


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '|'+'|'.join('---' for _ in headers)+'|',
                      *['| '+' | '.join(map(str,row))+' |' for row in rows]])


def metric_rows(items):
    rows=[]
    for ep, metric in items:
        a=STATS['endpoints']['relation']['primary'][ep][metric]
        b=STATS['endpoints']['logit0.1']['primary'][ep][metric]
        d=STATS['delta'][ep][metric]
        rows.append([ep,metric,f(a['mean']),f(b['mean']),f(d['mean'],True),ci(d)])
    return rows


headers=['端点','指标','候选','LOGIT0.1','差值','差值条件95%区间']
tables={
    'MAIN_SIX_TABLE':table(headers,metric_rows([(e,'map') for e in 'ONZ']+[(e,'average_precision') for e in 'ONZ'])),
    'PROBABILITY_DELTA_TABLE':table(headers,metric_rows([(e,c) for e in 'ONZ' for c in ('brier','log_loss')])),
    'FG_TABLE':table(headers,metric_rows([(e,'map') for e in ('F_first','F','G')])),
}
rows=[]
for order in ORDERS:
    for ep in 'ONZ':
        a=STATS['endpoints']['relation']['primary'][ep]['map']['per_order'][order]
        b=STATS['endpoints']['logit0.1']['primary'][ep]['map']['per_order'][order]
        rows.append([order,ep,f(a),f(b),f(a-b,True)])
tables['ORDER_MAP_TABLE']=table(['顺序','端点','候选MAP','LOGIT0.1 MAP','差值'],rows)
rows=[]
for order in ORDERS:
    t=INTERPRETATION['first_domain_trajectories'][order]
    rows.append([order+'/'+order[0],*[f(t['relation'][str(s)]['map']) for s in (1,2,3)],
                 f(t['logit0.1']['1']['map']),f(t['logit0.1']['3']['map'])])
tables['FIRST_DOMAIN_TABLE']=table(['顺序／首域','候选stage1','候选stage2','候选stage3','LOGIT stage1','LOGIT stage3'],rows)
rows=[]
for ep in 'ONZ':
    for arm,label in [('relation','候选'),('logit0.1','LOGIT0.1')]:
        r=INTERPRETATION['pooled'][arm]['stage-cal'][ep]
        rows.append([ep,label,*[r['count_totals'][k] for k in ('tp','fp','fn','tn')],f(r['pooled']['f1']),f(r['macro']['f1'])])
tables['POOLED_TABLE']=table(['端点','方法','TP','FP','FN','TN','合并F1','群宏F1'],rows)
rows=[]
for ep in 'ONZ':
    r=INTERPRETATION['probability_roles']['relation'][ep]
    rows.append([ep,*[f(r[role]['brier']) for role in ('raw','stage-cal','first-cal')],
                 *[f(r[role]['log_loss']) for role in ('raw','stage-cal','first-cal')]])
tables['CALIBRATION_TABLE']=table(['端点','raw Brier','stage-cal Brier','first-cal Brier','raw log-loss','stage-cal log-loss','first-cal log-loss'],rows)
tables['FULL_22_TABLES']='\n\n'.join('### A.'+str(i)+' '+ep+'\n\n'+table(headers,metric_rows([(ep,c) for c in COLS])) for i,ep in enumerate(EPS,1))

template=REPORT.read_text(encoding='utf-8')
if '<!-- MAIN_SIX_TABLE -->' in template:
    (OUT/'report_template.zh.md').write_text(template,encoding='utf-8')
else:
    template=(OUT/'report_template.zh.md').read_text(encoding='utf-8')
for name,value in tables.items():
    marker='<!-- '+name+' -->'
    assert template.count(marker)==1,name
    template=template.replace(marker,value)
template=template.replace('都显著小于raw诊断数值','都明显小于raw诊断数值')
assert '<!--' not in template
REPORT.write_text(template,encoding='utf-8')

with (OUT/'independent_all_endpoints.csv').open('w',encoding='utf-8',newline='') as stream:
    fields=['configuration','role','endpoint','metric','mean','ABC','BCA','CAB','conditional_95_lo','conditional_95_hi']
    writer=csv.DictWriter(stream,fieldnames=fields)
    writer.writeheader()
    count=0
    for arm,roles in STATS['endpoints'].items():
        for role,eps in roles.items():
            for ep in EPS:
                for metric in COLS:
                    r=eps[ep][metric]
                    writer.writerow(dict(configuration=arm,role=role,endpoint=ep,metric=metric,mean=r['mean'],
                        **r['per_order'],conditional_95_lo=r['conditional_95pct_interval'][0],conditional_95_hi=r['conditional_95pct_interval'][1]))
                    count+=1
    for ep in EPS:
        for metric in COLS:
            r=STATS['delta'][ep][metric]
            writer.writerow(dict(configuration='relation_minus_logit0.1',role='primary',endpoint=ep,metric=metric,mean=r['mean'],
                **r['per_order'],conditional_95_lo=r['conditional_95pct_interval'][0],conditional_95_hi=r['conditional_95pct_interval'][1]))
            count+=1
assert count==1386
print(json.dumps({'report_bytes':REPORT.stat().st_size,'report_lines':len(template.splitlines()),
                  'all_endpoints_csv_rows':count,'primary_appendix_rows':7*22},ensure_ascii=False,indent=2))
