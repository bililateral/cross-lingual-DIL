#!/usr/bin/env python3
"""Build the Chinese review from independently computed saved-result evidence."""
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path

WORK=Path(__file__).resolve().parents[2]
EVIDENCE=WORK/'evidence'
NUM=EVIDENCE/'agents/independent_metrics/numerical_v1'
OUT=WORK.parent/'deliverables'
OUT.mkdir(parents=True,exist_ok=True)
endpoints=json.loads((NUM/'independent_endpoints.json').read_text())
comparisons=json.loads((NUM/'independent_comparisons.json').read_text())
summary=json.loads((NUM/'summary.json').read_text())
assert summary['comparison_passes']=={'tenth_minus_quarter':23,'tenth_minus_seq':10,'logit_quarter_minus_quarter':23,'logit_quarter_minus_seq':16,'quarter_minus_seq':4}
NAMES={'seq':'SEQ','er':'ER1','half':'ER0.5','quarter':'ER0.25','tenth':'ER0.1','logit_quarter':'LOGIT0.25'}
COMPS=['tenth_minus_quarter','tenth_minus_seq','logit_quarter_minus_quarter','logit_quarter_minus_seq','quarter_minus_seq']
def compname(c):
    a,b=c.split('_minus_')
    return NAMES[a]+' − '+NAMES[b]
def fixed(x): return f'{x:.6f}'
def signed(x): return f'{x:+.6f}'
def interval(rec): return '['+', '.join(signed(x) for x in rec['conditional_95pct_interval'])+']'
def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in rows)

pass_table=table(['固定比较','独立判定','O MAP差','条件95%区间','本轮含义'],[
    [compname(c),f"{comparisons[c]['pass_count']}/23",signed(comparisons[c]['primary']['O']['map']['mean']),interval(comparisons[c]['primary']['O']['map']),interpret]
    for c,interpret in zip(COMPS,['按冻结规则选ER0.1为开发ER配置','局部旧域收益；完整保护未过','相对固定ER0.25的增量通过','局部收益；完整保护未过','原有效负结果保持'])])
core_table=table(['方法','O MAP','O R@5','N MAP','N R@5','Z MAP','Z R@5','F_first MAP↓','final_all MAP'],[
    [NAMES[a],*[fixed(endpoints[a]['primary'][e][m]['mean']) for e,m in [('O','map'),('O','recall_at_5'),('N','map'),('N','recall_at_5'),('Z','map'),('Z','recall_at_5'),('F_first','map'),('final_all','map')]]]
    for a in ['seq','er','half','quarter','tenth','logit_quarter']])
delta_rows=[]
for c in COMPS:
    for e,m in [('O','map'),('O','recall_at_5'),('N','map'),('N','recall_at_5'),('Z','map'),('Z','recall_at_5'),('F_first','map'),('final_all','map')]:
        r=comparisons[c]['primary'][e][m]
        delta_rows.append([compname(c),e,m,signed(r['mean']),interval(r)])
delta_table=table(['固定比较','端点','指标','观察差','条件95%区间'],delta_rows)
gate_names=[g['check'] for g in comparisons[COMPS[0]]['gates']]
gate_lookup={c:{g['check']:g for g in comparisons[c]['gates']} for c in COMPS}
gate_table=table(['原判据',*[compname(c) for c in COMPS]],[
    [g,*['通过' if gate_lookup[c][g]['pass'] else '未通过' for c in COMPS]] for g in gate_names]+[
    ['合计',*[f"{comparisons[c]['pass_count']}/23" for c in COMPS]]])
order_table=table(['O MAP差','ABC','BCA','CAB'],[
    [compname(c),*[signed(comparisons[c]['primary']['O']['map']['per_order'][o]) for o in ['ABC','BCA','CAB']]] for c in COMPS])
logit_failed=table(['端点','未通过指标','观察差','条件95%区间'],[
    [g['endpoint'],g['metric'],signed(g['observed_mean']),f"[{signed(g['ci_low'])}, {signed(g['ci_high'])}]"]
    for g in comparisons['logit_quarter_minus_seq']['gates'] if not g['pass']])
classification_table=table(['方法','端点','固定0.5 Recall','固定0.5 F1','候选Recall@5'],[
    [NAMES[a],e,*[fixed(endpoints[a]['primary'][e][m]['mean']) for m in ['recall','f1','recall_at_5']]]
    for a in ['seq','quarter','tenth','logit_quarter'] for e in ['O','N','Z']])
all22=[]
for e in ['O','N','Z']:
    all22.append('### 附表 '+e+'：主角色群宏均值\n\n'+table(['指标','SEQ','ER0.25','ER0.1','LOGIT0.25'],[
        [m,*[fixed(endpoints[a]['primary'][e][m]['mean']) for a in ['seq','quarter','tenth','logit_quarter']]]
        for m in endpoints['seq']['primary'][e]]))

template=(EVIDENCE/'src/review_template.zh.md').read_text()
values={'PASS_TABLE':pass_table,'CORE_TABLE':core_table,'DELTA_TABLE':delta_table,'GATE_TABLE':gate_table,'ORDER_TABLE':order_table,'LOGIT_FAILED_TABLE':logit_failed,'CLASSIFICATION_TABLE':classification_table,'ALL22_TABLES':'\n\n'.join(all22),'GENERATED_AT':dt.datetime.now(dt.timezone.utc).isoformat()}
for key,value in values.items(): template=template.replace('{{'+key+'}}',value)
assert '{{' not in template
(OUT/'REVIEW.zh.md').write_text(template,encoding='utf-8')
(EVIDENCE/'REVIEW.zh.md').write_text(template,encoding='utf-8')
rec={'path':str(OUT/'REVIEW.zh.md'),'bytes':(OUT/'REVIEW.zh.md').stat().st_size,'sha256':hashlib.sha256((OUT/'REVIEW.zh.md').read_bytes()).hexdigest(),'generated_at_utc':values['GENERATED_AT'],'source':'independent numeric JSON plus explicit review text, not project PASS output'}
(EVIDENCE/'report_generation.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(rec,ensure_ascii=False,indent=2))
