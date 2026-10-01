"""Label-free independent narrative cross-checks and disclosure tables; no new gates."""
from pathlib import Path
import csv, hashlib, json, math, re, time, os
R=Path('/mnt/data/bge_input'); O=Path('/mnt/data/bge_review_evidence/outputs')
J=R/'reports/seller_alias_continual/20260930/bge_continual_execution/20260930_143700/job'
E=J/'evaluation'
start=time.perf_counter()
REPORT_OUT=O/'report_crosscheck_v2'
REPORT_OUT.mkdir(exist_ok=False)
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(n,x):(REPORT_OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
ep=load(O/'independent_endpoints.json'); sc=load(O/'independent_score_and_count_summary.json'); first=load(O/'independent_first_learning.json'); updates=load(O/'independent_update_diagnostics.json')
methods=['frozen','seq','er','logit']; metrics=list(ep['seq']['primary']['O'])
checks=[]
doc=(R/'docs/SELLER_ALIAS_BGE_RESULT.zh.md').read_text().splitlines()
for ln,line in enumerate(doc,1):
    if not line.startswith('|'):continue
    cells=[x.strip() for x in line.strip('|').split('|')]
    if len(cells)==5 and cells[0] in metrics:
        for arm,text in zip(methods,cells[1:]):
            value=ep[arm]['primary']['final_all'][cells[0]]['mean'];error=abs(float(text)-value);assert error<=5.00001e-7,(ln,arm,cells[0],error)
            checks.append(dict(line=ln,section='final22',arm=arm,metric=cells[0],printed=float(text),independent=value,error=error))
    if len(cells)==7 and cells[0] in methods+['SEQ','ER','LOGIT']:
        arm=cells[0].lower()
        for text,(endpoint,metric) in zip(cells[1:],[(q,m) for q in ('O','N','Z') for m in ('map','recall_at_5')]):
            v=ep[arm]['primary'][endpoint][metric]['mean'];er=abs(float(text)-v);assert er<=5.00001e-7
            checks.append(dict(line=ln,section='ONZ',arm=arm,endpoint=endpoint,metric=metric,printed=float(text),independent=v,error=er))
    if len(cells)==5 and re.fullmatch(r'(SEQ|ER|LOGIT)／[ONZ]',cells[0]):
        arm,endpoint=cells[0].lower().split('／');endpoint=endpoint.upper()
        for text,metric in zip(cells[1:],['average_precision','roc_auc','brier','log_loss']):
            v=ep[arm]['primary'][endpoint][metric]['mean'];er=abs(float(text)-v);assert er<=5.00001e-7
            checks.append(dict(line=ln,section='probabilities',arm=arm,endpoint=endpoint,metric=metric,printed=float(text),independent=v,error=er))
assert len(checks)==88+24+36
conditions=list(csv.DictReader((O/'independent_23_conditions.csv').open()))
condition_rows=[]
for i in range(23):
    a,b=conditions[i],conditions[23+i];assert a['condition']==b['condition'] and a['operator']==b['operator']
    condition_rows.append(dict(index=i+1,condition=a['condition'],operator=a['operator'],er_minus_seq=float(a['value']),er_pass=a['pass']=='True',logit_minus_er=float(b['value']),logit_pass=b['pass']=='True'))
final_fixed=[]
for arm in ('seq','er','logit'):
    rows=[x for x in sc if re.fullmatch(r'(ABC|BCA|CAB)_'+arm+r'_stage3',x['point']) and x['role']=='stage-cal'];assert len(rows)==3
    counts={k:sum(x[k] for x in rows) for k in ('tp','fp','fn','tn')}
    denom=counts['tp']+counts['fp']
    final_fixed.append(dict(method=arm,records=rows,descriptive_reused_60_group_three_model_ledger=counts,group_macro_precision=ep[arm]['primary']['final_all']['precision']['mean'],group_macro_recall=ep[arm]['primary']['final_all']['recall']['mean'],pooled_precision=counts['tp']/denom if denom else 0.,pooled_recall=counts['tp']/3600,pooled_fpr=counts['fp']/64440,recall_at_5=ep[arm]['primary']['final_all']['recall_at_5']['mean'],scope='Pooling is descriptive across model evaluations; not 180 independent groups.'))
first_table=[]
col=load(E/'collected.json');dr={d:[i for i,x in enumerate(col['domains']) if x==d] for d in 'ABC'}
import numpy as np
for order in ('ABC','BCA','CAB'):
    one=dict(order=order,domain=order[0],initial_map=first[order]['initial_raw']['map'],first_map=first[order]['first_raw']['map'],initial_recall5=first[order]['initial_raw']['recall_at_5'],first_recall5=first[order]['first_raw']['recall_at_5'],F_first_map=ep['seq']['raw']['F_first']['map']['per_order'][order],G_map=ep['seq']['raw']['G']['map']['per_order'][order])
    for stage in (2,3):
        mat=np.load(E/col['points'][f'{order}_seq_stage{stage}']['raw']['matrix']['path'],allow_pickle=False)
        one[f'stage{stage}_first_domain_map']=float(mat[dr[order[0]],metrics.index('map')].mean())
    first_table.append(one)
# Specific document-navigation issue. Honor the explicitly supplied returned/job mapping.
inventory=load(R/'source_inventory.json')
source_to_archive={x['source_path']:x['archive_path'] for x in inventory['files']}
links=[]
for ln,line in enumerate(doc,1):
    for label,url in re.findall(r'\[([^\]]+)\]\(([^)]+)\)',line):
        if '://' in url:continue
        relative=(R/'docs'/url.split('#')[0]).resolve().relative_to(R).as_posix()
        present=(R/relative).exists();mapped=source_to_archive.get(relative)
        if not present and mapped and (R/mapped).exists():present=True
        # source_inventory is file-granular. A directory link can represent a
        # complete mapped subtree even when the Windows directory is not in ZIP.
        if not present:
            descendants=[v for k,v in source_to_archive.items() if k.startswith(relative.rstrip('/')+'/')]
            if descendants and all((R/v).is_file() for v in descendants):
                present=True
                mapped=os.path.commonpath(descendants)
        links.append(dict(line=ln,label=label,target=url,resolved=relative,present_or_manifest_mapped=present,mapped_archive=mapped))
unresolved=[x for x in links if not x['present_or_manifest_mapped']]
# No attempt to query live OS/server state; the unresolved observation claim is ancillary.
clipped=sum(round(x['clip_fraction']*288) for x in updates)
all_later=[x for x in updates if x['stage'] in (2,3)]
assert all(x['clip_fraction']==1 for x in all_later)
for row in final_fixed:
    if row['method']=='er': assert all(x['positive_predictions']==0 and x['max_score']<0 for x in row['records'])
result=dict(status='PASS_148_NARRATIVE_NUMBERS_AND_DISCLOSURES',narrative_numeric_values=len(checks),maximum_rounding_error=max(x['error'] for x in checks),narrative_rounding_tolerance=5.00001e-7,unresolved_document_links=unresolved,clip_threshold=1.0,later_stage_clipped_updates=18*288,all_clipped_updates=clipped,all_updates=6048,calibration_negative_workpoint_max_probabilities={x['point']:1/(1+math.exp(-x['max_score'])) for x in sc if '_er_stage3' in x['point'] and x['role']=='stage-cal'},formal_new_acceptance_rules=0,formal_label_access=0,elapsed_seconds=time.perf_counter()-start)
save('report_crosschecks.json',result);save('report_numeric_checks.json',checks);save('report_document_links.json',links);save('report_23_conditions.json',condition_rows);save('report_fixed_workpoints.json',final_fixed);save('report_first_learning.json',first_table)
print(json.dumps(result,ensure_ascii=False,indent=2));print('FIRST LEARNING');print(json.dumps(first_table,ensure_ascii=False,indent=2));print('FINAL FIXED POINTS');print(json.dumps(final_fixed,ensure_ascii=False,indent=2))
