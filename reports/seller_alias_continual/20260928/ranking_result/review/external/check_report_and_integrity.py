"""Cross-check human result tables against the independent arithmetic outputs.
Read-only source integrity; no project imports, formal labels, or model access.
"""
from __future__ import annotations
import csv, hashlib, json, re, sys, zipfile
from pathlib import Path
ROOT=Path('/mnt/data/ranking_result_reaudit_20260928/submitted')
OUT=Path('/mnt/data/ranking_result_reaudit_20260928/evidence')
D=json.loads((OUT/'independent_analysis/independent_results.json').read_text())
text=(ROOT/'docs/SELLER_ALIAS_RANKING_RESULT.zh.md').read_text()
C=D['comparisons']; primary=C['hard_minus_d']
N=0; DETAILS=[]
def check_numbers(line,expected):
 global N
 fields=[x.strip() for x in line.strip().strip('|').split('|')]
 nums=[float(n.replace('−','-')) for n in re.findall(r'[+−-]?\d+\.\d+', '|'.join(fields[1:]))]
 assert len(nums)==len(expected),(line,nums,expected)
 # The report explicitly rounds these tables to six decimals.
 for observed,actual in zip(nums,expected):
  assert observed==float(f'{actual:.6f}'),(line,observed,actual)
 N+=len(nums);DETAILS.append({'row':fields[0],'values':len(nums)})
name={'账号对AP':'average_precision','AP':'average_precision','梯形PR-AUC':'trapezoidal_pr_auc','ROC-AUC':'roc_auc','Recall@FPR≤1%':'recall_at_fpr_1pct','Brier':'brier','log_loss':'log_loss','固定0 precision':'precision','固定0 recall':'recall','固定0 F1':'f1','固定0 specificity':'specificity','固定0 balanced accuracy':'balanced_accuracy','固定0 MCC':'mcc','MAP':'map','MRR':'mrr','Recall@1':'recall_at_1','Recall@3':'recall_at_3','Recall@5':'recall_at_5','Recall@10':'recall_at_10','NDCG@1':'ndcg_at_1','NDCG@3':'ndcg_at_3','NDCG@5':'ndcg_at_5','NDCG@10':'ndcg_at_10'}
main_table=text.split('| 指标 | A：D | C：hard | C−A | 条件95%区间 |')[1].split('### 种子与域')[0]
for line in main_table.splitlines():
 if line.startswith('|'):
  key=line.split('|')[1].strip()
  if key in name:
   m=primary[name[key]];check_numbers(line,[m['reference_mean'],m['candidate_mean'],m['mean'],*m['conditional_95pct_interval']])
full_table=text.split('| 指标 | A | C | C−A |')[1].split('固定0下召回')[0]
full_count=0
for line in full_table.splitlines():
 if line.startswith('|'):
  key=line.split('|')[1].strip()
  if key in name:
   m=primary[name[key]];check_numbers(line,[m['reference_mean'],m['candidate_mean'],m['mean']]);full_count+=1
assert full_count==22
T={(r['run_id'],r['epoch'],r['split']):r for r in csv.DictReader((OUT/'independent_analysis/trajectories_22.csv').open())}
for s in ('s0','s1','s2'):
 line=next(x for x in text.splitlines() if x.startswith('| '+s))
 a=T[(s+'_d','6','valid')];c=T[(s+'_hard','6','valid')];i=int(s[1])
 check_numbers(line,[float(a['map']),float(c['map']),primary['map']['per_seed'][i],float(a['recall_at_5']),float(c['recall_at_5']),primary['brier']['per_seed'][i],primary['log_loss']['per_seed'][i]])
for prefix,comp in [('B−A：','schedule_minus_d'),('C−B：','hard_minus_schedule')]:
 line=next(x for x in text.splitlines() if x.startswith('| '+prefix));m=C[comp]
 check_numbers(line,[m['map']['mean'],*m['map']['conditional_95pct_interval'],m['recall_at_5']['mean'],*m['recall_at_5']['conditional_95pct_interval'],m['brier']['mean'],m['log_loss']['mean']])
# Independently check the report's six-decimal trajectory narrative, not new scoring.
trajectory=[]
for metric in ['map','recall_at_5']:
 for s in ('s0','s1','s2'):
  before=float(T[s+'_hard','3','valid'][metric]);after=float(T[s+'_hard','6','valid'][metric]);snippet=f'{before:.6f}→{after:.6f}'
  assert snippet in text,snippet;trajectory.append(snippet)
for s in ('s0','s1','s2'):
 for metric in ['brier','log_loss']:
  assert float(T[s+'_hard','6','valid'][metric])<float(T[s+'_hard','3','valid'][metric])
for metric in ['map','recall_at_5','average_precision','roc_auc']:
 assert all(x>0 for x in primary[metric]['per_seed'])
for metric in ['map','recall_at_5','brier','log_loss']:
 assert all(x>0 for x in primary[metric]['by_domain'].values())
assert sum(D['acceptance'].values())==9 and len(D['acceptance'])==13
assert all(D['perseed']['hard_minus_d']['s0'][m]['conditional_95pct_interval'][0]>0 for m in ['brier','log_loss'])
res=ROOT/'reports/seller_alias_continual/20260928/ranking_result'
identical=[]
for name_ in ['metrics.csv','comparisons.csv','trajectory.csv','primary_bootstrap.npy']:
 a=(res/'analysis'/name_).read_bytes();b=(res/'reviewer_analysis'/name_).read_bytes();assert a==b
 identical.append({'path':name_,'bytes':len(a),'sha256':hashlib.sha256(a).hexdigest()})
# Every submitted member, including the source inventory and original analysis,
# must still be exactly the uploaded bytes. New outputs have different paths.
zip_path=Path('/mnt/data/ranking_result.zip');z=zipfile.ZipFile(zip_path)
assert len(z.infolist())==253 and z.testzip() is None
for info in z.infolist():
 assert (ROOT/info.filename).read_bytes()==z.read(info),info.filename
result={'status':'PASS_REPORT_TABLES_AND_SOURCE_INTEGRITY','report_table_values':N,'rows':DETAILS,'complete_metric_rows':full_count,'trajectory_strings':trajectory,'all_253_original_members_unchanged':True,'original_vs_web_output_byte_equality':identical,'operations':{'formal_labels':0,'formal_texts':0,'model_loads':0,'training_updates':0,'project_modules_imported':False},'scope':'Report rounding/narrative consistency with independently recomputed saved outputs, not truth-level metric re-evaluation.'}
(OUT/'report_integrity_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
