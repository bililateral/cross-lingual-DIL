"""Final audit-artifact check; no model/training or formal label handling."""
from pathlib import Path
import csv, hashlib, json, re, zipfile
B=Path('/mnt/data/ranking_result_reaudit_20260928');E=B/'evidence';R=B/'submitted'
report=Path('/mnt/data/ranking_result_resubmission_review.zh.md')
assert report.read_bytes()==(E/'review_report.zh.md').read_bytes()
a=json.loads((E/'independent_analysis/independent_results.json').read_text())
rr=list(csv.DictReader((E/'primary_22_metrics.csv').open(encoding='utf-8-sig')))
assert len(rr)==22 and len({r['metric'] for r in rr})==22
text=report.read_text(); checks=0
for r in rr:
 expected=f"| {r['metric_zh']} | {float(r['reference']):.6f} | {float(r['candidate']):.6f} | {float(r['delta']):+.6f} | [{float(r['lower']):+.6f}, {float(r['upper']):+.6f}] |"
 assert expected in text, r['metric'];checks+=5
for i in range(1,11): assert f'### F{i}.' in text
assert sum(a['acceptance'].values())==9 and len(a['acceptance'])==13
assert len(re.findall(r'\| \*\*未通过\*\* \|',text))==4
assert (E/'primary_22_metrics.csv').read_bytes()==Path('/mnt/data/ranking_result_resubmission_metrics.csv').read_bytes()
with zipfile.ZipFile('/mnt/data/ranking_result.zip') as z:
 for n in z.namelist(): assert (R/n).read_bytes()==z.read(n),n
 assert len(z.namelist())==253
versions=[]
for p in sorted(E.glob('*.execution.json')):
 d=json.loads(p.read_text());assert d['exit_code']==0
 versions.append({'record':p.name,'exit_code':d['exit_code'],'python':d['python']})
for p in E.rglob('*'):
 if p.is_file(): assert p.suffix not in {'.pt','.bin','.safetensors','.jsonl'},p
res={'status':'PASS_FINAL_CURRENT_REAUDIT_ARTIFACT_CHECK','report_table_values_checked':checks,'findings_present':10,'acceptance_passed':False,'failed_conditions':4,'original_zip_members_unchanged':253,'completed_archived_executions':versions,'report_bytes':report.stat().st_size,'report_sha256':hashlib.sha256(report.read_bytes()).hexdigest(),'no_model_files_in_evidence':True,'scope':'Checks artifact consistency only; does not access formal labels/text or models.'}
(E/'final_evidence_checks.json').write_text(json.dumps(res,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(res,ensure_ascii=False,indent=2))
