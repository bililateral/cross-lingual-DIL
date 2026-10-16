#!/usr/bin/env python3
"""Check artifact consistency only; does not execute scientific code or change input."""
from pathlib import Path
import ast,csv,hashlib,json
r=Path('/mnt/data/h_external_review')
json_files=list(r.rglob('*.json'));scripts=list(r.rglob('*.py'))
for p in json_files:json.loads(p.read_text())
for p in scripts:ast.parse(p.read_text())
assert (r/'report.zh.md').read_bytes()==(r/'report.zh.txt').read_bytes()
f=json.loads((r/'findings.json').read_text())
assert len(f['groups']['SCIENTIFIC_BLOCKER'])==0 and len(f['groups']['REPRODUCIBILITY_DEFECT'])==1
scope=json.loads((r/'reading_scope.json').read_text())['files']
assert len(scope)==219 and all(x['integrity_checked'] for x in scope)
with (r/'checks/run01/recomputed_statistics.csv').open() as p:rows=list(csv.DictReader(p))
assert len(rows)==1650 and sum(x['mean']=='' for x in rows)==114
assert json.loads((r/'checks/run01/audit_result.json').read_text())['status']=='PASS_EXTERNAL_SAVED_EVIDENCE_CHECKS'
assert json.loads((r/'checks/run02/result.json').read_text())['status']=='FAIL_REVIEWER_RECONCILIATION'
assert json.loads((r/'checks/run03/result.json').read_text())['status']=='PASS_SAVED_REPORT_RECONCILIATION'
assert (r/'logs/reconcile_run02.exit_code.txt').read_text().strip()=='1'
assert (r/'logs/reconcile_run03.exit_code.txt').read_text().strip()=='0'
assert 'FileNotFoundError' in (r/'logs/reconcile_run02.stderr.txt').read_text()
assert (r/'logs/reconcile_run03.stderr.txt').stat().st_size==0
assert (r/'logs/audit_run01.stderr.txt').stat().st_size==0
print(json.dumps({'status':'PASS_DELIVERY_CONTENT_CHECK','json_files_parsed':len(json_files),'python_files_ast_parsed':len(scripts),'report_bytes':(r/'report.zh.md').stat().st_size,'report_copies_identical':True,'input_members_documented':219,'statistics_rows':1650,'NA_rows':114,'preserved_failed_review_script_exit':1,'preserved_successful_review_script_exit':0,'scientific_blockers':0,'minor_documentation_defects':1},ensure_ascii=False,indent=2))
