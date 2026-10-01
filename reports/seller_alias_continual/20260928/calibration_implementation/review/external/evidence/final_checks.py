"""Final artifact and submission checks; stdlib only, no research-array parsing."""
from pathlib import Path
import hashlib, json, zipfile
E=Path(__file__).resolve().parent
W=E.parent
S=W/'submission'
Z=W.parent/'calibration_review.zip'
sha=lambda b:hashlib.sha256(b).hexdigest()
original=Z.read_bytes()
assert len(original)==1286211
assert sha(original)=='77ab3765d0928a231fadae7cbb9d92049e0343c4a7518ea5478a0b2909bcf51d'
with zipfile.ZipFile(Z) as z:
    assert z.testzip() is None
    files=[n for n in z.namelist() if not n.endswith('/')]
    assert len(files)==103
    changed=[n for n in files if not (S/n).is_file() or (S/n).read_bytes()!=z.read(n)]
    assert not changed,changed
    extras=[p.relative_to(S).as_posix() for p in S.rglob('*') if p.is_file() and p.relative_to(S).as_posix() not in files]
    assert not extras,extras
expected={'verification_run':0,'supplied_audit_run':0,'independent_run_v1':1,'probability_diagnosis_run':0,'independent_run_v2':1,'independent_run_v3':0,'provenance_run':0}
runs={}
for name,code in expected.items():
    out=E/name
    record=json.loads((out/'execution.json').read_text())
    assert record['exit_code']==code==int((out/'exit_status.txt').read_text())
    for field in ('command.json','stdout.log','stderr.log','resource_usage.log'):
        assert (out/field).is_file(),(name,field)
    runs[name]={'exit_code':code,'wall_seconds':record['wall_seconds']}
sup=json.loads((E/'supplied_audit_artifacts/audit.json').read_text())
assert sup['contracts']=={'errors':0,'failures':0,'passed':16,'run':16,'skipped':0}
ind=json.loads((E/'independent_outputs/independent_results.json').read_text())
assert len(ind['tests_passed'])==10 and ind['tests_failed']==[]
assert sum(ind['numeric_counts'].values())==17316
assert ind['numeric_counts']['pipeline_valid_22_reference']==15840
for v in (1,2):
    old=json.loads((E/f'independent_outputs_v{v}/independent_results.json').read_text())
    assert len(old['tests_passed'])==9 and len(old['tests_failed'])==1
report=(W/'calibration_review_report.zh.md').read_bytes()
assert report==(W.parent/'calibration_review_report.zh.md').read_bytes()
text=report.decode()
for token in ('17,316','15,840','必须代码整改为0项','原九模型结果仍为9过4败','正式六映射'):
    assert token in text,token
findings=json.loads((E/'review_findings.json').read_text())
manifest={'status':'PASS_FINAL_ARTIFACT_CHECK','original_archive_bytes':len(original),'original_archive_sha256':sha(original),'original_members_verified_unchanged':len(files),'extra_submission_files':extras,'run_exit_codes_verified':runs,'final_unique_test_cases':26,'final_independent_numeric_comparisons':17316,'reviewer_failed_versions_retained':2,'report':{'path':'calibration_review_report.zh.md','bytes':len(report),'sha256':sha(report)},'formal_arrays_parsed_in_this_check':0,'formal_labels_read':0,'model_payloads_read':0,'formal_stage_run':False}
(E/'final_integrity.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(manifest,ensure_ascii=False,indent=2))
