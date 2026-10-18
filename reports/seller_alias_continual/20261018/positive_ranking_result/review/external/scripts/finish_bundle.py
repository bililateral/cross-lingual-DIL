"""Package complete authorized review artifacts; no project execution."""
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

BASE = Path('/workspace/scratch/0f307e99d302')
OUT = BASE / 'audit_output'
INPUT = BASE / 'review_input'
DELIVER = BASE / 'deliverables'
DELIVER.mkdir(exist_ok=True)


def read(p):
    return json.loads(p.read_text())


def identity(p, label=None):
    b = p.read_bytes()
    return {'path': label or str(p), 'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()}


manual = []
root_ranges = [
    ('workspace/reports/result_sync/report.zh.txt', 'full_text_semantic_read', [[1,193]]),
    ('workspace/docs/SELLER_ALIAS_POSITIVE_RANKING.zh.md', 'full_text_semantic_read', [[1,55]]),
    ('workspace/scripts/step28_positive_ranking_audit.py', 'full_text_semantic_read_before_execution', [[1,411]]),
    ('workspace/scripts/step28_positive_ranking_run.py', 'targeted_semantic_read_and_symbol_search', [[1,93],[357,476]]),
    ('workspace/scripts/step28_g_half_run.py', 'import_alias_read_and_symbol_search', [[1,39]]),
    ('workspace/scripts/step28_alias_calibration.py', 'targeted_semantic_read', [[1,149]]),
    ('workspace/scripts/step28_balanced_pooling_run.py', 'targeted_query_NA_summary_read', [[595,617],[648,681]]),
    ('workspace/scripts/step28_continual_population_evaluate.py', 'targeted_metric_definition_read', [[13,122],[225,241]]),
    ('workspace/scripts/step28_alias_ranking.py', 'symbol_search_only', []),
    ('workspace/scripts/step28_bge_continual_evaluate.py', 'symbol_search_only', []),
    ('workspace/reports/job/before_valid.json', 'selected_saved_fields_review', []),
    ('workspace/reports/job/completion.json', 'selected_saved_fields_review', []),
    ('workspace/reports/job/execution.json', 'selected_saved_fields_review', []),
    ('workspace/reports/qualification/native/result.json', 'selected_saved_field_summaries_review_not_tensor_replay', []),
    ('workspace/reports/result_sync/analysis_wallclock.json', 'complete_saved_json_read', []),
    ('workspace/reports/result_sync/closeout.json', 'complete_saved_json_read', []),
]
for rel, mode, ranges in root_ranges:
    p = INPUT / rel
    manual.append({**identity(p,rel), 'mode':mode, 'line_ranges':ranges})
for rel in ('agents/execution/report.zh.txt','agents/wiring/report.zh.txt','agents/diagnostics/report.zh.md'):
    manual.append({**identity(OUT/rel,'review_output/'+rel), 'mode':'full_text_subtask_report_review', 'line_ranges':[]})
manual.append({**identity(OUT/'supplied_replay/comparison.png','review_output/supplied_replay/comparison.png'),'mode':'visual_inspection','line_ranges':[]})
(OUT/'root_manual_read_scope.json').write_text(json.dumps({'note':'No claim of complete semantic review from hashing/AST alone; full source scopes read by specialists remain separately recorded.','files':manual},ensure_ascii=False,indent=2)+'\n')

merged = {}
def add(path, reader, observation):
    if path.startswith(str(INPUT)+'/'):
        path=path[len(str(INPUT))+1:]
    rec=merged.setdefault(path,{'path':path,'reader_observations':[]})
    for k in ('bytes','sha256'):
        if k in observation: rec[k]=observation[k]
    rec['reader_observations'].append({'reader':reader, **{k:v for k,v in observation.items() if k not in ('path','bytes','sha256')}})

for r in read(OUT/'input_inventory_verified.json'):
    add(r['path'],'root_input_identity',{**r,'mode':'all_payload_byte_identity_and_zip_payload_comparison'})
add('manifest.json','root_input_identity',{**identity(INPUT/'manifest.json','manifest.json'),'mode':'complete_manifest_parse_and_zip_byte_comparison'})
for r in manual: add(r['path'],'root',r)
for r in read(OUT/'independent_statistics/read_scope.json'):
    add('workspace/'+r['path'],'root_independent_statistics',{**r,'mode':'complete_saved_array_or_json_numeric_parse'})
for r in read(OUT/'agents/execution/read_scope.json')['files']: add(r['path'],'execution_specialist',r)
for r in read(OUT/'agents/wiring/read_scope.json'): add(r['path'],'wiring_specialist',r)
for r in read(OUT/'agents/wiring/mechanically_read_inputs.json'):
    if isinstance(r,dict) and 'path' in r: add(r['path'],'wiring_mechanical',r)
diagmanual=read(OUT/'agents/diagnostics/manual_read_scope.json')
for r in diagmanual['files']: add(r['path'],'diagnostics_specialist',r)
diagreads=read(OUT/'agents/diagnostics/attempt01/read_scope.json')
if isinstance(diagreads,dict): diagreads=diagreads.get('files',diagreads.get('inputs',[]))
for r in diagreads:
    if isinstance(r,dict) and 'path' in r:
        path=r['path']
        if not path.startswith(('workspace/','/')): path='workspace/'+path
        add(path,'diagnostics_mechanical',r)
(OUT/'reading_scope.json').write_text(json.dumps({'scope_note':'Byte identity, full semantic reading, targeted excerpts, and structured/array numeric parsing are distinguished per reader. Counts overlap. The supplied audit report also records 310 checked files. No raw labels/models/Memory read.','files':sorted(merged.values(),key=lambda r:r['path'])},ensure_ascii=False,indent=2)+'\n')

# Final unchanged-input check: this is artifact integrity, not another research run.
current=[]
for expected in read(INPUT/'manifest.json'):
    got=identity(INPUT/expected['path'],expected['path'])
    assert got==expected,expected['path']
    current.append(got)
assert len(current)==388
(OUT/'post_audit_input_verification.json').write_text(json.dumps({'verified_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'payload_count':388,'input_changes':0,'matches_original_manifest':True},indent=2)+'\n')
source_archive=BASE/'upload/positive_ranking_result_review.zip'
(OUT/'input').mkdir(exist_ok=True)
shutil.copyfile(source_archive,OUT/'input'/source_archive.name)
shutil.copyfile(INPUT/'manifest.json',OUT/'input/input_manifest.original.json')
script_paths=[OUT/'scripts/check_input_identity.py',OUT/'scripts/run_logged.py',OUT/'scripts/independent_statistics.py',INPUT/'workspace/scripts/step28_positive_ranking_audit.py']
(OUT/'root_executed_script_identities.json').write_text(json.dumps([identity(p) for p in script_paths],indent=2)+'\n')

report_name='S_pos_H_pos_external_review_20261018.zh.txt'
shutil.copyfile(OUT/report_name,DELIVER/report_name)
manifest_path=OUT/'artifact_manifest.json'
members=[]
for p in sorted(OUT.rglob('*')):
    if not p.is_file() or p==manifest_path or 'temporary' in p.relative_to(OUT).parts:
        continue
    assert not p.is_symlink()
    members.append(identity(p,str(p.relative_to(OUT))))
manifest_path.write_text(json.dumps({'note':'All review deliverable files except this manifest itself. Original authorized ZIP retained byte-for-byte under input/. Intermediate rendering cache excluded.','member_count_without_this_manifest':len(members),'files':members},ensure_ascii=False,indent=2)+'\n')
archive=DELIVER/'S_pos_H_pos_external_review_20261018.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for r in members: z.write(OUT/r['path'],r['path'])
    z.write(manifest_path,'artifact_manifest.json')
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert len(z.namelist())==len(members)+1
    assert len(z.namelist())==len(set(z.namelist()))
    for r in members:
        b=z.read(r['path'])
        assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']
    assert z.read('input/positive_ranking_result_review.zip')==source_archive.read_bytes()
sha_lines=[]
for p in (archive,DELIVER/report_name):
    sha_lines.append(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name)
(DELIVER/'S_pos_H_pos_external_review_20261018.sha256').write_text('\n'.join(sha_lines)+'\n')
print(json.dumps({'status':'COMPLETE_REVIEW_BUNDLE_VERIFIED','archive':identity(archive),'zip_members':len(members)+1,
                  'report':identity(DELIVER/report_name),'sha256_file':str(DELIVER/'S_pos_H_pos_external_review_20261018.sha256'),
                  'input_payloads_unchanged':388,'verified_archive_member_payloads':len(members)},ensure_ascii=False,indent=2))
