#!/usr/bin/env python3
"""Verify completed review artifacts and package original bytes plus evidence.

Packaging is an artifact operation, not a new scientific audit or experiment.
All recorded scientific runs retain their exact original streams and snapshots.
"""
import argparse
import collections
import datetime as dt
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

WORK=Path(__file__).resolve().parents[2]
E=WORK/'evidence'
P=WORK/'input'/'project'
DELIVER=WORK.parent/'deliverables'
ORIGINAL=WORK.parent/'upload'/'result_review(1).zip'

def sha(data): return hashlib.sha256(data).hexdigest()
def read(p): return json.loads(p.read_text())
def write(p,data):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')

def validate():
    blob=ORIGINAL.read_bytes()
    assert len(blob)==8729802 and sha(blob)=='87ff5ae41173ea9f68172736a11bd90bf54846fc4a2f0843b12edaa2965b9f7c'
    received=read(E/'input/received_members.json')
    assert len(received)==613
    for rec in received:
        path=WORK/'input'/rec['path']
        payload=path.read_bytes()
        assert len(payload)==rec['bytes'] and sha(payload)==rec['sha256'],rec['path']
    inv=read(WORK/'input/source_inventory.json')
    assert inv['file_count']==len(inv['files'])==612
    assert sum(r['bytes'] for r in inv['files'])==inv['total_bytes']==20199522
    assert set(r['path'] for r in received)=={r['path'] for r in inv['files']}|{'source_inventory.json'}
    for rec in inv['files']:
        payload=(WORK/'input'/rec['path']).read_bytes()
        assert len(payload)==rec['bytes'] and sha(payload)==rec['sha256'],rec['path']
    er=read(E/'agents/er_source/read_coverage.json')
    logit=read(E/'agents/logit_source/read_coverage.json')
    er_lookup={r['path']:r for r in er['full_read_sources']}
    roots={'low':P,'logit':P/'reports/seller_alias_continual/20261002/logit_weight_execution/20261002_153200/workspace'}
    manifests={'low':P/'reports/seller_alias_continual/20261002/er_low_execution/20261002_124018/job/run/manifest.json','logit':roots['logit']/'reports/job/run/manifest.json'}
    coverage=[]
    logit_lookup={r['path']:r for r in logit['logit_source_records']}
    for study in ('low','logit'):
        records=read(manifests[study])['source_files']
        assert len(records)==(31 if study=='low' else 34)
        for rec in records:
            path=roots[study]/rec['path']
            payload=path.read_bytes()
            assert len(payload)==rec['bytes'] and sha(payload)==rec['sha256']
            if study=='low':
                cov=er_lookup[rec['path']]
                assert cov['full_text_semantic_read'] and cov['sha256']==rec['sha256']
                reader='er_source:full'
            else:
                cov=logit_lookup[rec['path']]
                assert cov['sha256']==rec['sha256']
                if cov['full_read_by_logit_source_agent']:
                    reader='logit_source:full'
                else:
                    other=er_lookup[rec['path']]
                    assert other['full_text_semantic_read'] and other['sha256']==rec['sha256']
                    reader='er_source:identical-content full read reused explicitly'
            coverage.append({'study':study,'path':str(path.relative_to(WORK/'input')),'bytes':rec['bytes'],'sha256':rec['sha256'],'full_content_review_responsibility':reader})
    write(E/'combined_frozen_source_coverage.json',{'records':coverage,'counts':{'low':31,'logit':34},'note':'This validates the explicit reading records and version mapping; hash equality alone is not semantic review.'})
    run_rows=[]
    incomplete=[]
    for f in sorted(E.rglob('run.json')):
        m=read(f)
        if 'exit_code' not in m:
            incomplete.append(str(f.relative_to(E)))
            continue
        for name,rec in m['streams'].items():
            payload=(f.parent/name).read_bytes()
            assert len(payload)==rec['bytes'] and sha(payload)==rec['sha256'],str(f)+'/'+name
        for rec in m.get('sources',[]):
            payload=(f.parent/rec['snapshot']).read_bytes()
            assert len(payload)==rec['bytes'] and sha(payload)==rec['sha256'],rec['snapshot']
        run_rows.append({'record':str(f.relative_to(E)),'label':m['label'],'argv':m['argv'],'cwd':m['cwd'],'started_at_utc':m['started_at_utc'],'ended_at_utc':m['ended_at_utc'],'exit_code':m['exit_code'],'elapsed_seconds':m['elapsed_seconds'],'streams':m['streams'],'source_snapshots':m.get('sources',[])})
    write(E/'execution_index.json',{'complete_recorded_runs':len(run_rows),'failed_original_runs':sum(r['exit_code']!=0 for r in run_rows),'runs':run_rows,'in_progress_records_excluded':incomplete,'note':'Execution counts include review, source/metadata preparation and repeat attempts; they are not experiment or training-seed counts.'})
    # The first report assembler did not parse. Preserve its exact minimal repair.
    old=E/'runs/root_assemble_review_v1/source_snapshots/00_assemble_review.py'
    now=E/'src/assemble_review.py'
    if old.exists():
        diff=''.join(difflib.unified_diff(old.read_text().splitlines(True),now.read_text().splitlines(True),fromfile='assemble_review_failed_v1.py',tofile='assemble_review_corrected_v2.py'))
        (E/'root_report_assembler_v1_to_v2.diff').write_text(diff)
    report=(DELIVER/'REVIEW.zh.md').read_bytes()
    assert report==(E/'REVIEW.zh.md').read_bytes()
    assert '{{' not in report.decode()
    qa=read(E/'agents/audit_replay/final_report_review.json')
    # The exact QA structure is retained; confirm its reviewed identity appears.
    assert sha(report) in json.dumps(qa), 'Report changed after final content review'
    result={'input_original_bytes_unchanged':True,'input_members_unchanged':613,'root_manifest_payloads_verified':612,'frozen_source_content_coverage_verified':{'ER':31,'LOGIT':34},'recorded_completed_runs':len(run_rows),'failed_original_runs_retained':sum(r['exit_code']!=0 for r in run_rows),'in_progress_records_excluded':incomplete,'report':{'bytes':len(report),'sha256':sha(report)},'checked_at_utc':dt.datetime.now(dt.timezone.utc).isoformat()}
    write(E/'final_validation.json',result)
    return result

def pack():
    result=validate()
    assert not result['in_progress_records_excluded'],'Do not package incomplete run records'
    members={}
    for f in sorted(E.rglob('*')):
        if f.is_file(): members['evidence/'+str(f.relative_to(E))]=f
    members['input/result_review.zip']=ORIGINAL
    members['REVIEW.zh.md']=DELIVER/'REVIEW.zh.md'
    members['README.zh.md']=WORK/'README.zh.md'
    manifest={'created_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'scope':'All members except this manifest and its separate digest text, avoiding recursive self-hashes. The outer archive digest is delivered separately.','file_count':len(members),'total_bytes':sum(p.stat().st_size for p in members.values()),'files':[{'path':n,'bytes':p.stat().st_size,'sha256':sha(p.read_bytes())} for n,p in sorted(members.items())]}
    manifest_path=WORK/'EVIDENCE_MANIFEST.json'
    write(manifest_path,manifest)
    manifest_sha=sha(manifest_path.read_bytes())
    digest_path=WORK/'EVIDENCE_MANIFEST.sha256'
    digest_path.write_text(manifest_sha+'  EVIDENCE_MANIFEST.json\n')
    members['EVIDENCE_MANIFEST.json']=manifest_path
    members['EVIDENCE_MANIFEST.sha256']=digest_path
    target=DELIVER/'result_review_evidence.zip'
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name,path in sorted(members.items()): z.write(path,name)
    with zipfile.ZipFile(target) as z:
        assert len(z.infolist())==len(members)==len(set(z.namelist()))
        assert z.testzip() is None
        for rec in manifest['files']:
            payload=z.read(rec['path'])
            assert len(payload)==rec['bytes'] and sha(payload)==rec['sha256'],rec['path']
        assert sha(z.read('EVIDENCE_MANIFEST.json'))==manifest_sha
        assert sha(z.read('input/result_review.zip'))=='87ff5ae41173ea9f68172736a11bd90bf54846fc4a2f0843b12edaa2965b9f7c'
        assert z.read('REVIEW.zh.md')==(DELIVER/'REVIEW.zh.md').read_bytes()
    receipt={'archive':str(target),'bytes':target.stat().st_size,'members':len(members),'sha256':sha(target.read_bytes()),'all_member_crc_size_sha_verified':True,'original_input_sha256':sha(ORIGINAL.read_bytes()),'manifest':{'bytes':manifest_path.stat().st_size,'sha256':manifest_sha,'enumerated_payload_members':len(manifest['files'])},'report':result['report'],'completed_recorded_runs':result['recorded_completed_runs'],'failed_original_runs_retained':result['failed_original_runs_retained'],'packaged_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'packaging_command':'python -B '+str(Path(__file__).resolve())+' --pack','note':'This receipt is outside the archive to avoid a recursive outer-archive self-hash.'}
    write(DELIVER/'delivery_receipt.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False,indent=2))

if __name__=='__main__':
    a=argparse.ArgumentParser()
    a.add_argument('--pack',action='store_true')
    args=a.parse_args()
    if args.pack: pack()
    else: print(json.dumps(validate(),ensure_ascii=False,indent=2))
