#!/usr/bin/env python3
"""Administrative report/manifest assembly and byte verification only."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import sys
import time
import zipfile

BASE=Path('/workspace/scratch/2456d4553fd5')
WORK=BASE/'cublas_audit_work'
ROOT=WORK/'input'
OUT=BASE/'cublas_audit_output'
NAME='cublas_workspace_external_review'

def sha(b):return hashlib.sha256(b).hexdigest()
def save(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def ident(p,root=None):
    b=p.read_bytes()
    return {'path':str(p.relative_to(root)) if root else str(p),'bytes':len(b),'sha256':sha(b)}
def read_json(p):return json.loads(p.read_text())
def merge(ranges,last):
    result=[]
    for a,b in sorted(ranges):
        a=max(1,a);b=min(last,b)
        if a>b:continue
        if result and a<=result[-1][1]+1:result[-1][1]=max(result[-1][1],b)
        else:result.append([a,b])
    return result

def main():
    started=time.perf_counter()
    pre={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'command_argv':[sys.executable,*sys.argv],
         'source_identity':ident(Path(__file__)),'python':sys.version,'platform':platform.platform(),'cwd':os.getcwd(),
         'operation':'Assemble already completed static review and mechanically validate delivery bytes',
         'project_code_executed':False,'scientific_computation':False}
    save(OUT/'evidence/assembly/pre_execution.json',pre)
    verification=read_json(OUT/'input_verification.json')
    checks=read_json(OUT/'evidence/field_checks/result.json')
    assert checks['status']=='PASS_STATIC_SOURCE_AND_SAVED_FIELDS'
    operations=[json.loads(line) for line in (OUT/'evidence/operations.jsonl').read_text().splitlines()]
    assert all(row['exit_code']==0 for row in operations)
    members=[*verification['files'],verification['manifest_itself']]
    all_unchanged=[]
    source_zip=Path(verification['input_path'])
    assert ident(source_zip)['sha256']==verification['actual']['sha256']
    with zipfile.ZipFile(source_zip) as z:
        assert z.testzip() is None and len(z.namelist())==111
        for row in members:
            b=z.read(row['path']);assert len(b)==row['bytes'] and sha(b)==row['sha256']
            assert (ROOT/row['path']).read_bytes()==b
            all_unchanged.append({'path':row['path'],'bytes':len(b),'sha256':sha(b),'unchanged':True})
    save(OUT/'evidence/assembly/input_unchanged.json',{'member_count':111,'all_unchanged':True,'zip_crc_passed':True,'files':all_unchanged})
    requested={};logs={};searches={}
    for event in operations:
        for rec in event['records']:
            if event['operation']=='read':
                requested.setdefault(rec['path'],[]).extend(rec['read_ranges'])
                logs.setdefault(rec['path'],[]).append(event['stdout_path'])
            elif event['operation']=='search':
                searches.setdefault(rec['path'],[]).append({'pattern':rec['pattern'],'lines':rec['hit_lines'],'log':event['stdout_path']})
    mechanically_read={r['path'] for r in checks['mechanically_read_inputs']}
    reading=[]
    for row in sorted(members,key=lambda x:x['path']):
        p=row['path'];total=len((ROOT/p).read_bytes().splitlines());ranges=merge(requested.get(p,[]),total)
        scope='identity_only';notes=[]
        if ranges:scope='full_text_review' if ranges==[[1,total]] else 'selected_text_review'
        elif p in mechanically_read:scope='mechanical_fields_or_AST_only'
        elif p in searches:scope='search_hits_only'
        if p.endswith('/qualification/cpu_gate_workspace.json'):
            scope='selected_text_and_complete_mechanical_check'
            notes.append('A combined display truncated some source-list lines. Lines206-252 separately read in full; all source records mechanically checked. Requested ranges are not a full semantic-reading claim.')
        if p=='tests/test_step28_positive_ranking.py':
            notes.append('Complete affected test312-377 separately displayed. Some unchanged context in an earlier combined display was truncated. Six reused tests checked by AST, not rerun or newly audited for mathematics.')
        if p=='scripts/step28_positive_ranking_verify.py':
            notes.append('Entry, preflight, callback/return and main/budget ranges read. Unchanged native middle reused from prior review, not reread in full.')
        if '/review/repair/external/' in p:
            notes.append('Prior38d05b9a review evidence only; its original execution/reading ledger is not this round execution.')
        if p=='package_manifest.json':
            scope='mechanical_manifest_check';notes.append('Parsed for all payload identities; own identity checked separately.')
        if p.startswith('historical_sources/') or p.startswith('prior_repair_sources/') or p=='history/initial_repair.patch':
            notes.append('Old/new byte and patch/AST correspondence only unless a read range is listed. No project source executed.')
        reading.append({'path':p,'bytes':row['bytes'],'sha256':row['sha256'],'identity_verified':True,
            'scope':scope,'total_lines':total,'requested_display_ranges_bounded_to_file':ranges,
            'mechanical_field_or_ast_read':p in mechanically_read,'read_logs':sorted(set(logs.get(p,[]))),
            'searches':searches.get(p,[]),'notes':notes})
    scope_counts=dict(Counter(r['scope'] for r in reading))
    save(OUT/'reading_scope.json',{'input_members':111,'all_identity_verified':True,'scope_counts':scope_counts,
        'subagents_used':False,'principle':'Hash/AST/JSON processing does not imply full semantic reading. Truncated display limitations are marked.',
        'external_dependency_reading':'evidence/primary/read_log.json and web_reading.json','members':reading})
    columns=['path','bytes','sha256','identity_verified','scope','requested_display_ranges','mechanical_field_or_ast_read','notes']
    lines=['\t'.join(columns)]
    for r in reading:
        ranges=','.join(str(a) if a==b else f'{a}-{b}' for a,b in r['requested_display_ranges_bounded_to_file'])
        lines.append('\t'.join([r['path'],str(r['bytes']),r['sha256'],'true',r['scope'],ranges,
                               str(r['mechanical_field_or_ast_read']).lower(),' '.join(r['notes'])]))
    (OUT/'reading_scope.tsv').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (OUT/'input_members.tsv').write_text('path\tbytes\tsha256\n'+''.join(f"{r['path']}\t{r['bytes']}\t{r['sha256']}\n" for r in sorted(members,key=lambda x:x['path'])),encoding='utf-8')
    attempts=[]
    for number,directory in ((1,'field_checks_attempt01'),(2,'field_checks_attempt02'),(3,'field_checks')):
        data=read_json(OUT/'evidence'/directory/'result.json')
        operation=data['operation'];script=(OUT/'evidence'/directory/'check_workspace.py') if number<3 else WORK/'check_workspace.py'
        assert sha(script.read_bytes())==operation['source_sha256']
        assert operation['exit_code']==(1 if number<3 else 0)
        attempts.append({'attempt':number,'directory':'evidence/'+directory,'status':data['status'],'operation':operation})
    primary=read_json(OUT/'evidence/primary/result.json')
    assert primary['exit_code']==0
    ledger={'this_review_scientific_execution':{'project_tests':0,'project_modules_imported':0,'new_mathematical_or_numerical_experiments':0,
            'optimizer_calls':0,'GPU_calls':0,'BGE_weights_Memory_loads':0,'project_Linux_connections':0,
            'formal_data_reads':0,'network_installations':0,'subagents':0},
        'provided_project_ledger':checks['accounting'],
        'provided_attempt05_resource_fields':{k:checks['test_evidence']['saved_attempt05_result_without_source_list'][k] for k in ('peak_rss_bytes','peak_work_bytes','elapsed_seconds','torch')},
        'own_administrative_records':{'read_hash_invocations':len(operations),'read_hash_wall_seconds':sum(e['wall_seconds'] for e in operations),
            'read_hash_peak_process_maxrss_bytes':max(e['process_maxrss_bytes'] for e in operations),'read_hash_failures':0,
            'field_checks':attempts,'field_checks_total_wall_seconds':sum(a['operation']['wall_seconds'] for a in attempts),
            'field_checks_peak_process_maxrss_bytes':max(a['operation']['process_maxrss_bytes'] for a in attempts),
            'read_only_primary_fetch':primary},
        'mechanical_repairs':'evidence/field_check_repairs.json',
        'accounting_scope':'Listed helper process measurements only. No total-chat, editorial or transfer resource measurement is claimed. Two administrative failures do not add project scientific failures or modify52/300.',
        'packaging_measurement':'Internal evidence/assembly/result.json before packaging; final validation timing in external delivery.json.'}
    save(OUT/'resource_ledger.json',ledger)
    scripts=OUT/'evidence/scripts';scripts.mkdir(parents=True,exist_ok=True)
    for name in ('audit_io.py','check_workspace.py','read_primary.py','assemble.py'):shutil.copyfile(WORK/name,scripts/name)
    assert sha((scripts/'audit_io.py').read_bytes())==operations[0]['script_sha256']
    assert sha((scripts/'read_primary.py').read_bytes())==primary['operation']['source_sha256']
    save(OUT/'evidence/assembly/result.json',{'status':'ASSEMBLED_AND_INPUT_UNCHANGED','input_members':111,
        'source_count':40,'reading_scope_counts':scope_counts,'field_check_failures_retained':2,'final_field_check_exit_code':0,
        'script_identities_match_executed_versions':True,'report_bytes':(OUT/'report.zh.txt').stat().st_size,
        'wall_seconds_before_packaging':time.perf_counter()-started,
        'process_maxrss_bytes_before_packaging':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'scientific_computation':False})
    payloads=[p for p in sorted(OUT.rglob('*')) if p.is_file() and p!=OUT/'manifest.json']
    save(OUT/'manifest.json',{'type':'cublas_workspace_targeted_static_review','input':verification['actual'],
        'input_name':'cublas_workspace_review.zip','input_repackaged':False,'payload_count_excluding_manifest':len(payloads),
        'self_reference_policy':'Manifest excludes itself; external member TSV includes all ZIP members.',
        'files':[ident(p,OUT) for p in payloads]})
    files=[p for p in sorted(OUT.rglob('*')) if p.is_file()]
    bundle=BASE/(NAME+'.zip')
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in files:
            info=zipfile.ZipInfo(str(p.relative_to(OUT)),date_time=(2026,10,18,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16
            z.writestr(info,p.read_bytes(),compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
    listed=[ident(p,OUT) for p in files]
    tsv=BASE/(NAME+'_members.tsv')
    tsv.write_text('path\tbytes\tsha256\n'+''.join(f"{r['path']}\t{r['bytes']}\t{r['sha256']}\n" for r in listed),encoding='utf-8')
    external_report=BASE/(NAME+'.zh.txt');shutil.copyfile(OUT/'report.zh.txt',external_report)
    with zipfile.ZipFile(bundle) as z:
        assert z.testzip() is None and z.namelist()==[r['path'] for r in listed]
        for row in listed:
            b=z.read(row['path']);assert len(b)==row['bytes'] and sha(b)==row['sha256']
        manifest=json.loads(z.read('manifest.json'))
        assert manifest['files']==[r for r in listed if r['path']!='manifest.json']
        assert z.read('report.zh.txt')==external_report.read_bytes()
    tsv_lines=tsv.read_text().splitlines();assert len(tsv_lines)==len(listed)+1
    for line,row in zip(tsv_lines[1:],listed):assert line.split('\t')==[row['path'],str(row['bytes']),row['sha256']]
    delivery={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'VERIFIED_DELIVERABLE',
        'zip':{**ident(bundle),'members':len(listed)},'members_tsv':ident(tsv),'report':ident(external_report),
        'input':verification['actual'],'input_unchanged_members':111,'uncompressed_output_bytes':sum(r['bytes'] for r in listed),
        'verification':{'zip_crc':True,'all_member_identity':True,'internal_manifest':True,'external_members_tsv':True,'standalone_report_identical':True},
        'mechanical_field_failures_retained':2,'final_mechanical_field_status':checks['status'],
        'assembly_wall_seconds_through_verification':time.perf_counter()-started,
        'assembly_process_maxrss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'source_identity':pre['source_identity'],'exit_code':0,'scientific_experiments_performed':0,'read_scope_counts':scope_counts}
    save(BASE/(NAME+'_delivery.json'),delivery)
    print(json.dumps(delivery,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
