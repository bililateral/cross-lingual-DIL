#!/usr/bin/env python3
"""Administrative assembly only. No project imports, tests, model or math execution."""
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

BASE = Path('/workspace/scratch/2456d4553fd5')
WORK = BASE / 'repair_audit_work'
ROOT = WORK / 'input'
OUT = BASE / 'repair_audit_output'
NAME = 'positive_ranking_repair_external_review'

def digest(data):
    return hashlib.sha256(data).hexdigest()

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def ident(path, relative_to=None):
    content=path.read_bytes()
    return {'path':str(path.relative_to(relative_to)) if relative_to else str(path),
            'bytes':len(content), 'sha256':digest(content)}

def merged(ranges, maximum):
    values=[]
    for lo,hi in sorted(ranges):
        lo=max(1,lo); hi=min(maximum,hi)
        if lo>hi:
            continue
        if values and lo<=values[-1][1]+1:
            values[-1][1]=max(values[-1][1],hi)
        else:
            values.append([lo,hi])
    return values

def main():
    started=time.perf_counter()
    process_info={'started_utc':datetime.now(timezone.utc).isoformat(),
                  'command_argv':[sys.executable,*sys.argv], 'cwd':os.getcwd(),
                  'python':sys.version, 'platform':platform.platform(),
                  'source_identity':ident(Path(__file__)),
                  'operation':'Read prior static evidence; aggregate inventory/accounting; verify unchanged input; create and validate delivery ZIP',
                  'project_module_execution':False, 'scientific_computation':False,
                  'new_mathematical_input':None}
    write_json(OUT/'evidence/assembly/pre_execution.json',process_info)
    verification=json.loads((OUT/'input_verification.json').read_text())
    fields=json.loads((OUT/'evidence/field_checks/result.json').read_text())
    field_pre=json.loads((OUT/'evidence/field_checks/pre_execution.json').read_text())
    events=[json.loads(line) for line in (OUT/'evidence/mechanical_operations.jsonl').read_text().splitlines()]
    upstream=json.loads((OUT/'evidence/upstream/fetch_result.json').read_text())
    supplied=[*verification['files'],verification['manifest_itself']]
    with zipfile.ZipFile(verification['input_path']) as z:
        input_crc=z.testzip()
        assert input_crc is None
        assert len(z.namelist())==94
        unchanged=[]
        for row in supplied:
            original=z.read(row['path'])
            assert len(original)==row['bytes'] and digest(original)==row['sha256']
            assert (ROOT/row['path']).read_bytes()==original
            unchanged.append({'path':row['path'],'bytes':len(original),'sha256':digest(original),'unchanged':True})
    assert ident(Path(verification['input_path']))['sha256']==verification['actual']['sha256']
    write_json(OUT/'evidence/assembly/input_unchanged.json',{'all_94_unchanged':True,'zip_crc_pass':True,'files':unchanged})
    read_ranges={}
    logs={}
    for event in events:
        if event['operation']!='read':
            continue
        for row in event['records']:
            read_ranges.setdefault(row['path'],[]).extend(row['read_ranges'])
            logs.setdefault(row['path'],[]).append(event['stdout_path'])
    mechanically_read={row['path'] for row in fields['mechanically_read_inputs']}
    source_paths={row['path'] for row in fields['source_binding']['current_source_files']}
    changed_paths={row['path'] for row in fields['four_file_patch']}
    reading=[]
    for row in sorted(supplied,key=lambda x:x['path']):
        path=row['path']; lines=len((ROOT/path).read_bytes().splitlines())
        ranges=merged(read_ranges.get(path,[]),lines)
        scope='identity_only'
        if ranges:
            scope='full_text_review' if ranges==[[1,lines]] else 'selected_text_review'
        elif path in mechanically_read:
            scope='mechanical_only_no_semantic_review'
        notes=[]
        if path.endswith('/review/partial_page.json'):
            scope='background_excerpt_display_truncated'
            notes.append('Requested output was truncated by the tool display; no claim of complete semantic reading or recovery of the interrupted review.')
        if path.endswith('/s_improvement/implementation_scope.zh.txt'):
            scope='background_selected_review'
            notes.append('Background context only; combined display was partially truncated. Not an independent full re-review of the old implementation contract.')
        if path.endswith('/s_improvement/review/primary_disposition.zh.txt'):
            notes.append('Background selected review only; lines 1-14 separately redisplayed. Combined earlier output partially truncated.')
        if path=='tests/test_step28_positive_ranking.py':
            notes.append('Focused on changed diagnostics/preflight and affected regressions. Unchanged math tests compared mechanically, not re-experimented.')
        if path in source_paths:
            notes.append('Current 40-source binding member; identity and import closure mechanically checked.')
        if path in changed_paths or path.startswith('historical_sources/'):
            notes.append('In-memory patch/AST and old/new byte identity checked; supplied project code not executed.')
        if path=='package_manifest.json':
            notes.append('Manifest structure parsed for all member identities; bytewise/field reading only, no separate prose review.')
            scope='mechanical_manifest_review'
        reading.append({'path':path,'bytes':row['bytes'],'sha256':row['sha256'],
                        'identity_verified':True,'text_total_lines':lines,'scope':scope,
                        'displayed_read_ranges':ranges,'mechanical_field_or_ast_read':path in mechanically_read,
                        'read_log_paths':sorted(set(logs.get(path,[]))),'notes':notes})
    reading_result={'project_input_members':94,'identity_checked_members':94,
                    'scope_counts':dict(Counter(r['scope'] for r in reading)),
                    'principle':'Hash/AST/field inspection does not imply full semantic review; ranges are bounded by file length. Original commands and stdout are preserved.',
                    'additional_dependency_reference':'evidence/upstream/reading_and_retrieval.json',
                    'subagents_used_this_review':False,'members':reading}
    write_json(OUT/'reading_scope.json',reading_result)
    header='path\tbytes\tsha256\tidentity_verified\tscope\tdisplayed_read_ranges\tmechanical_field_or_ast_read\tnotes\n'
    rows=[]
    for row in reading:
        ranges=','.join(str(lo) if lo==hi else f'{lo}-{hi}' for lo,hi in row['displayed_read_ranges'])
        rows.append('\t'.join([row['path'],str(row['bytes']),row['sha256'],'true',row['scope'],ranges,
                              str(row['mechanical_field_or_ast_read']).lower(),' '.join(row['notes'])]))
    (OUT/'reading_scope.tsv').write_text(header+'\n'.join(rows)+'\n',encoding='utf-8')
    (OUT/'input_members.tsv').write_text('path\tbytes\tsha256\n'+''.join(f"{r['path']}\t{r['bytes']}\t{r['sha256']}\n" for r in sorted(supplied,key=lambda x:x['path'])),encoding='utf-8')
    field_op=fields['operation']
    ledger={'review_execution':{'new_math_or_numerical_experiments':0,'project_modules_imported':0,
             'project_cli_or_tests_run':0,'project_optimizer_calls':0,'gpu_calls':0,'model_or_memory_loads':0,
             'project_linux_connections':0,'formal_data_reads':0,'network_installations':0,'subagent_delegations':0},
             'provided_project_evidence':fields['saved_accounting_fields'],
             'provided_test_composition':fields['test_evidence_composition'],
             'own_recorded_administrative_operations':{
               'static_audit_invocations':len(events),'static_audit_wall_seconds':sum(e['wall_seconds'] for e in events),
               'static_audit_peak_process_maxrss_bytes':max(e['process_maxrss_bytes'] for e in events),
               'static_audit_nonzero_exits':[e['sequence'] for e in events if e['exit_code']!=0],
               'field_check':field_op,'dependency_read_only_fetch':upstream,
               'navigation_logs':['evidence/navigation_001.json','evidence/navigation_002.json','evidence/navigation_003.json']},
             'accounting_scope':'Only listed administrative subprocess records are measured. Metadata rereads, editorial reasoning, UI/transfer and total chat wall time are not claimed as an aggregate CPU benchmark. These operations are not project CPU qualification and do not change the supplied 49/300 second ledger.',
             'failures_and_repairs':'No new project/scientific run or failure. Static/read scripts succeeded. Failed webpage retrievals and display truncation are retained separately; not counted as project failures. No executed scientific repair.',
             'assembly_timing':'Assembly metadata is in evidence/assembly; final ZIP validation timing is in the external delivery JSON.'}
    write_json(OUT/'resource_ledger.json',ledger)
    for name in ('static_audit.py','check_repair_fields.py','fetch_primary_source.py','assemble_delivery.py'):
        dest=OUT/'evidence/scripts'/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(WORK/name,dest)
    assert digest((OUT/'evidence/scripts/static_audit.py').read_bytes())==events[0]['script_sha256']
    assert digest((OUT/'evidence/scripts/check_repair_fields.py').read_bytes())==field_pre['source_sha256']
    assert digest((OUT/'evidence/scripts/fetch_primary_source.py').read_bytes())==upstream['source_sha256']
    result={'status':'PASS_ADMINISTRATIVE_ASSEMBLY','input_members_unchanged':94,
            'source_bindings':fields['source_binding']['source_count'],
            'reading_scope_counts':reading_result['scope_counts'],
            'script_identity_checked_against_pre_execution':True,
            'report_bytes':(OUT/'report.zh.txt').stat().st_size,
            'wall_seconds_before_manifest_and_zip':time.perf_counter()-started,
            'process_maxrss_bytes_before_packaging':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'scientific_computation':False,'project_code_executed':False,
            'manifest_self_hash_policy':'Internal manifest excludes itself. External members TSV includes every ZIP member including manifest. ZIP identity is outside ZIP.'}
    write_json(OUT/'evidence/assembly/result.json',result)
    payloads=[p for p in sorted(OUT.rglob('*')) if p.is_file() and p!=OUT/'manifest.json']
    manifest={'type':'targeted_static_repair_review_deliverable','input':verification['actual'],
              'input_filename':'positive_ranking_repair_review.zip','input_repackaged':False,
              'payload_count_excluding_this_manifest':len(payloads),'files':[ident(p,OUT) for p in payloads]}
    write_json(OUT/'manifest.json',manifest)
    files=[p for p in sorted(OUT.rglob('*')) if p.is_file()]
    bundle=BASE/(NAME+'.zip')
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in files:
            item=zipfile.ZipInfo(str(p.relative_to(OUT)),date_time=(2026,10,18,0,0,0))
            item.compress_type=zipfile.ZIP_DEFLATED; item.external_attr=0o100644<<16
            z.writestr(item,p.read_bytes(),compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
    members=[ident(p,OUT) for p in files]
    tsv=BASE/(NAME+'_members.tsv')
    tsv.write_text('path\tbytes\tsha256\n'+''.join(f"{r['path']}\t{r['bytes']}\t{r['sha256']}\n" for r in members),encoding='utf-8')
    external_report=BASE/(NAME+'.zh.txt'); shutil.copyfile(OUT/'report.zh.txt',external_report)
    with zipfile.ZipFile(bundle) as z:
        assert z.testzip() is None
        assert z.namelist()==[r['path'] for r in members]
        for row in members:
            data=z.read(row['path'])
            assert len(data)==row['bytes'] and digest(data)==row['sha256']
        internal=json.loads(z.read('manifest.json'))
        assert internal['files']==[r for r in members if r['path']!='manifest.json']
        assert z.read('report.zh.txt')==external_report.read_bytes()
    parsed=tsv.read_text().splitlines()
    assert len(parsed)-1==len(members)
    for row,line in zip(members,parsed[1:]):
        assert line.split('\t')==[row['path'],str(row['bytes']),row['sha256']]
    delivery={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'VERIFIED_DELIVERABLE',
              'zip':{**ident(bundle),'members':len(members)},'members_tsv':ident(tsv),'report':ident(external_report),
              'input':verification['actual'],'input_member_unchanged_count':94,
              'validation':{'zip_crc':True,'all_zip_members_identity':True,'internal_manifest':True,
                            'external_members_tsv':True,'external_report_equals_internal':True},
              'uncompressed_output_bytes':sum(r['bytes'] for r in members),
              'assembly_wall_seconds_through_verification':time.perf_counter()-started,
              'assembly_process_maxrss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
              'source_identity':process_info['source_identity'],'exit_code':0,
              'not_a_project_run':True,'scientific_experiments_performed':0,
              'read_scope_counts':reading_result['scope_counts']}
    write_json(BASE/(NAME+'_delivery.json'),delivery)
    print(json.dumps(delivery,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
