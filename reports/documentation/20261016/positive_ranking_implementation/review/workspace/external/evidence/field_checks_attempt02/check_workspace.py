#!/usr/bin/env python3
"""Mechanical ZIP-member source/patch/AST and saved-field checks; no project execution."""
import ast
import contextlib
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import resource
import sys
import time
import traceback

BASE=Path('/workspace/scratch/2456d4553fd5')
ROOT=BASE/'cublas_audit_work/input'
OUT=BASE/'cublas_audit_output/evidence/field_checks'
Q='reports/documentation/20261016/positive_ranking_implementation/qualification/'
READ={}

def sha(b):return hashlib.sha256(b).hexdigest()
def read(path):
    b=(ROOT/path).read_bytes()
    READ[path]={'path':path,'bytes':len(b),'sha256':sha(b)}
    return b
def parsed(path):return json.loads(read(path))
def identity(path):
    read(path);return dict(READ[path])
def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def dump(node):return ast.dump(node,include_attributes=False)
def funcs(content):
    tree=ast.parse(content); result={}
    for node in tree.body:
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):result[node.name]=node
        elif isinstance(node,ast.ClassDef):
            for item in node.body:
                if isinstance(item,(ast.FunctionDef,ast.AsyncFunctionDef)):result[node.name+'.'+item.name]=item
    return result
def changes(a,b):
    x,y=funcs(a),funcs(b)
    return {'changed':[n for n in sorted(x.keys()&y.keys()) if dump(x[n])!=dump(y[n])],
            'added':sorted(y.keys()-x.keys()),'removed':sorted(x.keys()-y.keys())}
def blob_sha(b):return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()

def apply_patch_in_memory(path,base_for,target_for):
    lines=read(path).decode().splitlines(keepends=True); chunks=[]; current=[]
    for line in lines:
        if line.startswith('diff --git '):
            if current:chunks.append(current)
            current=[line]
        else:current.append(line)
    if current:chunks.append(current)
    records=[]
    for chunk in chunks:
        relative=re.fullmatch(r'diff --git a/(.+) b/(.+)\n?',chunk[0]).group(2)
        previous=base_for(relative);target=target_for(relative)
        old=read(previous);wanted=read(target); oldlines=old.decode().splitlines(keepends=True)
        prefixes=re.fullmatch(r'index ([0-9a-f]+)\.\.([0-9a-f]+) 100644\n?',chunk[1]).groups()
        rebuilt=[]; cursor=0; i=4; count=0
        while i<len(chunk):
            match=re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@',chunk[i]);assert match
            oldstart,oldcount,newstart,newcount=match.groups()
            oldstart=int(oldstart)-1; newstart=int(newstart)-1
            oldcount=int(oldcount or 1);newcount=int(newcount or 1)
            assert oldstart>=cursor
            rebuilt.extend(oldlines[cursor:oldstart]);cursor=oldstart
            assert len(rebuilt)==newstart
            i+=1;oldseen=0;newseen=0
            while i<len(chunk) and not chunk[i].startswith('@@ '):
                line=chunk[i];tag=line[0];payload=line[1:]
                assert tag in ' +-','Unexpected patch marker'
                if tag in ' -':
                    assert oldlines[cursor]==payload,(relative,cursor+1)
                    cursor+=1;oldseen+=1
                if tag in ' +':rebuilt.append(payload);newseen+=1
                i+=1
            assert (oldseen,newseen)==(oldcount,newcount)
            count+=1
        rebuilt.extend(oldlines[cursor:]); actual=''.join(rebuilt).encode()
        assert actual==wanted
        assert blob_sha(old).startswith(prefixes[0]) and blob_sha(wanted).startswith(prefixes[1])
        records.append({'path':relative,'base':identity(previous),'target':identity(target),'hunks':count,
                        'exact_reconstruction':True,'git_blob_prefixes_match':True,'function_ast_changes':changes(old,wanted)})
    return records

def rhs_string(tree,name):
    node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
    values=[n.value for n in ast.walk(node.value) if isinstance(n,ast.Constant) and isinstance(n.value,str)]
    assert len(values)==1
    return values[0]

def source_closure():
    run='scripts/step28_positive_ranking_run.py'; tree=ast.parse(read(run)); fn=funcs(read(run))['sources']
    initial=next(n.value for n in fn.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='paths' for t in n.targets))
    paths=set(ast.literal_eval(initial.generators[0].iter));todo=list(paths)
    while todo:
        current=todo.pop()
        for node in ast.walk(ast.parse(read(current))):
            names=[n.name for n in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) else []
            for name in names:
                if name and name.startswith('step28_'):
                    child='scripts/'+name+'.py'
                    if child not in paths:paths.add(child);todo.append(child)
    paths.update('schema/'+n.value for n in ast.walk(fn) if isinstance(n,ast.Constant) and isinstance(n.value,str) and n.value.startswith('step28_') and n.value.endswith('_policy.json'))
    paths.update(rhs_string(tree,name) for name in ('POLICY','AUTHORIZATION','CONTRACT'))
    paths.add(rhs_string(ast.parse(read('scripts/step28_h_consistency_run.py')),'BASELINE'))
    paths.add('scripts/run_step28_positive_ranking_linux_20261018.sh')
    return sorted(paths)

def records_for(paths,replace):
    return [{'path':p,'bytes':len(read(replace.get(p,p))),'sha256':sha(read(replace.get(p,p)))} for p in paths]

def resolve_saved(row):
    path=row.get('path',row.get('local'))
    possibilities=[path,'prior_repair_sources/'+path,'historical_sources/'+path,
        Q+'cpu/attempt01/sources/'+path,Q+'cpu/attempt02/sources/'+path]
    matches=[]
    for candidate in possibilities:
        if (ROOT/candidate).is_file():
            b=read(candidate)
            if len(b)==row['bytes'] and sha(b)==row['sha256']:matches.append(candidate)
    assert matches,('No corresponding supplied source bytes',path,row['sha256'])
    return {'logical_path':path,'bytes':row['bytes'],'sha256':row['sha256'],'matches_supplied_paths':matches}

def reference(row):
    actual=identity(row['path']);assert actual=={k:row[k] for k in ('path','bytes','sha256')}
    return {**actual,'matches':True}

def main():
    started=time.perf_counter(); OUT.mkdir(parents=True,exist_ok=True)
    pre={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'command_argv':[sys.executable,*sys.argv],
         'source_sha256':sha(Path(__file__).read_bytes()),'source_bytes':Path(__file__).stat().st_size,
         'python':sys.version,'platform':platform.platform(),'cwd':os.getcwd(),
         'operation':'In-memory patch/AST/source-identity and saved fields only',
         'project_code_executed':False,'scientific_computation':False}
    save(OUT/'pre_execution.json',pre)
    result={};stdout=io.StringIO();stderr=io.StringIO();status=0
    try:
        with contextlib.redirect_stdout(stdout):
            latest=['scripts/step28_positive_ranking_verify.py','tests/test_step28_positive_ranking.py']
            oldest=['scripts/step28_positive_ranking.py','scripts/step28_positive_ranking_run.py',*latest]
            new_patch=apply_patch_in_memory('changes.patch',lambda p:'prior_repair_sources/'+p,lambda p:p)
            old_patch=apply_patch_in_memory('history/initial_repair.patch',lambda p:'historical_sources/'+p,
                lambda p:'prior_repair_sources/'+p if p in latest else p)
            assert {r['path'] for r in new_patch}==set(latest)
            assert {r['path'] for r in old_patch}==set(oldest)
            assert new_patch[0]['function_ast_changes']=={'changed':['native_preflight'],'added':[],'removed':[]}
            assert new_patch[1]['function_ast_changes']=={'changed':['PositiveRankingTests.test_native_continuation_preflight'],'added':[],'removed':[]}
            result['patch_chain']={'38d05b9a_to_d007230b':new_patch,'d914f43c_to_38d05b9a':old_patch,
                                  'commit_scope':'Submitted commit labels; supplied bytes and Git blob prefixes verified, no project Git tree or host accessed.'}
            paths=source_closure();assert len(paths)==40
            current=records_for(paths,{})
            prior=records_for(paths,{p:'prior_repair_sources/'+p for p in latest})
            historical=records_for(paths,{p:'historical_sources/'+p for p in oldest})
            gate0=parsed(Q+'cpu_gate.json');gate4=parsed(Q+'cpu_gate_repair.json');gate5=parsed(Q+'cpu_gate_workspace.json')
            reports={n:parsed(Q+f'cpu/attempt{n:02d}/result/result.json') for n in range(1,6)}
            assert current==gate5['source_files']==reports[5]['source_files']
            assert prior==gate4['source_files']==reports[4]['source_files']
            assert historical==gate0['source_files']==reports[3]['source_files']
            differences=[a['path'] for a,b in zip(current,prior) if a!=b]
            assert differences==sorted(latest)
            saved_matches={f'attempt{n:02d}':[resolve_saved(row) for row in reports[n]['source_files']] for n in range(1,6)}
            result['source_bindings']={'count':40,'current':current,'prior_repair':prior,'historical':historical,
                'current_gate_and_attempt05_match':True,'prior_gate_and_attempt04_match':True,'historical_gate_and_attempt03_match':True,
                'current_changed_paths':differences,'current_unchanged_count':38,'all_attempt_source_matches':saved_matches}
            references=[]
            for g in (gate4,gate5):
                for key in ('inherited','executed','wallclock'):references.append(reference(g[key]))
            for a in gate0['attempts']:
                references.extend(reference(a[k]) for k in ('wallclock','result'))
            result['gate_reference_identities']=references
            test_path='tests/test_step28_positive_ranking.py';now_tests=funcs(read(test_path));before_tests=funcs(read('prior_repair_sources/'+test_path))
            reused=gate5['reused_tests'];executed=gate5['reexecuted_tests']
            assert len(reused)==6 and executed==['test_native_continuation_preflight']
            assert set(reused).isdisjoint(executed)
            qualified=set(reused+executed)
            assert qualified==set(gate5['qualified_tests'])==set(gate4['qualified_tests'])
            run_tree=ast.parse(read('scripts/step28_positive_ranking_run.py'))
            literal_tests=ast.literal_eval(next(n.value for n in run_tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CPU_TESTS' for t in n.targets)))
            actual_tests={n.split('.')[-1] for n in now_tests if n.startswith('PositiveRankingTests.test_')}
            assert qualified==set(literal_tests)==actual_tests
            reuse_rows=[]
            for name in reused:
                key='PositiveRankingTests.'+name;now=dump(now_tests[key]);before=dump(before_tests[key]);assert now==before
                reuse_rows.append({'name':name,'normalized_ast_sha256':sha(now.encode()),'unchanged':True})
            assert reports[5]['qualified_tests']==reports[5]['selected_tests']==executed
            assert reports[5]['status']=='PASS_HANDWRITTEN_ONLY' and reports[5]['mode']=='cpu'
            assert [reports[5][k] for k in ('tests_run','failures','errors','skipped')]==[1,0,0,0]
            fixture=parsed(Q+'cpu/attempt05/result/native_continuation_preflight.json')
            test=now_tests['PositiveRankingTests.test_native_continuation_preflight']
            order=ast.literal_eval(next(n.value for n in ast.walk(test) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='expected_cleanup' for t in n.targets)))
            assert fixture['cleanup_order']==order==['collect','synchronize','clear_workspaces','empty_cache','allocated','reserved']
            assert fixture['omitted_workspace_release_rejected'] is True and fixture['actual_gpu_calls']==0
            assert fixture['rejected_cases']==['apps','free','host','allocated','reserved','jobs']
            result['test_evidence']={'qualified_count':7,'reused':reuse_rows,'executed':executed,'only_one_rerun':True,
                'actual_class_and_declared_union_match':True,'saved_attempt05_result_without_source_list':{k:v for k,v in reports[5].items() if k!='source_files'},
                'fixture':fixture,'fixture_order_literal_matches_source':True,
                'scope':'Saved CPU evidence and static test text, not independently executed by this reviewer.'}
            walls=[];charged=0
            for n in range(1,6):
                w=parsed(Q+f'cpu/attempt{n:02d}/wallclock.json')
                assert w['prior_charged_seconds']==charged
                assert w['charged_seconds']==math.ceil(w['elapsed_seconds'])
                charged+=w['charged_seconds'];assert w['cumulative_charged_seconds']==charged
                assert w['deadline_epoch']==w['started_epoch']+300-w['prior_charged_seconds']
                assert w['exit_code']==(1 if n<3 else 0)
                walls.append({'attempt':n,**w})
            assert [w['charged_seconds'] for w in walls]==[3,3,24,19,3]
            assert charged==gate5['cumulative_charged_seconds']==52
            assert reports[5]['cumulative_charged_seconds']==51
            assert walls[-1]['command'][-2:]==['--tests',executed[0]]
            assert gate0['total_handwritten_optimizer_calls']==14
            assert gate4['attempt04_optimizer_calls']==8 and gate4['cumulative_optimizer_calls']==22
            assert gate5['attempt05_optimizer_calls']==0 and gate5['cumulative_optimizer_calls']==22
            result['accounting']={'outer_wallclock_records':walls,'current_project_cpu_charged_seconds':charged,'maximum_seconds':300,
                'remaining_seconds':300-charged,'attempt05_inner_cumulative_seconds':reports[5]['cumulative_charged_seconds'],
                'inner_does_not_replace_outer':True,'new_real_optimizer_calls':0,'cumulative_real_optimizer_calls':22,
                'cumulative_method_optimizer_calls':20,'cumulative_independent_reference_calls':2,
                'orchestration_simulations':1152,'orchestration_actual_optimizer_calls':0,
                'native_and_formal_actual_calls':0,'prior_two_failures_retained':True}
            deployments=[]
            for name in ('deployment.json','deployment02.json','deployment03.json','deployment04.json','deployment05.json'):
                d=parsed(Q+name);rows=[resolve_saved(row) for row in d['files']]
                assert len(rows)==41
                assert sum(row['bytes'] for row in rows)==d['total_bytes']
                if name=='deployment05.json':
                    expected_paths=set(paths)|{Q+'cpu_wrapper.py'}
                    assert {row['logical_path'] for row in rows}==expected_paths
                    assert all(row.get('readback_equal') is True for row in d['files'])
                deployments.append({'path':Q+name,'files':rows,'count':len(rows),'total_bytes':d['total_bytes'],'all_match_supplied_bytes':True})
            receipts=[]
            for name in ('receive_verification.json','receive04.json','receive05.json'):
                r=parsed(Q+name);items=r if isinstance(r,list) else r['files']
                rows=[resolve_saved(row) for row in items]
                if isinstance(r,dict) and 'total_bytes' in r:assert r['total_bytes']==sum(row['bytes'] for row in rows)
                receipts.append({'path':Q+name,'files':rows,'count':len(rows),'total_bytes':sum(row['bytes'] for row in rows),'all_match_supplied_bytes':True})
            assert receipts[-1]['count']==5 and receipts[-1]['total_bytes']==7689
            assert deployments[-1]['total_bytes']==701596
            result['deployment_and_receipts']={'deployment':deployments,'receive':receipts,
                'scope':'Saved transfer claims versus attached bytes only; no SFTP, SSH or producing-host authentication.'}
            history_delivery=parsed('reports/documentation/20261016/positive_ranking_implementation/review/repair/delivery.json')
            history_report=identity('reports/documentation/20261016/positive_ranking_implementation/review/repair/external/report.zh.txt')
            assert history_report['bytes']==history_delivery['report']['bytes'] and history_report['sha256']==history_delivery['report']['sha256']
            result['prior_report_identity']={**history_report,'matches_saved_prior_delivery':True,
                'scope':'Prior report only; old 62-member ZIP not included or reverified this round.'}
            result['status']='PASS_STATIC_SOURCE_AND_SAVED_FIELDS'
            print(json.dumps({'status':result['status'],'source_count':40,'unchanged_sources':38,'patch_files':2,
                'qualified_evidence':'6 reused + 1 executed','cpu_seconds':charged,'new_real_optimizer_calls':0,
                'cumulative_real_optimizer_calls':22,'deploy05_files':41,'deploy05_bytes':701596,
                'receive05_files':5,'receive05_bytes':7689,'project_code_executed':False},ensure_ascii=False,indent=2))
    except BaseException:
        status=1;result['status']='MECHANICAL_CHECK_FAILED';stderr.write(traceback.format_exc())
    result['mechanically_read_inputs']=[READ[p] for p in sorted(READ)]
    result['operation']={**pre,'exit_code':status,'wall_seconds':time.perf_counter()-started,
        'process_maxrss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
    save(OUT/'result.json',result)
    (OUT/'stdout.txt').write_text(stdout.getvalue(),encoding='utf-8');(OUT/'stderr.txt').write_text(stderr.getvalue(),encoding='utf-8')
    sys.stdout.write(stdout.getvalue());sys.stderr.write(stderr.getvalue());return status

if __name__=='__main__':raise SystemExit(main())
