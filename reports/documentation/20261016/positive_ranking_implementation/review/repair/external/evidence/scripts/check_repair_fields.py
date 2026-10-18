#!/usr/bin/env python3
"""Static AST/text/hash and existing-field comparisons only; no project execution."""
import ast
from datetime import datetime, timezone
import hashlib
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
ROOT=BASE/'repair_audit_work/input'
OUT=BASE/'repair_audit_output/evidence/field_checks'
PREFIX='reports/documentation/20261016/positive_ranking_implementation/'
QUAL=PREFIX+'qualification/'
CHANGED=['scripts/step28_positive_ranking.py','scripts/step28_positive_ranking_run.py',
         'scripts/step28_positive_ranking_verify.py','tests/test_step28_positive_ranking.py']
CACHE={}
START=time.perf_counter()

def sha(data):return hashlib.sha256(data).hexdigest()
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def raw(name):
    if name not in CACHE:CACHE[name]=(ROOT/name).read_bytes()
    return CACHE[name]
def obj(name):return json.loads(raw(name))
def ident(name):
    b=raw(name)
    return {'path':name,'bytes':len(b),'sha256':sha(b)}
def source_ident(name,historical=False):
    r=ident('historical_sources/'+name if historical and name in CHANGED else name)
    return {**r,'path':name}
def verify_record(record,path=None):
    path=path or record['path']
    actual=ident(path)
    return {**actual,'matches':all(actual[k]==record[k] for k in ('bytes','sha256'))}
def tree(name):return ast.parse(raw(name).decode('utf-8'))
def dump(node):return ast.dump(node,include_attributes=False)
def functions(t):
    r={}
    def walk(nodes,prefix=''):
        for n in nodes:
            if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)):
                r[prefix+n.name]=dump(n)
            elif isinstance(n,ast.ClassDef):walk(n.body,prefix+n.name+'.')
    walk(t.body)
    return r
def ast_changes(name):
    a=functions(tree('historical_sources/'+name));b=functions(tree(name))
    return {'changed_functions':[k for k in sorted(a.keys() & b.keys()) if a[k]!=b[k]],
            'added_functions':sorted(b.keys()-a.keys()),'removed_functions':sorted(a.keys()-b.keys())}

OUT.mkdir(parents=True,exist_ok=True)
pre={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'command_argv':[sys.executable,*sys.argv],
     'source_sha256':sha(Path(__file__).read_bytes()),'source_bytes':Path(__file__).stat().st_size,
     'python':sys.version,'platform':platform.platform(),'cwd':os.getcwd(),
     'operation':'AST text/identity and saved accounting fields only',
     'project_modules_imported':False,'scientific_computation':False}
save(OUT/'pre_execution.json',pre)
result={};exit_code=0
try:
    # Reconstruct the provided patch in memory against provided historical files.
    patch=raw('changes.patch').decode('utf-8').splitlines(keepends=True)
    starts=[i for i,s in enumerate(patch) if s.startswith('diff --git ')]+[len(patch)]
    patch_checks=[]
    for i,j in zip(starts,starts[1:]):
        block=patch[i:j]
        header=re.fullmatch(r'diff --git a/(\S+) b/(\S+)\n',block[0]);assert header and header[1]==header[2]
        name=header[1];old=raw('historical_sources/'+name);new=raw(name)
        old_lines=old.decode().splitlines(keepends=True);rebuilt=[];pos=0;k=0;hunks=0
        while k<len(block):
            h=re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@',block[k])
            if not h:k+=1;continue
            old_start=int(h[1])-1;old_count=int(h[2] or '1');new_count=int(h[4] or '1')
            rebuilt.extend(old_lines[pos:old_start]);pos=old_start;k+=1;seen_old=seen_new=0
            while k<len(block) and not block[k].startswith('@@ '):
                s=block[k];k+=1
                if s.startswith('\\ No newline'):continue
                assert s and s[0] in ' +-'
                if s[0] in ' -':
                    assert old_lines[pos]==s[1:],(name,pos+1,'patch old/context mismatch')
                    pos+=1;seen_old+=1
                if s[0] in ' +':rebuilt.append(s[1:]);seen_new+=1
            assert (seen_old,seen_new)==(old_count,new_count)
            hunks+=1
        rebuilt.extend(old_lines[pos:]);rebuilt=''.join(rebuilt).encode()
        indexes=re.match(r'index ([0-9a-f]+)\.\.([0-9a-f]+)',block[1]);assert indexes
        def blob_sha(data):return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        check={'path':name,'old':ident('historical_sources/'+name),'new':ident(name),
               'hunks':hunks,'rebuilt_new_matches_exact_bytes':rebuilt==new,
               'old_git_blob_prefix_matches':blob_sha(old).startswith(indexes[1]),
               'new_git_blob_prefix_matches':blob_sha(new).startswith(indexes[2]),
               'ast_function_changes':ast_changes(name)}
        assert all(check[key] for key in ('rebuilt_new_matches_exact_bytes','old_git_blob_prefix_matches','new_git_blob_prefix_matches'))
        patch_checks.append(check)
    assert [r['path'] for r in patch_checks]==CHANGED
    result['four_file_patch']=patch_checks
    result['commit_scope']='38d05b9a / d914f43c are submitted commit labels. Git blob prefixes and supplied bytes checked; no Git commit tree or project host accessed.'

    seeds=['scripts/step28_positive_ranking_run.py','scripts/step28_positive_ranking_verify.py',
           'tests/test_step28_positive_ranking.py','tests/test_step28_g_half.py','tests/test_step28_record_replay.py']
    paths=set(seeds);todo=list(paths)
    while todo:
        name=todo.pop()
        for n in ast.walk(tree(name)):
            names=[x.name for x in n.names] if isinstance(n,ast.Import) else [n.module] if isinstance(n,ast.ImportFrom) else []
            for m in names:
                if m and m.startswith('step28_'):
                    child='scripts/'+m+'.py'
                    if child not in paths:paths.add(child);todo.append(child)
    paths.update('schema/'+name for name in ('step28_g_half_policy.json','step28_record_replay_policy.json',
                 'step28_bge_continual_policy.json','step28_alias_ranking_policy.json','step28_chinese_base_policy.json',
                 'step28_positive_ranking_policy.json'))
    paths.update(['reports/documentation/20261016/h_consistency/baseline_sources.json',PREFIX+'authorization.json',
                  'docs/SELLER_ALIAS_POSITIVE_RANKING.zh.md','scripts/run_step28_positive_ranking_linux_20261018.sh'])
    current=[source_ident(p) for p in sorted(paths)];historical=[source_ident(p,True) for p in sorted(paths)]
    gate=obj(QUAL+'cpu_gate_repair.json');oldgate=obj(QUAL+'cpu_gate.json')
    cpu03=obj(QUAL+'cpu/attempt03/result/result.json');cpu04=obj(QUAL+'cpu/attempt04/result/result.json')
    assert len(current)==len(historical)==40
    assert current==gate['source_files']==cpu04['source_files']
    assert historical==oldgate['source_files']==cpu03['source_files']
    result['source_binding']={'source_count':40,'current_source_files':current,
        'historical_source_files':historical,'current_equals_repair_gate_and_attempt04':True,
        'historical_equals_old_gate_and_attempt03':True,
        'changed_source_paths':[a['path'] for a,b in zip(current,historical) if a!=b],
        'unchanged_source_count':sum(a==b for a,b in zip(current,historical))}
    refs=[verify_record(gate[k]) for k in ('inherited','executed','wallclock')]
    for r in oldgate['attempts']:
        refs.extend(verify_record(r[k]) for k in ('wallclock','result'))
    assert all(r['matches'] for r in refs)
    result['gate_reference_identities']=refs

    tnew=functions(tree(CHANGED[3]));told=functions(tree('historical_sources/'+CHANGED[3]))
    reused=[]
    for name in gate['reused_tests']:
        key='PositiveRankingTests.'+name
        same=tnew[key]==told[key];assert same
        reused.append({'test':name,'normalized_ast_sha256':sha(tnew[key].encode()),'body_ast_unchanged':same})
    declared=None
    for n in tree(CHANGED[1]).body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CPU_TESTS' for t in n.targets):
            declared=ast.literal_eval(n.value)
    actual_test_names=sorted(k.split('.',1)[1] for k in tnew if k.startswith('PositiveRankingTests.test_'))
    assert len(reused)==3 and len(gate['reexecuted_tests'])==4
    assert not set(gate['reused_tests']) & set(gate['reexecuted_tests'])
    assert set(gate['qualified_tests'])==set(declared)==set(actual_test_names)==set(gate['reused_tests']+gate['reexecuted_tests'])
    assert cpu04['selected_tests']==cpu04['qualified_tests']==gate['reexecuted_tests']
    assert (cpu04['tests_run'],cpu04['failures'],cpu04['errors'],cpu04['skipped'])==(4,0,0,0)
    assert cpu04['status']=='PASS_HANDWRITTEN_ONLY' and cpu04['unresolved_checks']==[]
    result['test_evidence_composition']={'qualified_count':7,'reused_tests':reused,
        'executed_tests':gate['reexecuted_tests'],'attempt04_selected_exactly_matches':True,
        'run_cpu_tests_and_actual_class_match_qualified_union':True,'not_seven_tests_rerun':True,
        'unchanged_B_capture_test_ast':tnew['PositiveRankingTests.test_capture_restore_and_next_update']==told['PositiveRankingTests.test_capture_restore_and_next_update']}

    deploy=obj(QUAL+'deployment04.json')
    deployment_files=[verify_record(r) for r in deploy['files']]
    assert len(deployment_files)==41 and all(r['matches'] for r in deployment_files)
    result['deployment04']={'top_level_keys':list(deploy),'files':deployment_files,
        'files_count':len(deployment_files),'all_supplied_file_identities_match':True,
        'scope':'Checks the supplied deployment receipts against attached bytes only, not remote host state.'}
    receive=obj(QUAL+'receive04.json')
    receive_files=[verify_record(r,r['local'].replace('\\','/')) for r in receive]
    assert len(receive_files)==8 and all(r['matches'] for r in receive_files)
    result['receive04']={'files':receive_files,'files_count':8,
        'total_bytes':sum(r['bytes'] for r in receive_files),'all_supplied_file_identities_match':True}

    wall=[]
    for attempt in ('01','02','03','04'):
        w=obj(QUAL+'cpu/attempt'+attempt+'/wallclock.json')
        assert math.ceil(w['elapsed_seconds'])==w['charged_seconds']
        wall.append({'attempt':'attempt'+attempt,**w})
    assert sum(w['charged_seconds'] for w in wall[:3])==30
    assert wall[-1]['prior_charged_seconds']==30 and wall[-1]['charged_seconds']==19
    assert sum(w['charged_seconds'] for w in wall)==gate['cumulative_charged_seconds']==49
    a=obj(QUAL+'cpu/attempt04/result/stage_a_empty_cache.json');b=obj(QUAL+'cpu/attempt04/result/capture_restore.json')
    native_mock=obj(QUAL+'cpu/attempt04/result/native_continuation_preflight.json')
    orchestration=obj(QUAL+'cpu/attempt04/result/orchestration.json')
    calls_a=sum(v['actual_optimizer_calls'] for v in a.values());calls_b=sum(v['actual_optimizer_calls'] for v in b.values())
    assert (calls_a,calls_b)==(2,6)
    assert calls_a+calls_b==gate['attempt04_optimizer_calls']==8
    assert oldgate['total_handwritten_optimizer_calls']+8==gate['cumulative_optimizer_calls']==22
    assert native_mock['actual_gpu_calls']==0 and orchestration['real_updates_in_this_fixture']==0
    result['saved_accounting_fields']={'wallclock_original_fields':wall,
        'current_cumulative_project_cpu_seconds':49,'attempt04_inner_cumulative_seconds':cpu04['cumulative_charged_seconds'],
        'inner_does_not_override_outer':True,'attempt04_capture_A_optimizer_calls':calls_a,
        'attempt04_capture_B_optimizer_calls':calls_b,'attempt04_total_real_calls':8,
        'retained_real_calls':14,'cumulative_real_calls':22,
        'mock_orchestration_calls':orchestration['scheduler_calls'],'mock_actual_optimizer_calls':0,
        'mock_preflight_actual_gpu_calls':0,'native_planned_optimizer_calls':12,'native_performed_calls':0,
        'formal_planned_optimizer_calls':1152,'formal_performed_calls':0,
        'attempt04_peak_rss_bytes':cpu04['peak_rss_bytes'],'attempt04_peak_work_bytes':cpu04['peak_work_bytes'],
        'scope':'Read supplied fields and checked arithmetic/identity. Did not rerun project CPU or GPU.'}
    result['saved_attempt04_fixture_results']={'stage_A':a,'stage_B':b,'native_mock':native_mock,'orchestration':orchestration}
    result['status']='PASS_STATIC_IDENTITY_PATCH_AND_SAVED_FIELD_CHECKS'
except BaseException:
    exit_code=1
    result.update(status='MECHANICAL_CHECK_FAILED',traceback=traceback.format_exc())
finally:
    result['operation']={'scientific_computation':False,'project_execution':False,
        'wall_seconds':time.perf_counter()-START,'process_maxrss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'timestamp_utc':datetime.now(timezone.utc).isoformat(),'exit_code':exit_code,
        'source_sha256':sha(Path(__file__).read_bytes())}
    result['mechanically_read_inputs']=[ident(name) for name in sorted(CACHE)]
    save(OUT/'result.json',result)
    (OUT/'stderr.txt').write_text(result.get('traceback',''),encoding='utf-8')
    summary={k:result[k] for k in ('status','operation')}
    if exit_code:summary['traceback']=result['traceback']
    else:
        summary.update(current_sources=40,unchanged_sources=36,patch_files=4,
                       qualified_tests=7,reused_tests=3,executed_tests=4,
                       cpu_charged_seconds=49,attempt04_real_optimizer_calls=8,total_real_optimizer_calls=22,
                       deployment_files=41,received_files=8,received_bytes=result['receive04']['total_bytes'])
    output=json.dumps(summary,ensure_ascii=False,indent=2)+'\n'
    (OUT/'stdout.txt').write_text(output,encoding='utf-8');sys.stdout.write(output)
    raise SystemExit(exit_code)
