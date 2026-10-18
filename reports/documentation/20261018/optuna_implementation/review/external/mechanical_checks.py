"""Attachment identity, saved fields, source snapshots and budget arithmetic only.

Never imports submitted modules or executes tests, models, metric reduction,
random simulation, or bootstrap. Python standard library only.
"""
from pathlib import Path
import ast, hashlib, io, json, math, tarfile, traceback, zipfile

ROOT = Path('/workspace/scratch/3e46faac4c35/optuna_review_input')
OUT = Path('/workspace/scratch/3e46faac4c35/optuna_static_review')
Q = 'reports/documentation/20261018/optuna_implementation/qualification/'

def sha(b):
    return hashlib.sha256(b).hexdigest()

def read(path):
    return json.loads((ROOT / path).read_bytes())

def write(name, value):
    (OUT / 'mechanical' / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def definition(text, name):
    for n in ast.walk(ast.parse(text)):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            return '\n'.join(text.splitlines()[n.lineno - 1:n.end_lineno])
    return None

def main():
    gate = read(Q + 'cpu_gate.json')
    current = {r['path']: r for r in gate['source_files']}
    matches = []
    for name, rec in current.items():
        b = (ROOT/name).read_bytes()
        matches.append(dict(path=name, matches=len(b)==rec['bytes'] and sha(b)==rec['sha256']))
    write('cpu_source_identity.json', dict(count=len(matches), all_match=all(r['matches'] for r in matches), files=matches))
    snapshots=[]
    tests=gate['qualified_tests']
    current_text=(ROOT/'tests/test_step28_optuna.py').read_text()
    for attempt in gate['attempts']:
        name=attempt['attempt']
        result=read(Q+name+'/result.json')
        wrapper=(ROOT/(Q+name+'.wrapper.txt')).read_text()
        info=dict(attempt=name, outer_wall_ns=attempt['wall_ns'], outer_charged_seconds=attempt['charged_seconds'],
            outer_exit_code=attempt['exit_code'], wrapper=wrapper,
            saved_result_fields={k:v for k,v in result.items() if k!='source_files'})
        archive=ROOT/(Q+name+'_sources.tar.gz')
        with tarfile.open(archive,'r:gz') as tar:
            files={m.name.removeprefix('./'):tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}
        source_records={r['path']:r for r in result['source_files']}
        rows=[]
        for path,b in files.items():
            r=source_records.get(path)
            rows.append(dict(path=path,bytes=len(b),sha256=sha(b),listed_in_result=r is not None,
                matches_saved_result=r is not None and r['bytes']==len(b) and r['sha256']==sha(b),
                matches_current=path in current and current[path]['sha256']==sha(b)))
        old_text=files['tests/test_step28_optuna.py'].decode('utf-8-sig')
        info.update(archive=dict(path=archive.relative_to(ROOT).as_posix(),bytes=archive.stat().st_size,sha256=sha(archive.read_bytes())),
            file_count=len(files), saved_source_count=len(source_records), manifest_membership_matches=set(files)==set(source_records),
            source_checks=rows, changed_from_current=[r['path'] for r in rows if not r['matches_current']],
            test_bodies_equal_current={t:definition(old_text,t)==definition(current_text,t) for t in tests},
            optimizer_reference_call_source=[dict(line=i,text=s.strip()) for i,s in enumerate(old_text.splitlines(),1) if 'ob.step()' in s])
        snapshots.append(info)
    write('cpu_attempts_and_snapshots.json',snapshots)
    arithmetic=dict(outer_wall_ns_sum=sum(a['wall_ns'] for a in gate['attempts']),
        outer_charged_seconds_sum=sum(a['charged_seconds'] for a in gate['attempts']),
        individually_ceiled_seconds_sum=sum(math.ceil(a['wall_ns']/1e9) for a in gate['attempts']),
        recorded_outer_wall_ns=gate['outer_wall_ns'],recorded_charge=gate['cumulative_charged_seconds'],
        remaining_cpu_budget_seconds=300-gate['cumulative_charged_seconds'],
        optimizer_call_accounting=dict(attempt01=dict(production=18,independent_reference=3),
            attempt02=dict(production=0,independent_reference=0),attempt03=dict(production=0,independent_reference=0),
            attempt04=dict(production=6,independent_reference=0),attempt05=dict(production=15,independent_reference=3),
            attempt06=dict(production=0,independent_reference=0), production_total=39,independent_reference_total=6,total_actual_cpu_steps=45,
            basis='Static successful test control flow and original saved test lists; no tests executed externally.'),
        development_budget_components=dict(cpu=300,native=1800,search=36*9000,replication=9*9000,control=300,evaluation=2400,analysis=600),
        development_total_seconds=300+1800+36*9000+9*9000+300+2400+600,
        development_total_hours=(300+1800+36*9000+9*9000+3300)/3600,
        prior_two_arm_projection_seconds=18844.60961239063+19235.266476024408,
        new_two_run_cap_seconds=2*9000,
        prior_two_arm_projection_exceeds_new_cap=(18844.60961239063+19235.266476024408)>2*9000,
        saved_analysis_field_count_components=dict(trial_absolute_and_old_A=45*(2*3*2*22*2+3*1*22*2),
            replication_per_seed=2*3*2*3*2*22*2, replication_three_seed=2*2*3*2*22*2),
        saved_analysis_field_count_total=45*(2*3*2*22*2+3*1*22*2)+2*3*2*3*2*22*2+2*2*3*2*22*2,
        non_actions=['No project imports','No scientific code/tests/model execution','No metric means recomputed','No bootstrap/resampling','No project server or external storage access'])
    write('budget_and_count_arithmetic.json',arithmetic)
    # Read saved S0 fields and alignment only, explicitly do not reduce the metric matrix.
    p='reports/seller_alias_continual/20261018/positive_ranking_result/analysis/summary.json'
    summary=read(p)
    base='reports/seller_alias_continual/20261015/balanced_pooling_result/job/'
    col=read(base+'evaluation/collected.json');split=read(base+'run/partition.json');matrix=read(base+'evaluation/B_S_raw_metrics.json')
    fields=dict(S0={d:summary['absolute']['B_S']['raw'][d]['map'] for d in 'AB'},
        group_ids_aligned=col['group_ids']==[r['group_uid'] for r in split['development']],
        domains_aligned=col['domains']==[r['domain'] for r in split['development']],
        metric_columns=matrix['columns'],row_count=len(matrix['values']),column_counts=sorted(set(map(len,matrix['values']))),
        saved_reduction_not_recomputed=True)
    write('saved_reference_fields.json',fields)
    # Verify received file hashes as recorded, without reading payload semantics.
    checksum_results=[]
    for checksum_name in ['receive.sha256','receive_delta.sha256','workspace.sha256','workspace_final.sha256']:
        text=(ROOT/(Q+checksum_name)).read_text()
        rows=[]
        for line in text.splitlines():
            if not line.strip(): continue
            expected,path=line.split(maxsplit=1);path=path.lstrip('*')
            candidates=[ROOT/path,ROOT/(Q+path)]
            f=next((c for c in candidates if c.is_file()),None)
            rows.append(dict(recorded_path=path,found=f is not None,
                actual_path=f.relative_to(ROOT).as_posix() if f else None,
                matches=f is not None and sha(f.read_bytes())==expected,
                note='Historical source list may intentionally differ from current sources' if checksum_name=='workspace.sha256' else None))
        checksum_results.append(dict(path=Q+checksum_name,records=rows))
    write('received_checksum_correspondence.json',checksum_results)
    summary_out=dict(source_count=len(matches),all_current_sources_match=all(r['matches'] for r in matches),
        attempts=[dict(attempt=a['attempt'],files=a['file_count'],source_manifest_matches=a['manifest_membership_matches'] and all(r['matches_saved_result'] for r in a['source_checks']),
                      changed_from_current=a['changed_from_current'],qualified_tests=a['saved_result_fields'].get('qualified_tests'),status=a['saved_result_fields'].get('status')) for a in snapshots],
        arithmetic=arithmetic,saved_reference_fields=fields,
        checksum_overview=[dict(path=a['path'],rows=len(a['records']),matched=sum(r['matches'] for r in a['records']),missing=sum(not r['found'] for r in a['records'])) for a in checksum_results])
    write('mechanical_summary.json',summary_out)
    print(json.dumps(summary_out,ensure_ascii=False,indent=2))

if __name__=='__main__':
    try:
        main()
    except BaseException:
        path=OUT/'execution/failures.json'
        failures=json.loads(path.read_text()) if path.exists() else []
        failures.append(dict(operation='mechanical_checks.py',kind='review_mechanical_failure',traceback=traceback.format_exc()))
        path.write_text(json.dumps(failures,ensure_ascii=False,indent=2)+'\n')
        raise
