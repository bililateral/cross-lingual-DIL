"""Read-only ZIP/receipt audit, not training or qualification.
Usage: python audit_checks.py INPUT_ZIP EXTRACTED_INPUT_DIR OUTPUT_JSON
No project module is imported. Only supplied archive bytes and numeric receipts are read.
"""
from __future__ import annotations
import hashlib, json, math, sys, zipfile
from decimal import Decimal
from pathlib import Path

def digest(b: bytes) -> str: return hashlib.sha256(b).hexdigest()
def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit('Usage: python audit_checks.py INPUT_ZIP EXTRACTED_INPUT_DIR OUTPUT_JSON')
    archive, root, output = map(Path, sys.argv[1:])
    base = root/'reports/seller_alias_continual/20261015/balanced_pooling_failure'
    old = root/'reports/documentation/20261014/balanced_pooling_implementation'
    load = lambda p: json.loads(p.read_text(encoding='utf-8'))
    raw = archive.read_bytes()
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        manifest = json.loads(z.read('review_manifest.json'))
        rows = []
        for r in manifest['entries']:
            b=z.read(r['path'])
            rows.append({'path':r['path'], 'bytes':len(b), 'sha256':digest(b),
                         'manifest_matches':len(b)==r['bytes'] and digest(b)==r['sha256'],
                         'extracted_matches_archive':(root/r['path']).read_bytes()==b})
        assert len(names)==len(set(names))==59
        assert len(rows)==58 and all(r['manifest_matches'] and r['extracted_matches_archive'] for r in rows)
        assert set(names)==set(r['path'] for r in rows)|{'review_manifest.json'}
    assert len(raw)==291160 and digest(raw)=='1fb624a61ee9e075fc11262c2eacecf3349330a08a026da1ea7cb03907247cd5'
    gate=load(old/'qualification/formal_gate.json'); execution=load(base/'evidence/reports/job/execution.json')
    receive=load(base/'receive_verification.json')
    norm=lambda a: sorted((x['path'],x['bytes'],x['sha256']) for x in a)
    assert norm(gate['source_files'])==norm(execution['sources'])==norm(receive['sources'])
    assert len(gate['source_files'])==29
    assert all((root/p).stat().st_size==size and digest((root/p).read_bytes())==sha for p,size,sha in norm(gate['source_files']))
    assert execution['gate']['sha256']==digest((old/'qualification/formal_gate.json').read_bytes())
    curated=[]
    for r in receive['files']:
        p=base/'evidence'/r['path']; present=p.is_file()
        match=None
        if present:
            match=p.stat().st_size==r['bytes'] and digest(p.read_bytes())==r['sha256']; assert match
        curated.append({'path':r['path'],'included_in_this_zip':present,'identity_matches_if_included':match})
    diag=[]
    for r in load(base/'diagnostic_cpu/receive.json')['files']:
        p=base/r['local_path']; present=p.is_file()
        match=None
        if present:
            match=p.stat().st_size==r['bytes'] and digest(p.read_bytes())==r['sha256']; assert match
        diag.append({'path':r['path'],'local_path':r['local_path'],'included':present,'matches':match,'declared_bytes':r['bytes']})
    failure=load(base/'evidence/reports/job/failure.json')
    residuals=[]
    for b in failure['solver_failure']['failed_blocks']:
        residuals.append({**b,'P_pass':b['marginal_error']<=1e-10,'Q_pass':b['feasible_error']<=1e-12,
                         'checked_gap_pass':b['gap_checked']<=5e-9,'P_limit_ratio':b['marginal_error']/1e-10})
        assert b['n']==b['m']==3 and b['iteration']==4096
        assert b['marginal_error']>1e-10 and b['feasible_error']<=1e-12 and b['gap_checked']<=5e-9
    af=load(base/'evidence/reports/job/run/updates/A_F.json'); logs=af['rows']; assert len(logs)==288
    steps=[1,48,96,144,192,240,288]; selected=[]
    for i,row in enumerate(logs,1):
        assert row['step']==i
        s=row['solver_summary']
        assert s['failures']==0 and s['marginal_error']<=1e-10 and s['feasible_error']<=1e-12 and s['gap_checked']<=5e-9
        if i in steps:selected.append({'step':i,**s})
    ledger=load(base/'cpu_ledger.json')
    assert ledger['cumulative_charged_seconds']==ledger['prior_charged_seconds']+sum(x['charged_seconds'] for x in ledger['attempts'])==97
    source_hand=[]
    for a, rel in [('attempt01','diagnostic_cpu/attempt01/handmade_diagnostic.py'),('attempt02','handmade_diagnostic.py')]:
        wrapper=load(base/f'diagnostic_cpu/{a}/wrapper.json')
        assert digest((base/rel).read_bytes())==wrapper['diagnostic_source_sha256']
        source_hand.append({'attempt':a,'source_path':rel,'source_sha256':wrapper['diagnostic_source_sha256'],'wrapper_matches':True})
    used=Decimal(str(failure['budget']['elapsed_seconds'])); remain=Decimal(86400)-used
    result={
        'scope':'READ_ONLY_ARCHIVE_AND_SAVED_NUMERIC_RECEIPTS_NOT_FORMAL_DATA_OR_TRAINING',
        'archive':{'bytes':len(raw),'members':len(names),'sha256':digest(raw),'manifest_payload_count':len(rows),'all_matches':True},
        'payloads':rows,
        'production_sources':{'count':29,'gate_execution_receive_archive_all_match':True,'paths':[r[0] for r in norm(gate['source_files'])]},
        'formal_receipt_curation':{'receiver_reports_files':len(receive['files']),'receiver_reports_bytes':sum(x['bytes'] for x in receive['files']),
            'included_files':sum(x['included_in_this_zip'] for x in curated),'entries':curated,
            'note':'Omitted receipts are known only through supplied inventory/hash statements, not independently read.'},
        'handwritten_receipt_curation':diag, 'handwritten_source_identity':source_hand,
        'formal_failure':{'status':failure['status'],'failure_count':failure['solver_failure']['failure_count'],
                         'phase':failure['budget']['phase'],'step':failure['solver_failure']['step'],'role':failure['solver_failure']['role'],
                         'view':failure['solver_failure']['view'],'blocks':residuals},
        'af_update_receipt':{'rows':len(logs),'all_recorded_solver_summaries_pass_original_limits':True,'selected_cumulative_summaries':selected},
        'observed_old_access':load(base/'evidence/reports/job/access.json'),
        'budget_arithmetic':{'cpu_charged':97,'cpu_remaining':203,'old_native_calls':8,'proposed_new_native_calls_max':3,
            'proposed_cumulative_native_calls_max':11,'old_native_charged_seconds':364,'native_remaining_seconds':536,
            'old_formal_seconds_decimal':str(used),'remaining_formal_seconds_decimal':str(remain),
            'remaining_formal_seconds_binary64':86400-failure['budget']['elapsed_seconds'],
            'old_successful_optimizer_calls_inferred_from_control_flow':5*288+247,'old_uncompleted_calls':1728-(5*288+247),
            'proposed_new_full_three_arm_calls':1728,'total_calls_if_new_full_run_completes':5*288+247+1728},
        'limits':'No live server inspection, formal K/P/Q reconstruction, model load, label read, optimizer call, or repaired implementation qualification.'}
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['archive','budget_arithmetic']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
