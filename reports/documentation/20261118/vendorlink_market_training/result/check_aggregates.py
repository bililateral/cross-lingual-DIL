"""Check already released FIT12 aggregates against the saved per-seed cells."""
from pathlib import Path
from datetime import datetime, timedelta, timezone
from math import isclose, isfinite
from statistics import mean, stdev
import hashlib
import json
import sys

en = Path('/home/yongpeng/cross-lingual/reports/documentation/20261118/vendorlink_market_training/execution')
c = en/'campaign'
out = c/'analysis/result'
def load(p):
    return json.loads(p.read_bytes())
def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def same(x,y):
    assert isfinite(x) and isfinite(y) and isclose(x,y,rel_tol=0,abs_tol=2e-12),(x,y)
def distribution(ref,values,short=False):
    assert len(values)==len(ref['values'])==3
    for x,y in zip(values,ref['values']): same(x,y)
    same(mean(values),ref['mean'])
    same(stdev(values),ref['sd'])
    bounds=ref['range'] if short else [ref['minimum'],ref['maximum']]
    same(min(values),bounds[0])
    same(max(values),bounds[1])
    if not short: assert ref['seeds']==[51,52,53]
assert digest(en/'formal_gate.json')=='7607f7cf20834a5fa4f9a34ebda0d92543537d33b87c704abf91d812b0555b1f'
assert load(c/'completion.json')['confirm_open'] is False and load(c/'completion.json')['dev_opening']=='whole_batch_once'
assert load(c/'completion.json')['decision']['sha256']==digest(out/'decision.json')
sys.path.insert(0,str(en/'workspace/scripts'))
from step35_market_accounting import billed_operation, MarketLedger
with billed_operation(en,'verify_released_aggregates',lambda: MarketLedger.settle_closed(c)):
    results=load(out/'results.json')
    metrics=load(out/'metrics.json')
    table=load(out/'metric_summary.json')
    pairs=load(out/'metric_pairs.json')['full_minus_comparator']
    d=load(out/'decision.json')
    valid=load(c/'training_complete.json')['tasks']
    for k,h in load(en/'formal_gate.json')['historical'].items():
        b=h['binding']
        p=Path(b['path'])
        assert digest(p)==b['sha256'] and p.stat().st_size==b['bytes']
        results[k]=load(p)[h['method']]
    cells=['R01','R11','R12','R21','R22','R23','R31','R32','R33']
    historic={'original_H_ER','original_beta_Full'}
    prepared={}
    for name,rows in results.items():
        assert [r['seed'] for r in rows]==[51,52,53]
        prepared[name]=[]
        for r in rows:
            vals=dict(r,Q=(r['R11']+r['R22'])/2,O=(r['R31']+r['R32'])/2,N=(r['R22']+r['R33'])/2,
                Z=r['R33'],GB=r['R22']-r['R12'],GC=r['R33']-r['R23'],FA=r['R11']-r['R31'],FB=r['R22']-r['R32'])
            vals['F']=vals['Q']-vals['O']
            for k in r: same(r[k],vals[k])
            if name not in historic:
                train='VendorLink_SEQ' if name.startswith('V_') else name
                for cell in cells: same(r[cell],metrics[f"{train}_s{r['seed']}:{name}:{cell}"]['macro_F1'])
            prepared[name].append(vals)
        for k,summary in d['summaries'][name].items():
            distribution(summary,[r[k] for r in prepared[name]])
    used_keys=set()
    for t in valid:
        readers=['V_CLS','V_SET','V_CLS_ads'] if t['method']=='VendorLink_SEQ' else [t['method']]
        for reader in readers:
            for cell in cells:
                key=f"{t['id']}:{reader}:{cell}"
                used_keys.add(key)
                assert len(metrics[key])==13
    assert set(metrics)==used_keys
    for n,rows in table.items():
        reader,cell=n.split(':')
        name='VendorLink_SEQ' if reader.startswith('V_') else reader
        for k,s in rows.items():
            distribution(s,[metrics[f'{name}_s{seed}:{n}'][k] for seed in (51,52,53)],True)
    for n,rows in pairs.items():
        m,cell=n.split(':')
        for k,s in rows.items():
            value=[a-b for a,b in zip(table['H_full:'+cell][k]['values'],table[n][k]['values'])]
            distribution(s,value,True)
    for m,parts in d['full_minus_comparator'].items():
        for k,s in parts.items():
            sign=-1 if k in ('F','FA','FB') else 1
            distribution(s,[sign*(a[k]-b[k]) for a,b in zip(prepared['H_full'],prepared[m])])
    for f,e in zip(prepared['H_full'],prepared['H_ER']):
        same(e['F']-f['F'],(f['O']-e['O'])-(f['Q']-e['Q']))
    prehistory=all(metrics[f'{m}_s{s}:{m}:{cell}']==metrics[f'H_full_s{s}:H_full:{cell}']
        for m in ('H_ER','H_SEQ') for s in (51,52,53) for cell in ('R01','R11','R12'))
    empirical=dict(prehistory_all13_exact=prehistory,full_er_joint_positive_seeds=[
        f['seed'] for f,e in zip(prepared['H_full'],prepared['H_ER']) if f['O']>e['O'] and f['F']<e['F']],
        full_er_lower_forgetting_seeds=[
        f['seed'] for f,e in zip(prepared['H_full'],prepared['H_ER']) if f['F']<e['F']],
        dream_full_negative_gain_seeds=[r['seed'] for r in prepared['H_full'] if r['GB']<0])
    end=(c/'completion.json').stat().st_mtime
    start=datetime.fromisoformat(load(en/'launch.json')['started']).timestamp()
    result=dict(native_decision_sha256=digest(out/'decision.json'),training_identities=len(valid),current_readers=len(table)//9,
        seed_matrix_cells=len(metrics),per_cell_metrics=13,individual_metric_values=sum(len(x) for x in metrics.values()),
        summary_cells=len(table),metric_summary_fields=sum(len(x) for x in table.values()),
        full_comparator_cells=len(pairs),metric_pair_fields=sum(len(x) for x in pairs.values()),
        primary_summary_series=sum(len(v) for v in d['summaries'].values()),
        primary_paired_series=sum(len(v) for v in d['full_minus_comparator'].values()),
        reported_seeds=[51,52,53],maximum_absolute_tolerance=2e-12,empirical=empirical,
        wall_seconds_from_displayed_second=int(end)-int(start),access='saved open aggregates and operational receipts only')
b=load(c/'budget.json')
result.update(observed=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),settled_seconds=b['used_seconds'])
print(json.dumps(dict(check=result,budget=load(c/'budget.json'),budget_binding=dict(path=str(c/'budget.json'),
    bytes=(c/'budget.json').stat().st_size,sha256=digest(c/'budget.json')))))
