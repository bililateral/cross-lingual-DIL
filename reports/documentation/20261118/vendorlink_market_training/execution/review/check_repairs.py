"""Two small, CPU-only counterexamples from the independent prelaunch review."""
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from operations import read, write
from step35_market_accounting import MarketLedger, billed_operation
from step35_market_worker import paired_report
from step35_vendorlink_account_policy import SEEDS, summarize


def check_costs(folder):
    campaign=folder/'campaign'
    campaign.mkdir()
    (folder/'evidence').mkdir()
    prior=dict(training=0,gpu=0,auxiliary=4)
    ledger=MarketLedger(campaign,prior,0,[],caps=dict(training=600,gpu=300,auxiliary=200))
    with billed_operation(folder,'synthetic_observation'):
        pass
    try:
        with billed_operation(folder,'synthetic_failed_delivery'):
            raise ValueError('Authored failure')
    except ValueError:
        pass
    ledger.reserve('training',80)
    once=read(campaign/'budget.json')
    ledger.sync_external(force=True)
    assert once==read(campaign/'budget.json') and once['used_seconds']['auxiliary']==6
    assert sorted(e['success'] for e in once['events'])==[False,True]
    ledger.charge('auxiliary','authored_prior_work',140,True)
    try:
        ledger.reserve('training',1)
    except TimeoutError:
        denied=True
    else:
        denied=False
    assert denied
    return dict(external_charged_seconds=2,successful_events=1,failed_events=1,idempotent=True,
                denied_new_dispatch_with_insufficient_auxiliary=True)


def check_attribution():
    def rows(old):
        cells=dict(R01=.20,R11=.80,R12=.60,R21=.76,R22=.80,R23=.60,R31=old,R32=old,R33=.83)
        return [dict(seed=s,**summarize(cells)) for s in SEEDS]
    results={k:rows(v) for k,v in dict(H_full=.79,H_ER=.73,H_SEQ=.70,V_CLS=.75,V_SET=.75).items()}
    good=paired_report(results,{})
    assert good['full_application'] is True and good['prehistory_integrity']
    for r in results['H_full']:
        r['R01']=.21
    held=paired_report(results,{})
    gates=('full_er','full_seq','meaningful_scenario','external','scientific_gates_joint')
    assert all(good[k] is True and held[k] is True for k in gates)
    assert held['interpretation_requires_identity_review'] and held['full_application'] is None
    assert all(abs(x-.01)<1e-12 for k in ('H_ER','H_SEQ') for x in held['prehistory_H_macro_f1_differences'][k]['R01'])
    return dict(unchanged_scientific_gates=list(gates),consistent_support=True,
                anomalous_R01_scientific_joint=True,anomalous_attribution=None,differences_retained=True)


if __name__=='__main__':
    dest=Path(sys.argv[1])
    assert len(os.sched_getaffinity(0))==1 and not os.environ['CUDA_VISIBLE_DEVICES']
    with tempfile.TemporaryDirectory(dir=dest) as tmp:
        evidence=dict(accounting=check_costs(Path(tmp)),attribution=check_attribution())
    write(dest/'result.json',dict(passed=True,**evidence,model_updates=0,formal_data_reads=0))
