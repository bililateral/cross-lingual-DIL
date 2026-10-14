"""Clarify a missing formal S residual field in our own reference output.
No scientific statistic, frozen input, or original run01 output is changed.
Usage: python clarify_reference_reporting.py --run01 DIR --out NEW_DIR
"""
from pathlib import Path
import argparse, hashlib, json, sys, traceback

def main():
    p=argparse.ArgumentParser();p.add_argument('--run01',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    raw=(a.run01/'execution_evidence.json').read_bytes();d=json.loads(raw)
    assert d['training']['B_S']['mean_R']==0.0
    d['training']['B_S']['mean_R']=None
    d['training']['B_S']['mean_R_status']='NOT_LOGGED_IN_FORMAL_S_TRAINING; null is not measured zero'
    d['training']['B_S']['mean_weighted_R_status']='NOT_AN_OPTIMIZATION_TERM; recorded zero denotes excluded contribution, not a measured residual'
    d['clarification']={'source':'run01/execution_evidence.json','source_sha256':hashlib.sha256(raw).hexdigest(),'changed_field':'training.B_S.mean_R','reason':'Our check_saved.py used get(response,0) when summarizing all arms. S formal log has no response field. Descriptive S R exists only in common diagnostics. This clarification changes no arithmetic check or result.'}
    (a.out/'execution_evidence.corrected.json').write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    note={'status':'REFERENCE_REPORTING_CLARIFICATION_ONLY','failed_execution':False,'scientific_statistics_changed':False,'frozen_input_changed':False,'original_run01_preserved':True,'path':'run01/execution_evidence.json:training.B_S.mean_R','original_value':0.0,'clarified_value':None,'reason':d['clarification']['reason']}
    (a.out/'clarification.json').write_text(json.dumps(note,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(note,ensure_ascii=False,indent=2))
if __name__=='__main__':
    try: main()
    except BaseException:
        traceback.print_exc();sys.exit(1)
