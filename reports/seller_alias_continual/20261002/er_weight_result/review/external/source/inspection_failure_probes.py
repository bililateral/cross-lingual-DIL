"""Reproduce two inspection mistakes only. Not training/audit-result failures.
Original exploratory tool invocations did not use run_logged; the present replay
is timestamped as a replay, not assigned the original invocation time.
"""
import argparse,json,pathlib
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['missing_stderr','point_training_key'],required=True);a=p.parse_args()
n=pathlib.Path('/mnt/data/er_review_input/reports/seller_alias_continual/20261001/er_weight_execution/20261001_144452/job')
print('INSPECTION_FAILURE_REPLAY',a.mode,flush=True)
if a.mode=='missing_stderr':print((n/'stderr.log').read_text())
else:
 m=json.loads((n/'run/manifest.json').read_text());v=json.loads((n/'run'/m['points']['ABC_half_stage2']['path']).read_text());print(v['training'])
