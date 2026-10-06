"""Independent local access-attempt and reused resource-guard checks.
Every group, ledger and failure here is handwritten; no formal loader is called.
"""
import os,sys,time,json,tempfile,unittest
from pathlib import Path
from unittest import mock
R=Path(__file__).resolve().parents[1];C=R/'sources/current'
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',RELATION_STARTED_EPOCH=str(time.time()))
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});sys.dont_write_bytecode=True
sys.path[:0]=[str(C/'tests'),str(C/'scripts')]
import step28_relation_revision_run as a
import test_step28_relation_memory_run as tests
r=a.runner;d=r.data
out={'origin':'independent_web_handwritten_failures','formal_inputs':False,'cases':{}}
for split,ledgerkey in [('train','train'),('development','valid')]:
 with tempfile.TemporaryDirectory(prefix='WEB_ACCESS_ONLY_',dir=C/'reports') as temp:
  job=Path(temp);d.write_json(job/'access.json',dict(train=0,valid=0,heldout=0,owners=0))
  calls=[]
  def loader(groups,c,split):calls.append(split);raise RuntimeError('handwritten parser failure')
  try:r.previous.parse_once(job,[],{},split,loader=loader)
  except RuntimeError as exc:assert 'handwritten' in str(exc)
  else:raise AssertionError('failure not propagated')
  ledger=d.read_json(job/'access.json');assert ledger[ledgerkey]==1
  try:r.previous.parse_once(job,[],{},split,loader=loader)
  except ValueError as exc:assert 'already been attempted' in str(exc)
  else:raise AssertionError('second parse allowed')
  assert len(calls)==1
  out['cases'][split]={'attempt_consumed_before_failure':True,'second_attempt_rejected':True,'loader_calls':len(calls),'ledger':ledger}
with mock.patch.object(tests,'run',r):
 suite=unittest.TestSuite([tests.IntegrationTests('test_resource_paths_snapshot_and_stop_on_ledger_failure')])
 with (R/'outputs/05_resource_guard.unittest.txt').open('w') as f:
  result=unittest.TextTestRunner(stream=f,verbosity=2).run(suite)
 assert result.wasSuccessful()
out['resource_guard']={'original_test_bound_to_new_runner':True,'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'status':'PASS_WEB_CPU_ONLY'}
# Distinguish saved working-point pooled counts from group-macro averages.
from fractions import Fraction
# Two symbolic groups demonstrate noncommutativity without actual labels.
counts=[dict(tp=1,fp=0,fn=19,tn=358),dict(tp=15,fp=30,fn=5,tn=328)]
macro=sum(Fraction(2*x['tp'],2*x['tp']+x['fp']+x['fn']) for x in counts)/2
merged={k:sum(x[k] for x in counts) for k in counts[0]}
pooled=Fraction(2*merged['tp'],2*merged['tp']+merged['fp']+merged['fn'])
assert macro!=pooled
out['macro_vs_pooled_example']={'counts':counts,'macro_f1':float(macro),'pooled_f1':float(pooled),'not_effects':True}
(R/'outputs/05_access_and_budget_guards.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n');print(json.dumps(out,indent=2,ensure_ascii=False))
