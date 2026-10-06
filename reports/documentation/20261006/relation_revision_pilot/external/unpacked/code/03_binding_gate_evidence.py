"""Independent real-import namespace/source/gate audit. NEVER call execute/native.
Temporary gate objects below are explicitly web-only schema fixtures, not approvals.
"""
import os,sys,json,ast,zipfile,hashlib,tempfile,copy,inspect
from pathlib import Path
R=Path(__file__).resolve().parents[1]; C=R/'sources/current'
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
sys.dont_write_bytecode=True;sys.path[:0]=[str(C/'scripts'),str(C/'tests')]
import step28_relation_revision_run as a
import step28_relation_revision as rev
import step28_relation_revision_admission as adm
r=a.runner;d=rev.original.data
out={'origin':'web_independent_import_checks','formal_execute_called':False,'native_called':False,'formal_gate_issued':False}
plain=['train','checkpoint','blind_gate','collect','finalize','comparisons','read_roles','complete','recover_statistics','execute','supervised','watchdog','expected_points']
bind={name:getattr(r,name).__globals__ is r.__dict__ for name in plain}
assert all(bind.values())
assert r.Budget.check.__globals__ is r.__dict__
assert r is not a.prior and r.method is not a.prior.method
assert r.method.update is rev.update and rev.update.__globals__ is rev.__dict__
assert a.prior.method is rev.original
assert a.inherited_policy.__globals__ is r.__dict__ and a.inherited_sources.__globals__ is r.__dict__
assert r.expected_points()==[f'{o}_relation_revision_stage{s}' for o in ('ABC','BCA','CAB') for s in (1,2,3)]
assert a.prior.expected_points()==[f'{o}_relation_stage{s}' for o in ('ABC','BCA','CAB') for s in (1,2,3)]
tree=ast.parse(Path(a.prior.__file__).read_text())
# Top-level aliases data/base/core/parent were resolved BEFORE the namespace binding.
attrs=sorted({n.attr for top in tree.body if isinstance(top,(ast.FunctionDef,ast.ClassDef)) for n in ast.walk(top) if isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name) and n.value.id=='method'})
assert all(hasattr(r.method,attr) for attr in attrs)
out['binding']={'inherited_function_globals':bind,'prior_unmutated':True,'new_update_identity':True,'method_attributes_used':attrs,'new_point_names':r.expected_points(),'policy':a.policy()}
sources=a.sources(); native=adm.native_sources()
cpu_rec=json.loads((C/'reports/documentation/20261006/relation_revision_pilot/evidence/reports/cpu/result.json').read_text())
gpu_rec=json.loads((C/'reports/documentation/20261006/relation_revision_pilot/evidence/reports/native/result.json').read_text())
assert sources==cpu_rec['source_files']; assert native==gpu_rec['scientific_sources']
# Independently collect all loaded local production modules and dynamically loaded workflow.
loaded={Path(v.__file__).resolve().relative_to(C).as_posix() for v in sys.modules.values() if getattr(v,'__file__',None) and Path(v.__file__).resolve().is_relative_to(C/'scripts')}
loaded.add(Path(r.__file__).resolve().relative_to(C).as_posix())
assert loaded.issubset({x['path'] for x in sources})
out['sources']={'formal_count':len(sources),'native_count':len(native),'formal_matches_cpu':True,'native_matches_gpu':True,'loaded_local_production_files':sorted(loaded),'loaded_closure_covered':True}
# Root deployed source list, K1 deployed list, and returned evidence all match supplied bytes.
for tag,folder in [('admission','reports/documentation/20261006/relation_revision_pilot/evidence'),('k1','reports/documentation/20261006/relation_revision/k1')]:
 p=C/folder
 rows=json.loads((p/'manifest.json').read_text())
 resolutions=[]
 for row in rows:
  q=C/row['path']
  if not q.is_file():
   q=(p/row['path']) if (p/row['path']).is_file() else (R/'sources/history'/row['path'])
  payload=q.read_bytes()
  assert len(payload)==row['bytes'] and hashlib.sha256(payload).hexdigest()==row['sha256'],row['path']
  resolutions.append({'declared_path':row['path'],'resolved_source':q.relative_to(R/'sources').as_posix()})
 out[tag+'_deployment']={'count':len(rows),'matches_declared_deployed_bytes':True,'resolutions':resolutions}
recovered=C/'reports/documentation/20261006/relation_revision_pilot/evidence'
rows=json.loads((recovered/'recovery_manifest.json').read_text())
archive=C/'reports/documentation/20261006/relation_revision_pilot/admission_evidence.zip'
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None
 assert {x.filename for x in z.infolist() if not x.is_dir()}=={x['path'] for x in rows}|{'recovery_manifest.json'}
 assert z.read('recovery_manifest.json')==(recovered/'recovery_manifest.json').read_bytes()
 for row in rows:
  payload=z.read(row['path']);assert len(payload)==row['bytes'] and hashlib.sha256(payload).hexdigest()==row['sha256']
  assert payload==(recovered/row['path']).read_bytes()
out['returned_evidence']={'payload_count':len(rows),'archive_bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'all_match':True}
# Real submitted evidence is read, but the approval object is a local temporary fixture.
with tempfile.TemporaryDirectory(prefix='WEB_ONLY_GATE_FIXTURE_',dir=C/'reports') as temp:
 t=Path(temp);g=t/'fixture.json';job=t/'not_an_actual_job';p=a.policy()
 nr=recovered/'reports/native/result.json';cr=recovered/'reports/cpu/result.json'
 gate={'status':'APPROVED_RELATION_REVISION_PILOT','source_files':sources,'job':job.relative_to(C).as_posix(),'runtime':p['runtime'],'supervision':p['supervision'],'review_disposition':'NO_OPEN_BLOCKERS','native':d.record(nr,C),'integration_cpu':d.record(cr,C),'fixture_only':True,'authority':'NONE_WEB_AUDIT_ONLY_DO_NOT_EXECUTE'}
 d.write_json(g,gate);assert a.validate_gate(job,g)==p
 results={'current_submitted_evidence_accepted':True}
 for key,value in [('status','APPROVED_RELATION_PILOT'),('review_disposition','OPEN_BLOCKER'),('job','wrong_job')]:
  altered=copy.deepcopy(gate);altered[key]=value;d.write_json(g,altered)
  try: a.validate_gate(job,g)
  except ValueError: results[key+'_mismatch_rejected']=True
  else: raise AssertionError(key)
 for kind,record in [('native',gpu_rec),('integration_cpu',cpu_rec)]:
  for fault in ('wrong_mode','wrong_source'):
   modified=copy.deepcopy(record)
   if fault=='wrong_mode':modified['mode']='cpu' if kind=='native' else 'native'
   elif kind=='native':modified['scientific_sources']['scripts/step28_relation_revision.py']='0'*64
   else:modified['source_files'][0]['sha256']='0'*64
   f=t/(kind+'_'+fault+'.json');d.write_json(f,modified)
   altered=copy.deepcopy(gate);altered[kind]=d.record(f,C);d.write_json(g,altered)
   try:a.validate_gate(job,g)
   except ValueError:results[kind+'_'+fault+'_rejected']=True
   else:raise AssertionError((kind,fault))
 out['gate_cases']=results
# Operational numerical evidence is checked as a receipt, not recreated on BGE.
x=gpu_rec['native'];v=x['optimizer']
assert v['adam_step']==1 and all(q=={'texts':448,'minimum':256,'maximum':256} for q in x['input_shapes'])
assert len(x['probe_max_parameter_changes'])==3 and all(v>0 for v in x['probe_max_parameter_changes'])
assert all(len(g)==3 and all(value>0 for value in g) for g in x['history_only_gradient_norms'].values())
# FP32 R weight multiply happens before FP64 addition: preserve correct rounding interpretation.
import numpy as np
weighted=float(np.float32(np.float32(.1)*np.float32(v['history_rank'])))
errors={'history_total_vs_fp32_weighted_R':abs(v['history_total']-v['history_compressed']-weighted),'current_B':abs(v['current_total']-v['current_bce']-v['current_rank']-.5*v['current_hard']),'total':abs(v['total']-v['current_total']-v['history_total'])}
assert max(errors.values())<1e-6
out['native_receipt']={'arithmetic_errors':errors,'max_reserved_GiB':gpu_rec['max_reserved_bytes']/2**30,'max_allocated_GiB':gpu_rec['max_allocated_bytes']/2**30,'rss_GiB':gpu_rec['peak_rss_bytes']/2**30,'update_seconds_including_extra_VJPs':x['update_seconds'],'dtypes':x['dtypes'],'not_reexecuted':True}
(R/'outputs/03_binding_gate_evidence.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(out,ensure_ascii=False,indent=2))
