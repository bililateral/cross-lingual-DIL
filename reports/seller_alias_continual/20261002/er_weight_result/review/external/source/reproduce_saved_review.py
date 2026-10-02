"""Offline replay of this external review, into a NEW directory only.
Requires existing numpy; torch is used only by a three-parameter synthetic toy.
Does not install packages, connect to a server, or run a formal training entry.
"""
from __future__ import annotations
import argparse,datetime,hashlib,json,os,pathlib,platform,subprocess,sys,time,zipfile
EXPECTED='5de5711fbe6d20107e8607e8c4aec85a2ac47293e845af9c8c6ece1358631db0'
NEW='reports/seller_alias_continual/20261001/er_weight_execution/20261001_144452/job'
OLD='reports/seller_alias_continual/20260930/bge_continual_execution/20260930_143700/job'
RES='reports/seller_alias_continual/20261002/er_weight_result'
def main()->None:
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=pathlib.Path,required=True);parser.add_argument('--cpu',type=int,default=None);args=parser.parse_args()
 bundle=pathlib.Path(__file__).resolve().parent.parent;source=bundle/'source';archive=bundle/'original/er_weight_result_review.zip'
 assert hashlib.sha256(archive.read_bytes()).hexdigest()==EXPECTED,'Received ZIP identity changed'
 allowed=sorted(os.sched_getaffinity(0));cpu=min(allowed) if args.cpu is None else args.cpu
 if cpu not in allowed:raise ValueError(f'CPU {cpu} is not available; allowed={allowed}')
 work=args.output.resolve();work.mkdir(parents=True,exist_ok=False);project=work/'input';project.mkdir();results=work/'results';results.mkdir();logs=work/'logs';logs.mkdir()
 with zipfile.ZipFile(archive) as z:
  assert len(z.infolist())==365 and z.testzip() is None
  for info in z.infolist():
   target=(project/info.filename).resolve()
   if not target.is_relative_to(project):raise ValueError('Unexpected archive path')
  z.extractall(project)
 env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
 def run(name:str,command:list[str])->None:
  dest=logs/name;dest.mkdir();started=datetime.datetime.now(datetime.timezone.utc).isoformat();clock=time.monotonic()
  with (dest/'stdout.log').open('wb') as so,(dest/'stderr.log').open('wb') as se:
   process=subprocess.run(command,cwd=project,env=env,stdout=so,stderr=se,check=False)
  meta=dict(command=command,cwd=str(project),started_at_utc=started,finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-clock,exit_code=process.returncode,python=sys.version,platform=platform.platform(),cpu=cpu,origin='OFFLINE_REPLAY_NOT_PROJECT_LINUX')
  (dest/'execution.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta),flush=True)
  if process.returncode:raise RuntimeError(f'{name} failed; preserved {dest}')
 run('independent',[sys.executable,str(source/'independent_result_audit.py'),'--project',str(project),'--zip',str(archive),'--output',str(results/'independent_v1'),'--cpu',str(cpu)])
 # The historical directory basename is retained for compatibility with the
 # supplemental reader; actual CPU, never inferred from this name, is in logs.
 run('supplied_audit',[sys.executable,'-B','scripts/step28_er_weight_audit.py','--project',str(project),'--job',str(project/NEW),'--baseline',str(project/OLD),'--inventory',str(project/RES/'return_inventory.json'),'--output',str(results/'supplied_audit_cpu0'),'--cpu',str(cpu)])
 run('hand_reference',[sys.executable,str(source/'hand_objective_reference_v3.py'),'--project',str(project),'--output',str(results/'hand_reference_v3'),'--cpu',str(cpu)])
 run('supplemental',[sys.executable,str(source/'supplemental_result_audit_v2.py'),'--project',str(project),'--independent',str(results/'independent_v1'),'--output',str(results/'supplemental_v2'),'--cpu',str(cpu)])
 with zipfile.ZipFile(archive) as z:
  for info in z.infolist():assert (project/info.filename).read_bytes()==z.read(info.filename)
 print('PASS_OFFLINE_REPLAY_SAVED_RESULTS_AND_HAND_FIXTURES_INPUT_UNCHANGED',flush=True)
if __name__=='__main__':main()
