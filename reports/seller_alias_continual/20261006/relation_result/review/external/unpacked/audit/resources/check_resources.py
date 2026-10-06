"""Read-only stdlib audit of submitted receipts; no Torch, model or formal loader.

Actual model/cache payloads are deliberately absent. Their recorded identities
are cross-compared, never represented as payload hashes computed here.
"""
from pathlib import Path
from datetime import datetime
import hashlib
import json

ROOT = Path('/workspace/scratch/f4d639b3b473/relation_memory_result_review')
RES = ROOT / 'reports/seller_alias_continual/20261006/relation_result'
OUT = Path(__file__).parent

def read(path):
    return json.loads(path.read_text())

def verify(path, record):
    raw = path.read_bytes()
    assert len(raw) == record['bytes'], str(path)
    assert hashlib.sha256(raw).hexdigest() == record['sha256'], str(path)

completion = read(RES/'job/completion.json')
resource = read(RES/'job/resource.json')
execution = read(RES/'job/execution.json')
gate = read(RES/'gate.json')
manifest = read(RES/'job/run/manifest.json')
collected = read(RES/'job/evaluation/collected.json')
evaluation = read(RES/'job/evaluation/evaluation.json')
deployment = read(ROOT/'reports/documentation/20261005/relation_memory/deployment.json')
cpu = read(ROOT/'reports/documentation/20261005/relation_memory/cpu/result.json')
policy = read(RES/'source/schema/step28_relation_memory_policy.json')
native_path = ROOT/gate['native']['path']
native = read(native_path)
verify(native_path, gate['native'])
verify(ROOT/gate['integration_cpu']['path'], gate['integration_cpu'])
verify(RES/'gate.json', execution['gate'])
verify(RES/'job/evaluation/evaluation.json', completion['evaluation'])

sources = gate['source_files']
assert len(sources) == 23
for record in sources:
    verify(RES/'source'/record['path'], record)
for rows in (execution['sources'], manifest['source_files'], collected['source_files'],
             evaluation['source_files'], deployment['source_files'], cpu['source_files']):
    assert rows == sources
core = RES/'source/scripts/step28_relation_memory.py'
assert hashlib.sha256(core.read_bytes()).hexdigest() == native['scientific_sources']['scripts/step28_relation_memory.py']
assert gate['runtime'] == policy['runtime']

wrapper = dict(line.split('=',1) for line in (RES/'job.wrapper.txt').read_text().splitlines())
wall = (datetime.fromisoformat(wrapper['ended_at']) - datetime.fromisoformat(wrapper['started_at'])).total_seconds()
assert wall == float(wrapper['total_wall_seconds']) == 13850
assert wrapper['exit_code'] == '0'
assert 0 <= wall-completion['budget']['elapsed_seconds'] < 2
assert completion['status'] == 'COMPLETE_RELATION_FIXED_POINT'
assert evaluation['status'] == 'STATISTICS_COMPLETE_REQUIRES_VALID_COMPLETION'
assert completion['physical_updates'] == manifest['physical_updates'] == 2592
assert manifest['gradient_group_presentations'] == 4320
assert completion['worth_matched_replay'] is False
assert read(RES/'job/access.json') == completion['access'] == dict(train=1,valid=1,heldout=0,owners=0)
before = read(RES/'job/before_valid.json')
assert before['status'] == 'PASS_COMPLETE_BLIND_GATE' and before['points'] == 9
assert before['access'] == dict(train=1,valid=0,heldout=0,owners=0)
verify(RES/'job'/before['manifest']['path'], before['manifest'])

logs = [json.loads(line) for line in (RES/'job.console.txt').read_text().splitlines() if line.startswith('{')]
expected = [(o,s,k) for o in ('ABC','BCA','CAB') for s in (1,2,3) for k in range(24,289,24)]
assert [(r['order'],r['stage'],r['step']) for r in logs] == expected
assert len(logs) == 108
peak_keys = ['peak_cuda_allocated_bytes','peak_cuda_reserved_bytes','peak_observed_bytes','peak_rss_bytes']
for key in ['elapsed_seconds',*peak_keys]:
    sequence = [r[key] for r in logs] + [completion['budget'][key],resource[key]]
    assert all(a <= b for a,b in zip(sequence,sequence[1:])), key
limits = {'peak_cuda_reserved_bytes':28*2**30,'peak_rss_bytes':64*2**30,
          'peak_observed_bytes':24*2**30,'elapsed_seconds':86400}
assert all(resource[key] < value for key,value in limits.items())

inventory = read(RES/'inventory.json')
inventory_map = {r['path']:r for r in inventory['files']}
models = {Path(r['path']).name:r for r in inventory['retained_model_files']}
model_bytes = 0
memory_bytes = []
record_crosschecks = []
for name in [f'{o}_relation_stage{s}' for o in ('ABC','BCA','CAB') for s in (1,2,3)]:
    point_path = RES/'job/run/points'/(name+'.json')
    point = read(point_path)
    verify(point_path, manifest['points'][name])
    assert point['full_restore_verified'] is True
    assert point['intermediate_deleted_bytes'] == point['full_state']['bytes']
    model = models[name+'.pt']
    assert (model['bytes'],model['sha256']) == (point['model']['bytes'],point['model']['sha256'])
    memory = inventory_map['job/run/'+point['memory']['path']]
    assert (memory['bytes'],memory['sha256']) == (point['memory']['bytes'],point['memory']['sha256'])
    assert memory['bytes'] == point['memory_bytes'] <= 2**20
    model_bytes += model['bytes']
    memory_bytes.append(memory['bytes'])
    record_crosschecks.append({'name':name,'model_identity_records_agree':True,'cache_identity_records_agree':True})

report = {'scope':'Web workspace stdlib-only receipt/source/log checks; no Torch/GPU/model/formal data access',
    'status':'PASS_SUBMITTED_RESOURCE_RECEIPT_CROSSCHECKS',
    'source_files_hashed_here':23, 'native_core_matches_frozen_core':True,
    'source_collections_equal':['gate','execution','manifest','collected','evaluation','deployment','repaired_cpu'],
    'wrapper_wall_seconds':wall,'python_elapsed_seconds':completion['budget']['elapsed_seconds'],
    'console_progress_records':len(logs),'ordered_complete_24_step_logs':True,
    'final_resource':resource, 'resource_gib':{k:resource[k]/2**30 for k in peak_keys},
    'all_observed_peaks_and_time_monotone':True, 'all_contract_bounds_pass':True,
    'access_before_valid':before['access'],'access_final':completion['access'],
    'failure_or_recovery_paths_in_submitted_inventory':[r['path'] for r in inventory['files'] if 'failure' in r['path'] or 'recovery' in r['path']],
    'model_record_count':len(models),'model_record_total_bytes':model_bytes,
    'stage_memory_record_range_bytes':[min(memory_bytes),max(memory_bytes)],
    'payload_caveat':'Model and cache payloads absent by design; only their submitted identities were cross-compared, no payload hashes recomputed.',
    'record_crosschecks':record_crosschecks}
(OUT/'resource_checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
