"""Final immutable submission/output checks. Byte reads and saved metadata only."""
from pathlib import Path
import zipfile,json,hashlib,sys
root=Path(sys.argv[1]); ev=Path(sys.argv[2]); archive=Path('/mnt/data/calibration_result_review.zip')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for name in z.namelist():
        assert (root/name).read_bytes()==z.read(name),name
    original_members=len(z.namelist())
assert (ev/'review_report.zh.md').read_bytes()==Path('/mnt/data/calibration_formal_result_review.zh.md').read_bytes()
assert (ev/'outputs/independent_v1/independent_results.json').read_bytes()==Path('/mnt/data/calibration_formal_result_independent_results.json').read_bytes()
assert (ev/'outputs/independent_v1/all_22_means.csv').read_bytes()==Path('/mnt/data/calibration_formal_result_22_metrics.csv').read_bytes()
expected={'initial_inventory_probe_capture_replay':1,'extract':0,'source_verification':0,'supplied_help':0,'supplied_replay':0,'independent_v1':0,'report_edge_checks':0,'build_report':0}
logs={}
for p in sorted((ev/'logs').glob('*/execution.json')):
    d=json.loads(p.read_text()); assert int((p.parent/'exit_status.txt').read_text())==d['exit_code']; assert (p.parent/'stdout.log').exists() and (p.parent/'stderr.log').exists()
    logs[p.parent.name]={'exit_code':d['exit_code'],'elapsed_seconds':d['elapsed_seconds'],'command':d['command']}
    assert d['exit_code']==expected.get(p.parent.name,0),p
result={'status':'PASS','original_archive_bytes':archive.stat().st_size,'original_archive_sha256':sha(archive),'all_original_members_byte_identical':original_members,'public_report_matches':True,'public_json_matches':True,'public_csv_matches':True,'executions':logs,'note':'Official initial failures copied separately under history/submitted; reviewer metadata-probe capture is explicitly a replay. No labels, fitting, model loading, or project source changes.'}
(ev/'outputs/final_integrity.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False,indent=2))
