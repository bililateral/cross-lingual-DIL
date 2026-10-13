#!/usr/bin/env python3
"""Archive identity, exact receipt times, log summary, and reader-only diff."""
from pathlib import Path
import collections, datetime, decimal, difflib, hashlib, json, platform, zipfile

BASE=Path(__file__).resolve().parents[1]
INPUT=BASE/'input'
OUT=BASE/'independent'/'supplementary'
OUT.mkdir(exist_ok=True)

def read(name): return json.loads((INPUT/name).read_text(encoding='utf-8-sig'))
def digest(data): return hashlib.sha256(data).hexdigest()

def main():
    archive=BASE/'input_archive.zip'
    raw=archive.read_bytes()
    assert len(raw)==4285123 and digest(raw)=='261b82d9b7d26b1e93c38cdc6ac63512acc8876fd78329e7b1dfe6a2c7ded3f0'
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        names=z.namelist()
        assert len(names)==len(set(names))==303
        records=[]
        for info in z.infolist():
            value=z.read(info.filename)
            assert value==(INPUT/info.filename).read_bytes()
            records.append({'path':info.filename,'bytes':len(value),'sha256':digest(value)})
        declared=read('input_manifest.json')['files']
        actual={r['path']:r for r in records}
        assert len(declared)==302
        assert all(r==actual[r['path']] for r in declared)
    totals={}
    for kind,count in [('cpu',4),('gpu',2)]:
        seconds=[]
        for n in range(1,count+1):
            lines=(INPUT/f'qualification/evidence/{kind}/attempt{n:02}/wrapper.txt').read_text().splitlines()
            seconds.append(decimal.Decimal(lines[3])-decimal.Decimal(lines[1]))
        totals[kind]={'individual_seconds':[str(v) for v in seconds],
          'total_seconds_exact_decimal':str(sum(seconds)),
          'sum_per_attempt_ceiling':sum(int(v.to_integral_value(rounding=decimal.ROUND_CEILING)) for v in seconds)}
    lines=(INPUT/'job.console.txt').read_text().splitlines()
    events=[json.loads(line) for line in lines if line.startswith('{')]
    assert not any(any(token in line for token in ['Traceback','KeyError','TypeError','FAIL']) for line in lines)
    updates=collections.defaultdict(list)
    for row in events:
        if row.get('event')=='updates': updates[row['point']].append(row['completed'])
    assert set(updates)=={'A_shared','B_S','B_C_cf','B_C_perm'}
    assert all(values[0]==1 and values[-1]==288 and values==sorted(set(values)) for values in updates.values())
    starts=read('qualification/evidence/formal/execution_at_start.json')
    current=read('job/execution.json')
    assert starts==current
    source=read('job/completion.json')['source_files']
    assert read('job/run/startup.json')['source_files']==source
    assert read('job/run/progress.json')['source_files']==source
    assert read('qualification/evidence/formal/access_at_start.json')=={'heldout':0,'owners':0,'train':0,'valid':0}
    old=(INPUT/'analysis_attempt01/source.py').read_text().splitlines()
    new=(INPUT/'step28_crossfit_result_audit.py').read_text().splitlines()
    diff='\n'.join(difflib.unified_diff(old,new,fromfile='input/analysis_attempt01/source.py',tofile='input/step28_crossfit_result_audit.py',lineterm=''))+'\n'
    (OUT/'linux_reader_fix.diff').write_text(diff)
    # Historical native source changes are logged as identity differences only.
    native1=read('qualification/evidence/gpu/attempt01/result/result.json')
    native2=read('qualification/evidence/gpu/attempt02/result/result.json')
    oldhash={r['path']:r['sha256'] for r in native1['source_files']}
    changed=[r['path'] for r in native2['source_files'] if r['sha256']!=oldhash[r['path']]]
    wrapper=(INPUT/'job.wrapper.txt').read_text().splitlines()
    elapsed=(datetime.datetime.fromisoformat(wrapper[2])-datetime.datetime.fromisoformat(wrapper[0])).total_seconds()
    assert wrapper[1]=='exit_code=0' and elapsed==15025
    result={'status':'PASS_SUPPLEMENTARY_READ_ONLY_CHECKS','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'platform':platform.platform(),'source_sha256':digest(Path(__file__).read_bytes()),
      'archive':{'bytes':len(raw),'sha256':digest(raw),'members':303,'payloads':302,'CRC':'PASS','all_extracted_bytes_equal':True},
      'exact_qualification_wall_times':totals,
      'console':{'lines':len(lines),'json_events':len(events),'event_counts':dict(collections.Counter(r.get('event') for r in events)),
        'main_progress_by_point':{k:{'logged_points':len(v),'first':v[0],'last':v[-1]} for k,v in updates.items()},
        'exception_markers':0,'formal_wrapper_seconds':elapsed},
      'source_at_start_and_completion_equal':True,'native1_to_native2_changed_source_paths':changed,
      'linux_saved_result_reader_diff_hunks':sum(line.startswith('@@') for line in diff.splitlines()),
      'boundaries':'All checks use archived supplied files; no server, model, labels, Memory, or F/O payload access.'}
    (OUT/'archive_members_verified.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
    (OUT/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
