#!/usr/bin/env python3
"""Verify the original submission remains byte-identical after permitted audit reads."""
import argparse,datetime,hashlib,json,pathlib,shutil,zipfile
p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--root',required=True);p.add_argument('--evidence',required=True);a=p.parse_args()
archive=pathlib.Path(a.archive);root=pathlib.Path(a.root);out=pathlib.Path(a.evidence)
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    checked=[]
    names=set(z.namelist())
    for name in sorted(names):
        value=z.read(name);actual=(root/name).read_bytes()
        assert actual==value,name
        checked.append({'path':name,'bytes':len(actual),'sha256':hashlib.sha256(actual).hexdigest(),'unchanged':True})
    extras=sorted(p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.relative_to(root).as_posix() not in names)
    assert all(x.startswith('reports/reviewer_test_result_replay/') for x in extras), extras
# Preserve the original, supplied readonly audit's actual streams as history too.
histbase=pathlib.Path('reports/seller_alias_continual/20260928/test_result')
for rel in ['analysis_command.json','analysis_execution/stdout.log','analysis_execution/stderr.log','analysis_execution/exit_status.txt','analysis_execution/resource_usage.log','analysis_execution/started.txt','analysis_execution/finished.txt']:
    src=root/histbase/rel;dst=out/'history/project_readonly_analysis'/rel
    dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
receipt={'status':'PASS','checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
'archive':{'name':archive.name,'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()},
'original_members':len(checked),'all_original_members_byte_identical':True,
'new_files_only_in_permitted_replay_output':extras,'old_analysis_unchanged':True,'files':checked}
target=out/'outputs/final_source_verification.json';target.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='files'},ensure_ascii=False,indent=2))
