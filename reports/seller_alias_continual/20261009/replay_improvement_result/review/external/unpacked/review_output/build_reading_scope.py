"""Consolidate actual reading/processing records; do not equate hashing to review."""
from collections import Counter,defaultdict
import hashlib
import io
import json
from pathlib import Path
import zipfile

OUT=Path(__file__).resolve().parent;BASE=OUT.parent;ROOT=BASE/'input'
human=defaultdict(list);machine=defaultdict(set)
for line in (OUT/'reading_log.jsonl').read_text().splitlines():
    r=json.loads(line);human[r['path']].append(r)
for path in OUT.glob('*_run_*.json'):
    for r in json.loads(path.read_text()).get('input_files',[]):machine[r['path']].update(r.get('modes',[]))
for name in ('display_audit.json','provenance_audit.json'):
    for r in json.loads((OUT/name).read_text()).get('inputs',[]):machine[r['path']].update(r.get('modes',[]))

def range_text(ranges):
    if not ranges:return ''
    ranges=sorted(ranges);merged=[]
    for s,e in ranges:
        if merged and s<=merged[-1][1]+1:merged[-1][1]=max(merged[-1][1],e)
        else:merged.append([s,e])
    return ', '.join(str(s) if s==e else f'{s}-{e}' for s,e in merged)

def entry(path,container='P0',name=None):
    rel=str(path.relative_to(BASE));records=human.get(rel,[]);ranges=[];details=[];full=False
    for r in records:
        mode=r['mode'];d=r['detail']
        if mode=='full_text_read':full=True
        if isinstance(d,dict) and 'start_line' in d:ranges.append((d['start_line'],d['end_line']))
        else:details.append({'mode':mode,'detail':d})
    if ranges and path.suffix in ('.py','.txt','.md','.sh'):
        lines=path.read_text(encoding='utf-8-sig').splitlines()
        covered={i for s,e in ranges for i in range(s,e+1)}
        if all(i in covered or not text.strip() for i,text in enumerate(lines,1)):full=True
    modes=sorted(machine.get(rel,[]))
    # "Full" only means visible text read, never whole-file bytes hashed.
    h='全文阅读' if full else '节选阅读' if ranges or any('fields_read' in r['mode'] or 'content_read' in r['mode'] for r in records) else '结构/字面搜索' if records else '未人工阅读'
    if 'all_saved_array_values_processed' in modes:c='全部保存数组值检查/复算'
    elif 'all_JSON_fields_processed' in modes:c='JSON机器读取与列明字段核验（不表示每一字段均语义重审）'
    elif 'all_formatted_fields_verified' in modes:c='全部展示字段逐项核对'
    elif 'all_log_lines_scanned' in modes:c='全部日志行扫描'
    elif any('AST_' in x for x in modes):c='AST/运行设置字面搜索'
    elif any('nested_archive' in x for x in modes):c='容器身份与选定成员读取'
    elif modes:c='字节/来源身份或收据核验'
    else:c='仅外层清单身份核验'
    b=path.read_bytes()
    return {'container':container,'path':name or str(path.relative_to(ROOT)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),
            'human_scope':h,'line_ranges':range_text(ranges),'human_detail':details,'machine_scope':c,'machine_modes':modes,
            'identity_scope':'外层包清单已逐项核对' if container=='P0' else '列明来源容器/已读成员字节'}

rows=[entry(p) for p in sorted(ROOT.rglob('*')) if p.is_file()]
hr=json.loads((OUT/'historical_inputs.json').read_text())
selected={x['path']:x for x in hr['selected_members']}
outer=zipfile.ZipFile(ROOT/'background/historical_result_input.zip')
inner=zipfile.ZipFile(io.BytesIO(outer.read('background/original_result_input.zip')))
for container,z in [('P1',outer),('P2',inner)]:
    for info in z.infolist():
        if info.is_dir():continue
        chosen=(container=='P1' and info.filename=='background/original_result_input.zip') or (container=='P2' and info.filename in selected)
        hp=BASE/'history'/info.filename
        if container=='P2' and hp.exists() and str(hp.relative_to(BASE)) in human:
            row=entry(hp,container,info.filename);rows.append(row);continue
        record=selected.get(info.filename) if container=='P2' else None
        rows.append({'container':container,'path':info.filename,'bytes':info.file_size,'sha256':record['sha256'] if record else hr['inner_sha256'] if chosen else None,
                     'human_scope':'未人工阅读','line_ranges':'','human_detail':[],
                     'machine_scope':'选定保存矩阵/登记身份计算' if chosen and container=='P2' else '已核实内层容器身份，读取选定成员' if chosen else '未展开/未读取，只有ZIP成员目录信息',
                     'machine_modes':[],'identity_scope':'选定成员另核绑定清单' if chosen else '不声称已逐件哈希或语义检查'})
summary={container:dict(Counter(r['human_scope'] for r in rows if r['container']==container)) for container in ('P0','P1','P2')}
result={'scope':'当前实例的实际阅读范围。未递归展开 P2/history/implementation_review_input.zip。机器处理不能等同全文人工阅读。',
        'outer_archive_sha256':'5b410fa7b80fb9fd89a02555a898014f3877e53620dfc170a2ffcc02870da2d1',
        'containers':{'P0':'本轮上传包','P1':'P0/background/historical_result_input.zip','P2':'P1/background/original_result_input.zip'},
        'summary':summary,'member_count':len(rows),'files':rows}
(OUT/'READING_SCOPE.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
lines=['逐文件实际阅读范围','',result['scope'],'','计数：'+json.dumps(summary,ensure_ascii=False),'']
for r in rows:
    lines.append(f"{r['container']}/{r['path']} | {r['bytes']} bytes | SHA256={r['sha256'] or '未另核'}")
    lines.append(f"  人工：{r['human_scope']}；行：{r['line_ranges'] or '见结构/字段说明'}；机器：{r['machine_scope']}")
    if r['human_detail']:lines.append('  说明：'+json.dumps(r['human_detail'],ensure_ascii=False))
(OUT/'READING_SCOPE.zh.txt').write_text('\n'.join(lines)+'\n')
print(json.dumps({'member_count':len(rows),'summary':summary},ensure_ascii=False,indent=2))
