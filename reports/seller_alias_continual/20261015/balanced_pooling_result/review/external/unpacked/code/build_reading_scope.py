#!/usr/bin/env python3
"""Build per-file read-scope and check included context bindings; this is metadata processing only."""
import csv, hashlib, json, zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT.parent/'input'
def merged(xs):
    out=[]
    for a,b in sorted(xs):
        if out and a<=out[-1][1]+1:out[-1][1]=max(out[-1][1],b)
        else:out.append([a,b])
    return out
reads={}
for line in (ROOT/'raw/manual_reads.jsonl').read_text().splitlines():
    r=json.loads(line);reads.setdefault(r['path'],[]).append([r['start'],r['end']])
prog={r['path']:r['purposes'] for r in json.loads((ROOT/'raw/attempt02/programmatic_read_scope.json').read_text())}
supp=json.loads((ROOT/'raw/supplemental_evidence.json').read_text())
for n in supp['programmatic_read_files']:prog.setdefault(n,[]).append('targeted supplemental structured evidence; see supplemental_evidence.py')
rows=[]
for r in csv.DictReader((ROOT/'raw/attempt02/input_members.tsv').open(),delimiter='\t'):
    n=r['path'];p=INPUT/n
    notes=[]
    if n in reads:notes.append('按所列范围提取并针对本轮结果语义审读；不是整仓逐行重审。提取日志可能包含重复窗口，范围不等于对每个字节作语义证明。')
    if n.endswith('.npy'):notes.append('完整数值数组读取；形状/有限性全查，具体重算用途见程序，不代表从正式标签重建。')
    if n=='result/analysis/comparison.png':notes.append('实际打开图像：上半部两域最终MAP差；下半部仅B阶段两域valid MAP六轮；并非全部诊断图。')
    if n=='result/analysis/comparison.pdf':notes.append('仅字节身份；没有渲染/阅读PDF。对应PNG已查看。')
    if n.endswith('qualification/gpu/attempt01/result/result.json'):notes.append('读取30来源、费用/三更新、两组件梯度位移、恢复/下一状态比较摘要和分离困难表证据；不重跑原生或逐张量模型比较。')
    if not notes:notes.append('操作范围以programmatic_operations为准；没有列出的语义审读不作已完成声明。')
    rows.append(dict(path=n,bytes=int(r['bytes']),sha256=r['sha256'],text_extracted_ranges=merged(reads.get(n,[])),programmatic_operations=prog.get(n,[]),scope_note=' '.join(notes)))
(ROOT/'reading_scope.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
with (ROOT/'reading_scope.tsv').open('w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');w.writeheader()
    for r in rows:w.writerow({**r,'text_extracted_ranges':';'.join(f'{a}-{b}' for a,b in r['text_extracted_ranges']),'programmatic_operations':'; '.join(r['programmatic_operations'])})
g=json.loads((INPUT/'result/formal_gate.json').read_text());bindings=[]
def walk(x,field):
    if isinstance(x,dict):
        if set(['path','bytes','sha256']).issubset(x):
            name='context/'+x['path'];p=INPUT/name
            if p.exists():
                blob=p.read_bytes();assert len(blob)==x['bytes'] and hashlib.sha256(blob).hexdigest()==x['sha256']
                bindings.append({'field':field,'input_path':name,'present':True,'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest()})
            elif not field.startswith('source_files'):
                bindings.append({'field':field,'reference_path':x['path'],'present':False,'disposition':'Not re-read from payload; bounded inheritance via included closed dispositions, no assertion of direct rehash.'})
        else:
            for k,v in x.items():walk(v,field+'.'+k if field else k)
    elif isinstance(x,list):
        for i,v in enumerate(x):walk(v,f'{field}[{i}]')
walk(g,'')
(ROOT/'raw/context_gate_bindings.json').write_text(json.dumps(bindings,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'input_members':len(rows),'files_with_text_extraction':len(reads),'context_binding_present':sum(r['present'] for r in bindings),'context_binding_inherited_not_present':sum(not r['present'] for r in bindings),'scope':'Input membership/reading metadata; no qualification or scientific recalculation'},ensure_ascii=False,indent=2))
