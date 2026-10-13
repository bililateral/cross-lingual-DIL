"""Build per-member reading ranges from this review's recorded reads; hashes != reading."""
from pathlib import Path
import json, hashlib
root=Path(__file__).resolve().parents[1]
source=root/'reference_input'
events=[json.loads(x) for x in (root/'audit/read_events.jsonl').read_text().splitlines()]
seen={}
for e in events:seen.setdefault(e['path'],[]).append((e['start'],e['end']))
def merge(ranges):
 out=[]
 for a,b in sorted(ranges):
  if b<a:continue
  if out and a<=out[-1][1]+1:out[-1][1]=max(b,out[-1][1])
  else:out.append([a,b])
 return out
def reason(path, status):
 if status=='FULL_TEXT':return '全文读取；重复出现的阅读区间已合并。'
 if status=='EMPTY':return '实测空文件；不存在可读正文，不当作缺失内容。'
 if path.endswith('.tar.gz'):return '二进制归档逐成员机械核对；29来源及授权与当前相同。各内部文本的阅读范围见其外层同路径条目。'
 if path.endswith('source_manifest.json') or path=='INPUT_MANIFEST.json':return '完整结构化解析并逐项校验路径、大小和SHA；不把身份清单的机械处理等同科研正文阅读。'
 if status=='PARTIAL_TEXT':
  if path.startswith('scripts/'):return '只读当前入口调用到的函数及必要上下文；未读区间为旧入口／旧算法或无本轮具体矛盾的历史实现，不宣称全文复审。'
  if path.startswith('tests/'):return '只读本轮复用的手写输入、小网络和直接参考函数及必要上下文；其余旧测试未重审或执行。'
  if path.startswith('schema/'):return '读取继承配置中与本轮模型／输入／优化等有关段；其余历史实验包装按未变边界复用，仅校验完整文件字节。'
  if 'CROSSFIT' in path:return '只读旧A/B编排、读取、恢复、评价依据；不重审其折外目标或旧结果。'
  return '只读适用于当前实现及审查边界的条款；其余通用／历史部分不扩审。'
 if '/query_replay/review/external/unpacked/repro/' in path:return '旧静态审查算例材料，仅核对随包字节并保留；本轮没有重新执行或据此宣称当前实现合格。'
 if '/query_replay/review/external/unpacked/' in path:return '旧静态审查附属材料，仅核对随包字节；当前承接依据为已全文阅读的review及minimal_edits和主审处置。'
 if '/qualification/gpu/' in path:return '旧C_cf GPU原始结果，不是新C_q资格；本轮未重算旧测量，当前预算只独立核对新公式与调用数。'
 if '/20261012/' in path:return '旧C_cf审查／资格背景未重开；本轮A/B行为依据读旧合同相关段和实际复用函数。'
 if path.startswith('scripts/'):return '机械核对来源／函数导航；此旧入口或未激活分支未作正文逐行审查，不作为新方法资格证明。'
 if 'recovery' in path:return '旧静态回收凭据未重开核验；仅保留原件与输入清单机械匹配。'
 if path.startswith('docs/'):return '旧实验文档未通读；本轮以当前C_q合同及实际直接依赖为准，不追溯全历史。'
 return '仅作本次归档身份机械核对，未据该未读正文作独立事实或效果裁决。'
rows=[]
for p in sorted(source.rglob('*')):
 if not p.is_file():continue
 rel=p.relative_to(source).as_posix();raw=p.read_bytes()
 try:total=len(raw.decode('utf-8').splitlines())
 except UnicodeDecodeError:total=None
 ranges=merge(seen.get(rel,[]));read_n=sum(b-a+1 for a,b in ranges)
 status='BINARY_MECHANICAL' if total is None else ('EMPTY' if total==0 else ('FULL_TEXT' if read_n==total else ('PARTIAL_TEXT' if ranges else 'MECHANICAL_ONLY')))
 unread=[]
 if total:
  start=1
  for a,b in ranges:
   if a>start:unread.append([start,a-1])
   start=max(start,b+1)
  if start<=total:unread.append([start,total])
 rows.append({'path':rel,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'total_lines':total,'status':status,'read_ranges':ranges,'unread_ranges':unread,'mechanical_identity_verified':True,'note':reason(rel,status)})
assert len(rows)==80
out={'scope':'All 80 current ZIP members. UTF-8 one-based inclusive lines; archive hashing is not reading. Empty files and binary archives distinct.','records':rows}
(root/'audit/reading_scope.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
fmt=lambda rs:'；'.join(str(a) if a==b else f'{a}–{b}' for a,b in rs) or '无'
lines=['# 逐文件实际阅读范围','', '列全80个输入成员。行号为原UTF-8文本1起始；完整身份机械核对不等于通读。源文件的已读范围合并重复窗口，未读补集明确列出。JSON长单行的全文结构化展示记为该原始一行。未把旧算例或旧GPU结果当本轮新证据。','', '| 文件 | 状态／总行数 | 已读行 | 未读行 | 范围说明 |','|---|---|---|---|---|']
for r in rows:lines.append(f"| `{r['path']}` | {r['status']} / {r['total_lines'] if r['total_lines'] is not None else '二进制'} | {fmt(r['read_ranges'])} | {fmt(r['unread_ranges']) if r['total_lines'] is not None else '不适用'} | {r['note']} |")
(root/'audit/reading_scope.zh.md').write_text('\n'.join(lines)+'\n')
from collections import Counter
print(json.dumps(dict(Counter(r['status'] for r in rows)),ensure_ascii=False))
