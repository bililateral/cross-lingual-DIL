"""Format saved fields only. Run on Linux with the completed evaluation."""
from pathlib import Path
import json
import sys

r=Path(sys.argv[1]);x=json.loads((r/'result.json').read_text());cols=x['columns']
names=('S_default','H_default','S_tuned','H_pos_tuned')
fields={'O':'最终两个旧域','N':'第二/三阶段新域刚学完','Z':'最后新域','A_acq':'两个旧域刚学完',
        'F':'旧域刚学完减最终','F_first':'首域刚学完减最终','D2':'学第二域造成的首域变化','D3':'学第三域造成的两旧域变化','final_mean':'最终三域均值'}
def fmt(v):return 'N/A' if v is None else f'{v:.9f}'
lines=['四系统旧开发集完整指标：仅回顾分析，非独立确认。',
       '60个共同群；9固定轨迹/系统。下列表覆盖22指标、3角色、9端点与全部逐阶段/域/顺序/种子。',
       '排序主读raw；概率/阈值分类主读stage-cal；first-cal诊断。Brier/log_loss越低越好。',
       '差值均为所列前者减后者。F等为字面初始减最终；对损失指标，正值反而表示改善。',
       '95%区间为固定开发集及已选轨迹的条件区间，未调整全部辅助比较的多重性。','']
for role in ('raw','stage-cal','first-cal'):
    for field,label in fields.items():
        lines += [f'[{role}] {field}：{label}','指标\t'+'\t'.join(names)]
        for col in cols:lines.append(col+'\t'+'\t'.join(fmt(x['absolute'][n][role][field][col]['mean']) for n in names))
        lines.append('')
for key,roles in x['comparisons'].items():
    lines.append('方法差值：'+key)
    for role,fs in roles.items():
        for field,entries in fs.items():
            lines.append(f'[{role}] {field}：差值\t95%条件下界\t上界')
            for col in cols:
                v=entries[col];lines.append(col+'\t'+fmt(v['mean'])+'\t'+'\t'.join(map(fmt,v['conditional_95pct_interval'])))
for name,roles in x['supplementary_classification'].items():
    lines.append('未定义覆盖和辅助合并分类：'+name)
    for role,fs in roles.items():
        for field,v in fs.items():
            lines.append(f'[{role}] {field}（群评价实例含重复，非独立群数）')
            for col in ('precision','mcc'):
                c=v['coverage'][col];lines.append(f"{col}\t有效{c['valid']}/{c['total']}\t未定义{c['undefined']}\t{100*c['undefined_fraction']:.4f}%")
            lines.append('合并计数指标/时点差：'+'\t'.join(k+'='+fmt(y) for k,y in v['pooled_metrics'].items()))
            lines.append('正项计数：'+json.dumps(v['positive_terms']['counts'])+'；负项计数：'+json.dumps(v['negative_terms']['counts'] if v['negative_terms'] else None))
for rid,roles in x['per_run'].items():
    lines.append('单轨迹完整端点：'+rid)
    for role,fs in roles.items():
        lines.append('role='+role+'\n端点\t'+'\t'.join(cols))
        for field,values in fs.items():lines.append(field+'\t'+'\t'.join(map(fmt,values)))
for rid,roles in x['cells'].items():
    lines.append('单轨迹逐阶段/实际域九格：'+rid)
    for role,cells in roles.items():
        lines.append('role='+role+'\n阶段_实际域\t'+'\t'.join(cols))
        for cell,v in cells.items():lines.append(cell+'\t'+'\t'.join(map(fmt,v['values'])))
(r/'tables.zh.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('EXPORTED',len(lines),'lines')
