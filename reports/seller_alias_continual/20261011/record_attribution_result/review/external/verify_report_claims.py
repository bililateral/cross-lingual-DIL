"""Check the main report's displayed tables against independently recomputed values.
Only reads uploaded report/JSON and the reviewer's saved NumPy results.
"""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import re


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--results',type=Path,required=True)
    a=p.parse_args();root=a.root;out=a.results
    report=(root/'report.zh.txt').read_text(encoding='utf-8');lines=report.splitlines()
    e=json.loads((out/'recomputed_endpoints.json').read_text())
    stage=list(csv.DictReader((out/'recomputed_stage_domain.csv').open(encoding='utf-8-sig')))
    st={(r['point'],r['role'],r['actual_domain']):r for r in stage}
    observations=[]
    def check(line,label,value,expected):
        digits=len(value.split('.')[1]);tol=10**(-digits)
        error=float(value)-float(expected)
        observations.append(dict(line=line,label=label,reported=value,expected=float(expected),
                                 max_last_digit_rounding_or_truncation=tol,absolute_error=abs(error),
                                 matches=abs(error)<=tol+1e-14))
    section=None
    for i,line in enumerate(lines,1):
        if line.startswith('方法          O'):section='endpoints';continue
        if line.startswith('比较           O'):section='differences';continue
        if line.startswith('方法       首域'):section='acquisition';continue
        if line.startswith('                         LOGIT0.1'):section='probability';continue
        if section in ('endpoints','acquisition') and re.match(r'^(LOGIT0\.1|R0|S)\s+',line):
            arm,values=line.split(maxsplit=1);nums=re.findall(r'[+-]?\d+\.\d+',values)
            if section=='endpoints':
                for ep,val in zip(('O','N','Z','A2','final_all','F_first','F','G'),nums,strict=True):
                    check(i,arm+'/primary/'+ep+'/map',val,e['absolute'][arm]['primary'][ep]['map']['mean'])
            else:
                fields=[('first_acquired',1,0),('first_final',3,0),('second_acquired',2,1),('second_final',3,1)]
                for (label,s,arrival),val in zip(fields,nums,strict=True):
                    expected=sum(float(st[f'{o}_{arm}_stage{s}','raw',o[arrival]]['map']) for o in ('ABC','BCA','CAB'))/3
                    check(i,arm+'/'+label,val,expected)
            if arm=='S':section=None
        elif section=='differences' and re.match(r'^(R0−LOGIT|S−R0|S−LOGIT)\s',line):
            name,body=line.split(maxsplit=1);name={'R0−LOGIT':'R0-LOGIT0.1','S−R0':'S-R0','S−LOGIT':'S-LOGIT0.1'}[name]
            nums=re.findall(r'[+-]?\d+\.\d+',body)
            for j,ep in enumerate(('O','N','Z')):
                v=e['difference'][name]['primary'][ep]['map']
                for k,expected in enumerate((v['mean'],*v['conditional_95pct_interval'])):check(i,name+'/'+ep+'/'+('mean','lo','hi')[k],nums[3*j+k],expected)
        elif line.startswith(('ABC（C新）','BCA（A新）','CAB（B新）')):
            order=line[:3];groups=line.split('：',1)[1].rstrip('。').split('；')
            for arm,body in zip(('LOGIT0.1','R0','S'),groups,strict=True):
                nums=re.findall(r'[+-]?\d+\.\d+',body.split(' ',1)[1])
                for d,v in zip('ABC',nums,strict=True):check(i,order+'/'+arm+'/final/'+d,v,float(st[f'{order}_{arm}_stage3','raw',d]['map']))
        elif section=='probability':
            labels={'O Recall@5':('O','recall_at_5'),'N Recall@5':('N','recall_at_5'),
                    'O pair AP':('O','average_precision'),'N pair AP':('N','average_precision'),
                    'O stage-cal log_loss':('O','log_loss'),'N stage-cal log_loss':('N','log_loss'),
                    'O stage-cal Brier':('O','brier'),'N stage-cal Brier':('N','brier'),
                    'O group-macro F1':('O','f1'),'N group-macro F1':('N','f1'),
                    'O group-macro recall':('O','recall'),'O specificity':('O','specificity')}
            for prefix,(ep,met) in labels.items():
                if line.startswith(prefix):
                    nums=re.findall(r'[+-]?\d+\.\d+',line[len(prefix):])
                    for arm,val in zip(('LOGIT0.1','R0','S'),nums,strict=True):check(i,prefix+'/'+arm,val,e['absolute'][arm]['primary'][ep][met]['mean'])
    # Independent AST-only reconstruction of the 29-source declared dependency closure.
    paths={'scripts/step28_record_attribution_run.py','scripts/step28_record_attribution_verify.py',
           'scripts/step28_replay_diagnostics.py','tests/test_step28_record_attribution.py'}
    todo=list(paths)
    while todo:
        name=todo.pop();tree=ast.parse((root/'source'/name).read_text())
        for node in ast.walk(tree):
            mods=[x.name for x in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) else []
            for mod in mods:
                if mod and mod.startswith('step28_'):
                    child='scripts/'+mod+'.py'
                    if child not in paths:paths.add(child);todo.append(child)
    paths.update('schema/'+x for x in ('step28_record_attribution_policy.json','step28_record_replay_policy.json',
                 'step28_bge_continual_policy.json','step28_alias_ranking_policy.json','step28_chinese_base_policy.json'))
    paths.update(('docs/SELLER_ALIAS_RECORD_ATTRIBUTION.zh.md','scripts/run_step28_record_attribution_linux_20261009.sh','tests/test_step28_record_replay.py'))
    source_list=json.loads((root/'job/execution.json').read_text())['sources']
    closure_equal=paths=={r['path'] for r in source_list}
    # Mechanical native budget projection using the saved component times only.
    n=json.loads((root/'qualification/gpu/attempt01/result/result.json').read_text())['native']
    ms=n['architectures'];tr=sum(2160*v['paired_update_seconds']+9*max(0.,v['first_current_allocation_seconds']-v['paired_update_seconds']/2) for v in ms.values())
    fw=sum(n['forward_group_counts'][arm]*max(v['worst_eval_group_seconds'],v['source_seconds']) for arm,v in ms.items())
    proxy=9072*max(v['save_two_groups_seconds'] for v in ms.values())
    projection=1.25*(tr+fw+3600+proxy)
    failures=[r for r in observations if not r['matches']]
    result={'scope':'Main displayed tables only, plus source dependency closure and saved-measurement arithmetic. No new scientific experiment.',
            'checked_table_cells':len(observations),'inconsistent_cells':failures,'observations':observations,
            'source_dependency_closure':{'count':len(paths),'equals_declared_29':closure_equal},
            'native_projection':{'training_seconds':tr,'forward_seconds':fw,'array_save_proxy_seconds':proxy,
                                 'allowance_seconds':3600,'margin_factor':1.25,'recomputed_seconds':projection,
                                 'saved_seconds':n['projected_total_seconds'],'absolute_error':abs(projection-n['projected_total_seconds'])},
            'classification':'One nonblocking report transcription defect' if len(failures)==1 else 'Inspect detailed mismatches'}
    (out/'report_table_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='observations'},ensure_ascii=False,indent=2))
    assert len(observations)==126
    assert closure_equal and abs(projection-n['projected_total_seconds'])<1e-9


if __name__=='__main__':main()
