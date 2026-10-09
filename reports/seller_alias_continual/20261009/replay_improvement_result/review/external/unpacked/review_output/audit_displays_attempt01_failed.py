"""Check saved displays, pooled counts, and scalar summation without new science."""
from collections import Counter
from datetime import datetime
import ast
import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np
import audit_saved as a

start=time.monotonic()
result={'environment':'ChatGPT web review workspace; not project Linux','status':'RUNNING'}
try:
    e=a.js(a.INPUT/'result/job/evaluation/evaluation.json')
    d=a.js(a.INPUT/'result/job/evaluation/diagnostics/overfitting.json')
    summary=a.js(a.INPUT/'result/analysis/output/summary.json')
    cols=list(a.METRICS); eps=('O','N','Z','final_all','A2','F_first','F','G')
    def fmt(x):
        lo,hi=x['conditional_95pct_interval']
        return f"{x['mean']:.9f} [{lo:.9f}, {hi:.9f}]"
    lines=['Saved results, original conditional 95% intervals; no new resampling.',
           'Primary: raw ranking/curve metrics plus stage-cal probability/classification metrics.',
           'O old; N newly learned; Z final new; A2 stage2 new; F_first/F lower is better; G higher is better.',
           'The three orders are not three seeds; shared first-stage curves are aliases.']
    for role in ('primary','raw','stage-cal','first-cal'):
        for ep in eps:
            lines+=['',f'ABSOLUTE {role} {ep}','metric\t'+'\t'.join(a.ARMS)]
            lines += [m+'\t'+'\t'.join(fmt(e['endpoints'][arm][role][ep][m]) for arm in a.ARMS) for m in cols]
    for name,c in e['comparisons'].items():
        lines+=['',f"COMPARISON {name}: {sum(c['checks'].values())}/23",'FAILED: '+', '.join(c['assessment']['failed'])]
        for role in ('primary','raw','stage-cal','first-cal'):
            for ep in eps:lines += [f'{role} {ep}']+[m+'\t'+fmt(c['delta'][role][ep][m]) for m in cols]
    a.expect(('\n'.join(lines)+'\n').encode()==a.payload(a.INPUT/'result/analysis/output/all_metrics.txt','all_formatted_fields_verified'),'all_metrics_text_exact')
    curve_rows=[]
    for point,roles in d['points'].items():
        for role,domains in roles.items():
            for domain,v in domains.items():
                if not isinstance(v,dict) or 'interpretation' not in v:continue
                for m,label in v['interpretation'].items():
                    k=v['columns'].index(m)
                    row={'point':point,'role':role,'domain':domain,'metric':m,'interpretation':label,
                         'train_groups':v['fit_or_cache_groups'],'valid_groups':v['valid_groups'],
                         'train_curve':[x[k] for x in v['train_raw_curve']],'valid_curve':[x[k] for x in v['valid_raw_curve']]}
                    for f in ('epoch1_to_6_train_benefit','epoch1_to_6_valid_benefit','epoch1_to_6_gap_widening'):
                        row[f]={'mean':v[f]['mean'][k],'conditional_95pct_interval':v[f]['conditional_95pct_interval'][k]}
                    curve_rows.append(row)
    a.compare(curve_rows,a.js(a.INPUT/'result/analysis/output/curves.json'),'curves_display_values')
    lines=['All 228 physical phase/role/domain/metric decisions, copied from saved diagnostics.',
           'Positive benefit = higher MAP/R5 or lower log_loss. Epoch1 -> epoch6 fixed in advance.']
    for row in curve_rows:
        lines += ['', ' '.join(row[k] for k in ('point','role','domain','metric','interpretation')),
                  'train: '+', '.join(f'{v:.9f}' for v in row['train_curve']),
                  'valid: '+', '.join(f'{v:.9f}' for v in row['valid_curve'])]
        for f in ('epoch1_to_6_train_benefit','epoch1_to_6_valid_benefit','epoch1_to_6_gap_widening'):lines.append(f+': '+fmt(row[f]))
    a.expect(('\n'.join(lines)+'\n').encode()==a.payload(a.INPUT/'result/analysis/output/all_curves.txt','all_formatted_fields_verified'),'all_curves_text_exact')
    a.compare([r for r in curve_rows if r['interpretation']=='有相应迹象'],summary['supported_diagnostics'],'supported_diagnostic_list')
    for arm in a.ARMS:a.compare(e['endpoints'][arm]['primary'],summary['primary'][arm],'summary_primary')
    for key,c in e['comparisons'].items():
        want={'passed_count':sum(c['checks'].values()),'total_checks':len(c['checks']),'failed':c['assessment']['failed'],
              'primary':c['delta']['primary'],'raw_probability':{ep:{m:c['delta']['raw'][ep][m] for m in ('brier','log_loss')} for ep in ('O','N','Z')}}
        a.compare(want,summary['comparisons'][key],'summary_comparisons',key)
    # Resolve small first-audit teacher mean roundoff with explicit left-to-right
    # additions of the six stored scalars. This does not change original outputs.
    teacher=a.js(a.INPUT/'result/analysis/output/teacher_errors.json');fold_diff=0.;builtin_diff=0.;scalar_count=0
    for row in teacher['rows']:
        source=a.js(a.INPUT/'result/job/run/diagnostics'/f"{row['point']}_epoch{row['epoch']}.json")
        for key in ('D0','D1','residual_cross_mean','residual_difference_mse'):
            if row[key] is None:continue
            values=[x[key] for x in source['teacher_errors']];acc=0.
            for val in values:acc+=val
            fold_diff=max(fold_diff,abs(acc/len(values)-row[key]));builtin_diff=max(builtin_diff,abs(sum(values)/len(values)-row[key]));scalar_count+=1
    a.expect(fold_diff==0.,'explicit_order_teacher_means_exact')
    # Verify saved micro/pooled statistics from counts, without reconstructing labels.
    col=a.js(a.INPUT/'result/job/evaluation/collected.json');pooled=0
    for p,roles in col['points'].items():
        for role,rec in roles.items():
            counts=a.js(a.verified(a.INPUT/'result/job/evaluation',rec['counts']))
            obj=e['absolute_stage_results'][p][role]['fixed_half_classification']
            for dom,idx in [('all',range(60)),*[(dom,[i for i,d in enumerate(col['domains']) if d==dom]) for dom in 'ABC']]:
                c={k:sum(counts[i][k] for i in idx) for k in ('tp','fp','fn','tn')}
                v=a.classification(c)
                expected=obj['pooled_all'] if dom=='all' else obj['by_domain'][dom]
                wanted={**c,'precision':v['precision'],'recall':v['recall'],'f1':v['f1'],'fpr':1-v['specificity']}
                a.compare(wanted,expected,'pooled_count_statistics',p+'/'+role+'/'+dom);pooled+=1
    # Parse console receipts, preserving the distinction from complete model state.
    raw=a.payload(a.INPUT/'result/job.console.txt','all_log_lines_scanned').decode()
    records=[];nonjson=[]
    for lineno,line in enumerate(raw.splitlines(),1):
        try:obj=json.loads(line)
        except json.JSONDecodeError:nonjson.append([lineno,line]);continue
        if isinstance(obj,dict):records.append(obj)
    updates=[x for x in records if x.get('event')=='updates'];by=Counter(x['point'] for x in updates)
    a.expect(set(by)==set(col['points']),'console_physical_point_coverage')
    a.expect(all(n==13 for n in by.values()),'console_expected_progress_ticks')
    a.expect(all([x['completed'] for x in updates if x['point']==p]==[1,*range(24,289,24)] for p in by),'console_update_sequence')
    wr=a.payload(a.INPUT/'result/job.wrapper.txt','all_wrapper_lines_processed').decode().splitlines()
    duration=(datetime.fromisoformat(wr[2])-datetime.fromisoformat(wr[0])).total_seconds()
    a.expect(wr[1]=='exit_code=0' and duration==79962,'wrapper_completion_duration')
    # Settings: record where assignments live; do not infer process runtime values.
    assignments=[]
    for p in sorted((a.INPUT/'result/source/scripts').glob('*.py')):
        raw=a.payload(p,'AST_search_for_numeric_runtime_settings').decode();tree=ast.parse(raw)
        for n in ast.walk(tree):
            if isinstance(n,(ast.Assign,ast.Expr)):
                segment=ast.get_source_segment(raw,n) or ''
                if ('use_deterministic_algorithms' in segment or 'allow_tf32' in segment or 'cudnn.benchmark' in segment) and len(segment)<300:
                    owners=[x.name for x in ast.walk(tree) if isinstance(x,(ast.FunctionDef,ast.ClassDef)) and x.lineno<=n.lineno<=x.end_lineno]
                    assignments.append({'path':str(p.relative_to(a.INPUT)),'line':n.lineno,'enclosing_definitions':owners,'statement':segment})
    result.update(status='PASS',all_metrics_text_exact=True,all_curves_text_exact=True,
                  displayed_absolute_metric_entries=5*4*8*22,displayed_comparison_metric_entries=7*4*8*22,
                  physical_diagnostic_rows=len(curve_rows),pooled_count_blocks=pooled,
                  teacher_scalar_count=scalar_count,teacher_builtin_sum_max_difference=builtin_diff,teacher_explicit_addition_max_difference=fold_diff,
                  wrapper_elapsed_seconds=duration,console_update_lines=len(updates),console_point_count=len(by),console_non_JSON_lines=nonjson,
                  numeric_setting_assignments=assignments,formal_effective_runtime_values='NOT_RECORDED_IN_SUPPLIED_EXECUTION_RECEIPT',
                  numeric_comparisons=dict(a.STATS),inputs=list(a.SEEN.values()))
except BaseException:
    result.update(status='FAIL',traceback=traceback.format_exc());print(result['traceback'])
result['elapsed_seconds']=time.monotonic()-start
a.save('display_audit.json',result)
print(json.dumps({k:v for k,v in result.items() if k not in ('inputs','console_non_JSON_lines')},ensure_ascii=False,indent=2))
raise SystemExit(0 if result['status']=='PASS' else 1)
