"""Present and cross-check already opened ablation metrics; never load labels or models."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path


def mean(values):
    return None if any(v is None for v in values) else math.fsum(values) / len(values)


def pretty(value, scale=100):
    return 'N/A' if value is None else f'{value * scale:.4f}'


def table(headers, rows):
    return '\n'.join(['|' + '|'.join(headers) + '|', '|' + '|'.join(['---'] * len(headers)) + '|']
                     + ['|' + '|'.join(map(str, row)) + '|' for row in rows]) + '\n\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--formal', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    root, out = args.formal, args.output
    assert json.loads((root / 'completion.json').read_text())['status'] == 'COMPLETE_PENDING_RESULT_REVIEW_MODELS_RETAINED'
    receipt = json.loads((root / 'analysis/result.json').read_text())
    source = root / 'confirmation/summary.json'
    blob = source.read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    assert receipt['status'] == 'ANALYSIS_COMPLETE'
    assert digest == receipt['summary']['sha256'] and len(blob) == receipt['summary']['bytes']
    s = json.loads(blob)
    assert len(s['per_run']) == 72 and len(s['per_cell']) == 1296
    columns = s['metric_columns']; mi = columns.index('map')
    arms = ['Full', 'ER', 'KD', 'SEQ', 'NoRegroup', 'MeanPool', 'StandardRank', 'AccountPool']
    fields = ['A_acq', 'O', 'F', 'N', 'Z', 'final_mean', 'F_first']
    roles = ['raw', 'stage-cal', 'first-cal']
    terms = {'A_acq': [('1_1', .5), ('2_2', .5)], 'O': [('3_1', .5), ('3_2', .5)],
             'F': [('1_1', .5), ('2_2', .5), ('3_1', -.5), ('3_2', -.5)],
             'N': [('2_2', .5), ('3_3', .5)], 'Z': [('3_3', 1)],
             'final_mean': [('3_1', 1/3), ('3_2', 1/3), ('3_3', 1/3)],
             'F_first': [('1_1', 1), ('3_1', -1)]}
    comparisons = 0
    for uid, values in s['per_run'].items():
        for role in roles:
            for field in fields:
                for i, metric in enumerate(columns):
                    entries = [(s['per_cell'][f'{uid}_{role}_{cell}']['mean'][i], c) for cell, c in terms[field]]
                    expected = None if any(v is None for v, _ in entries) else math.fsum(v*c for v, c in entries)
                    if expected is not None and field in ('F', 'F_first') and metric in ('brier', 'log_loss'):
                        expected *= -1
                    actual = values[role][field][i]
                    assert (actual is None and expected is None) or (actual is not None and expected is not None and abs(actual-expected) < 1e-12), (uid, role, field, metric)
                    comparisons += 1
    details = {'summary_sha256': digest, 'primary_pass': s['primary_pass'], 'primary_checks': s['primary_checks'],
               'formula_crosschecks': comparisons, 'scope': 'Saved metric arithmetic only; no inference, labels, new bootstrap or changed gate',
               'contrasts_stability': {}, 'full_er_cells': {}, 'full_er_acquisition_pairs': [], 'subgroups': {}, 'na_coverage': {}}
    for contrast in s['contrasts']:
        a, b = contrast.split('-')
        row = {}
        for field in fields:
            per_seed = {str(seed): s['stratified'][a]['seed'][str(seed)]['raw'][field][mi] - s['stratified'][b]['seed'][str(seed)]['raw'][field][mi] for seed in (4, 5, 6)}
            per_order = {order: s['stratified'][a]['order'][order]['raw'][field][mi] - s['stratified'][b]['order'][order]['raw'][field][mi] for order in ('ABC', 'BCA', 'CAB')}
            paired = {f'{order}_s{seed}': s['per_run'][f'{order}_s{seed}_{a}']['raw'][field][mi] - s['per_run'][f'{order}_s{seed}_{b}']['raw'][field][mi] for order in ('ABC', 'BCA', 'CAB') for seed in (4, 5, 6)}
            row[field] = {'seed': per_seed, 'order': per_order, 'paired': paired,
                          'positive_pairs': sum(x > 0 for x in paired.values()), 'negative_pairs': sum(x < 0 for x in paired.values())}
        details['contrasts_stability'][contrast] = row
    for cell in ('1_1', '2_1', '2_2', '3_1', '3_2', '3_3'):
        rows = []
        for order in ('ABC', 'BCA', 'CAB'):
            for seed in (4, 5, 6):
                a = s['per_cell'][f'{order}_s{seed}_Full_raw_{cell}']; b = s['per_cell'][f'{order}_s{seed}_ER_raw_{cell}']
                rows.append({'order': order, 'seed': seed, 'domain': a['domain'], 'Full': a['mean'][mi], 'ER': b['mean'][mi], 'delta': a['mean'][mi]-b['mean'][mi]})
        details['full_er_cells'][cell] = {'Full': mean([x['Full'] for x in rows]), 'ER': mean([x['ER'] for x in rows]), 'delta': mean([x['delta'] for x in rows]), 'rows': rows}
    for order in ('ABC', 'BCA', 'CAB'):
        for seed in (4, 5, 6):
            diffs = [s['per_cell'][f'{order}_s{seed}_Full_raw_{c}']['mean'][mi] - s['per_cell'][f'{order}_s{seed}_ER_raw_{c}']['mean'][mi] for c in ('1_1', '2_2')]
            details['full_er_acquisition_pairs'].append({'order': order, 'seed': seed, 'R11_difference': diffs[0], 'R22_difference': diffs[1], 'A_acq_difference': mean(diffs)})
    details['first_domain_all_22_roles_equal'] = all(s['per_cell'][f'{order}_s{seed}_Full_{role}_1_1']['mean'] == s['per_cell'][f'{order}_s{seed}_ER_{role}_1_1']['mean'] for order in ('ABC', 'BCA', 'CAB') for seed in (4, 5, 6) for role in roles)
    for arm in ('Full', 'StandardRank'):
        details['subgroups'][arm] = {}
        for positives in ('1', '2'):
            result = {}
            for field in fields:
                vals = []
                for order in ('ABC', 'BCA', 'CAB'):
                    for seed in (4, 5, 6):
                        v = []
                        for cell, coef in terms[field]:
                            groups = s['ranking_query_subgroups'][f'{order}_s{seed}_{arm}_raw_{cell}'][positives]['group_values']
                            v.append(([mean([g[i] for g in groups]) for i in range(3)], coef))
                        vals.append([math.fsum(x[i]*coef for x, coef in v) for i in range(3)])
                result[field] = [mean([v[i] for v in vals]) for i in range(3)]
            details['subgroups'][arm][positives] = result
    for role in roles:
        by = {}
        for arm in arms:
            cells = [v for k, v in s['per_cell'].items() if f'_{arm}_{role}_' in k]
            by[arm] = {metric: {'defined_groups': sum(c['defined_groups'][i] for c in cells), 'total_group_evaluations': len(cells)*40,
                              'defined_cells': sum(c['mean'][i] is not None for c in cells), 'total_cells': len(cells)} for i, metric in enumerate(columns) if metric in ('precision', 'mcc')}
        details['na_coverage'][role] = by
    main = '# 中文H消融：完整保存指标表\n\n单位：能力为%，差值/遗忘为百分点；Brier和log-loss用原单位。N/A原样传播。所有区间均沿用原5000次条件配对结果，没有新重采样。\n\n'
    for role in roles:
        main += f'## {role}：八配置全22指标\n\n'
        for metric_i, metric in enumerate(columns):
            scale = 1 if metric in ('brier', 'log_loss') else 100
            main += f'### {metric}\n\n' + table(['配置'] + fields, [[arm] + [pretty(s['absolute'][arm][role][f][metric_i], scale) for f in fields] for arm in arms])
    main += '## Full−ER及八项次级比较：raw MAP\n\n除Full−ER唯一联合主张外，以下区间均为探索性，不作多重比较校正的确认胜出。\n\n'
    for contrast, roles_data in s['contrasts'].items():
        main += f'### {contrast}\n\n' + table(['端点', '差值pp', '95%条件区间pp', '正/负配对(共9)', 'seed4/5/6差值pp'], [[f, pretty(roles_data['raw'][f]['map']['mean']), '['+', '.join(pretty(x) for x in roles_data['raw'][f]['map']['conditional_interval'])+']', f"{details['contrasts_stability'][contrast][f]['positive_pairs']}/{details['contrasts_stability'][contrast][f]['negative_pairs']}", '/'.join(pretty(v) for v in details['contrasts_stability'][contrast][f]['seed'].values())] for f in fields])
    main += '## Full−ER初学分解\n\n' + table(['顺序','seed','R11差pp','R22差pp','A_acq差pp'], [[x['order'],x['seed'],pretty(x['R11_difference']),pretty(x['R22_difference']),pretty(x['A_acq_difference'])] for x in details['full_er_acquisition_pairs']])
    main += '## 单/双正例排序补充\n\n同次开放下的描述性均值，未追加区间。\n\n'
    main += table(['配置','每query正例数','端点','MAP%/pp','MRR%/pp','Recall@5%/pp'], [[arm,p,f]+[pretty(v) for v in values] for arm,ps in details['subgroups'].items() for p,fs in ps.items() for f,values in fs.items()])
    main += '## Precision/MCC覆盖\n\n合并混淆矩阵指标保留于原summary.per_cell；不替换未定义的群均值。\n\n'
    main += table(['角色','配置','指标','可定义群次/总群次','可定义格/总格'], [[role,arm,metric,f"{x['defined_groups']}/{x['total_group_evaluations']}",f"{x['defined_cells']}/{x['total_cells']}"] for role,arms0 in details['na_coverage'].items() for arm,ms in arms0.items() for metric,x in ms.items()])
    (out/'tables.zh.md').write_text(main, encoding='utf-8')
    details['presentation_seconds'] = time.monotonic()-started
    (out/'analysis_details.json').write_text(json.dumps(details, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'formula_crosschecks': comparisons, 'first_domain_all_22_roles_equal': details['first_domain_all_22_roles_equal'], 'primary_pass': s['primary_pass'], 'seconds': details['presentation_seconds']}))


if __name__ == '__main__':
    main()
