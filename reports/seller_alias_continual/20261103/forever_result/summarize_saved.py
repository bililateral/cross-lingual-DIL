"""Mechanical checks and tables from saved JSON fields; no research imports/inference/resampling."""
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct
import time

ROOT = Path(__file__).resolve().parent
NAMES = ('H_pos_tuned', 'FOREVER_group_fixed')
ROLES = ('raw', 'stage-cal', 'first-cal')
COEFFICIENTS = {
    'O': {'3_1': .5, '3_2': .5},
    'A_acq': {'1_1': .5, '2_2': .5},
    'F': {'1_1': .5, '2_2': .5, '3_1': -.5, '3_2': -.5},
    'N': {'2_2': .5, '3_3': .5}, 'Z': {'3_3': 1.},
    'F_first': {'1_1': 1., '3_1': -1.},
    'final_mean': {'3_1': 1/3, '3_2': 1/3, '3_3': 1/3},
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(name, value):
    (ROOT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def mean(values):
    return None if any(v is None for v in values) else math.fsum(values)/len(values)


def close(left, right):
    if left is None or right is None:
        assert left is right, (left, right)
    else:
        assert math.isfinite(left) and math.isfinite(right) and abs(left-right) < 1e-11, (left, right)


def linear(cells, coefficients, column):
    values = [cells[key][column] for key in coefficients]
    return None if any(v is None for v in values) else math.fsum(cells[key][column]*weight for key, weight in coefficients.items())


def fmt(value):
    return 'N/A' if value is None else f'{value:.9f}'


def npy_shape(path):
    with path.open('rb') as stream:
        assert stream.read(6) == b'\x93NUMPY'
        version = tuple(stream.read(2))
        length = struct.unpack('<H' if version == (1, 0) else '<I', stream.read(2 if version == (1, 0) else 4))[0]
        header = ast.literal_eval(stream.read(length).decode('latin1'))
        assert header['descr'] in ('<f8', '<f4'), header
        assert not header['fortran_order']
        assert path.stat().st_size-stream.tell() == math.prod(header['shape'])*int(header['descr'][-1])
        return header['shape']


def main():
    started = time.perf_counter()
    manifest, receipt = read(ROOT/'transfer_manifest.json'), read(ROOT/'receive.json')
    assert receipt['status'] == 'ALL_FILES_SIZE_SHA256_MATCH'
    for rec in manifest['files']:
        path = ROOT/rec['local']
        assert path.stat().st_size == rec['bytes']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == rec['sha256'], rec['local']
    saved = read(ROOT/'campaign/analysis/result.json')
    conf = read(ROOT/'campaign/confirmation/result.json')
    gate = read(ROOT/'formal_gate.json')
    first = next(iter(conf['runs'].values()))['1_1']['metrics']['raw']['matrix']['path']
    columns = read(ROOT/'campaign/confirmation'/first)['columns']
    assert len(columns) == 22
    runs, na, cell_rows, matrices = {}, {}, [], 0
    for identity, points in conf['runs'].items():
        order, seed, name = identity.split('_', 2)
        assert name in NAMES and order in ('ABC', 'BCA', 'CAB') and seed in ('s4', 's5', 's6')
        runs[identity] = {}
        for role in ROLES:
            means = {}
            for key, cell in points.items():
                assert cell['domain'] == order[int(key[2])-1]
                assert cell['ids'] == [uid for uid, d in zip(conf['groups'], conf['domains']) if d == cell['domain']]
                score = cell['scores'][role]
                score_path = ROOT/'campaign/confirmation'/score['path']
                assert score_path.stat().st_size == score['bytes'] and hashlib.sha256(score_path.read_bytes()).hexdigest() == score['sha256']
                assert npy_shape(ROOT/'campaign/confirmation'/score['path']) == (40, 378)
                query = cell['metrics'][role]['queries']
                query_path = ROOT/'campaign/confirmation'/query['path']
                assert query_path.stat().st_size == query['bytes'] and hashlib.sha256(query_path.read_bytes()).hexdigest() == query['sha256']
                assert npy_shape(query_path) == (40, 28, 10)
                rec = cell['metrics'][role]['matrix']
                path = ROOT/'campaign/confirmation'/rec['path']
                assert path.stat().st_size == rec['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest() == rec['sha256']
                matrix = read(path)
                assert matrix['columns'] == columns and len(matrix['values']) == len(matrix['counts']) == 40
                assert all(len(row) == 22 for row in matrix['values'])
                assert all(c['tp']+c['fn'] == 20 and c['tn']+c['fp'] == 358 for c in matrix['counts'])
                values = [mean([row[j] for row in matrix['values']]) for j in range(22)]
                old = saved['per_run_actual_domain_cells'][f'{identity}_{role}_{key}']
                assert old['domain'] == cell['domain'] and old['counts'] == matrix['counts']
                for left, right in zip(values, old['values']):
                    close(left, right)
                means[key] = values
                for j, column in enumerate(columns):
                    item = na.setdefault(f'{name}/{role}/{column}', dict(defined=0, undefined=0, final_defined=0, final_undefined=0))
                    undefined = sum(row[j] is None for row in matrix['values'])
                    item['undefined'] += undefined
                    item['defined'] += 40-undefined
                    if key.startswith('3_'):
                        item['final_undefined'] += undefined
                        item['final_defined'] += 40-undefined
                cell_rows.append(dict(run=identity, role=role, key=key, domain=cell['domain'], values=values))
                matrices += 1
            runs[identity][role] = {field: [linear(means, weights, j) for j in range(22)] for field, weights in COEFFICIENTS.items()}
    assert matrices == 324
    for name in NAMES:
        selected = [r for r in runs if r.endswith(name)]
        assert len(selected) == 9
        for role in ROLES:
            for field in COEFFICIENTS:
                for j in range(22):
                    close(mean([runs[r][role][field][j] for r in selected]), saved['absolute'][name][role][field][j])
    for role in ROLES:
        for field in COEFFICIENTS:
            for j, column in enumerate(columns):
                h, f = (saved['absolute'][name][role][field][j] for name in NAMES)
                entry = saved['difference'][role][field][column]
                close(None if h is None or f is None else h-f, entry['mean'])
                bounds = entry['conditional_interval']
                assert bounds is None or bounds == [None, None] or all(math.isfinite(v) for v in bounds) and bounds[0] <= bounds[1]
    raw = saved['difference']['raw']
    checks = dict(old=raw['O']['map']['conditional_interval'][0]>0,
                  forgetting=raw['F']['map']['conditional_interval'][1]<0,
                  new=raw['N']['map']['mean']>=0, last=raw['Z']['map']['mean']>=0)
    assert checks == saved['checks'] and all(checks.values()) == saved['passed']
    training = []
    for path in sorted((ROOT/'campaign').glob('*/run/completion.json')):
        row = read(path)
        assert row['source_files'] == gate['sources']
        logs = [json.loads(line) for line in path.with_name('updates.jsonl').read_text(encoding='utf-8').splitlines()]
        counts = Counter(x['role'] for x in logs)
        assert counts['current'] == row['counts']['current'] == 864 and counts['replay'] == row['counts']['replay']
        assert [x['adam_step'] for x in logs] == list(range(1, len(logs)+1))
        current, replay = ([x for x in logs if x['role'] == role] for role in ('current', 'replay'))
        assert all(x['displacement']['total']>0 and x['regularizer'] is None for x in current)
        assert all(x['displacement'] is None and x['encoder_lr']==1e-5 and x['head_lr']==.001 and .0005<=x['regularizer']['beta']<=.003 for x in replay)
        assert all(x['regularizer']['norms'][part]['regularizer']>0 for x in replay for part in ('encoder','head'))
        retired = read(path.with_name('recovery_retirement.json'))
        assert retired['status']=='DELETED' and retired['latest']['metadata']['state']['phase']=='complete'
        params = retired['latest']['metadata']['state']['parameters']
        training.append(dict(run=path.parent.parent.name, counts=row['counts'],
            stage_counts={k:v['counts'] for k,v in row['points'].items()},
            events=dict(Counter(x.get('event') for x in replay)),
            beta_range=[min(x['regularizer']['beta'] for x in replay),max(x['regularizer']['beta'] for x in replay)],
            parameter_count=sum(p['numel'] for p in params), parameter_bytes=sum(p['bytes'] for p in params),
            history_bytes=retired['latest']['metadata']['memory']['bytes'],
            resources=row['resources'], first_update_epoch=logs[0]['wall_time_epoch'], last_update_epoch=logs[-1]['wall_time_epoch'],
            regularizer_norm_range={part:[min(x['regularizer']['norms'][part]['regularizer'] for x in replay),max(x['regularizer']['norms'][part]['regularizer'] for x in replay)] for part in ('encoder','head')}))
    by_order, by_seed, by_domain = {}, {}, {}
    for name in NAMES:
        for dimension, keys, output in (('order', ('ABC','BCA','CAB'), by_order), ('seed', ('s4','s5','s6'), by_seed)):
            for key in keys:
                ids = [r for r in runs if r.endswith(name) and r.split('_')[0 if dimension=='order' else 1] == key]
                output[f'{name}/{key}'] = {field: mean([runs[r]['raw'][field][12] for r in ids]) for field in COEFFICIENTS}
        for domain in 'ABC':
            selected = [r for r in cell_rows if r['run'].endswith(name) and r['role']=='raw' and r['domain']==domain]
            acquired = [r['values'][12] for r in selected if r['key'][0] == r['key'][2]]
            final = [r['values'][12] for r in selected if r['key'].startswith('3_')]
            assert len(acquired)==len(final)==9
            by_domain[f'{name}/{domain}'] = dict(acquired=mean(acquired), final=mean(final), decline=mean(acquired)-mean(final))
    write('summary.json', dict(columns=columns, per_run=runs, by_order=by_order, by_seed=by_seed, by_domain=by_domain, defined_counts=na, training=training))
    text = ['FOREVER 与 H_pos_tuned：完整保存结果表', '数值来自已保存 JSON 的字段及简单算术；没有从预测重算指标或重新抽样。', '', '一、22指标：最终三域、旧域、初学、遗忘、新域、末域与首域变化；H−FOREVER及原95%条件区间。']
    for role in ROLES:
        for field in COEFFICIENTS:
            text.extend(['', f'{role} / {field}', '指标\tH_pos_tuned\tFOREVER_group_fixed\tH−FOREVER\t原95%条件区间'])
            for j, column in enumerate(columns):
                entry = saved['difference'][role][field][column]
                interval = 'N/A' if entry['conditional_interval'] is None else '['+', '.join(fmt(x) for x in entry['conditional_interval'])+']'
                text.append('\t'.join([column, fmt(saved['absolute'][NAMES[0]][role][field][j]), fmt(saved['absolute'][NAMES[1]][role][field][j]), fmt(entry['mean']), interval]))
    text.extend(['', '二、逐轨迹六格 RAW MAP', '运行\tR11\tR21\tR22\tR31\tR32\tR33\tO\tA_acq\tF\tN\tZ\t最终三域'])
    for identity in sorted(runs):
        cells = {r['key']:r['values'][12] for r in cell_rows if r['run']==identity and r['role']=='raw'}
        values = [cells[k] for k in ('1_1','2_1','2_2','3_1','3_2','3_3')]+[runs[identity]['raw'][k][12] for k in ('O','A_acq','F','N','Z','final_mean')]
        text.append(identity+'\t'+'\t'.join(fmt(v) for v in values))
    for title, source in (('三、顺序 RAW MAP',by_order),('四、种子 RAW MAP',by_seed),('五、实际域 RAW MAP（末到域含零后续时段）',by_domain)):
        text.extend(['', title])
        for key, values in source.items(): text.append(key+'\t'+'\t'.join(k+'='+fmt(v) for k,v in values.items()))
    text.extend(['', '六、N/A覆盖：前两列为所有六格，后两列为最终三个域；计数为重复群次，不是独立样本数。', '方法/口径/指标\t有效\t未定义\t最终有效\t最终未定义'])
    for key, counts in na.items(): text.append(key+'\t'+'\t'.join(str(v) for v in counts.values()))
    (ROOT/'tables.zh.txt').write_text('\n'.join(text)+'\n',encoding='utf-8')
    result = dict(status='PASS_SAVED_FIELD_AND_IDENTITY_CHECKS', files=len(manifest['files']), matrices=matrices, metric_columns=22,
        absolute_fields=2*3*7*22, difference_fields=3*7*22, original_intervals_recomputed=False,
        scores_re_evaluated=False, labels_opened=False, models_loaded=False, research_modules_imported=False,
        checks=checks, passed=all(checks.values()), elapsed_seconds=time.perf_counter()-started)
    write('check_saved.json',result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
