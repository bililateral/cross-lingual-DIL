"""Assemble the review report from already verified saved-statistic outputs."""
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path

BASE = Path('/workspace/scratch/0f307e99d302')
OUT = BASE / 'audit_output'
IN = BASE / 'review_input'


def read(path):
    return json.loads(path.read_text())


def fmt(x, signed=False, digits=12):
    if x is None:
        return 'N/A'
    return format(x, ('+' if signed else '') + '.' + str(digits) + 'f')


def interval(rec, signed=True):
    if rec['mean'] is None:
        return 'N/A'
    lo, hi = rec['conditional_95pct_interval']
    return fmt(rec['mean'], signed) + ' [' + fmt(lo, signed) + ', ' + fmt(hi, signed) + ']'


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(map(str, headers)) + ' |',
                      '| ' + ' | '.join('---' for _ in headers) + ' |',
                      *['| ' + ' | '.join(map(str, row)) + ' |' for row in rows]])


stat = read(OUT / 'independent_statistics/result.json')
diag = read(OUT / 'agents/diagnostics/attempt01/independent_results.json')
exe = read(OUT / 'agents/execution/attempt02/mechanical_summary.json')
wire = read(OUT / 'agents/wiring/wiring_replay_result.json')
stats = {}
with (OUT / 'independent_statistics/all_metrics_independent.csv').open(newline='') as f:
    for r in csv.DictReader(f):
        key = tuple(r[k] for k in ('kind', 'point_or_contrast', 'role', 'domain', 'metric'))
        stats[key] = {'mean': float(r['mean']) if r['mean'] else None,
                      'conditional_95pct_interval': [float(r['lower_95']), float(r['upper_95'])] if r['lower_95'] else None}
arms = ('S', 'G_half', 'S_pos', 'H_pos')
roles = ('raw', 'stage-cal', 'first-cal')
contrasts = ('S_pos-S', 'H_pos-S', 'H_pos-G_half', 'H_pos-S_pos')
collected = read(IN / 'workspace/reports/job/evaluation/collected.json')
columns = collected['columns']


def name(text):
    return text.replace('G_half', 'H')


def get(kind, subject, role, domain, metric):
    return stats[kind, subject, role, domain, metric]


fill = {}
fill['EXECUTION_TABLE'] = table(['端点', '实际更新行', 'current/history群呈现', '末端Adam', '捕获', '严格判读'], [
    ['A_S_pos', 288, '288 / 0', 288, 6, 3], ['B_S_pos', 288, '288 / 288', 576, 6, 6],
    ['A_H_pos', 288, '288 / 0', 288, 6, 3], ['B_H_pos', 288, '288 / 288', 576, 6, 6],
    ['合计', 1152, '1152 / 576 = 1728', '每臂576', 24, 18]])
fill['PROJECT_RESOURCE_TABLE'] = table(['项目', '保存实际量/保守计费', '既定上限'], [
    ['CPU资格累计', '52秒（保存项目账）', '300秒'],
    ['原生实际调用/外层计费', '12调用 / 650秒', '12调用 / 1200秒'],
    ['正式外层', '12661秒', '57600秒（含合同内分析）'],
    ['原独立分析', '2.385307759秒；计3秒', '计入正式总额'],
    ['正式+原分析', '12664秒', '57600秒'],
    ['正式CUDA reserved峰值', '8652849152字节', '30064771072字节 / 28GiB'],
    ['正式RSS峰值', '6715924480字节', '68719476736字节 / 64GiB'],
    ['正式输出峰值', '9130164115字节', '17179869184字节 / 16GiB']])
loss_rows = []
for point in ('A_S_pos', 'B_S_pos', 'A_H_pos', 'B_H_pos'):
    m = wire['formal']['points'][point]['measurements']
    loss_rows.append([point, fmt(m['current_supervised_contribution']['mean']),
                      fmt(m['history_supervised_contribution']['mean']) if point.startswith('B') else '不适用',
                      fmt(m['history_teacher_contribution']['mean']) if point.startswith('B') else '0',
                      '288/288' if point.startswith('B') else '0/0；A无历史'])
fill['LOSS_TABLE'] = table(['阶段', 'current监督贡献均值', '0.1 history监督均值', '0.5D0均值', '历史教师正值数'], loss_rows)
native_rows = []
for arm in ('S_pos', 'H_pos'):
    for phase in ('current', 'paired'):
        for part in ('encoder', 'head'):
            r = exe['native']['arms'][arm]['effects'][phase][part]
            native_rows.append([arm + '/' + phase, part, format(r['post_clip_gradient_max_delta'], '.12g'),
                                format(r['parameter_max_delta'], '.12g'),
                                format(r['adam_max_exp_avg_delta'], '.12g'), format(r['adam_max_exp_avg_sq_delta'], '.12g')])
fill['NATIVE_EFFECT_TABLE'] = table(['臂/资格阶段', '组件', '裁剪后梯度最大差', '参数最大差', 'Adam一阶矩最大差', 'Adam二阶矩最大差'], native_rows)
fill['MAP_TABLE'] = table(['比较', '域', 'MAP差 [条件95%区间]', '正/负/零群'], [
    [name(r['contrast']), r['domain'], interval(r), f"{r['positive_groups']}/{r['negative_groups']}/{r['zero_groups']}"]
    for r in stat['map_contrasts']])
fill['A_END_TABLE'] = table(['A末比较', '旧A MAP差 [95%区间]', '尚未训练B MAP差 [95%区间]'], [
    [name(c), interval(get('initial_contrast', c, 'raw', 'A', 'map')), interval(get('initial_contrast', c, 'raw', 'B', 'map'))] for c in contrasts])
fill['OLD_CHANGE_TABLE'] = table(['方法', '旧A raw MAP：原始B末−A末 [95%区间]'], [
    [name(a), interval(get('old_change', a, 'raw', 'A', 'map'))] for a in arms])
fill['FORGETTING_TABLE'] = table(['比较', '旧A raw MAP方向一致遗忘差 [95%区间]'], [
    [name(c), interval(get('forgetting_contrast', c, 'raw', 'A', 'map'))] for c in contrasts])
fill['SUBGROUP_TABLE'] = table(['比较', '每查询正例数', '域', 'MAP差 [95%区间]'], [
    [name(r['contrast']), r['positives'], r['domain'], interval(r)] for r in stat['subgroup_contrasts']])
fill['PROBABILITY_TABLE'] = table(['比较', '角色', '域', 'Brier差 [95%区间]', 'log-loss差 [95%区间]'], [
    [name(c), role, d, interval(get('final_contrast', c, role, d, 'brier')), interval(get('final_contrast', c, role, d, 'log_loss'))]
    for c in contrasts for role in roles for d in 'AB'])
fill['DIAGNOSTIC_TABLE'] = table(['端点', '分支/域', '指标', '训练收益 [95%区间]', '验证收益 [95%区间]', '差距扩大 [95%区间]', '严格判读'], [
    [r['point'], r['role'] + '/' + r['domain'], r['metric'],
     interval({'mean': r['train_benefit'], 'conditional_95pct_interval': r['train_ci']}),
     interval({'mean': r['valid_benefit'], 'conditional_95pct_interval': r['valid_ci']}),
     interval({'mean': r['gap_widening'], 'conditional_95pct_interval': r['gap_ci']}), r['decision']]
    for r in diag['diagnostic_rows']])

findings = {
    'review_scope': 'New fixed S_pos/H_pos results and correspondence to preserved execution/source evidence',
    'input_sha256': '4eeeb5df0e93b6c32bf352f42b656b945a8d4e4f71078b73fa4df0e015ae63f7',
    'verdict': 'VALID_WITHIN_SAVED_EVIDENCE_SCOPE_PRIMARY_OBJECTIVE_NOT_MET',
    'SCIENTIFIC_BLOCKER': [], 'REPRODUCIBILITY_DEFECT': [],
    'primary_passed': False,
    'confirmed_results': [
        {'id':'O01','type':'PRIMARY_OBJECTIVE_NOT_MET','finding':'S_pos−S B末raw MAP两域下界均未严格>0；旧A近零负下界保持负值。','evidence':['E02','E06','R01','R02']},
        {'id':'O02','type':'EXECUTION_CORRESPONDENCE','finding':'40来源、255原件、52基线与1152/1728/4/24/18及访问/退出对应；未发现新矛盾。','evidence':['E03','E04','E05','R03']},
        {'id':'O03','type':'ACTIVE_OBJECTIVE_WIRING','finding':'四位置新rank进入共享VJP；0.1历史监督与0.5D0分开；既有原生encoder/head/Adam和正式日志共同支持实际作用。','evidence':['E04','E05','E09','R04']},
        {'id':'O04','type':'LIMITED_SECONDARY_BENEFIT','finding':'H_pos旧A对S/S_pos有条件正向差；对H两域总MAP均未确定；次级不能覆盖主失败。','evidence':['E06','R02']},
        {'id':'O05','type':'NEGATIVE_PROBABILITY_EVIDENCE','finding':'S_pos−S及H_pos−H两域raw Brier/log-loss区间全正；校准只改善部分概率表现，不改变排序。','evidence':['E06','R02','R05']},
        {'id':'O06','type':'GENERALIZATION_GAP','finding':'18差距扩大区间下界全正；严格判读17未见明确迹象、1证据不足，不等于没有过拟合。','evidence':['E07','R05']},
        {'id':'O07','type':'DESCRIPTIVE_SUBGROUP_COST','finding':'H_pos−H新B双正例MAP差区间全负；同一整群抽样，描述性而非交互或因果证据。','evidence':['E06','R02','R05']},
        {'id':'O08','type':'FORGETTING_NOT_ESTABLISHED','finding':'四方法旧A MAP原始变化与四个遗忘差区间均跨0，不确证灾难性遗忘或改进遗忘。','evidence':['E06','R02']},
    ],
    'interpretation_limits': [
        {'id':'L01','finding':'未从正式标签重建AP/ROC/PR-AUC/Brier/log-loss或查询相关性；原值是保存标签派生证据。','requires_new_execution':False},
        {'id':'L02','finding':'诊断epoch1已完成48步；epoch1→6不覆盖阶段入口至首轮结束；cache/A只含6个存活群。','requires_new_execution':False},
        {'id':'L03','finding':'单/双正例区间是否跨0的差异不等于交互检验；辅助区间不是同时置信区间。','requires_new_execution':False},
        {'id':'L04','finding':'A获得表是A末状态及候选−对手比较，不是已独立核算相对预训练初始模型的获得量。','requires_new_execution':False},
        {'id':'L05','finding':'重复开发集、单种子、固定顺序的条件bootstrap不覆盖重训/选择不确定性及真实市场有效性。','requires_new_execution':False},
        {'id':'L06','finding':'原生张量作用与服务器保管为保存原件，未在本地重新产生张量或实时确认服务器状态。','requires_new_execution':False},
    ],
    'reasonable_followup':['保留有效未达标结果与完整S主要对照地位','摘要明确列示既有诊断/标签复核/分组解释边界；无须项目代码补丁'],
    'OUT_OF_SCOPE_OVERDESIGN':['按已见曲线选epoch','临时调权重/温度/教师','新增候选/种子/对手','添加新显著性或交互成功门','重训旧S/H','请求正式标签/模型/Memory','连接项目服务器','全仓回归/通用恢复框架/网络加固'],
    'input_modified': False, 'project_patch_generated': False, 'project_server_connected': False,
    'formal_labels_reparsed': False, 'model_or_memory_loaded': False, 'new_training_executed': False,
    'external_web_sources_used': False,
}
(OUT / 'findings.json').write_text(json.dumps(findings, ensure_ascii=False, indent=2) + '\n')
fill['FINDINGS_TABLE'] = table(['编号', '分类', '发现/结论', '依据'], [
    [r['id'], r['type'], r['finding'], ','.join(r['evidence'])] for r in findings['confirmed_results']])

local_rows, resource_records = [], []
for code, title in [('input_identity','输入身份与全部载荷'),('supplied_audit_replay','原样随包审计重放'),('independent_statistics','主审独立统计')]:
    r = read(OUT / 'execution' / code / 'execution.json')
    resource_records.append({'review_task': code, **r})
    local_rows.append([title, r['exit_code'], format(r['elapsed_seconds'], '.9f'), str(r['child_peak_rss_kib_linux']) + ' KiB', 'execution/' + code])
for r in read(OUT / 'agents/execution/execution_ledger.json')['attempts']:
    resource_records.append({'review_task': 'execution_' + r['attempt'], **r})
    local_rows.append(['执行证据 ' + r['attempt'], r['exit_code'], format(r['elapsed_seconds'], '.9f'), str(r['peak_rss_bytes']) + '字节', 'agents/execution/' + r['attempt']])
r = read(OUT / 'agents/wiring/wiring_replay.execution.json')
resource_records.append({'review_task': 'wiring', **r})
local_rows.append(['目标接线保存证据', r['exit_code'], format(r['resource']['wall_seconds'], '.9f') + '（脚本内计）', str(r['resource']['peak_rss_kib']) + ' KiB', 'agents/wiring'])
r = read(OUT / 'agents/diagnostics/attempt01.execution.json')
resource_records.append({'review_task': 'diagnostics', **r})
local_rows.append(['独立诊断', r['exit_code'], format(r['outer_elapsed_seconds'], '.9f'), '47444 KiB', 'agents/diagnostics'])
fill['LOCAL_EXECUTION_TABLE'] = table(['核算进程', 'exit', '墙钟秒', '峰值RSS', '原件目录'], local_rows)
(OUT / 'resource_and_execution_record.json').write_text(json.dumps({
    'scope':'Review computation records only; never project-native or formal execution charges',
    'local_runs': resource_records, 'project_resources_from_preserved_evidence': exe['formal_resources'],
    'review_failure_count': 1, 'separate_exploratory_field_error': True,
    'walltime_not_whole_session': True, 'overlapping_checks_not_independent_samples': True,
}, ensure_ascii=False, indent=2) + '\n')

evidence = [
    ['E01','input/positive_ranking_result_review.zip；内含manifest.json、request.zh.txt','唯一获准输入及全部载荷身份'],
    ['E02','workspace/docs/SELLER_ALIAS_POSITIVE_RANKING.zh.md（17–23、45–55行）','目标、数量、主要/次级判据、分组与解释边界'],
    ['E03','workspace/reports/documentation/20261016/positive_ranking_implementation/qualification/；review/workspace/','执行许可、CPU/native/formal gate及已关闭实现意见'],
    ['E04','workspace/reports/qualification/native/result.json；workspace/reports/launch/native_*.json','真实调用、梯度/参数/Adam、恢复、第二臂资源及投影'],
    ['E05','workspace/reports/job/{execution,before_valid,completion,access,resource}.json；run/；job.console.txt；job.wrapper.txt','正式源绑定、调度/更新/捕获、退出/访问/资源原件'],
    ['E06','workspace/reports/job/evaluation/；workspace/reports/documentation/20261016/h_consistency/baseline_sources.json及其52来源','保存查询/指标/计数/比较和固定S/H对照'],
    ['E07','workspace/reports/job/run/diagnostics/；workspace/reports/job/diagnostics_evaluation/','全部24捕获、18判读、盲分数与事后指标'],
    ['E08','workspace/reports/result_sync/','执行者报告、独立分析、回传/计费/保管观察'],
    ['E09','workspace/scripts/step28_positive_ranking*.py、step28_record_replay.py、step28_g_half*.py、step28_replay_diagnostics.py、step28_balanced_pooling_run.py、step28_alias_calibration.py、step28_continual_population_evaluate.py','直接相关目标/共享VJP/入口/统计/诊断代码；具体实际范围见reading_scope.json'],
    ['R00','input_identity.json；input_inventory_verified.json；execution/input_identity/','本审查输入字节核对原件'],
    ['R01','supplied_replay/；execution/supplied_audit_replay/','原样审计成功重放及原始stdout/stderr/exit/资源'],
    ['R02','independent_statistics/；scripts/independent_statistics.py；execution/independent_statistics/','主审独立重构2640统计与16分组、N/A和阈值对应'],
    ['R03','agents/execution/','执行来源专项、2078断言及完整失败/成功原件'],
    ['R04','agents/wiring/','目标接线、原生作用与正式分项核对'],
    ['R05','agents/diagnostics/','24/18独立诊断、概率、分组、曲线与解释界限'],
]
fill['EVIDENCE_TABLE'] = table(['索引', '原件/审查产物位置', '支持内容'], evidence)
(OUT / 'evidence_index.json').write_text(json.dumps([dict(zip(('id','path','supports'), row)) for row in evidence], ensure_ascii=False, indent=2) + '\n')

absolute_sections = []
for stage in 'AB':
    for role in roles:
        header = ['指标'] + [name(a) + '/' + d for d in 'AB' for a in arms]
        rows = [[k] + [fmt(get('absolute', stage + '_' + a, role, d, k)['mean'], digits=9) for d in 'AB' for a in arms] for k in columns]
        absolute_sections.append(stage + '末 / ' + role + ' / 各域20群宏平均\n\n' + table(header, rows))
fill['ABSOLUTE_APPENDIX'] = '\n\n'.join(absolute_sections)
count_rows = []
for stage in 'AB':
    for a in arms:
        point = stage + '_' + a
        for role in roles:
            for d in 'AB':
                tp, fp, fn, tn = stat['confusion_totals'][point][role][d]
                precision = tp / (tp + fp) if tp + fp else None
                missing = ','.join(k for k in ('precision','mcc') if get('absolute',point,role,d,k)['mean'] is None) or '无'
                count_rows.append([name(point), role, d, f'{tp}/{fp}/{fn}/{tn}', fmt(precision, digits=9),
                                   fmt(tp/(tp+fn),digits=9), fmt(2*tp/(2*tp+fp+fn),digits=9), fmt(fp/(fp+tn),digits=9), missing])
fill['COUNTS_APPENDIX'] = '全部48组混淆计数与汇总率\n\n' + table(['端点','角色','域','TP/FP/FN/TN','pooled precision','pooled recall','pooled F1','pooled FPR','macro N/A列'], count_rows)
curve_rows = []
for point in ('A_S_pos','B_S_pos','A_H_pos','B_H_pos'):
    for d in 'AB':
        for epoch, row in enumerate(diag['valid_curves'][point][d], 1):
            curve_rows.append([point,d,epoch,*[fmt(x,digits=12) for x in row]])
assert len(curve_rows) == 48
fill['CURVES_APPENDIX'] = '24捕获对应的48个验证域曲线点（不用于选轮）\n\n' + table(['端点','验证域','epoch','MAP','Recall@5','log-loss'],curve_rows)

template = (OUT / 'report_template.zh.txt').read_text()
for key, value in fill.items():
    template = template.replace('{{' + key + '}}', value)
assert '{{' not in template
report = OUT / 'S_pos_H_pos_external_review_20261018.zh.txt'
report.write_text(template)
print(json.dumps({'report':str(report),'bytes':report.stat().st_size,'characters':len(template),
                  'sha256':hashlib.sha256(report.read_bytes()).hexdigest(),'complete_statistic_rows':len(stats),
                  'absolute_metric_tables':6,'counts_rows':len(count_rows),'valid_curve_rows':len(curve_rows)},ensure_ascii=False,indent=2))
