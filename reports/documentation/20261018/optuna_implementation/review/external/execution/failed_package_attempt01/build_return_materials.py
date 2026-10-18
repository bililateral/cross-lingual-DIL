"""Build the review deliverable from static text and saved mechanical results.

Reviewer authored; standard library only; never imports or runs project code.
"""
from pathlib import Path
import csv
import hashlib
import io
import json
import re
import shutil

ROOT = Path('/workspace/scratch/3e46faac4c35/optuna_review_input')
OUT = Path('/workspace/scratch/3e46faac4c35/optuna_static_review')
P = 'reports/documentation/20261018/optuna_plan/'
I = 'reports/documentation/20261018/optuna_implementation/'
Q = I + 'qualification/'

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

report = (OUT/'report.zh.txt').read_text(encoding='utf-8')
identity = json.loads((OUT/'mechanical/input_identity.json').read_text())
arithmetic = json.loads((OUT/'mechanical/budget_and_count_arithmetic.json').read_text())

# Explicit main-review adjudication. Detailed narrative is copied from the report.
stages = {
 'N01': ('before_native_execution', '原生执行前', 'confirmed_resource_preflight_omission', ['F02','F05']),
 'N02': ('before_formal_execution', '正式执行前', 'confirmed_runtime_resource_guard_omission', ['F05']),
 'N03': ('before_formal_dispatch', '正式子进程调度前', 'confirmed_failure_cleanup_gap', ['F03','F05']),
 'N04': ('before_formal_evaluation_completion', '正式完整评价前', 'confirmed_required_output_omission', ['F04']),
 'N05': ('before_formal_feedback_analysis_acceptance', '正式反馈/独立分析验收前', 'confirmed_verification_coverage_gap', ['F03','F04']),
 'N06': ('current_report_correction', '当前交付报告纠正', 'confirmed_reporting_count_error', ['F02','F05']),
 'N07': ('before_formal_anchor_acceptance', '正式默认锚点前', 'confirmed_historical_reference_binding_gap', ['F01']),
 'N08': ('before_formal_controller_execution', '正式控制器执行前', 'confirmed_controller_time_guard_and_charge_gap', ['F05']),
}
locations = {
 'N01': [('scripts/step28_optuna_verify.py',85,103),('scripts/step28_optuna_verify.py',229,235),('scripts/step28_positive_ranking_verify.py',55,60),('scripts/step28_g_half_verify.py',113,120)],
 'N02': [('scripts/step28_optuna_run.py',70,82),('scripts/step28_optuna_run.py',237,300),('scripts/step28_optuna_search.py',195,205),('scripts/step28_optuna_verify.py',215,216)],
 'N03': [('scripts/step28_optuna_search.py',106,130),('scripts/step28_optuna_search.py',171,178)],
 'N04': [(P+'proposal.zh.txt',52,52),(P+'proposal.zh.txt',70,70),('scripts/step28_optuna_run.py',303,351),('scripts/step28_replay_diagnostics.py',130,208)],
 'N05': [('scripts/step28_optuna_verify.py',25,82),('scripts/step28_optuna_run.py',222,222),('tests/test_step28_optuna.py',266,283)],
 'N06': [(I+'report.zh.txt',35,35),(Q+'cpu_gate.json',356,356),('tests/test_step28_optuna.py',65,111),('tests/test_step28_optuna.py',134,147)],
 'N07': [('scripts/step28_optuna_run.py',168,178),('scripts/step28_optuna_run.py',33,51),('scripts/step28_optuna_search.py',72,91),('scripts/step28_h_consistency_run.py',19,19),('scripts/step28_h_consistency_run.py',78,113)],
 'N08': [('scripts/step28_optuna_search.py',5,18),('scripts/step28_optuna_search.py',95,103),('scripts/step28_optuna_search.py',153,160),('scripts/step28_optuna_search.py',202,211),('scripts/step28_optuna_control.py',75,103)],
}

new = []
for ident in stages:
    match = re.search(r'^'+ident+r' /.*?(?=\nN\d\d /|\n十、)', report, re.M|re.S)
    if not match:
        raise ValueError('Missing report finding '+ident)
    body = match.group(0).strip()
    def field(label):
        m = re.search(r'^'+label+'：(.*)$', body, re.M)
        return m.group(1) if m else None
    stage, stage_zh, kind, prior = stages[ident]
    new.append(dict(id=ident, priority='P1' if ident=='N02' else 'P2', status='OPEN_FOR_PRIMARY_DISPOSITION',
        kind=kind, title_zh=field('标题'), stage=stage, stage_zh=stage_zh, related_previous_findings=prior,
        confidence='high_static', observed_runtime_failure=False,
        locations=[dict(path=p,lines=[a,b],sha256=digest(ROOT/p)) for p,a,b in locations[ident]],
        trigger_zh=field('触发'),impact_zh=field('影响'),minimum_change_zh=field('最小修改'),disposition_zh=field('处置'),
        complete_finding_zh=body,
        blocks_current_authorized_targeted_implementation=False,
        blocks_native_execution=(ident=='N01'),
        blocks_formal_acceptance=(ident in ('N02','N03','N04','N05','N07','N08')),
        invalidates_existing_cpu_wall_charge=False,
        external_scientific_reproduction=False))

prior = []
prior_titles = {
 'F01':'随机根、旧 s0 兼容与新 seed 身份',
 'F02':'lambda/LR 与完整 S 正式分支',
 'F03':'持续三 study、精确反馈、失败与硬截止',
 'F04':'九次集中开放、45/37 与全评价',
 'F05':'集中评价预算、资源与退出端点',
 'F06':'新 9000 秒及 114 小时时间准入',
 'F07':'新 80 群和最终确认',
 'F08':'有限机会、因果与创新解释边界',
}
statuses = {'F01':'IMPLEMENTED_WITH_FORMAL_COMPATIBILITY_PENDING','F02':'IMPLEMENTED_CPU_APPLICABLE_NATIVE_PENDING',
 'F03':'CORE_IMPLEMENTED_SPECIFIC_GAPS_OPEN','F04':'OPEN_REQUIRED_OUTPUTS_INCOMPLETE',
 'F05':'CORE_ACCOUNTING_IMPLEMENTED_RESOURCE_AND_RELEASE_GAPS_OPEN','F06':'NOT_ESTABLISHED_INDEPENDENTLY_OPEN',
 'F07':'DEFERRED_FINAL_CONTRACT_ONLY','F08':'ACCEPTED_WITH_INTERPRETATION_LIMITS'}
for ident in prior_titles:
    m = re.search(r'^'+ident+r' [^\n]+$', report,re.M)
    prior.append(dict(id=ident,title_zh=prior_titles[ident],latest_status=statuses[ident],
        judgment_zh=m.group(0) if m else None,
        linked_new_findings=[n['id'] for n in new if ident in n['related_previous_findings']]))

findings = dict(schema_version='1.0',review_date_utc='2026-10-18',language='zh-CN',
    input_identity={k:identity[k] for k in ('bytes','sha256','members','payloads','expected_identity_matches')},
    requested_model='GPT-6 / Pro', environment_self_description='GPT-6 Astra Pro',
    ui_model_selector_observed=False,backend_routing_verified=False,
    collaboration=dict(auxiliary_agents=['objectives_native','control_budget','run_evaluation'],explicit_model_override=False,
        final_adjudication='root',single_instance_only_claim=False),
    decision=dict(status='CONTINUE_TARGETED_IMPLEMENTATION_NOT_ACCEPTED_AS_COMPLETE',
        worth_continuing=True,worth_applying_for_planned_native_after_targeted_fixes_and_primary_review=True,
        native_pass=False,formal_execution_allowed=False,new_9000_second_qualification_established=False,
        shared_capacity_established=False,development_114_hour_feasibility_established=False,
        new_data_or_final_evaluation_authorized=False,
        summary_zh='主科学接线成立，八项资源/释放/计时/输出/证据和口径问题须分阶段处置；有限修正值得继续，新时间资格独立未建立。'),
    actual_review_execution=dict(static_only=True,project_imports=0,project_test_runs=0,model_loads=0,
        metric_recomputations=0,bootstrap_runs=0,project_server_connections=0,web_lookups=0,
        library_or_other_destination_uploads=0,prohibited_dataset_accesses=0,
        operations=['text_read','zip_and_tar_member_bytes','sha256_and_size','AST_structure_and_function_text_comparison','saved_field_extraction','budget_and_count_arithmetic']),
    previous_findings=prior,new_findings=new,
    cpu_evidence=dict(applicable_test_count=7,outer_wall_seconds=77.254095395,charged_seconds=82,cap_seconds=300,
        original_claimed_actual_optimizer_calls=39,production_optimizer_steps=39,independent_reference_optimizer_steps=6,
        total_actual_cpu_optimizer_steps=45,native_bge_gpu_calls=0,formal_training_runs=0,
        actual_cpu_six_call_arm='H_pos',mocked_six_call_resource_and_shape_checks=True,
        source_files=46,snapshots=6,snapshot_files_each=46,attempt02_failure_preserved=True,
        synthetic_feedback_slots=36,synthetic_saved_analysis_slots=45,
        synthetic_slots_are_training=False),
    time_admission=dict(status='NOT_ESTABLISHED',old_two_arm_projection_seconds=38079.87608841504,
        new_two_run_cap_seconds=18000,retained_model='legacy_1.25_allowance1800',
        per_run_cap_seconds=9000,shared_caps_seconds=dict(control=300,evaluation=2400,analysis=600),
        total_budget_seconds=410400,old_actual_mean_or_cpu_seconds_are_not_admission=True,
        L_measurement_boundary_caveat_zh='新 L 在 pretrained_archive 校验后开始；未来资格须明确覆盖该成本，不能仅因公式名相同宣称测量边界相同。'),
    report_path='report.zh.txt',evidence_index_path='evidence_index.tsv',reading_scope_path='reading_scope.json',
    mechanical_failures_path='execution/failures.json')
write_json(OUT/'findings.json',findings)

# The raw line logs record requested ranges. Manual attestation is deliberately separate.
logs = []
for path in sorted(OUT.glob('reads_*.jsonl')):
    logs.extend(json.loads(line) for line in path.read_text().splitlines() if line)
root_full = [
 'AGENTS.md',P+'implementation_scope.zh.txt',P+'budget_amendment.zh.txt',P+'proposal.zh.txt',
 P+'review/primary_disposition.zh.txt',P+'review/external/review/report.zh.txt',P+'review/external/review/findings.json',
 'docs/SELLER_ALIAS_OPTUNA.zh.md',I+'report.zh.txt',I+'authorization.json',I+'implementation.diff',
 Q+'cpu_gate.json','schema/step28_optuna_policy.json',
 'scripts/step28_optuna.py','scripts/step28_optuna_control.py','scripts/step28_optuna_run.py',
 'scripts/step28_optuna_search.py','scripts/step28_optuna_verify.py','tests/test_step28_optuna.py',
 'scripts/step28_positive_ranking.py','scripts/step28_record_replay.py','scripts/step28_replay_diagnostics.py',
 'scripts/step28_g_half.py','scripts/run_step28_optuna_cpu_linux_20261018.sh',
 Q+'receive.json',I+'review/preparation_failure.json',Q+'attempt02/unittest.txt']
def merge_ranges(values):
    out=[]
    for start,end in sorted(values):
        if out and start<=out[-1][1]+1: out[-1][1]=max(out[-1][1],end)
        else: out.append([start,end])
    return out
scope_rows=[]
for item in identity['payload_checks']:
    path=item['path'];calls=[row for row in logs if row['path']==path]
    by_reader={reader:merge_ranges([(x['start'],x['end']) for x in calls if x['reader']==reader]) for reader in sorted({x['reader'] for x in calls})}
    category=('main_full_semantic_read' if path in root_full else 'requested_text_read_ranges' if calls else 'identity_only_or_saved_field_mechanical')
    notes=[]
    if path==I+'review_request.zh.txt':
        category='user_supplied_complete_request_and_attachment_text';notes.append('Full content also supplied by user; one bad relative-path read was preserved as failure.')
    if path.startswith(Q):notes.append('CPU saved evidence/snapshot/checksum fields mechanically inspected as recorded; no scientific rerun.')
    if path.endswith('_sources.tar.gz'):notes.append('All 46 source members compared to own result manifest; selected changed source/test definitions inspected statically.')
    if path.endswith('workspace.sha256'):notes.append('45/46 match current; prior test snapshot difference retained, not overwritten.')
    if calls and path not in root_full:notes.append('These are requested ranges; output may have truncated. Do not infer full semantic coverage solely from range union.')
    scope_rows.append(dict(path=path,bytes=item['bytes'],sha256=item['sha256'],identity_verified=item['matches_manifest'],
        category=category,main_full_semantic_attested=path in root_full,requested_ranges_by_reader=by_reader,notes=notes))
scope=dict(schema_version='1.0',input_payload_count=140,all_payload_identity_verified=True,
    main_core_read_attestation_zh='主审完整读过所列核心合同、前轮审查、五入口、当前测试和关键直接依赖；在批量截断后以分段补读完成。旧依赖只按实际定向范围，不冒称整个仓库或140文件全文语义阅读。',
    requested_read_log_limit_zh='reads_*.jsonl 是工具请求范围，不是每个字节均可见的证明；本字段与人工完整阅读确认分开。',
    exclusions=['No project/test execution','No formal data/labels','No model or Memory loading','No metric recomputation or bootstrap','No prior conversation lookup'],
    main_full_semantic_paths=root_full,files=scope_rows)
write_json(OUT/'reading_scope.json',scope)
with (OUT/'reading_scope.tsv').open('w',encoding='utf-8',newline='') as stream:
    writer=csv.writer(stream,delimiter='\t');writer.writerow(['path','bytes','sha256','identity_verified','category','main_full_semantic_attested','requested_ranges_by_reader','notes'])
    for row in scope_rows:
        writer.writerow([row[k] if not isinstance(row[k],(dict,list)) else json.dumps(row[k],ensure_ascii=False) for k in ('path','bytes','sha256','identity_verified','category','main_full_semantic_attested','requested_ranges_by_reader','notes')])

evidence=[]
def add(ref,title,path,start=None,end=None,scope='input'):
    evidence.append(dict(ref=ref,title_zh=title,path=path,start=start,end=end,scope=scope))

add('E00','实际ZIP与140载荷身份','mechanical/input_identity.json',scope='review')
add('E01','新增实施规格，本轮首次审查',P+'implementation_scope.zh.txt',1,80)
add('E01','冻结方案选择与复验要求',P+'proposal.zh.txt',39,80)
add('E01','实际已获准CPU/代码范围',I+'authorization.json',1,19)
add('E01','当前合同','docs/SELLER_ALIAS_OPTUNA.zh.md',1,19)
add('E02','预算补充并非前轮已审',P+'budget_amendment.zh.txt',1,75)
add('E03','TrialSpec/根/Memory/目标/LR全部新方法','scripts/step28_optuna.py',1,241)
add('E03','原rank与hard目标入口','scripts/step28_alias_ranking.py',89,129)
add('E03','固定H pooling','scripts/step28_g_half.py',18,49)
add('E04','真实Entry、初始/恢复/历史锚点','scripts/step28_optuna_run.py',85,179)
add('E05','控制器、精确选择、预算、真实sampler与开放门','scripts/step28_optuna_control.py',1,209)
add('E06','新单run资源检查','scripts/step28_optuna_run.py',54,82)
add('E06','真实单run与集中开放','scripts/step28_optuna_run.py',237,300)
add('E07','真实许可、时间准入、子进程、退出、生命周期和总调度','scripts/step28_optuna_search.py',1,218)
add('E08','数组结构、全评价和反馈来源','scripts/step28_optuna_run.py',181,236)
add('E08','三seed保存汇总实际覆盖','scripts/step28_optuna_run.py',303,351)
add('E08','逐run诊断评价','scripts/step28_replay_diagnostics.py',130,208)
add('E09','七项CPU证据与最新来源映射',Q+'cpu_gate.json',1,367)
add('E09','受影响手写测试全文','tests/test_step28_optuna.py',1,318)
add('E09','独立保存结果分析实际覆盖','scripts/step28_optuna_verify.py',25,82)
add('E09','保存CPU字段而非外部重算','mechanical/saved_cpu_fields.json',scope='review')
add('E09','全部attempt结果/外层日志/快照对应','mechanical/cpu_attempts_and_snapshots.json',scope='review')
add('E10','待许可真实原生六调用代码','scripts/step28_optuna_verify.py',85,206)
add('E10','CPU/native/analysis入口与资源门','scripts/step28_optuna_verify.py',209,277)
add('E10','旧原生预检含continuing清理、磁盘','scripts/step28_positive_ranking_verify.py',25,70)
add('E10','原有archive/磁盘/保存辅助','scripts/step28_g_half_verify.py',80,120)
add('E11','实际监督解析先记attempt','scripts/step28_bge_continual_run.py',76,87)
add('E11','七文件读取/C先排除边界','scripts/step28_balanced_pooling_run.py',89,205)
add('E11','三个真实评价角色及训练配置/日程','scripts/step28_bge_continual.py',20,85)
add('E11','完整22指标定义列','scripts/step28_continual_population_evaluate.py',1,45)
add('E12','两个旧公共入口的受影响改动',I+'implementation.diff',1,95)
add('E12','实际阶段日程/训练/捕获','scripts/step28_g_half_run.py',177,235)
add('E12','真实checkpoint/恢复/Memory及完整单run','scripts/step28_g_half_run.py',236,369)
add('E12','实际盲完整性与新身份适配','scripts/step28_positive_ranking_run.py',219,358)
add('E12','真实loader的初始化seed','scripts/step28_chinese_base.py',89,134)
add('E12','实际两组optimizer构造','scripts/step28_continual_population.py',1,61)
add('E12','中性诊断保存与恢复','scripts/step28_replay_diagnostics.py',1,129)
add('E13','保留成本模型','scripts/step28_g_half_run.py',163,174)
add('E13','预算、外层秒数及物理step计数算术','mechanical/budget_and_count_arithmetic.json',scope='review')
add('E14','保存S0字段与矩阵结构，未重算均值','mechanical/saved_reference_fields.json',scope='review')
add('E15','旧结果有效但主要门失败的已保存主审','reports/seller_alias_continual/20261018/positive_ranking_result/review/primary_disposition.zh.txt',1,28)
add('E16','返回清单对应与历史差异保留','mechanical/received_checksum_correspondence.json',scope='review')
add('E16','47载荷加两份清单算术','mechanical/received_payload_totals.json',scope='review')
add('E16','attempt02实际失败',Q+'attempt02/unittest.txt',1,17)
add('E16','本包准备阶段失败原件',I+'review/preparation_failure.json',1,20)
add('E16','本轮审查机械失败','execution/failures.json',scope='review')
add('E17','当前run sources不递归绑定旧报告','scripts/step28_optuna_run.py',18,51)
add('E17','旧52来源固定链','scripts/step28_h_consistency_run.py',15,20)
add('E17','旧S/G_half的baseline_runs绑定','scripts/step28_h_consistency_run.py',78,113)
add('E17','52来源内的旧S manifest','reports/documentation/20261016/h_consistency/baseline_sources.json',500,521)
add('E17','52来源内的旧G_half manifest','reports/documentation/20261016/h_consistency/baseline_sources.json',656,678)

(OUT/'evidence').mkdir(exist_ok=True)
extracts=['审查证据摘录（只读取附件文本及审查机械保存字段；未执行项目代码）',
          '每段标明原附件相对路径、整文件SHA256与原文件行号；review路径指本返回包内审查材料。',
          '这些是必要定位摘录，不冒称所有旧依赖全文重审。']
for item in evidence:
    source=(ROOT if item['scope']=='input' else OUT)/item['path']
    item.update(bytes=source.stat().st_size,sha256=digest(source))
    if item['start'] is not None:
        lines=source.read_text(encoding='utf-8-sig').splitlines()
        if not 1<=item['start']<=item['end']<=len(lines):
            raise ValueError('Invalid excerpt range '+str(item))
        extracts += ['',f"[{item['ref']}] {item['title_zh']}",f"附件：{item['path']} | SHA256 {item['sha256']} | {item['start']}–{item['end']}/{len(lines)}"]
        extracts += [f'{i}: {lines[i-1]}' for i in range(item['start'],item['end']+1)]
    else:
        extracts += ['',f"[{item['ref']}] {item['title_zh']}",f"审查文件：{item['path']} | {item['bytes']} bytes | SHA256 {item['sha256']}"]
(OUT/'evidence/source_extracts.zh.txt').write_text('\n'.join(extracts)+'\n',encoding='utf-8')
with (OUT/'evidence_index.tsv').open('w',encoding='utf-8',newline='') as stream:
    writer=csv.writer(stream,delimiter='\t');writer.writerow(['ref','title_zh','scope','path','start','end','bytes','sha256'])
    for row in evidence:writer.writerow([row.get(k) for k in ('ref','title_zh','scope','path','start','end','bytes','sha256')])

# Preserve selected original failed evidence as byte-for-byte copies.
original_paths=[Q+'attempt02/unittest.txt',Q+'attempt02/result.json',Q+'attempt02.console.txt',
                Q+'attempt02.wrapper.txt',I+'review/preparation_failure.json']
copied=[]
for path in original_paths:
    relative=path.removeprefix(I)
    dest=OUT/'evidence/original_failures'/relative
    dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/path,dest)
    copied.append(dict(input_path=path,output_path=dest.relative_to(OUT).as_posix(),bytes=dest.stat().st_size,
                       sha256=digest(dest),bytes_equal=(ROOT/path).read_bytes()==dest.read_bytes()))
write_json(OUT/'evidence/original_failures/copy_manifest.json',copied)

write_json(OUT/'execution/review_execution.json',dict(
    scientific_execution=False,
    mechanical_failure_count=len(json.loads((OUT/'execution/failures.json').read_text())),
    raw_tool_read_logs='reads_*.jsonl',
    limitations_zh=[
      '部分批量输出截断，核心按分段补读；完整阅读确认与工具请求范围分开。',
      '未捕获所有成功shell命令的独立stdout/stderr档案；工具结果在会话中，返回提供实际机械程序、结果和失败记录。',
      '失败工具暴露combined output，不将可见错误片段伪称为独立完整stderr。',
      '目录不存在的静态查询不证明项目现场缺失该目录。',
      '检查、生成与打包只使用Python标准库；任何输入源码仅按文本或字节读取。'],
    review_mechanical_scripts=['mechanical_checks.py','supplemental_mechanical_checks.py','build_return_materials.py'],
    not_uploaded_to_external_storage=True))

(OUT/'README.zh.txt').write_text('''本返回包的主裁决是 report.zh.txt 和 findings.json。

阅读建议：先读报告第一、三、九节及 CPU 第七节；再按 evidence_index.tsv / evidence/source_extracts.zh.txt 定位原附件源码。完整范围见 reading_scope.json/tsv。contributions 是辅助原意见，已由主审裁决；若措辞不同，以完整报告和 findings 为准。

机械核对程序只用标准库，对文件大小/SHA、ZIP/tar成员、AST函数文本、保存字段和计数/预算算术操作。它们不导入项目模块、不运行项目测试、不重算指标或bootstrap。mechanical_checks.py 与 supplemental_mechanical_checks.py 顶部配置输入/输出路径；复核者可改为其本地附件提取目录和独立输出目录。程序中39/6/45的控制流计数由静态人工裁决显式列出，不冒称执行跟踪。

原附件未改动。本次审查三次机械路径错误见 execution/failures.json；项目attempt02失败与一次旧候选包准备失败的必要原件副本在 evidence/original_failures。所有原件副本按字节核对。source snapshots 的全体46文件身份在 mechanical/cpu_attempts_and_snapshots.json 中，本返回包无需再嵌套原2MB输入ZIP。

manifest.json列出除自身和 members.tsv 之外的所有返回载荷。members.tsv列出所有载荷及manifest的字节/SHA，不列自身以避免循环哈希。外层同名 .identity.json 和 .members.tsv 对最终ZIP全部成员再次机械列明。ZIP身份不写入内部manifest，避免自引用。

本次交付仅在当前会话提供下载。没有执行科研、连接项目主机、读受禁数据或向其他目的地上传。当前模型选择/后台路由未被独立验证，报告已区分用户请求、环境自述和未观察事实。
''',encoding='utf-8')
print(json.dumps(dict(new_findings=len(new),previous_findings=len(prior),input_scope_rows=len(scope_rows),
    main_full_semantic_paths=len(root_full),evidence_entries=len(evidence),original_failure_copies=len(copied)),ensure_ascii=False))
