#!/usr/bin/env python3
"""Assemble the human review and transparent reading/execution records from saved audits."""
from pathlib import Path
import csv,json,hashlib,datetime
R=Path('/mnt/data/h_review_input'); O=Path('/mnt/data/h_external_review')
def J(p):return json.loads(p.read_text())
def write(name,x):
 p=O/name;p.parent.mkdir(exist_ok=True,parents=True)
 p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def text(name,s):(O/name).write_text(s,encoding='utf-8')
audit=J(O/'checks/run01/audit_result.json');supp=J(O/'checks/run03/result.json');ident=J(O/'checks/input_identity.json')
with (O/'checks/run01/recomputed_statistics.csv').open() as f: rows=list(csv.DictReader(f))
ix={tuple(r[k] for k in ['kind','point_or_contrast','role','domain','metric']):r for r in rows}
def val(kind,point,role,domain,metric):
 r=ix[(kind,point,role,domain,metric)]
 if not r['mean']:return 'N/A'
 return f"{float(r['mean']):+.10f} [{float(r['lower_95']):+.10f}, {float(r['upper_95']):+.10f}]"
def abval(point,domain):
 r=ix[('absolute',point,'raw',domain,'map')]
 return f"{float(r['mean']):.9f} [{float(r['lower_95']):.9f}, {float(r['upper_95']):.9f}]"
diag=J(O/'checks/run01/diagnostics.json')
finaltable='| 比较 | 旧A：最终MAP差及95%条件区间 | 新B：最终MAP差及95%条件区间 |\n|---|---:|---:|\n'
for arm in ['S','G']:
 finaltable+=f"| H−{arm} | {val('final_contrast','G_half-'+arm,'raw','A','map')} | {val('final_contrast','G_half-'+arm,'raw','B','map')} |\n"
abtable='| 完整训练方法 | B末旧A MAP及区间 | B末新B MAP及区间 |\n|---|---:|---:|\n'
for arm in ['S','G','G_half']:abtable+=f"| {'H' if arm=='G_half' else arm} | {abval('B_'+arm,'A')} | {abval('B_'+arm,'B')} |\n"
probtable='| 比较 | 角色 | 域 | Brier差及区间（负为好） | log-loss差及区间（负为好） |\n|---|---|---|---:|---:|\n'
for other in ['S','G']:
 for role in ['raw','stage-cal','first-cal']:
  for domain in 'AB':probtable+=f"| H−{other} | {role} | {domain} | {val('final_contrast','G_half-'+other,role,domain,'brier')} | {val('final_contrast','G_half-'+other,role,domain,'log_loss')} |\n"
diagtable='| 阶段／角色／域 | 指标 | 训练收益 | valid收益 | 差距扩大量 | 差距扩大95%条件区间 |\n|---|---|---:|---:|---:|---:|\n'
for d in diag:diagtable+=f"| {d['point']}/{d['role']}/{d['domain']} | {d['metric']} | {d['train_benefit']:+.9f} | {d['valid_benefit']:+.9f} | {d['gap_widening']:+.9f} | [{d['gap_ci'][0]:.9f}, {d['gap_ci'][1]:.9f}] |\n"
classtable='| 方法 | 角色 | TP | FP | FN | TN | 汇总Precision | 汇总Recall | 汇总F1 |\n|---|---|---:|---:|---:|---:|---:|---:|---:|\n'
for arm in ['S','G','G_half']:
 for role in ['raw','stage-cal','first-cal']:
  d=audit['classification']['B_'+arm][role]['all'];prec='N/A' if d['precision'] is None else f"{d['precision']:.9f}"
  classtable+=f"| {'H' if arm=='G_half' else arm} | {role} | {d['tp']} | {d['fp']} | {d['fn']} | {d['tn']} | {prec} | {d['recall']:.9f} | {d['f1']:.9f} |\n"
# Evidence keys identify input files / original JSON fields, not synthetic file-search citations.
evidence={
 'E01':{'topic':'收到的附件及内部身份','sources':['manifest.json','context/baseline_sources.json','result/inventory.json','result/analysis_inventory.json','result/receive_verification.json'],'review_outputs':['checks/input_identity.json','evidence/input_members.json','checks/run01/audit_result.json#/identity','checks/run03/result.json']},
 'E02':{'topic':'授权、资格继承、门文本差异','sources':['context/formal_gate.json#/kind,/scope,/authorization,/gpu,/runtime,/supervision','context/cpu_receipt.json#/qualified_tests,/attempts','context/native_result.json#/native','context/native_disposition.zh.txt','context/implementation_primary.zh.txt','result/job/run/sources/reports/documentation/20261015/g_half_implementation/authorization.json','result/job/run/sources/schema/step28_g_half_policy.json','result/job/run/sources/scripts/step28_g_half_run.py:573-609'],'review_outputs':['checks/run01/audit_result.json#/identity','checks/run03/result.json#/gate_text_inconsistency']},
 'E03':{'topic':'初始、连续性、旧对照配对、全部逐步目标','sources':['result/job/run/manifest.json#/initials,/starts,/points,/training,/public_inputs,/pretrained_archive','result/job/run/updates/A_G_half.json','result/job/run/updates/B_G_half.json','baseline/run/manifest.json','baseline/run/partition.json','baseline/run/updates/{A_S,B_S,A_G,B_G}.json','result/job/run/sources/scripts/step28_g_half.py:34-103','result/job/run/sources/scripts/step28_record_replay.py:159-210','result/job/run/sources/scripts/step28_g_half_run.py:275-469'],'review_outputs':['checks/run01/audit_result.json#/training','checks/run01/program_read_files.json']},
 'E04':{'topic':'恢复、诊断末轮、盲门与一次开放','sources':['result/job/run/points/{A_G_half,B_G_half}.json','result/job/run/scores/*.npy','result/job/run/diagnostics/*','result/job/before_valid.json','result/job/access.json','result/job/completion.json','result/job/run/sources/scripts/step28_g_half_run.py:210-273,334-479,610-658','result/job/run/sources/scripts/step28_replay_diagnostics.py:1-208'],'review_outputs':['checks/run01/numeric_differences.json','checks/run01/audit_result.json#/diagnostics']},
 'E05':{'topic':'逐群/查询/计数与1650行独立配对统计','sources':['result/job/evaluation/collected.json','result/job/evaluation/*_metrics.json','result/job/evaluation/*_queries.npy','result/job/evaluation/*_counts.json','baseline/evaluation/*','result/job/evaluation/evaluation.json','result/analysis/all_metrics.csv','result/analysis/summary.json','result/step28_g_half_result_audit.py','result/job/run/sources/scripts/step28_g_half_run.py:482-570','result/job/run/sources/scripts/step28_balanced_pooling_run.py:595-616,648-682','result/job/run/sources/scripts/step28_continual_population_evaluate.py:28-102'],'review_outputs':['checks/run01/recomputed_statistics.csv','checks/run01/registered_draws.npy','checks/run01/audit_result.json#/statistics','checks/run03/result.json']},
 'E06':{'topic':'MAP终点、获得、变化、遗忘方向','sources':['result/job/evaluation/evaluation.json#/comparisons,/initial_comparisons,/forgetting_comparisons,/old_changes,/success','context/baseline_loss_correction.csv','result/job/run/sources/scripts/step28_g_half_run.py:550-570'],'review_outputs':['evidence/key_map_statistics.csv','evidence/loss_forgetting_contrasts.csv','evidence/selected_secondary_contrasts.csv']},
 'E07':{'topic':'概率、校准、阈值及不同统计对象','sources':['result/job/run/maps/{A_G_half,B_G_half}.json','result/job/run/manifest.json#/points/*/maps','result/job/evaluation/evaluation.json#/absolute','context/evaluation_definition.zh.txt:104-137'],'review_outputs':['evidence/final_probability_contrasts.csv','checks/run01/audit_result.json#/classification']},
 'E08':{'topic':'九诊断、六轮曲线及D0/D1','sources':['result/job/diagnostics_evaluation/overfitting.json','result/job/diagnostics_evaluation/*_valid_metrics.npy','result/job/run/diagnostics/*','result/analysis/summary.json#/diagnostic_summary,/valid_curves,/residuals','result/job/run/sources/scripts/step28_g_half.py:122-133','result/job/run/sources/scripts/step28_replay_diagnostics.py'],'review_outputs':['checks/run01/diagnostics.json','checks/run01/diagnostic_curves.csv','checks/run01/residuals.json','checks/run03/result.json']},
 'E09':{'topic':'实际资源、累计账、关闭记录','sources':['result/job.wrapper.txt','result/job.console.txt','result/job/completion.json#/budget','result/analysis_wallclock.json','context/native_wallclock.json','context/cpu_receipt.json','result/closeout.json','context/native_result.json#/native/measurements,/native/projection'],'review_outputs':['checks/run01/audit_result.json#/budget','checks/run03/result.json#/progress_events']},
 'E10':{'topic':'论文与适用边界，继承而非重新文献搜索','sources':['result/report.zh.txt','context/static_primary.zh.txt','context/baseline_primary.zh.txt','context/implementation_external.zh.txt:307-338','result/job/run/sources/docs/SELLER_ALIAS_G_HALF.zh.md'],'review_outputs':['findings.json','reading_scope.zh.md']}
}
write('evidence/evidence_index.json',evidence)
rd={'id':'RD-01','category':'REPRODUCIBILITY_DEFECT','severity':'LOW_NON_BLOCKING_DOCUMENTATION','status':'CONFIRMED_METADATA_INCONSISTENCY','title':'正式门的说明性scope保留了原生资格文字','evidence':['E02','context/formal_gate.json#/scope','context/formal_gate.json#/kind,/runtime,/supervision,/authorization','result/job/run/sources/reports/documentation/20261015/g_half_implementation/authorization.json#/user_verbatim'],'observed':'scope写Unique six-call...no formal data;600 seconds，但kind=FORMAL且runtime=28800秒、正式监督字段与政策完全一致；用户明确批准门控序列直到唯一固定训练。','impact':'降低单独阅读该门文件的人类可理解性；不构成未授权训练或统计失效，不影响机器按结构化字段校验。','minimal_disposition':'在本次主审处置/收尾说明追加一条勘误，说明scope保留原生模板措辞，正式有效范围由已绑定授权、kind、job、runtime、supervision、gpu共同确定；保持原门与所有哈希不变。','requires_training':False,'requires_reanalysis':False,'requires_restricted_assets':False}
rs=[]
for ident_,title,e,ob,act in [
 ('RS-01','保留固定目标失败与旧A局部收益','E06','双域H-S MAP门false；H-G/B直接门false；A最终收益有正区间。','保留现有判据及原始终点；不换平均/容忍门，不用R5/AP替代MAP。'),
 ('RS-02','不把完整轨迹差异解释为孤立抗遗忘机制','E03/E06','H-S A末已有优势；H-G遗忘差CI全负但最终MAP差跨零。','同时展示获得、终点与遗忘差，沿用原稿的相关性/因果边界。'),
 ('RS-03','概率与阈值代价必须和排序收益同页保留','E07','raw损失代价、stage-cal主角色局部收益、first-cal误报代价均存在。','原稿已披露，保留三角色和宏/汇总区分；不重新调阈值或角色。'),
 ('RS-04','九个未见明确迹象不等于没有过拟合','E08','九项差距扩大CI全部为正；B fit MAP强增益较少转移至valid；A raw损失两侧恶化。','保留九项分量与限定语，不回选epoch。'),
 ('RS-05','数值复核与资格/泛化证明分开','E01/E04/E05/E10','1650行重算正确，但未重建标签原值、模型/Memory、校准拟合或后端执行。','继续使用已披露的边界；发布时注明单seed/AB、重复开发、20群条件CI及未多重校正。')]:
 rs.append({'id':ident_,'category':'REASONABLE_FOLLOW_UP','severity':'NON_BLOCKING','status':'ALREADY_ADDRESSED_IN_SUBMITTED_REPORT_KEEP','title':title,'evidence':[e],'observed':ob,'minimal_disposition':act,'requires_new_compute':False})
ood=[{'id':'OOD-01','category':'OUT_OF_SCOPE_OVERDESIGN','request_rejected':'为显著性新增seed、顺序、臂、温度、系数、epoch选择或预算','reason':'本轮固定候选结果审查；当前失败不授权新探索。'}, {'id':'OOD-02','category':'OUT_OF_SCOPE_OVERDESIGN','request_rejected':'补传正式标签/文本、旧权重/Memory、封存/private资产；重载或重训S/G','reason':'当前43实际保存输入已足以做预定比较，受限载荷不属于网页复核前提。'}, {'id':'OOD-03','category':'OUT_OF_SCOPE_OVERDESIGN','request_rejected':'重审F求解器、旧项目全部历史、通用认证/验证系统或敌对环境审计','reason':'没有与本结果相关的新矛盾要求扩大范围。'}]
findings={'review_status':'COMPLETE_RESULT_REVIEW_WITH_ONE_MINOR_DOCUMENTATION_NOTE','scientific_verdict':'支持本固定H未达目标但旧A等局部收益保留；不支持整体优于S/G或论文主贡献充分。','groups':{'SCIENTIFIC_BLOCKER':[],'REPRODUCIBILITY_DEFECT':[rd],'REASONABLE_FOLLOW_UP':rs,'OUT_OF_SCOPE_OVERDESIGN':ood},'category_note':'没有新发现足以否定本次结果的科学阻断。RD-01仅是当前formal_gate说明字段的文本矛盾，不把历史待批准文字当作现在未批准。','reviewer_execution_error':{'id':'REVIEWER-ERR-01','project_defect':False,'status':'CLOSED','failed_program':'checks/reconcile_summary_v1.py','exit_code':1,'cause':'审查者猜写点号分隔文件名；实际路径为下划线且collected.json提供了准确路径。','fix':'v2仅改用collected.json中的matrix.path，并更换输出目录保留失败现场。','successful_program':'checks/reconcile_summary_v2.py','successful_exit_code':0,'impact':'补充summary核对首版中断，不影响已成功的1650行独立统计；没有项目文件变更或预算重置。'},'model_identity':{'requested_selector':'GPT 6 Pro（用户指定）','selector_independently_observed_this_turn':False,'self_description':'GPT-6 Astra Pro','backend_route_verified':False},'authority':'外部审查意见，不替代主执行者独立判断。'}
write('findings.json',findings)
# Per-member levels. A read by a program may be identity-only, never called a complete scientific reading.
full_text=['request.zh.txt','result/report.zh.txt','context/static_primary.zh.txt','context/implementation_primary.zh.txt','context/native_disposition.zh.txt','context/baseline_primary.zh.txt','result/step28_g_half_result_audit.py','result/job/run/sources/docs/SELLER_ALIAS_G_HALF.zh.md','result/job/run/sources/scripts/step28_g_half.py','result/job/run/sources/scripts/step28_g_half_run.py','result/job/run/sources/scripts/step28_replay_diagnostics.py','result/job/run/sources/schema/step28_g_half_policy.json','result/job/run/sources/reports/documentation/20261015/g_half_implementation/authorization.json','context/baseline_loss_correction.csv']
targeted={
 'context/research_discipline.zh.txt':'1-40及节标题/权限相关关键词定位；未声称全文重新审查',
 'context/evaluation_definition.zh.txt':'指标/概率角色相关检索，104-140连续段；旧三域汇总条款只作历史边界',
 'context/implementation_external.zh.txt':'结论/目录与相关发现定位；282-365连续段，重点F01-F06、原生未执行的历史时态及旧43项当时未附载荷；未重读全文所有推导',
 'result/job/run/sources/scripts/step28_record_replay.py':'159-210 VJP及update；234-310 Memory相关；其他仅定位/身份',
 'result/job/run/sources/scripts/step28_balanced_pooling_run.py':'89-198公开输入/标签附接/划分；595-616逐查询；648-682缺失与统计；不审F求解器',
 'result/job/run/sources/scripts/step28_bge_continual.py':'70-82学习率日程',
 'result/job/run/sources/scripts/step28_continual_population_data.py':'247-258种子/日程及序列化接口定位',
 'result/job/run/sources/scripts/step28_alias_ranking.py':'89-131自身top5与监督目标',
 'result/job/run/sources/scripts/step28_bge_continual_evaluate.py':'1-110旧公共评价接口/布局；非全面重审',
 'result/job/run/sources/scripts/step28_continual_population_evaluate.py':'1-106曲线/分类/检索定义；严格N/A由当前入口及保存计数另核对'
}
prog=J(O/'checks/run01/program_read_files.json');members=J(O/'evidence/input_members.json');sc=[]
for m in members:
 n=m['path'];semantic=[]
 if n in full_text:semantic.append('TEXT_READ_FULL_CURRENT_OR_INHERITED_SCOPE')
 if n in targeted:semantic.append('TEXT_READ_TARGETED')
 if n.endswith('.npy'):semantic.append('FULL_SAVED_ARRAY_LOADED_AND_CHECKED_NO_LABEL_RECONSTRUCTION')
 if ('/evaluation/' in n or '/updates/' in n or '/diagnostics/' in n or '/diagnostics_evaluation/' in n) and n.endswith('.json'):semantic.append('JSON_PARSED_ALL_ROWS_RELEVANT_FIELDS_CHECKED')
 if n in ['result/analysis/all_metrics.csv','context/baseline_loss_correction.csv']:semantic.append('ALL_NUMERIC_ROWS_CHECKED')
 if n=='result/analysis/summary.json':semantic.append('ALL_REPORTED_SECTIONS_RECONCILED')
 if n=='result/analysis/comparison.png':semantic.append('IMAGE_VISUALLY_INSPECTED_MAP_AND_CURVES')
 if n=='result/analysis/comparison.pdf':semantic.append('HASH_ONLY_NOT_RENDERED_PNG_AND_DATA_USED_INSTEAD')
 if (n in ['manifest.json','context/baseline_sources.json','context/formal_gate.json','context/cpu_receipt.json','context/native_result.json','context/native_wallclock.json','result/inventory.json','result/analysis_inventory.json','result/receive_verification.json','result/analysis_wallclock.json','result/closeout.json','result/job/run/manifest.json','baseline/run/manifest.json','result/job/run/partition.json','baseline/run/partition.json','result/job/before_valid.json','result/job/access.json','result/job/completion.json','result/job/execution.json','result/job/run/points/A_G_half.json','result/job/run/points/B_G_half.json','result/job/run/maps/A_G_half.json','result/job/run/maps/B_G_half.json'] and not semantic):semantic.append('JSON_PROVENANCE_AND_SELECTED_FIELDS_NOT_ALL_SEMANTICS')
 if n in ['result/job.console.txt','result/job.wrapper.txt']:semantic.append('EXECUTION_TEXT_OR_PROGRESS_CHECKED')
 if not semantic:semantic=['IDENTITY_ONLY_NOT_FULL_SEMANTIC_REVIEW']
 sc.append({**m,'integrity_checked':True,'review_modes':semantic,'targeted_scope':targeted.get(n),'listed_in_run01_byte_read_log':n in prog})
write('reading_scope.json',{'scope_note':'219项全部核对实际字节/SHA，不把哈希等同内容理解；210项主脚本read_files只是字节访问日志。具体范围如下。','files':sc,'unverified':['项目模型或Memory载荷未下载、未反序列化、未在本环境重新恢复','AP/ROC/Brier/log-loss逐群原值及逐查询相关性真值未从标签重新生成','校准拟合未用标签独立重跑；只核对映射记录/系数/分数变换/排序/计数','正式每一步所有参数梯度/位移未保存或逐参数复核；继承已关闭原生资格','正式逐步A1的实际分配数组未从未提供文本/标签重建；核对全部保存分配SHA及外生UID配对','无项目服务器连接；历史关闭/资源快照不等于现在远程实测','comparison.pdf仅身份核对，未解析/渲染；PNG和底层数表已检查','未新做F求解器复审、旧全项目回归或近邻文献搜索','当前网页模型选择器及后台路由未独立观察/验证']})
text('reading_scope.zh.md','''# 实际阅读范围与未验证边界

所有219个ZIP成员均完成内容字节、大小与SHA核对；根manifest的218个载荷全部相符。`reading_scope.json`逐文件列出阅读层级、范围、大小与SHA。`program_read_files.json`的210项是脚本字节访问记录，其中包含哈希读取，不是210项全文科学审读。

## 文本精读

完整读取本次request、结果报告、当前H合同、H核心133行、H正式运行器673行、诊断公共模块208行、Linux独立结果分析器，以及实施主审、原生处置、旧结果主审与静态主审。完整读取当前H policy、用户批准JSON和旧18行损失方向表。既有实施外审原文、科研纪律和旧评价合同按相关段落定向读取，未全量重审旧项目。

## 数值和来源

读取18个40×22指标矩阵、18套40×4计数、18个40×28×10查询数组，以及H两端点所有保存分数、12轮fit/cache/blind分数和12个已开放valid指标数组。核对全部288+288步H日志、四份288步S/G日志、划分、UID/域/列/角色、初始和连续性、出生Memory摘要、43基线映射、33来源及门/回执。重算1650行、9诊断及126个曲线均值，核对summary全部报告分区。读取JSON的“全部”不表示独立证明其记录的真实设备行为。

## 图与未改变依赖

comparison.png实际目视核对了两个最终MAP对比面板及B阶段fit/cache-valid曲线。comparison.pdf只核验字节身份，没有声称渲染或逐页分析PDF。未变化的F求解器只核对来源身份，不重审其内部算法。其他依赖的逐文件具体行段见JSON。

## 没有做的事

没有连接项目Linux、加载模型/Memory、读取正式文本或标签、重开heldout/owners、重训或重做CPU/GPU资格。没有从标签重建AP/ROC/Brier/log-loss和逐查询相关性；没有重新拟合校准。逐步A1只比对保存分配身份，未对未提供的实际记录重新分配。没有逐步逐参数重建全部576次权重更新。原生能力与此前已关闭反例按来源绑定继承。历史现场关闭记录不是现在服务器实测；本轮没有独立看到网页选择器或后台路由。

普通静态阅读命令的终端显示没有逐条复制为独立测量日志；实际隔离数值程序、其成功/失败输出及输入身份均保存在checks/与logs/。没有把读取次数/断言数量当成科学正确性的替代证据。
''')
ledger={'separation':'仅网页隔离保存证据核算，非项目py310原生执行；不修改项目CPU17、GPU320或正式6050秒账。','primary_environment':audit['environment'],'executions':[{'id':'inspection','program':'checks/inspect_inputs.py','status':'COMPLETED','exit_code_observed_from_tool':0,'stdout':'logs/inspect_inputs.stdout.txt','stderr':'logs/inspect_inputs.stderr.txt','dedicated_exit_code_file_initially_captured':False,'timing':'未单独采集'},{'id':'run01','program':'checks/audit_saved_v1.py','command':'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python checks/audit_saved_v1.py --input /mnt/data/h_review_input --output /mnt/data/h_external_review/checks/run01','exit_code_observed_from_tool':0,'stdout':'logs/audit_run01.stdout.txt','stderr':'logs/audit_run01.stderr.txt','dedicated_exit_code_file_initially_captured':False,'body_elapsed_seconds':audit['elapsed_seconds'],'status':audit['status']},{'id':'run02','program':'checks/reconcile_summary_v1.py','command':"OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python checks/reconcile_summary_v1.py --input /mnt/data/h_review_input --review /mnt/data/h_external_review --archive '/mnt/data/review_input(20261016-014443).zip'",'exit_code':1,'exit_code_file':'logs/reconcile_run02.exit_code.txt','stdout':'logs/reconcile_run02.stdout.txt','stderr':'logs/reconcile_run02.stderr.txt','body_elapsed_seconds':J(O/'checks/run02/result.json')['elapsed_seconds'],'status':'REVIEWER_FILENAME_ERROR_PRESERVED'},{'id':'run03','program':'checks/reconcile_summary_v2.py','command':"OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python checks/reconcile_summary_v2.py --input /mnt/data/h_review_input --review /mnt/data/h_external_review --archive '/mnt/data/review_input(20261016-014443).zip'",'exit_code':0,'exit_code_file':'logs/reconcile_run03.exit_code.txt','stdout':'logs/reconcile_run03.stdout.txt','stderr':'logs/reconcile_run03.stderr.txt','body_elapsed_seconds':supp['elapsed_seconds'],'status':supp['status']}],'timing_note':'代码计时从NumPy导入后开始，是检查主体时间；未把它冒充完整浏览器/解释器墙钟或项目计费。','retrieval':'Files语义检索两次未得到ZIP解析文本，随后读取已挂载原ZIP；不把零检索当成文件缺失。','failures':'保留一项真实审查者补充核对失败；没有项目运行失败或额外项目尝试。'}
write('execution_ledger.json',ledger)
text('README.zh.md','''# H完成结果外部审查产物

首先阅读`report.zh.md`（`report.zh.txt`内容相同）。`findings.json`提供分级发现，`reading_scope.json`逐文件给出实际阅读范围与输入SHA，`evidence/evidence_index.json`将正文E01—E10定位到原包路径/字段。

`checks/audit_saved_v1.py`是独立保存证据核查，`checks/run01/`有1650行重算、注册抽样索引、九诊断、126曲线均值和数值差。输入是用户已提供的原ZIP安全解包目录，不是项目正式数据。脚本不导入项目模块或加载模型/Memory。

补充summary核对首版`checks/reconcile_summary_v1.py`曾因审查者误写文件名退出1；失败源码、stdout/stderr/result均保留在run02。`reconcile_summary_v2.py`改为读取collected.json的准确路径，run03成功；两版本差异见patch。不要运行错误v1作为推荐入口。

可复现成功核对的命令（需Python、NumPy，输出应指向新的空目录）：

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python checks/audit_saved_v1.py --input /path/to/extracted_input --output /path/to/new_output/checks/run01
```

补充v2需要审查目录有上述run01输出与`evidence/`目录：

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python checks/reconcile_summary_v2.py --input /path/to/extracted_input --review /path/to/new_output --archive /path/to/original_review_input.zip
```

主脚本固定使用已登记PCG64(20260930)5000×2×20，不重新挑统计方案。不要在用户项目原生环境中把此脚本当成新的资格任务执行，也不据此重新计费/清零项目账。

本ZIP不重复包含用户原输入ZIP或任何模型/Memory/标签载荷。内部payload_manifest.json列出其余载荷；外部members.tsv列出ZIP全部成员（包括内部manifest）大小/SHA，避免自引用哈希。交付JSON提供ZIP字节数/SHA。
''')
# The report contains all decision-relevant reasoning, not merely a status summary.
report=f'''# 唯一 G_half（H）完成结果：独立外部科学审查

**审查结论：本次固定H轨迹和保存结果支持“目标未达成，但旧A等局部收益保留”。不支持“H整体超过S/G”，也不支持“论文主贡献已经充分”。**

本轮没有发现足以否定该运行、预定配对比较或1650行统计的新`SCIENTIFIC_BLOCKER`。发现一项**低风险、非阻断的文档一致性`REPRODUCIBILITY_DEFECT`**：当前formal_gate的说明性scope仍是六调用原生资格措辞，详见第十节。它不改变已明确批准的正式范围，也不要求重训、重新评价或修改冻结门。

审查只针对本包已发生的运行、保存证据、统计和结论。已关闭的实现与原生资格按具体来源复用；不重审未变F求解器，不新增候选、预算、训练或受限载荷访问。本报告是外部意见，不替代主执行者的独立处置。

## 一、模型身份、输入身份与证据级别

用户指定的网页选择器名称为**GPT 6 Pro**；本轮我没有独立看到选择器截图或读取其状态，不能把用户指定当成亲自验证的页面事实。本实例自述为**GPT-6 Astra Pro**；后台实际路由、档位及历史会话身份未独立验证。历史审查中关于页面选型的记录只适用于其各自时点。

收到附件：`review_input(20261016-014443).zip`，**2,798,254字节，219个ZIP成员**。

```text
SHA256 c1831a58ba41c7ff46c77b782aad9789730d234e9bf50faa836094cd23cc3880
```

ZIP CRC正常；成员无重复或不安全路径；根manifest列明218项载荷，全部实际大小/SHA相符，没有额外未列载荷。219个解包文件又与ZIP原成员逐字节一致。本次消息没有另给事前包SHA，因此本项建立的是**实际收到包的身份及内部完整性**，不宣称与未送达的另一份本地原件已独立比对。[E01]

`result`回传清单的147文件共4,102,971字节；`baseline`的43个实际输入共2,873,124字节；33个冻结来源逐项匹配运行、授权门、CPU/原生回执、盲门、收集、评价及冻结副本。哈希证明的是身份和不变性；实际行为另由目标分解、调度、分数、诊断及资格内容支持。[E01—E04]

本审查区分四层证据：①亲自复核的原包字节和保存数组；②由已冻结来源及记录构成的行为一致性；③按未变来源继承的项目Linux原生资格；④本轮未独立验证的真值/模型载荷和真实设备行为。不会以第一层冒充后两层。

## 二、执行轨迹、实际目标与盲门

### 2.1 同源未训练起点与自身连续性

H的`manifest.initials.G_half`与43基线映射、旧S/G实际manifest及原生资格中的初始model/Adam身份一致。共同model标识为`13b535e1d66a0f13f0d87461835bd3ba0cee69c531307884b471f8702bd14e4b`，空Adam标识为`de5b1864dc195dd65286e083567e3fd8bb6a146a9ecfcc3bae6b3fdda8c32d20`。正式输入、预训练归档的保存身份记录及完整fit/calibration/development划分也逐项相同。[E03]

H的`starts.G_half`与自己的`points.A_G_half.continuity`整个对象相同，而不是只比较模型哈希；其中包含模型、Adam、RNG/模式和Memory身份。B的历史六群、seen=48、出生表身份/来源在A末、B前后和B端点一致，历史draw计数0→288。审查脚本独立重建已登记Algorithm R与当前/历史调度，核对六群名单、全部UID和计数；未加载实际Memory。[E03]

这里的“同源初始”指未训练模型/空Adam一致；后文“初始A优势”指**已经完成A训练的A末能力**，两者不冲突。

### 2.2 576不是从completion里照抄的数字

H有两份完整288行更新日志；阶段step从1至288，Adam末计数分别288和576。当前日程每阶段6轮×48fit群；B另有288个真实历史群分支，因而梯度群呈现为288+2×288=864。26条稀疏控制台进度与两阶段[1,24,…,288]完全对应、耗时单调；这些进度是交叉证据，不代替逐步日志。[E03、E09]

源码`record.update`对当前和可用历史分支分别求梯度，累加后裁剪并实际调用一次`optimizer.step()`。本次保存记录中576个当前群分支加288个历史群分支，共864个record-gradient范数均为正，combined gradient范数均为正且有限；B的288个D0全部正值。没有将手写原生夹具的计数336冒充正式更新，原生仍只有六次真实调用。[E02、E03]

两阶段第288步encoder学习率按固定日程均为0，head仍0.001。因此应表述为**576次optimizer调用**，不能写成“576步每一个参数都变化”。原生记录已区分encoder/head真实梯度和位移，也明确原pooler两个参数不参与既定句向量路径；本轮没有逐步重建所有参数的梯度或位移。

### 2.3 实际公式与历史分支

本次源码中每账号对评分为：

\\[
H(K)=\\tfrac12\\operatorname{{mean}}(K)+\\tfrac12\\log\\left(\\tfrac1N\\sum_{{i=1}}^N e^{{K_i}}\\right),\\qquad \\tau=1.
\\]

H进入自然A0和重组A1的监督目标，并由各自H评分选自己的top5；没有借用S/G的top5。单群监督是两视图损失的平均。出生教师是该H轨迹保存记录表的H投影，D0为自然A0账号对输出的均方差，不是把mean分支误差平方和G分支误差平方各半混合。D1仅诊断。[E03]

当前阶段A：`total=current.supervised`。阶段B：

\\[
J=J_{{current}}+0.1J_{{history}}+0.5D_0.
\\]

逐步独立核对结果：A总目标最大差0；B总目标与上述标量分解最大绝对差**3.2782554449×10⁻⁸**，处于固定FP32运算容差内。B当前监督均值2.2189363626、历史监督均值2.2621102242、D0均值0.1253264465；`mse1=null`，未进入训练。出生表没有随B更新刷新。[E03]

这些证据支持当前保存运行确实走了预定目标及历史路径，但不是每一步都保存并重算了正式K、标签、完整自动微分图。H两分量、top5分界、正确教师/非零D0、VJP至encoder/head的原生作用按已关闭CPU/原生具体资格继承。不能把“总梯度非零”单独当成两分量或教师正确的证明。

### 2.4 恢复和十二诊断的行为证据

每个端点均保存原始raw与12群calibration分数，以及full和inference恢复后这两种分数。两个端点×两恢复路径×两分数组，共**八个数组对**逐元素一致。stage-cal和first-cal保存分数与正斜率仿射变换精确相符，完整排序和并列均保持。两端点各第6轮诊断的blind原始分数与最终raw端点精确相同；末轮MAP、R5、log-loss的40群值与端点指标对应。[E04]

两阶段分别有6个诊断点，epoch、stage_step=48×epoch、Adam计数、48fit群、B阶段6个cache群及40个valid盲预测群均吻合。源码明确向诊断传入H评分回调与H残差池化回调，不是只替换预测回调而让D0/D1退回旧评分。[E04、E08]

**未做的恢复验证：**没有在网页加载两份正式推理权重或Memory，也没有重新构造项目Adam和下一步更新；本次的精确结论是“保存的恢复后分数与原分数一致”。恢复代码及原生下一步1967张量/1,301,039,501元素逐位重现的既有证据按来源复用，不能冒充本轮新重载。

### 2.5 盲门确实包含行为条件，但访问账不是操作系统审计

`open_valid`先检查`train=1,valid=0,heldout=0,owners=0`，调用包含端点、恢复分数、Memory/调度、十二诊断及43旧对照配对的`blind_gate`，写出`before_valid.json`，之后才调用一次开发标签附接/解析。before_valid引用的完成manifest字节正确；最终访问账是`train=1,valid=1,heldout=0,owners=0`。[E04]

因此，冻结调用顺序、前后计数、完整manifest与保存结果相互支持“完成盲门后统一开放valid”。这里的计数是该入口的解析尝试账，不等于独立操作系统文件访问审计；盲预测阶段可以使用合同授权的开发公开输入，valid=0不应解释为从未接触任何开发公开内容。本轮没有重新运行会读模型/Memory的生产盲门，也没有通过新标签读取来证明这些记录。

## 三、旧S/G实际配对与归因范围

43份基线不是汇总表或文件名目录：实际含S/G各A/B端点×三角色的12套指标、查询与计数，共36文件，加旧manifest、partition、collected及4份逐步日志。所有43项按`baseline_sources.json`的明确根映射核对原相对路径、大小和SHA。[E01、E03、E05]

H/旧对照的40个群UID、实际域A/B、22指标列顺序、raw/stage-cal/first-cal角色一致；每域20群。每个指标矩阵40×22，每个查询数组40×28×10，每组计数涵盖20正对、358负对。fit/calibration与development群相互分离。全部当前和历史UID、每步当前/历史A1分配SHA与S、G一致，并非只比较随机种子。[E03、E05]

这形成**相同外生数据及重组/抽样安排下的完整训练系统配对**。H仍会改变自身评分、top5、监督梯度、出生教师投影、随后K和最终分数；旧S/G的K也各自经过训练。故本结果不能归因为“保持同一K，仅在推理时换池化”，也不能用同K时mean≤H≤G的代数夹逼推出最终MAP夹逼或某个孤立保持机制的因果优势。A1实际记录分配没有在本网页重建；这里机械验证的是正式保存的全部分配身份与外生序列一致。

## 四、1650行统计：独立重算及不可替代的边界

我另写了`checks/audit_saved_v1.py`，只使用标准库和NumPy；未导入生产模块，未调用生产finalize/blind_gate，未访问模型、Memory、正式标签或文本。其输入是本包保存数组、计数、映射和日志。[E05]

固定一次重建`Generator(PCG64(20260930)).integers(0,20,(5000,2,20))`，域轴按实际A/B，同一组抽样索引用于所有臂、端点、角色和对比。两臂差先在同UID群上形成，遗忘差也先在同群上构成差的差，再使用共同抽样求均值及线性2.5%/97.5%分位数。**未相减两个置信区间端点，未把5000次重复当成5000个独立群。**注册抽样数组原始int64数值字节SHA为`b1a539f80631940936dbd78b7141db5b37d96eeb9283a28c964afc733eda552e`；它不等同于带.npy头部文件的SHA。[E05]

复算行数构成：792绝对量+264最终比较+264A末比较+132遗忘比较+198旧域变化=**1650行**。其中114行N/A严格一致。主核算29258个数值标量比较最大差**5.5511151231×10⁻¹⁶**；该数字是保存统计/诊断的浮点组合差，不是逐步FP32目标残差或模型重现误差。LinuxCSV和生产JSON的全部对应行均核对；另一次补充核对把summary所有分区、群胜负、诊断及26条进度完整对齐，最大差2.2204460493×10⁻¹⁶。[E05]

18套逐查询值先各自对28查询平均，精确对应其10列群级检索指标。18套混淆计数重新计算precision/recall/F1/specificity/balanced accuracy/MCC；H另可由保存分数≥0独立核对预测正例总数tp+fp，无需标签。分母为0的precision/MCC保持N/A；任一所需群操作数缺失就使该固定群均值和区间N/A，不drop群、不填0、不分臂nanmean。先汇总计数再算比率可能有定义，这不代表群宏平均也有定义。[E05、E07]

**不可替代边界：**AP、梯形PR-AUC、ROC、Brier、log-loss的逐群原值仍来自已审生产评价；逐查询AP等相关性真值没有从正式标签重新重建。混淆计数可验证内部算术，但TP/FP的真值归属不是无标签就能独立重建。校准只核对保存映射、作用范围、变换和排序，未用校准标签重新拟合。本次结论是保存输入身份、内部行为一致性和统计组合通过，不是独立从标签复评全部结果。

## 五、主目标与必须保留的局部收益

### 5.1 最终绝对能力

{abtable}

### 5.2 直接配对比较

{finaltable}

固定判据是**B末H−S的A、B两域raw MAP条件区间下界都严格大于0**。A满足、B不满足，`success.passed=false`正确。H−G的新B下界也不大于0，`direct_G_new_B_positive=false`正确。[E06]

H−S旧A增益约2.798个百分点具有本固定条件下的正向证据；B点估计约−0.152个百分点、区间跨零。跨零既不证明有害，也不证明等价或非劣；不能用“基本持平”暗代未批准容忍门。H−G两域最终点估计为正但区间均跨零，不能宣布整体优于G。两域平均即使为正也不替代原双域门。

逐群描述也对齐：H−S旧A14升/6降、新B10升/10降；H−G旧A12升/8降、新B13升/7降。此类胜负计数不替代配对区间，也不额外产生独立成功证据。

### 5.3 获得、变化和遗忘是不同估计量

A末H−S/A MAP已为**+0.0215783689 [0.0012742634,0.0424026477]**。A末H−S/B为+0.0235416374 [0.0085597886,0.0373401700]；此时B尚未训练，属于最终统一开放的前向预测，不是B末学习成功。H−G/A末为−0.0094078455 [−0.0259881361,0.0040313303]。[E06]

旧A原指标B末−A末：H +0.0107957205、S +0.0043929486、G −0.0044461643，各自区间均跨零。按高值好指标F=A末−B末定义：

- H−S遗忘差：**−0.0064027719 [−0.0269174734,0.0121127932]**，未确定。
- H−G遗忘差：**−0.0152418848 [−0.0288753197,−0.0020143676]**，全负，支持H的旧A变化相对G更有利。

后者与H−G最终MAP差跨零可以同时成立。点估计恒等式为：

\\[
\\Delta_{{final}}=\\Delta_{{A-end}}-\\Delta_{{forgetting}}
=-0.0094078455-(-0.0152418848)=0.0058340393.
\\]

各项相关性不同，必须对同群的差或差的差直接重采样；上述恒等式不能用来相减区间端点。H−S的A优势在A末已经出现，所以不能把最终+0.0279811408全部归为“更少遗忘”。H−G相对变化证据也不能孤立归因于D0、混合池化或某个抗遗忘因果机制。[E03、E06]

本轮Brier/log-loss遗忘已按**B末−A末**定义（增大为坏），再做H−对照；`old_changes`则始终是原指标B−A。新12行损失遗忘比较独立复算正确，旧18行补正表的均值取负、区间端点取负并交换也逐行核对，但未改旧冻结文件。[E06]

## 六、其他指标、概率系统与误报代价

H−S/B的R5为**+0.0214285714 [0.0044419643,0.0392857143]**，但Recall@FPR≤1%为**−0.0300000000 [−0.0600000000,−0.0025000000]**。前者是每查询前五召回，后者是群内pair排序的低FPR工作点；它们可以一升一降，不能互相抵消或替代MAP门。[E06、E07]

H−G/B的群内pair AP为+0.013467416、区间[0.005724856,0.021239869]，梯形PR-AUC也正向；这不等于逐查询MAP已确定改善。不同查询分母和群内所有pair共同排序的权重不同。H−S/A多个检索/曲线指标方向一致有描述价值，但高度相关，不能计成多个独立确认。

### 6.1 三个角色不能后选

raw用于主排序，其sigmoid概率是辅助；stage-cal是预定主概率系统；first-cal只诊断。H的A首映射a=0.6714437041、b=2.1476662320；B阶段映射a=0.6238589075、b=1.5562173641；B的first仍为自身A映射。每次校准范围为当前域12群、4536对、240正对。映射对所有域统一作用，不按评价域另选。[E07]

{probtable}

因此必须同时保留：H−S的raw Brier/log-loss在A/B均有恶化证据；H−G的raw log-loss在两域也更差。stage-cal下H−S旧A的两种损失改善，新B未确定；H−G的stage-cal两域两损失全跨零。first-cal下H−G/B Brier改善只是诊断结果，不能越位替代主概率角色或MAP。

正仿射变换保存排序，所以概率代价和排序收益能够并存；stage-cal比较评价的是“模型＋当阶段校准映射”整个系统，不能据此声称原始模型概率没有代价或纯编码器抗遗忘得到证明。

### 6.2 固定0.5阈值：汇总计数而非群宏平均

下表合并A/B，共800正对、14320负对；它不是22指标表中的群宏比率。[E07]

{classtable}

H raw的旧A只有1TP/0FP，新B0TP/0FP；极低误报伴随极低召回。stage-cal的H相对S多14TP也多2FP，相对G少4TP也少9FP；这是实际权衡，不是某方法阈值指标全面胜出。H first-cal共117TP/77FP，召回提高伴随明显误报增加，且其角色仍仅诊断。没有重新调阈值。

## 七、九项诊断：标签正确，宽泛解释会错误

九项原严格判读均为“未见明确迹象”，重算正确。该标签所排除的仅是原规则要求的**训练改善、valid下降、差距扩大三者共同成立**；它不是“无过拟合”证明。事实上九项差距扩大量的条件区间**全部为正**。[E08]

以下收益方向统一为“正值改善”：MAP/R5为末−初，log-loss为初−末。每项判读仍均为原标签，不重新定义诊断门。

{diagtable}

最重要的两个反对性解释检查：

**B/fit/B MAP：**fit从0.566572升至0.645118，valid仅0.502209→0.504348。fit收益+0.0785454135，valid收益+0.0021392018、区间[−0.0118141643,0.0162992516]；差距扩大+0.0764062116 [0.0589519095,0.0934388851]。支持训练收益较少转移至未见群的具体观察，但不足以确认一个独立失效机制。

**A/fit/A raw log-loss：**fit 0.232110→0.318067，valid 0.226895→0.328597，二者均变差；不满足“训练改善”，所以仍得该原标签。这绝不能翻译成概率训练健康、没有概率代价。A的MAP则两侧均提高，因而不满足valid下降。B/cache/A MAP差距扩大也有正区间，与最终旧A局部收益并不矛盾。

诊断统计将固定训练/cache群与valid群分开重采样，复用登记的每个阶段/角色种子，不把fit群当valid配对群。六轮曲线、各训练/valid分量区间及126个曲线均值均核对，没有仅比较最终标签或事后挑首尾之外的epoch。曲线中的局部波动不授权重选最佳轮。

D0在B训练目标里，D1只诊断。B诊断D0均值首轮0.1295585482→末轮0.1686388071，D1 0.0968865007→0.1736463900；交叉项与残差差均方满足`difference=D0+D1−2cross`及非负/Cauchy一致性。它们是A0_eval的保存残差描述，和训练dropout/更新时点下的逐步D0不是同一个统计对象。未从缺失的正式K/T重算每个残差，不按不同方法的D绝对大小判断保持优劣。[E03、E08]

## 八、资源、权限与历史时态

原wrapper采用+08:00时区，**2026-10-16 00:00:08至01:40:56，退出0，6048秒**。Linux保存结果分析完整进程1.526959896秒，保守记2秒，正式累计**6050/28800秒**；余22750秒仅为原合同内收尾，不是新运行许可。CPU保留**17/180秒**，其中原临时夹具覆写错误及两轮继承事实不删；原生保持**6/6调用、320/600秒**。[E02、E09]

正式峰值CUDA reserved 8,650,752,000字节（约8.06GiB），RSS 6,846,410,752（约6.38GiB），输出6,516,177,771（约6.07GiB）。保存记录在原28/64/10GiB限内。最大形状投影独立复算为19043.417693秒、9,527,013,935输出字节，含原1446/36前向项、I/O、1800秒预留和1.25时间余量。实耗小于保守投影不是少跑证据；576逐步日志和12诊断另有核对。不能仅凭差值断言GPU性能提升原因，本轮未分解全部耗时。[E09]

结果包显示两推理权重共2,608,793,332字节、两Memory共690,219字节保留于项目Linux；这只是保存身份/留存记录，本轮没有下载或重载。临时full状态按源码在恢复后删除。closeout中的空进程/无failure是其记录时点的事实，不冒称我现在远程观察到了服务器。

旧合同/静态主审/实施外审里的“待批准”“尚未GPU”等按各自历史时点解释。最新明确用户批准、原生完成、正式门和正式完成记录构成连续链，不把旧待执行话语恢复成现状；也不据这条链追改原历史失败或计算账。

## 九、研究与论文边界

本次证据条件于**s0、单一A→B、两个域各20个开发群、固定28账号/27候选的受控合成任务**。原RNG命名使用ABC不代表本次训练了C。5000次整群bootstrap只量化该保存比较在现有群重采样下的条件变化，不覆盖训练seed、顺序、候选选择、反复valid开发或真实市场分布的不确定性。22指标、三角色、多种对比未作全家族/同时校正，不能把若干区间合计成确认性成功。[E05、E10]

本任务每查询已有1或2个相关候选，不能外推开放世界无正候选、极大候选池或真实全市场识别。阈值Precision依赖此候选集合和类别比例。本次未新增独立留出或真实标签；也不要求补传这些内容作为审查前提。

继承已关闭贡献定位：固定mean/LogAvgExp混合本身没有在本轮产生新聚合原理证据；本轮不重新做近邻文献检索，也不因构件已知就否认任务化系统的一切研究价值。问题在于**当前结果未通过自己的双域目标，且没有孤立机制因果证据或前瞻泛化确认**，不足以将H/G提升为论文主算法或宣称主贡献充分。完整S仍是主要内部对照；H未达标既不抹去S的有效证据，也不反向证明S的最终论文贡献已成立。[E10]

## 十、分级发现与最小处置

### SCIENTIFIC_BLOCKER：当前已证未决项0

没有发现会使本次运行或固定统计失效的新科学阻断。原稿的最终裁决与保存证据相符。若将其改写成“整体优于S/G”“没有过拟合/概率代价”“孤立抗遗忘机制已证明”或“论文主贡献充分”，那将形成结论与证据冲突；当前稿没有这样写，所以不能给原稿强加这些错误。

### REPRODUCIBILITY_DEFECT：RD-01，低风险说明字段矛盾

**具体字段：**`context/formal_gate.json#/scope`写`Unique six-call G_half handwritten native qualification under approved contract; no formal data; 600 seconds, no automatic retry.`，但同一对象`kind=FORMAL`，job、runtime=28800、supervision与正式policy一致，并绑定明确批准直到唯一固定训练的用户原话、CPU与已完成GPU结果。[E02]

**实际影响：**单独阅读该文件的人会在正式/原生范围之间遇到矛盾。这不是把一份旧报告的历史“待执行”当作现状，而是当前formal_gate同一对象内的描述字段与结构化字段不一致。现有授权原话、合同、结构化校验及原生处置足以确认正式范围，故不构成未授权训练、不使数值无效。

**最小处置：**在本次主执行者处置或收尾附注中说明scope保留了原生模板措辞，正式有效范围由已绑定授权及kind/job/runtime/supervision/gpu确定。**保留原门字节和所有哈希不变，不重训、不重开标签、不重置预算，不要求新认证系统。**本项只需文档解释，结果判读可以按现有证据成立。

### 合理后续建议：均不要求新增科研运行

RS-01保留固定门false和旧A局部收益；RS-02同时展示A末获得、B末终点和遗忘差，阻止孤立机制归因；RS-03概率三角色、固定阈值及宏/汇总代价同列；RS-04保留九项训练/valid/差距分量而非只给标签；RS-05保留保存值核算、原生资格、单seed/重复开发和论文边界。**这些要求原结果报告基本已经做到，最小处置是保留而不是返工。**完整结构化记录见findings.json。

### OUT_OF_SCOPE_OVERDESIGN：不采纳

不为获得显著性追加seed、顺序、臂、训练、温度/系数/学习率搜索或epoch选择；不补传正式文本/标签、模型/Memory、private/封存材料；不重训S/G、重审F求解器或旧项目全部历史；不扩展通用验证/认证平台。RD-01不应成为这些扩展的借口。

## 十一、实际阅读、隔离程序及审查者自身错误

完整阅读本次报告、H合同与当前H核心/运行器、诊断相关代码、Linux分析器、当前授权与政策、实施主审、原生处置和相关旧主审；旧实施外审全文只定向读相关段落，未全量重审旧实现。全部219文件做身份核对，33来源全做字节验证，但未声称33个源文件全部逐行科学复审。逐文件模式/行段与SHA列在`reading_scope.json`，人类摘要在`reading_scope.zh.md`。[E01—E10]

`comparison.png`实际目视核对了MAP两对比和B阶段fit/cache-valid曲线；其底层数值另行核算。PDF副本只核对哈希，没有声称渲染/逐页阅读PDF。

隔离主程序Python {audit['environment']['python']} / NumPy {audit['environment']['numpy']}，未导入项目模块。主检查主体耗时{audit['elapsed_seconds']:.6f}秒；计时从依赖导入后开始，不是完整网页墙钟，更不是项目py310资格。源码、注册抽样输入、219输入成员身份、重算表、差异、stdout/stderr与执行记录均保留。

**真实审查者错误也保留：**补充summary核对首版把实际下划线文件名猜写成`B_G_half.raw.metrics.json`，发生FileNotFoundError，退出1。它是我的文件定位错误，不是包缺失或项目科研缺陷。v2改为从collected.json读取准确`matrix.path`，换输出目录保留失败现场；随后成功。失败主体耗时{J(O/'checks/run02/result.json')['elapsed_seconds']:.6f}秒，成功主体{supp['elapsed_seconds']:.6f}秒。两个版本、patch、原始双流日志、退出码和result均在包中；没有覆盖或把失败算到项目预算。最初静态阅读的终端显示未逐条另存为测量日志，这一限制也在执行账中说明。

## 十二、最终裁决

**接受原报告的受限科学结论：本次固定H完整轨迹有效完成，预定双域MAP目标未通过；旧A最终排序、部分校准、相对G的旧A变化以及个别辅助指标的局部收益应保留，同时保留新B未达标、低FPR召回与raw概率代价和泛化差距。**

不认可整体超过S/G、孤立抗遗忘机制或论文主贡献充分的表述。仅提出RD-01说明字段的非阻断勘误，不要求新的训练/统计/权限门。主执行者应基于原始证据和本审查独立处置。

---

## 证据定位索引

正文[E01]等不是虚构的网页文件引用；它们对应原输入包的路径/JSON字段/源码行，以及本次真实重算输出。精确机器可读索引见`evidence/evidence_index.json`。

'''
for key,record in evidence.items():
 report+=f"**[{key}] {record['topic']}**："+'；'.join('`'+x+'`' for x in record['sources'])+'。本审查输出：'+'；'.join('`'+x+'`' for x in record['review_outputs'])+'。\n\n'
text('report.zh.md',report);text('report.zh.txt',report)
print(json.dumps({'report_characters':len(report),'report_bytes':len(report.encode()),'input_members_scope':len(sc),'scientific_blockers':0,'nonblocking_reproducibility_defects':1,'reviewer_failed_attempts_preserved':1},ensure_ascii=False))
