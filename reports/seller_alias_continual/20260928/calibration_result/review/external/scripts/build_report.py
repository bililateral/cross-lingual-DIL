from pathlib import Path
import json,hashlib,shutil,datetime
B=Path('/mnt/data/calibration_result_audit_20260928');P=B/'project';E=B/'evidence';D=json.loads((E/'outputs/independent_v1/independent_results.json').read_text());S=json.loads((E/'outputs/source_verification.json').read_text());X=json.loads((E/'outputs/report_edge_checks/report_edge_results.json').read_text())
base=P/'reports/seller_alias_continual/20260928'
# Preserve this stage's submitted failures and fixes as original bytes.
hist=[
 'calibration_execution/20260928_140128/started.txt','calibration_execution/20260928_140128/finished.txt','calibration_execution/20260928_140128/stdout.log','calibration_execution/20260928_140128/stderr.log','calibration_execution/20260928_140128/exit_status.txt','calibration_execution/20260928_140128/resource_usage.log',
 'calibration_execution/launch_correction/decision.json','calibration_execution/launch_correction/launch.json',
 'calibration_result/audit_initial/step28_alias_calibration_result.py','calibration_result/stderr.log','calibration_result/stdout.log','calibration_result/exit_status.txt','calibration_result/execution.json','calibration_result/command.json','calibration_result/resource_usage.log',
 'calibration_result/audit_execution/stderr.log','calibration_result/audit_execution/stdout.log','calibration_result/audit_execution/exit_status.txt','calibration_result/audit_execution/execution.json','calibration_result/audit_execution/command.json','calibration_result/audit_execution/resource_usage.log',
 'calibration_execution/20260928_140752/started.txt','calibration_execution/20260928_140752/finished.txt','calibration_execution/20260928_140752/stdout.log','calibration_execution/20260928_140752/stderr.log','calibration_execution/20260928_140752/exit_status.txt','calibration_execution/20260928_140752/resource_usage.log']
for n in hist:
 out=E/'history/submitted'/n;out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(base/n,out)
for n in ['step28_alias_calibration_result.py','step28_alias_pooling_result.py']:
 out=E/'history/submitted_current'/n;out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(P/'scripts'/n,out)
# Save the actual supplied replay outputs separate from the user's original analysis.
for p in (P/'reports/calibration_external_saved_audit').iterdir():
 dest=E/'outputs/supplied_replay'/p.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
errors={
 'this_review':[{'stage':'initial archive metadata probe','error':'KeyError: no root source_inventory.json','cause':'Reviewer assumed a root-level inventory before listing the full archive','remedy':'Use actual reports/seller_alias_continual/20260928/calibration_result/review/source_inventory.json; no archive/source edits','first_occurrence':'Initial container tool before run_logged helper existed','preservation':'Exact Python body retained; same read-only probe re-executed with raw stdout/stderr and exit1 at logs/initial_inventory_probe_capture_replay. This replay is not mislabelled as the first process log.'}],
 'independent_numeric_reference_revisions':[], 'independent_numeric_first_run_exit':0,
 'submitted_history':[{'stage':'formal launch','status':'resolved','classification':'REPRODUCIBILITY_DEFECT (historical invocation error, corrected)','exit':1,'cause':'relative CLI authorization at record() before parse_once','correction':'absolute paths and fresh directory; same26 scientific sources, same scope; zero label parses/fits on failed launch supported by traceback and receipt'}, {'stage':'saved-only result analyst','status':'resolved','classification':'REPRODUCIBILITY_DEFECT (historical read-schema error, corrected)','exit':1,'cause':'extra ["metrics"] level under per_seed_comparisons','correction':'one field access removed; preserved original script and failures; current script exit0; no fit/label reload'}],
 'previous_implementation_reference_failures':'Two older handmade probability-reference failures are supplied in the user archive; they were not rerun or relabelled as this result review. Original complete332-member archive is not supplied in full here.'}
(E/'history/error_and_revision_history.json').write_text(json.dumps(errors,ensure_ascii=False,indent=2)+'\n')
refs=[{'id':'EXT1','title':'SciPy 1.15.3 L-BFGS-B','url':'https://docs.scipy.org/doc/scipy-1.15.3/reference/optimize.minimize-lbfgsb.html','checked':'ftol relative reduction and gtol projected gradient stopping; no paper or model reproduction'}, {'id':'EXT2','title':'scikit-learn Probability calibration','url':'https://scikit-learn.org/stable/modules/calibration.html','checked':'Proper losses measure more than calibration alone; not an empirical guarantee.'}]
(E/'outputs/external_sources.json').write_text(json.dumps(refs,ensure_ascii=False,indent=2)+'\n')
readfiles={
 'docs/SELLER_ALIAS_CALIBRATION_RESULT.zh.md':'Full report132lines; five displayed tables independently checked',
 'docs/SELLER_ALIAS_CALIBRATION.zh.md':'Full frozen contract',
 'reports/documentation/20260928/calibration_plan/decision.json':'Actual approved scope/user answer',
 'schema/step28_alias_calibration_policy.json':'Fixed constants and scope',
 'scripts/step28_alias_calibration.py':'Complete method',
 'scripts/step28_alias_calibration_run.py':'Complete execute/fit/restore/collect/finalize path',
 'scripts/step28_alias_calibration_result.py':'Complete corrected saved-result checker',
 'scripts/step28_alias_pooling_result.py':'Read-only imports/classes/numeric helpers lines1–115; old unrelated run not invoked',
 'scripts/step28_alias_ranking.py':'Shared comparisons and13conditions185–222',
 'scripts/step28_chinese_base.py':'Fixed/automatic classification680–726; per-seed comparison775–785 and related code',
 'scripts/step28_continual_population_evaluate.py':'Curve/classification/query ordering21–100 and group assembly225–228',
 'scripts/run_step28_alias_calibration_linux_20260928.sh':'Complete launcher',
 'reports/seller_alias_continual/20260928/calibration_implementation/review/external/calibration_review_report.zh.md':'Complete274line prior F1–F10 report, no reenactment of prior tests',
 'reports/seller_alias_continual/20260928/calibration_implementation/review/disposition.json':'Full disposition, source list verified'}
(E/'outputs/source_read_register.json').write_text(json.dumps([{'path':n,'sha256':hashlib.sha256((P/n).read_bytes()).hexdigest(),'scope':v} for n,v in readfiles.items()],ensure_ascii=False,indent=2))

cols=list(D['means']['d_raw']);labels=['AP','梯形PR-AUC','ROC-AUC','Recall@FPR≤1%','Brier','log_loss','precision（群宏）','recall','F1（群宏）','specificity','balanced accuracy','MCC','MAP','MRR','Recall@1','Recall@3','Recall@5','Recall@10','NDCG@1','NDCG@3','NDCG@5','NDCG@10'];vs=('d_raw','d_calibrated','hard_raw','hard_calibrated')
means_table='| 指标 | A原始 | A校准 | C原始 | C校准 |\n|---|---:|---:|---:|---:|\n'+'\n'.join('| '+lab+' | '+' | '.join(f'{D["means"][v][k]:.9f}' for v in vs)+' |' for lab,k in zip(labels,cols))
primary=D['comparisons']['hard_calibrated_minus_d_calibrated'];raw=D['comparisons']['hard_calibrated_minus_d_raw']
guards=[('MAP平均增益≥0.01',primary['map']['mean']),('MAP条件95%下界>0',primary['map']['conditional_95pct_interval'][0]),('三种子MAP各自提高',' / '.join(f'{v:+.9f}' for v in primary['map']['per_seed'])),('平均R5严格提高',primary['recall_at_5']['mean']),('固定s0 R5严格提高',primary['recall_at_5']['per_seed'][0])]
for k,lab in [('average_precision','AP'),('roc_auc','ROC-AUC'),('brier','Brier'),('log_loss','log_loss')]:
 guards.extend([(f'主要比较平均{lab}'+('不增加' if k in ('brier','log_loss') else '不下降'),primary[k]['mean']),(f'主要比较s0 {lab}'+('不增加' if k in ('brier','log_loss') else '不下降'),primary[k]['per_seed'][0])])
for k in ('brier','log_loss'):guards.extend([(f'对原始A平均{k}不增加',raw[k]['mean']),(f'对原始A固定s0 {k}不增加',raw[k]['per_seed'][0])])
guard_table='| 判据 | 观察差或边界 | 判定 |\n|---|---:|---|\n'+'\n'.join('| '+k+' | '+(v if isinstance(v,str) else f'{v:+.9f}')+' | 通过 |' for k,v in guards)
prob_table='| 比较 | ΔBrier［条件95%］ | Δlog_loss［条件95%］ |\n|---|---|---|\n'
for name,lab in [('hard_calibrated_minus_d_calibrated','C校准−A校准（主）'),('hard_calibrated_minus_d_raw','C校准−A原始'),('hard_calibrated_minus_hard_raw','C校准−C原始'),('d_calibrated_minus_d_raw','A校准−A原始')]:
 v=D['comparisons'][name];vals=[]
 for k in ('brier','log_loss'):
  q=v[k];vals.append(f'{q["mean"]:+.9f} [{q["conditional_95pct_interval"][0]:+.9f}, {q["conditional_95pct_interval"][1]:+.9f}]')
 prob_table+='| '+lab+' | '+' | '.join(vals)+' |\n'
seed_table='| 种子 | ΔMAP | ΔR5 | ΔBrier | Δlog_loss |\n|---|---:|---:|---:|---:|\n'+'\n'.join('| '+s+' | '+' | '.join(f'{primary[k]["per_seed"][i]:+.9f}' for k in ('map','recall_at_5','brier','log_loss'))+' |' for i,s in enumerate(('s0','s1','s2')))
fit_table='| 模型 | a | b | 迭代/目标调用 | 投影梯度最大值 | 停止 | NLL：初始→最终 |\n|---|---:|---:|---:|---:|---|---|\n'
for rid,f in D['fits'].items():fit_table+=f'| {rid.replace("_d"," A").replace("_hard"," C")} | {f["a"]:.10f} | {f["b"]:.10f} | {f["optimizer_iterations"]}/{f["objective_calls"]} | {f["projected_gradient_max"]:.6g} | {f["stop"]} | {f["initial_nll"]:.9f}→{f["final_nll"]:.9f} |\n'
fixed_table='| 固定s0系统 | TP | FP | 合并precision | recall | 合并F1 | 群宏precision | specificity |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
for v,lab in zip(vs,('A原始','A校准','C原始','C校准')):
 r=D['fixed_classification']['s0_'+v];p=r['pooled'];fixed_table+=f'| {lab} | {p["tp"]} | {p["fp"]} | {p["precision"]:.9f} | {p["recall"]:.9f} | {p["f1"]:.9f} | {r["macro"]["precision"]:.9f} | {p["specificity"]:.9f} |\n'
report=f'''# 冻结C排序后的概率校准：正式结果独立科研外审

日期：2026-09-28。审查对象：本次 `calibration_result_review.zip`；有效作业 `calibration_execution/20260928_140752/job`。此报告不沿用实现审查作为效果证明，也未启动正式拟合、标签解析、模型加载、test或持续学习。

## 0. 三个独立结论

**结果有效性外审通过；本次冻结17项开发验收全部通过；具备准备另行授权的固定s0独立合成test合同及入口的依据。** 必须先修复的新增科研／复现缺陷为0。

这里“通过”限于上传证据可核验的实际计算与保存结果。它支持：在当前多轮开发过的中文合成valid上，C相对A的旧排序收益保持，且校准C满足相对校准A及原始A的AP／AUC和概率损失观察值保护。它不表示C自身获得新排名、不表示总体或全部阈值非退化、不表示独立test通过，也不授权将联合三域模型用作持续学习首域初态。

原九模型未经校准的9过4败保持不变；六个校准映射与既有18份Linux权重应按既定保留决定保存。本次没有远端现存性复查，不能把包内无权重说成Linux丢失。

| 类别 | 本轮确认 | 最小处置 |
|---|---|---|
| SCIENTIFIC_BLOCKER | 未发现使结果或上述限定结论失效的新增项 | 不改17项，不外推总体或真实市场 |
| REPRODUCIBILITY_DEFECT | 两个真实历史执行／分析错误已经纠正；未发现剩余结果缺陷 | 保存原错误、调用修正、单行分析差异与有效结果；不要求重读标签 |
| OUT_OF_SCOPE_OVERDESIGN | 不据此创造整改 | 不追加重训、校准族搜索、种子、环境安装、无关加固或自动test |

下文类别是审查维度；“未确认缺陷”的检查不构成阻断项。本地主执行者仍负责逐项回读处置。

## 1. 本次附件、来源与证据身份

本次实测ZIP **4,717,975字节**，SHA-256：

`819361383e6692c16aafce16439f7a3b21e9fb439ff23795ecd063c5cb9c95c2`

共291成员，即290份来源和位于 `reports/seller_alias_continual/20260928/calibration_result/review/source_inventory.json` 的清单；290份来源合计6,737,866字节。逐成员大小、SHA和ZIP完整性一致。该SHA是本次实测，不冒称比对了用户本条未提供的预期SHA。

原实现ZIP实际为1,286,211字节，SHA仍为`77ab3765d0928a231fadae7cbb9d92049e0343c4a7518ea5478a0b2909bcf51d`；其103成员及102来源清单一致。当前26份冻结来源与内嵌原ZIP、两份授权、preparation/prevalid/collected/evaluation/completion记录一致。未修改这些文件或历史状态文字。

82份正式执行／回传记录合计2,397,392字节，逐一与return_inventory及return_sync对应。原完整332成员网页实现证据没有被重新全量提供；只读了本包提供的原报告、F1—F10处置和节选，不声称核对了未提供的其余成员。

证据：`outputs/source_verification.json`、`outputs/source_read_register.json`。正式文件内容身份以实际包内来源为准，不以执行者摘要或PASS作为唯一证据。

## 2. 网页实际执行与独立性

网页现有Linux、Python3.13.5、NumPy2.3.5；CPU亲和性[0]，三个BLAS环境变量均1，实际进程Threads:1。SciPy1.17.0仅查询安装版本，本次未调用优化器。未安装或升级任何依赖。

| 网页实际执行 | 结果 |
|---|---|
| 提交入口先查看--help，再在新目录运行 | 退出0；24,120数值、1,512盲计数、576群顺序一致，最大差1.1102230246251565e−16 |
| 不导入项目模块的独立保存结果实现 | 首次数值运行退出0；253,390数值对照，最大差5.551115123125783e−16 |
| 完整映射与排列 | 217,728个分数用Python标量a*z+b精确恢复；576个全对群顺序／并列及16,128个查询候选列表一致 |
| 统计与计数 | 四比较×22指标及全部逐种子区间、1,512盲计数、自动诊断区间、17判据一致 |
| 文档与边界补充 | 5个表格162个显示数值吻合；阈值舍入实际边界核对通过 |
| 全来源检查 | 290来源及嵌套102来源对应；26冻结文件不变 |

独立数值最大差仍低于预定1e−12；没有参考数值修订、没有放宽指标容限。提交分析入口这次输出的两个CSV与原Linux有效分析逐字节一致。具体耗时以各 `logs/*/execution.json` 的scope为准，不与正式1.40秒混写。

独立实现使用标准库数学、Python稳定排序、手写标量仿射恢复、独立混淆计数代数、顺序生成域内抽样和频数权重、显式线性分位数；没有调用项目的指标、bootstrap、验收或拟合函数来生成期望值。按同一随机种子抽取同一群样本是比较的必要条件，不是把项目结果当答案。

仅计算保存矩阵及盲分数。没有正式标签或商品文本读取、反推个体真值、模型加载、编码器更新、重新拟合或服务器访问。66份实际NumPy文件被解码，含24份原始／变换分数及42份本轮或历史／复制指标矩阵；文件重复用途不会增加独立数据量。253,390项比较不是新增样本或实验数。

## 3. 正式执行、访问和失败史

有效进程记录为北京时间2026-09-28 14:08:50开始，14:08:51结束，退出0；GNU time1.40秒，内部1.269744616933167秒。单CPU核0，无GPU，RSS99,228KiB；采样产物峰值2,366,368字节。当前有效job58文件合计2,371,157字节，最终completion等文件会使最终字节数高于前一次采样；两者均远低于256MiB，不能把采样值写成精确连续峰值。RSS不是文件预算。

preparation记录Python3.10.19、NumPy2.2.6；项目既有CPU证据记录SciPy1.15.3。网页未访问服务器确认安装实物，不把外部NumPy2.3.5当成项目版本。

train_access记录UTC06:08:50.109527，development_access为06:08:50.636709，均parse_attempts=1、heldout/owners=0。代码先六拟合、六JSON／12分数恢复、原校准矩阵对应、prevalid来源绑定，再进入development解析。记录与路径支持新的train/development各一次；不能宣称独立监控了远端全部进程或证明服务器绝无其他访问。

### F1. 科学目标与比较对象
**审查维度：SCIENTIFIC_BLOCKER；已确认缺陷：否。**

受影响主张：是否回答“相对D候选排序提升，并保护基本识别”。依据：用户decision的question_scope；合同§1、3、5；`step28_alias_calibration.py:15–29、150–179`；`docs/SELLER_ALIAS_CALIBRATION_RESULT.zh.md:1–132`。

本次固定A/C、E6、三配对种子，主要C校准−A校准；另以原始A保护概率。它落实了用户明确选择的“保留C相对D收益”，不是进一步优化C排序。原A和C排序列逐群保持，故本轮不是第二份独立排名证据。没有新方法创新或持续学习效果主张。

最小处置：无需变更科学比较。正式表述限定为当前合成valid开发结果；历史原模型9过4败仍保留。

### F2. 来源、身份、访问与首次启动错误
**审查维度：REPRODUCIBILITY_DEFECT；确认历史错误已纠正，当前剩余缺陷：否。**

依据：`step28_alias_calibration_run.py:57–138、141–206、381–434`；有效preparation/prevalid/fitted及alignment；两份authorization；`calibration_execution/20260928_140128/stderr.log`；`launch_correction/decision.json`。

六模型origin与原run manifest的s0/s1/s2、d/hard、E6、原scores及state/model记录对应。36校准群与60valid群及域顺序匹配旧partition/collected；公开账号ID分别1008、1680且不交叉，每群28个排序账号，代码按字典序组合378对。真值文件未提供，无法重新核实每一行标签及其生成正确性；原始指标精确重放是有力一致性证据，不是替代真值。

初次14:06:18退出1，真实栈停在R:403的`data.record(authorization_path, ROOT)`，早于R:407的`parse_once`和R:409的拟合。相对授权路径与绝对ROOT不兼容。失败目录在本包没有access或fit记录，纠正回执报告二者为0；历史分数预检已经读取，不应说成该进程“没有读任何数据”。

改为绝对CLI授权路径、新输出目录后执行有效作业；两授权仅time/job/readiness_report/launch_correction变更，26科学来源、方法、门槛、预算和标签额度一致。没有按观察效果重跑。最小处置已完成：保存原失败与纠正，不修改冻结科学来源，也不要求新标签访问或重训。

### F3. 实际拟合目标、梯度与停止
**审查维度：SCIENTIFIC_BLOCKER／REPRODUCIBILITY_DEFECT；已确认新增缺陷：否。**

依据：方法`step28_alias_calibration.py:32–58、97–147`；运行器`:209–253`；六`fit.json`；`outputs/independent_v1/independent_results.json`的fits。

实际目标为 L(a,b)=mean[softplus(a*z+b)−y*(a*z+b)]，未加权、未截断、原二值比例，36×378=13,608对、720正对。梯度分别为mean[(sigmoid(a*z+b)−y)*z]及mean[sigmoid(a*z+b)−y]。静态公式与数学推导一致，原手工实现审查保持关闭；本次不在无标签包中重拟合。

六记录的初值(1,0)、边界a∈[0.001,100]／b∈[−100,100]、maxiter200/maxfun2000/maxls40/maxcor10/ftol1e−12/gtol1e−8一致。未命中边界，无替代求解器或最优映射选择。trajectory长度、末次参数、NLL和投影梯度与正式记录精确对应；校准矩阵保存Brier均值与fit内诊断一致。

{fit_table}

s1 C、s2 C由相对下降ftol停止，实际投影梯度略高于gtol1e−8，却满足冻结接收界1e−6。真实最后相对下降与该原因一致，不冒称六个均达gtol。SciPy的两类停止语义见外部核对EXT1。

独立检查额外用已披露总正对数与盲概率均值核对截距梯度分量，没有推断任何单对标签。其绝对值均不大于保存的投影梯度最大值。斜率梯度、真实完整NLL或实际最优点仍无法在没有正式标签时重算；保存NLL与校准矩阵log_loss的吻合只是两份保存统计的交叉验证。

拟合未截断NLL和旧概率截断log_loss是不同定义，本次两类校准统计数值在机器精度内一致，不证明任何极端输入也必然相等。优化NLL不保证Brier下降；本次Brier成功来自已保存valid证据，不是由优化目标推演。

最小处置：无代码修复；继续保留真实停止原因、拟合与评价定义以及无标签核验边界。

### F4. 完整映射恢复、保序与角色矩阵
**审查维度：SCIENTIFIC_BLOCKER／REPRODUCIBILITY_DEFECT；已确认缺陷：否。**

依据：方法`:61–94`；运行器`:225–280、284–343`；`fitted.json`、`prevalid.json`、全部分数与新旧矩阵。

使用六实际JSON参数，对12角色完整原float32分数先升float64，再逐元素Python标量a*z+b，217,728值与正式float64数组精确一致。独立检查576群的378对完整顺序及并列块，另核对16,128个查询27候选的公开ID并列规则，均保持。这里不是仅查看a>0或抽样几条。

calibration与development两角色的raw/calibrated14个曲线／检索列逐群精确相同；全部原始22列与对应旧矩阵精确相同，旧valid复制矩阵也一致。排名使用logit，不使用饱和或打印概率。

六映射保存恢复在valid解析之前，由代码顺序、prevalid对fitted的实物SHA及保存集合共同支持；不能把JSON时间戳中不存在的逐映射完成时间虚构出来。最小处置：无需重跑BGE或重新载入模型，本阶段恢复对象已经是标量及分数。

### F5. 完整采集、统计及主执行者只读分析修正
**审查维度：REPRODUCIBILITY_DEFECT；确认历史分析错误已修正，当前剩余缺陷：否。**

依据：运行器`:284–366`；分析器`step28_alias_calibration_result.py:37–193`；其纯工具依赖`step28_alias_pooling_result.py:28–115`；旧初版、失败日志及单行diff。

本次12份60×22 valid矩阵和固定0计数先进入完整collected，再核对原矩阵、排名、比较和判定。collect当然要先计算逐群指标才能保存；“先保存”不是禁止这些必要指标运算。正式作业没有发生统计失败，本次也没有通过伪造故障来替代正常结果；恢复性质的旧手工审查不重开。

主执行者初次分析误取`per_seed_comparisons[name][seed]["metrics"][metric]`，实际结构是直接`[seed][metric]`；保留初版，修正只有这一行。它不在26份正式科学来源内，也没有改变正式输出。原初次退出1、有效退出0均已核对；网页先查看帮助，再在新目录运行有效入口退出0，未覆盖原analysis。

该入口确实独立于正式评价过程而仅读保存结果，但复用旧只读工具，不能把复跑它称为完全独立原理核验。本审查另写不导入项目模块的实现，且补齐其未独立复算的旧自动诊断区间及实际查询顺序检查。最小处置已完成，无须再拟合或解析标签。

### F6. 四比较、逐种子及条件bootstrap
**审查维度：SCIENTIFIC_BLOCKER；已确认缺陷：否。**

依据：`step28_alias_ranking.py:185–205`；`step28_chinese_base.py:680–726、775–785`；运行器`:344–365`；独立四比较输出。

同群三个固定种子的差先平均，三域各20群有放回重采样5000次、种子20260927，域等权，线性分位数。独立程序顺序构造域内抽样的频数矩阵，核对全部22指标、四比较、逐种子区间和诊断区间；另以显式展开群样本和math.fsum对部分重复结果进行交叉检查。没有把账号对、查询或3×60种子群当成独立样本。

主要MAP为+0.016499599270，条件95%[0.010473378356,0.022457469323]；R5为+0.016369047619，条件95%[0.004759424603,0.027480158730]。校准相对自身的14个排序/曲线差为0。重新输出旧排名区间不是新独立重复证据。

区间条件于固定模型、实际映射与当前合成数据，不包含映射重拟合、模型重训、数据重生成及反复开发valid的选择不确定性；多指标与辅助比较无额外多重校正。最小处置：保持限定，不引入其他bootstrap次数、cosine版本或事后采样单位。

### F7. 17项开发验收与两条A参照
**审查维度：SCIENTIFIC_BLOCKER；已确认缺陷：否。**

依据：方法`:150–179`、排序方法`:208–222`、合同§5、实际evaluation及独立输出。

{guard_table}

全部17项独立复算为真，failed为空，与正式完成记录一致。不是只复核一个PASS状态。

{prob_table}

对照A自身的Brier和log_loss也降低，三个种子观察值都获益，因此并非把校准A做坏而制造通过。C校准还必须同时不劣于原始A，两条参照确实生效。通过是合同规定的观察值保护，不是总体非劣效或每种子每域每查询的保证。

最小处置：可以记为本轮17项开发验收通过；不回写旧未经校准结果，不选择B、E3、s1或raw/calibrated赢家。

### F8. 固定0.5分类和旧低误报诊断
**审查维度：SCIENTIFIC_BLOCKER；已确认缺陷：否。**

依据：评价器`:58–100`；运行器`:300–311、356–364`；正式counts；独立`fixed_classification`及`report_edge_results.json`。

{fixed_table}

s0 C校准固定p≥0.5相当于原logit约−0.626602132，而非原零点。对原始C，TP从18增至152、FP从0增至59，召回和F1提高；对校准A，precision和specificity降低。概率与AP/AUC保护不能被扩大成“全部阈值指标非退化”。宏precision和合并precision均已核查，不能互相代替。

更细的观察：s2主比较固定0.5的recall差−0.000833333、群宏F1差−0.002836172；总体均值recall/F1增加不能掩盖该种子。这不属于17项中的新增失败，也不应临时加门槛。

原严格诊断确实保持旧原logit和原阈值的同一决策集合。s0 C为TP72/FP8，recall0.06，三域仍不满足旧绝对门。独立还发现实际s1 A的数值边界：若错误地把变换分数与变换后的旧阈值直接比较，会额外纳入93对、涉及46群；正式代码没有这样做，仍保留原集合。这是保留原空间比较必要性的实证，不是现有实现缺陷；没有标签，不能把这93对称为已验证误报。

最小处置：维持原诊断实现和固定0.5披露；不追加阈值搜索或用旧门否定本轮候选验收。

### F9. 报告准确性与局部不确定性
**审查维度：SCIENTIFIC_BLOCKER；已确认缺陷：否。**

结果报告5表162个显示数值、参数与停止原因经过独立舍入核对，无发现错填、删掉失败指标或混淆宏／合并率。

{seed_table}

三域平均概率差相对校准A均降低。需保留的个体不确定性不止R5：s0和s2 R5区间跨零；s2的主比较log_loss差虽为−0.001373428，条件95%为[−0.002754423,+0.000080546]，同样跨零。当前全文和完整逐seed区间没有宣称全部区间都排除零，因此这不是新阻断；本审查在此补充，以免摘要被过度解读。

Brier和log_loss不是纯校准误差，降低不等于证明退化根因唯一或概率在现实市场已准确。标准资料EXT2亦明确区分概率proper scores与单纯可靠性。映射后新增的结果是当前valid上的概率损失改善，不是新排序机制或论文创新。

最小处置：不改正式数字或合同，只保持上述限定。无需为本次审查新增ECE、真实市场、无正候选或大候选库实验。

### F10. test准备、持续学习及未验证行为
**审查维度：OUT_OF_SCOPE_OVERDESIGN／结论边界；已确认新增缺陷：否。**

依据：合同§5–6、用户decision、报告§1/8、完成记录的下一阶段标记。本轮结果有效性与17项开发验收均已满足，可以准备固定s0独立合成test的合同和入口，之后另行授权；本外审不替用户制定或执行test，也没有重新分配最终留出用途。

尚未验证：独立test效果、test实现和划分用途；从原始文本经Linux权重到映射概率的新的端到端推理；正式个体真值、完整梯度和NLL的无标签独立重算；当前远端18权重存在性；真实市场、全库检索或无匹配拒识；持续学习方法及其首域初始化。

当前模型和映射已利用三域，不可因本次通过就当作只学过持续学习首域的初态。多轮开发valid的偏差不由条件区间消除。36校准群曾用于旧阈值，不是全新数据；合成20/378同控比例、27候选且有正例的条件限制概率与检索解释。最小处置：保持上述未验证范围，不额外重训、补seed或提前进行持续学习设计。

## 4. 完整22指标：三种子／60群宏平均

{means_table}

AP不是梯形PR-AUC。Recall@1/5是相关账号找回比例，不是首位命中率或至少命中一个的概率。AP/AUC等是各群内计算后宏平均，不是跨群合并曲线。固定0.5分类的宏值在表中，合并计数另列。全部精度、四比较及逐seed区间在独立JSON和CSV内。

## 5. 本次错误保存及执行边界

本次独立数值实现首次运行通过，没有为通过修改参考数值、正式阈值或容限。最初审查者探查压缩包时误以为清单在根目录，出现KeyError；列举实际成员后使用嵌套的结果审查清单。该错误不涉及科学计算或包损坏。原Python探查体已保存，并在有日志捕获器后原样重执行，stdout/stderr与退出1见`logs/initial_inventory_probe_capture_replay`；不冒称这个重捕获进程是首次进程日志。

项目两次失败保留在`history/submitted/`：正式启动的相对路径错误，以及只读分析的结构访问错误；原代码、单行diff、原stderr/stdout/退出码、有效日志均保留。原实现审查的两次手工数值参考失败属前阶段，本包有节选，本次未重跑或冒称新失败。

网页没有运行正式`execute`、没有调用`parse_once`、没有访问新CSV、文本或模型。对zip字节及保存数组的实际读取已经披露，不能说成“没有读取任何正式分数”；本次确实读取了获准保存的logit和矩阵。

## 6. 最终归档建议

> 本轮六个概率映射、一次train/development及valid评价完成；附件身份、冻结来源、映射恢复、保序、统计与17项判据经独立保存证据审查通过。C保留相对D/A的既有排序收益，同时在当前中文合成valid上满足相对校准A和原始A的概率保护，17/17开发验收通过。固定0.5仍存在precision、specificity和个别种子分类指标取舍，不能称全部基本识别指标或总体非退化。历史原九模型9过4败保持不变；保存六映射和既有18权重。可准备固定s0独立合成test合同并另获授权，尚无test结论或持续学习资格。

外审为建议，本地主执行者负责逐项核对；本次必须先修复项为0，不增加无关审批／签名／系统加固仪式。

## 7. 下载证据与复算方式

独立主入口：`scripts/independent_saved_audit.py`；完整原命令、环境、stdout/stderr/退出码见`logs/independent_v1/`。它需要当前项目根及本次保存结果，不需要任何标签或模型。`outputs/independent_v1/`含独立结果、22指标、四比较、固定0.5合并表、群抽样频数及四组主平均bootstrap输出；`outputs/report_edge_checks/`含报告162数值及阈值边界核查；`outputs/source_verification.json`包含全部来源哈希对应。

在已有Python/NumPy环境，以一个可用CPU运行并使用新的输出目录：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
  taskset -c <一个可用CPU> python -B scripts/independent_saved_audit.py \\
  --root <本次结果包解压后的项目根> --out <新的审查输出目录>
```

不得把该命令替换成正式拟合入口。不要为了版本一致而安装或变更项目环境；实际版本和偏差应记录。证据包SHA清单覆盖包内所有实际文件（清单自身除外），ZIP实物大小／SHA另给回执。

外部技术核查只用于解释：EXT1为SciPy1.15.3官方L-BFGS-B停止准则；EXT2为scikit-learn官方Probability calibration中proper scores与校准本身的区别。完整来源地址及用途在`outputs/external_sources.json`。未由这些资料推断本研究效能。
'''
(E/'review_report.zh.md').write_text(report,encoding='utf-8')
Path('/mnt/data/calibration_formal_result_review.zh.md').write_text(report,encoding='utf-8')
shutil.copyfile(E/'outputs/independent_v1/independent_results.json','/mnt/data/calibration_formal_result_independent_results.json')
shutil.copyfile(E/'outputs/independent_v1/all_22_means.csv','/mnt/data/calibration_formal_result_22_metrics.csv')
readme='''# 本次校准正式结果审查证据

review_report.zh.md 为完整报告。本包是本次网页实际工作，不是上轮332成员实现证据的重新命名。

- scripts/independent_saved_audit.py：不导入项目模块，仅保存数值审查；无需模型、标签、SciPy拟合。
- logs/*/：每次实际命令、单CPU线程设置、版本、stdout、stderr和退出码。即使stderr为空也保存。
- outputs/independent_v1/：首次独立数值运行PASS；所有输出为保存结果的派生统计，不是新增真值或训练。
- outputs/supplied_replay/：提交入口在新目录的实际输出；与本审查独立实现分开。
- outputs/report_edge_checks/：显示表格及s1 A映射阈值舍入检查。93个新增预测正例不是93个已知误报。
- history/submitted/：项目原始首次失败、修正与有效日志的字节副本。逐文件SHA可与用户原包核对。
- history/error_and_revision_history.json：区分项目两次失败、审查者最初清单位置探查，以及前阶段已有参考修订。

初次清单探查在统一捕获器创建之前出错；其原Python体已保留并原样重捕获。该捕获日志明确是重现，不冒充最早进程日志。独立数值脚本本次未修订或放宽容差。

代码需要用户本次结果包的解压项目树，不需要未上传正式文本、CSV、owners或模型。包内不包含这些资产。只读JSON中保存的计数不允许被用于反推出个体真值。

SHA256SUMS.json列出除清单自身以外的全部成员；外层ZIP的摘要在独立receipt中。报告中的现存Linux权重边界不构成网页重新检查服务器的声明。
'''
(E/'README.zh.md').write_text(readme)
print('report_chars',len(report),'bytes',len(report.encode()),'saved history files',len(hist),'independent_number_comparisons',D['numeric_comparisons'])
