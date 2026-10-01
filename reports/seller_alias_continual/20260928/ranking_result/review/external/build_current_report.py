#!/usr/bin/env python3
"""Build this resubmission audit report from freshly recomputed evidence only."""
from pathlib import Path
import csv, json, shutil, hashlib

BASE=Path('/mnt/data/ranking_result_reaudit_20260928')
EV=BASE/'evidence'; ROOT=BASE/'submitted'; IA=EV/'independent_analysis'
REPORT=Path('/mnt/data/ranking_result_resubmission_review.zh.md')
PUBLIC_CSV=Path('/mnt/data/ranking_result_resubmission_metrics.csv')
result=json.loads((IA/'independent_results.json').read_text())
rows=list(csv.DictReader((IA/'comparisons_22.csv').open()))
primary=[r for r in rows if r['comparison']=='hard_minus_d']
assert len(primary)==22 and result['checks_passed']==9 and not result['acceptance_passed']
name={
'average_precision':'账号对AP','trapezoidal_pr_auc':'梯形PR-AUC','roc_auc':'ROC-AUC',
'recall_at_fpr_1pct':'Recall@FPR≤1%','brier':'Brier','log_loss':'log_loss',
'precision':'固定0 precision（群宏）','recall':'固定0 recall','f1':'固定0 F1（群宏）',
'specificity':'固定0 specificity','balanced_accuracy':'固定0 balanced accuracy','mcc':'固定0 MCC',
'map':'MAP','mrr':'MRR','recall_at_1':'Recall@1','recall_at_3':'Recall@3','recall_at_5':'Recall@5',
'recall_at_10':'Recall@10','ndcg_at_1':'NDCG@1','ndcg_at_3':'NDCG@3','ndcg_at_5':'NDCG@5','ndcg_at_10':'NDCG@10'}

def f(v): return f'{float(v):.6f}'
def s(v): return f'{float(v):+.6f}'
def metric_table():
    z=['| 指标 | A | C | C−A | 条件95%区间 |','|---|---:|---:|---:|---|']
    for r in primary:
        z.append(f"| {name[r['metric']]} | {f(r['reference'])} | {f(r['candidate'])} | {s(r['delta'])} | [{s(r['lower'])}, {s(r['upper'])}] |")
    return '\n'.join(z)

with PUBLIC_CSV.open('w',encoding='utf-8-sig',newline='') as out:
    w=csv.DictWriter(out,fieldnames=['comparison','metric','metric_zh','reference','candidate','delta','lower','upper'])
    w.writeheader()
    for r in primary:w.writerow({**r,'metric_zh':name[r['metric']]})
shutil.copy2(PUBLIC_CSV,EV/'primary_22_metrics.csv')

checks=result['acceptance']; comp=result['comparisons']['hard_minus_d']
checkdefs=[
('map_minimum_observed_gain','三配对平均ΔMAP≥0.01',s(comp['map']['mean'])),
('map_interval_above_zero','MAP条件95%区间下界>0',s(comp['map']['conditional_95pct_interval'][0])),
('map_improves_each_seed','三个配对各自ΔMAP>0',' / '.join(s(v) for v in comp['map']['per_seed'])),
('recall_at_5_mean_strictly_improves','三配对平均ΔRecall@5>0',s(comp['recall_at_5']['mean'])),
('recall_at_5_s0_strictly_improves','固定s0 ΔRecall@5>0',s(comp['recall_at_5']['per_seed'][0])),
('average_precision_mean_non_degradation','平均AP不下降',s(comp['average_precision']['mean'])),
('average_precision_s0_non_degradation','固定s0 AP不下降',s(comp['average_precision']['per_seed'][0])),
('roc_auc_mean_non_degradation','平均ROC-AUC不下降',s(comp['roc_auc']['mean'])),
('roc_auc_s0_non_degradation','固定s0 ROC-AUC不下降',s(comp['roc_auc']['per_seed'][0])),
('brier_mean_non_degradation','平均Brier不增加',s(comp['brier']['mean'])),
('brier_s0_non_degradation','固定s0 Brier不增加',s(comp['brier']['per_seed'][0])),
('log_loss_mean_non_degradation','平均log_loss不增加',s(comp['log_loss']['mean'])),
('log_loss_s0_non_degradation','固定s0 log_loss不增加',s(comp['log_loss']['per_seed'][0])),
]
checktable='| 冻结条件 | 观察差／边界 | 结论 |\n|---|---:|---|\n'+'\n'.join(f"| {label} | {value} | {'通过' if checks[key] else '**未通过**'} |" for key,label,value in checkdefs)
seedtable='| 配对 | ΔMAP | ΔR5 | ΔAP | ΔAUC | ΔBrier | Δlog_loss |\n|---|---:|---:|---:|---:|---:|---:|\n'
seedtable+='\n'.join('| '+seed+' | '+' | '.join(s(comp[k]['per_seed'][i]) for k in ['map','recall_at_5','average_precision','roc_auc','brier','log_loss'])+' |' for i,seed in enumerate(['s0（固定）','s1','s2']))
domaintable='| 域 | ΔMAP | ΔR5 | ΔAP | ΔAUC | ΔBrier | Δlog_loss |\n|---|---:|---:|---:|---:|---:|---:|\n'
domaintable+='\n'.join('| '+domain+' | '+' | '.join(s(comp[k]['by_domain'][domain]) for k in ['map','recall_at_5','average_precision','roc_auc','brier','log_loss'])+' |' for domain in ['A','B','C'])
aux='| 比较 | 指标 | 平均差 | 条件95%区间 |\n|---|---|---:|---|\n'
aux+='\n'.join(f"| {label} | {name[k]} | {s(c[k]['mean'])} | [{s(c[k]['conditional_95pct_interval'][0])}, {s(c[k]['conditional_95pct_interval'][1])}] |" for key,label in [('schedule_minus_d','B−A（解释）'),('hard_minus_schedule','C−B（解释）')] for c in [result['comparisons'][key]] for k in ['map','recall_at_5','average_precision','roc_auc','brier','log_loss'])

text='''# 九模型排序训练：同包重新提交的独立结果外审

审查日期：2026-09-28。对象：本条重新上传的 `ranking_result.zip`。本报告是依据当前附件重新核验及实际重运行的审查，不沿用上次未完整呈现的回复作为结果依据。

## 0. 总判定与处置

**结果有效性外审通过；冻结完整验收未通过（9/13通过、4/13失败）；按用户最新明确指令保留全部18份Linux推理权重。**

本轮C在当前已多轮开发的中文合成valid上改善了候选排序及群宏账号对AP／ROC-AUC；同时均值与固定s0的Brier、log_loss上升，固定0的召回与F1下降。不能把完整未通过说成排序没有改善，也不能因概率损失绝对差较小而追认完整通过。用户决定保留C为后续候选，不等于已经证明其收益足以抵消风险，更不自动授权校准、改阈值、重训、test或持续学习设计。

| 分类 | 本轮确证情况 | 最小处置 |
|---|---|---|
| SCIENTIFIC_BLOCKER | 未发现使本轮保存结果及所限定结论失效的新增科研缺陷 | 保留四项护栏失败及全部边界；不改成功线 |
| REPRODUCIBILITY_DEFECT | 未发现本轮保存证据、复算或身份对应中的新增缺陷 | 不要求补训、标签重读或恢复已按设计删除的临时Adam状态 |
| OUT_OF_SCOPE_OVERDESIGN | 新场景、补种子、自动校准／阈值搜索、无关加固或重复流程不属于本轮整改 | 本次不提出这些执行要求 |

必须先修复的代码／数值缺陷为0。下面F1—F10是逐项审查发现，不将“发现”全部冒称为“缺陷”。阴性护栏结果本身不是实现错误。

## 1. 本次附件、来源与阅读范围

实际大小 **7,579,246字节**，实际SHA-256：

```
6d01ada467bc705044d31008302fb6a96aa6fd9fbb7817db4166dd06254c9474
```

ZIP共有253成员：252来源＋`source_inventory.json`。逐成员大小／SHA全部一致，ZIP完整性无错。作业子集161文件、8,894,602字节与回传清单相符。18份实际冻结来源与启动记录对应，并与已挂载训练前附件的对应文件逐字节一致。审查末尾全部253个原始成员仍与本次上传ZIP一致，原`analysis`未覆盖。

完整阅读了本次request、冻结合同、结果报告、直接相关训练前完整回复／F1—F10报告及主审处置；追踪了方法、运行器、输入、训练、优化器、保存恢复、指标与纯结果分析依赖，以及授权、访问、完成、资源、权重和同步记录。对长篇历史交接文档只读直接相关段落，不声称重新审查了所有历史研究。此前加权汇总负结果不重开。

下文定位简称：

- **M**：`scripts/step28_alias_ranking.py`。
- **R**：`scripts/step28_alias_ranking_run.py`。
- **D**：`scripts/step28_chinese_base.py`。
- **P**：`scripts/step28_continual_expression_run.py`。
- **E**：`scripts/step28_continual_population_evaluate.py`。
- **K**：`scripts/step28_continual_population.py`。
- **AN**：`scripts/step28_alias_ranking_result.py`，纯函数依赖`step28_alias_pooling_result.py`。
- **J**：`reports/seller_alias_continual/20260927/ranking_execution/20260927_114646/job/`。
- **S**：`reports/seller_alias_continual/20260928/ranking_result/`。

数值及原始身份记录的可追溯入口见证据包的`submission_verification.json`、`independent_analysis/independent_results.json`、`additional_saved_checks.json`和逐项CSV。

## 2. 网页实际执行及独立性

本次用既有Linux、Python **3.13.5**、NumPy **2.3.5**，固定单CPU，独立进程记录`Threads: 1`；没有安装依赖、改变项目环境、下载／载入模型或新增训练。此环境不是项目Linux py310。

| 本次实际执行 | 核对范围 | 结果 |
|---|---|---|
| `verify_submission.py` | ZIP身份、全部来源、作业清单 | 退出0 |
| 提交的AN，输出到新`reviewer_analysis` | 35,496数值、5,184盲阈值计数、7,776损失组合 | 退出0 |
| `independent_result_audit.py` | 不导入项目模块的独立数值与证据对应实现 | 退出0；161,516数值比较 |
| `additional_saved_checks.py` | 冻结源、18保存点记录、全精度阈值、保留／资源链 | 退出0 |
| `check_report_and_integrity.py` | 人类报告数值、轨迹、Linux／网页输出及来源不变性 | 退出0 |

独立计算实现沿用了审查者先前写成的实现，本次已检查其代码并对当前新解压附件重新执行；并非把旧PASS文件复制成本轮结果。`script_provenance.json`记录其来源SHA和仅路径修改。新增本次辅助记录检查及打包源码一并保存。提交AN与独立实现分开执行，后者只导入标准库和NumPy，不调用项目的bootstrap／指标模块；采用逐次逐域抽样、独立计数代数等路径验证保存统计。

独立计算核对54份盲分数、54份矩阵（18valid＋18fit＋18calibration）、三种比较／各种子／各域汇总、全部13判据；主bootstrap的5000×22＝110,000数值逐项比较。指标及区间最大差 **4.440892098500626e−16**，固定容限仍为1e−12。5,184是盲分数对应的预测正例总数检查行数，不是5,184条新增标签验证。

将保存损失分量以float64相加，最大差2.980232238769531e−7。进一步按实际float32加法树重建，7,776条总损失全部精确一致；这是加法舍入，不改变指标容限、原损失或阈值。报告表格138个数字、6个轨迹字符串均核对，22指标显示行齐全。原Linux与本次AN输出的三份CSV及主bootstrap数组逐字节一致。

所有实际命令、stdout／stderr、GNU time、退出码和环境在证据包保存。另有一次交互式元数据展示命令把整数`files`误作列表调用`len()`而失败；修正只涉及审查者展示代码，随后`inspect_run_records.py`退出0，未改提交来源或任何科学数值。保留了该工具输出错误片段及说明，没有将其伪装成项目错误，也不声称未归档的交互命令具有完整原始日志。

本次正式文本／标签读取0，owners／test访问0，模型加载0，训练更新0，远端服务器访问0。比较次数不代表新增样本或科研实验。

## 3. F1—F10逐项审查

### F1. 研究目标、输入与有效授权

**受影响主张：** 这九个模型回答的是候选排序与账号对识别的受控改进问题，而不是已完成持续学习或真实市场身份验证。

**证据位置：** 合同§1—5；M:46–63；P:58–128；D:110–191；`ranking_execution/authorization.json`；J中的分区、access、manifest。

实际保留多记录、屏蔽标题／描述，逐商品编码后两通道均值／总体标准差和归一化，原对称评分头不变。账号／域／商品身份用于分区和索引检查，不作为模型特征。每群28账号、378无向对，144拟合群、36训练内校准群；60valid群、每查询同群27候选且1或2个合成相关账号。输入路径和分区记录相符。

冻结合同的准备期“尚未获准”不是当前状态。后续正式授权记录于2026-09-27 11:44:28+08，早于11:50:06启动，限定九模型／7776调用、train／valid各一次和test／owners禁止；其来源／政策／处置摘要对应。不能要求改写历史冻结合同才能使合法运行有效。

**判断与最小处置：** 未发现新增SCIENTIFIC_BLOCKER或REPRODUCIBILITY_DEFECT；保持任务及阶段边界即可。公开群ID的隔离与配对已复核，但没有owners或原文，不能说本次独立验证了控制者级隔离、所有屏蔽效果或合成真值正确性。

### F2. 目标、梯度、学习率和九模型配对

**受影响主张：** C是在相同表示与监督边界下改变训练日程和对前排已知负例的重视，而不是更换数据或新增编码器前向。

**证据位置：** M:66–182；D:165–191；R:147–277；九臂manifest、`updates.npy`和J/`train.log`。

原BCE只在378个无向对上计算一次，全候选rank仍保留。困难项排除自身及全部正例，从当前训练标签允许的负例中稳定选前5个；选择索引不求导，被选logit保持梯度。每个查询内部平均其全部正例×5负例，再28查询等权，不把200比较直接混成一组平均。目标是`BCE+rank+0.5×hard`，不等价于精确优化MAP或Recall@5。没有新模型参数，也没有新增编码器前向。

九模型均新建于同一预训练档案，三组配对的初态摘要、群顺序SHA及dropout流一致，独立重建全部日程通过；每群训练六次，每模型864调用、总计7776。运行和源码共同支持约定配对随机条件，不是逐个保存并验证了每次CUDA随机抽样值。

B/C预热87步、峰值1e−5、其后衰减，第864次编码器lr为0，头仍1e−3。A编码器lr恒2e−5。累计lr分别0.01728与0.00432，不能解释成参数位移四分之一；B/C均只有863个编码器正lr步骤。324条每24步日志及7776行更新数组一致。45个指定观察步（每臂1、2、87、88、433）共90个编码器／头观察记录非零有限梯度及参数变化；它们不是每一步、每参数、每个损失项的独立梯度证明。

**判断与最小处置：** 未发现目标、归约、更新预算或配对的新缺陷。正式观察记录不能由旧CPU微型测试冒充；反过来，本次也不需要重复未变训练前用例或要求额外种子。相同optimizer次数不等于相同所有计算开销，时间差不能唯一归因于困难项。

### F3. 18个保存恢复点、54分数及valid前检查

**受影响主张：** E3／E6使用预定真实保存模型，E3后延续原Adam和日程，valid没有先于完整前置核验读取。

**证据位置：** R:67–116、280–386；K:157–210；J/各臂points和`evaluation/preparse_verification.json`；S/`weight_inventory.json`。

18个点的run_id、epoch、完成步数432／864、来源与lr、下一步lr、完整模型／Adam摘要和实际恢复记录对应。完整状态恢复后fit／calibration／development三角色盲分数逐项相等，再保存推理权重；推理文件实际重载并核对模型状态摘要。准确区别：全角色前向重放发生在完整模型＋Adam恢复后；推理文件重载后是模型状态核对，不冒称又进行了另一轮前向。

每点三份分数：开发集60×378、fit144×378、calibration36×378，合计54；对应评价矩阵分别60×22、144×22、36×22，合计54。文件哈希、形状、角色和点身份全部重核一致。valid gate逐点核验分数／模型实际大小SHA及有限性，全部完成后才调用development标签解析；不要求错误地描述为先一次性检查全部54分数再检查全部18模型。

**判断与最小处置：** 未见模型错绑、缺点或前置次序新缺陷。网页没有权重，只能交叉核对运行时实物记录、源码约束及后续清单，不能宣称现在独立重载过18模型。完整Adam临时文件完成验证后按设计移除；保留的是18份推理权重，不是18份可直接恢复训练的完整Adam检查点。

### F4. 一次标签边界、18valid矩阵保全及R1／R2／R3

**受影响主张：** 一次合法监督采集得到的结果可供后处理复算，不因统计失败自动消费新标签。

**证据位置：** P:107–128；R:227–277、389–488；J/访问、collected和evaluation记录；D:680–750。

train一次父解析供九模型共享，校准群不进入梯度；development在完整gate后一次解析。18份完整valid矩阵及固定0计数、E6校准计数、群域顺序、来源先落盘至collected，再由finalize进行三比较、bootstrap和验收。本次读取的是保存矩阵／计数／分数，不读取或反推正式真值。

`collect()`仍有单份矩阵落盘后的轻量分类／均值计算，故准确主张是“完整采集后比较、抽样和验收”，不是“完整18矩阵之前不存在任何统计算术”。合并分类计数先求和再算比率，群宏另列；资源统计保留既定作用域。没有发现已关闭R1／R2／R3回归，也没有重开此前加权汇总负结果。

**判断与最小处置：** 无必须整改。标签次数是运行记录与实际执行路径支持的事实，不是网页对远端一切访问的独立监控。仅依据盲分数总数不能验证每个TP／FP的真值身份。

### F5. 提交分析入口和独立复算的正确性与覆盖

**受影响主张：** 原报告的保存汇总、区间、验收及解释不是只靠执行者PASS支撑。

**证据位置：** AN及直接纯函数依赖；本次`independent_result_audit.py`、`additional_saved_checks.py`、全部execution/log/JSON、各比较CSV。

独立实现核对三比较（C−A、B−A、C−B）、各seed／domain／point／epoch、全部22指标汇总、计数的macro与pooled计算及自动诊断，所有13判据一致。根据盲分数在固定0及已保存全精度阈值处的比较，核对5,184行预测正例总数；没有重拟合阈值，没有把四舍五入展示阈值当实际阈值。最大差处于1e−12数值容限之内，原Linux与网页CSV／主抽样数组字节一致。

按独立频数／循环路径复算保存矩阵，可以检查聚合及统计实现，但不能凭此验证truth-dependent的每群AP、ROC、MAP或概率损失。本次没有任何正式标签重建。loss按float32加法树精确一致是“组合公式”核验，不是对7776步原始样本重新计算BCE／rank／hard。

**判断与最小处置：** 未发现提交分析入口引入的影响结论错误或虚报真值重算。数字复算通过不是对所有原数据语义的证明；已明确限制后，无需增加标签重读或模型推理。

### F6. 主比较、完整22指标和13验收

**受影响主张：** 排序收益成立且概率保护失败，二者需并列而不互相抵消。

**证据位置：** 合同§5；M:185–222；R:416–459；J/evaluation；S/结果报告及本次全部CSV。

''' + checktable + '''

主MAP差为 **+0.016499599270**，条件95%区间 **[+0.010473378356，+0.022457469323]**；R5差为 **+0.016369047619**，区间 **[+0.004759424603，+0.027480158730]**。MAP相对约3.39%、R5约1.64个百分点。不能把不同指标的绝对量级相减来证明整体收益大于概率代价。

以下完整22项为三种子60群宏平均；Brier／log_loss低为好。固定0对应logit≥0，未改阈值。区间为保存模型条件下的配对差区间，不是各模型单独区间。

''' + metric_table() + '''

AP是非插值Average Precision，不等于梯形PR-AUC；参考官方定义[R1]。账号对AP／AUC等以群内378对计算后宏平均，不是把全部群账号对混成一个全局AP／AUC。Recall@1/5是相关账号找回比例，双正例查询找回一个贡献1/2，不是Hit@K或至少命中一次的概率。NDCG@1在当前二元相关性下对应首位相关命中，但不能替换Recall@1的名称。

**判断与最小处置：** 9过4败准确，不存在以排序成功追認完整通过的依据，也不应说候选排序没有改善。四个概率失败项必须保留，不因用户保留权重而回写规则；这不是实现缺陷。

### F7. 各种子、分域、辅助比较与E3→E6

**受影响主张：** 不能用s1、B或E3替代事前固定s0、C、E6；不能把所有局部改善说成总体稳定优势。

**证据位置：** 各臂训练／评价矩阵；AN比较与轨迹；本次`per_seed_comparisons_22.csv`、`by_domain_comparisons_22.csv`、`trajectories_22.csv`。

''' + seedtable + '\n\n' + domaintable + '''

三个seed的MAP／R5／AP／AUC观察差均正，但不等于每个seed每域每查询都提高。s0 R5条件区间约[−0.005365,+0.024107]，s2约[−0.007150,+0.024107]；合同要求观察值严格为正而不要求其单seed区间下界为正，因此不是漏判，但不可夸大。s1两项概率指标改善不能替代固定s0。三域种子平均概率损失都增加也不推出每群均增加。

''' + aux + '''

B日程本身有部分排序收益；C−B在固定日程下进一步提高MAP／AP，R5和AUC增量区间跨零，Brier／log_loss增量区间为正。B−A不能唯一分离降低峰值与日程形状的贡献，辅助比较没有多重校正，不能升格为多个确认性成功。B在s0的R5差约−0.000893，且两项概率损失增加，亦不能事后宣布B完整通过。

C的valid MAP E3→E6：s0 0.504528→0.505727；s1 0.505167→0.499026；s2 0.500590→0.504105。s1拟合改善而valid排序下降支持局部泛化分化的观察，其余两个seed上升，不支持“全部过拟合”或“全部持续改善”。三个C的valid Brier／log_loss从E3到E6都下降，但E6相对A存在上述退化；一个是同模型轨迹，一个是跨配置比较，不能混同。不同目标总损失的绝对值也不能直接跨A／C作优劣比较。

**判断与最小处置：** 原报告相关方向及限定准确。没有确定唯一失效机制，不换seed／arm／epoch；不要求追加调参或新种子。

### F8. 概率损失、固定0计数与“可保留候选”的含义

**受影响主张：** 小幅概率损失不自动等于纯校准偏差、不等于无损，也不意味着C不值得保留。

**证据位置：** E:28–100；D:282–305、680–750；保存计数及本次`fixed_zero_confusion.csv`；官方概率校准说明[R2]。

平均Brier差约+0.000321，条件区间[+0.000115,+0.000521]；平均log_loss差约+0.001879，区间[+0.000528,+0.003324]。固定s0两项区间也高于0。Brier和log_loss综合衡量概率预测质量，不能直接当作纯校准误差；[R2]明确说明它们同时受可靠性／区分及数据不确定性影响。

数学上，对所有账号对logit加同一常数可保留排名、差值排序项及查询softmax，但改变sigmoid概率和BCE；因此排序与概率指标分离并不矛盾。这只是说明两目标不等价，不能据此认定本轮退化唯一来自平移，也不保证任何后续校准必能修复。仅换分类阈值而不改变概率，不会改变Brier／log_loss。

固定s0的原始计数提供了避免误读macro precision的具体例子：

| 固定logit≥0 | TP | FP | FN | TN | 合并precision | 合并recall | 群宏precision | 无预测正例群 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 102 | 26 | 1098 | 21454 | 0.796875 | 0.085000 | 0.787222 | 3/60 |
| C | 18 | 0 | 1182 | 21480 | 1.000000 | 0.015000 | 0.250000 | 45/60 |

C只报18个正例，保存计数没有FP；45群完全不报正例，按既定规则precision记0，拉低群宏。不能把群宏下降解释成报出的正例更不准；也不能凭合并precision=1掩盖召回／F1下降或宣称分类全面改善。

训练内冻结校准阈值的极低误报诊断另列。s0 A／C valid合并召回0.038333／0.060000、FPR0.000279／0.000372；三个C召回0.0600／0.0450／0.0425，未达旧0.5诊断目标。该诊断不替代本轮检索判据，也不与曲线的Recall@FPR≤1%混用。全精度阈值已用保存盲分数核对，没有重新校准或新标签访问。

**判断与最小处置：** 原报告已区分这些事实，无强制更改。用户选择保留权重是候选管理决定，不是已经规定概率容忍界限、证明业务效用占优或通过非劣效。不得把该决定推演成自动概率校准／test授权。

### F9. 条件bootstrap及结论边界

**受影响主张：** 区间描述的是当前固定模型与保存群体的有限不确定性，而不是训练算法总体或真实市场的保证。

**证据位置：** 合同§5；M:185–205；本次独立draw／frequency对照；结果报告§8。

主比较先在同一群内计算三个配对差并取平均，再在三域各20群中整群有放回抽样，5000次，随机种子20260927，三域等权和线性分位数。没有把3×60 seed群、1680查询或378账号对当相互独立样本。独立实现逐重复产生300,000个群抽样位置（5000×60），核对全部110,000主统计数值。

区间条件于固定三个训练种子模型、当前生成数据及阈值，不覆盖重新训练、重新生成、重估阈值或反复开发valid的选择偏差。本次MAP区间下界高于0.01是这一条件计算中的事实，但不等于面向算法总体、未知市场或业务预算的至少0.01增益保证。辅助和逐指标区间未经多重校正。

本次只有中文合成、同群27候选且保证存在相关账号的受控证据，不支持全库、无正候选拒识、真实控制者识别、总体非劣效、持续学习效果或论文方法创新。没有正式原始文本／owners，不能验证合成规则与现实控制关系的对应。

**判断与最小处置：** 原文已披露核心边界，无确证越界。强制扩大候选库、加入无匹配查询、收集真实数据或多跑seed属于OUT_OF_SCOPE_OVERDESIGN，不作为本轮整改。

### F10. 耗时资源、权重证据与最新保留决定

**受影响主张：** 本轮按授权完成并在预算范围，18权重存在的最新可见证据是保留回执，不是旧删除意图。

**证据位置：** J/训练日志、GNU time、exit和completion／manifest；S/`weight_inventory.json`、`weight_retention.json`、`retention_verification.json`；本次`additional_saved_checks.json`及权重交叉CSV。

北京时间2026-09-27 11:50:06启动，2026-09-28 02:15:28完成；时间戳差14:25:22，GNU time14:25:21，exit0。计时作用域造成秒级差异，不是运行失败。正式记录单GPU、单CPU；最大RSS6,936,424KiB。CUDA allocated8,540,458,496字节（约7.95GiB），reserved9,472,835,584字节（约8.82GiB），是该进程分配器记录而非整卡总占用。

作业内采样最大27,435,480,672字节，保守共存上界＝18最终权重23,516,221,872＋最大临时完整Adam3,910,960,965＋最终小型结果8,894,602＝27,436,077,439字节，低于32GiB。采样不当成精确瞬时峰值，后续外审材料也不冒充原训练预算内产物。

18份推理权重的保存记录、valid前实物核验及新鲜盘点大小／SHA一致。用户看到结果后明确保留，`weight_retention.json`于2026-09-28 10:05:48+08记录该决定；10:08:32+08的后续回执记录18份存在、大小／mtime对应、删除0、仅Linux。保留决定之后的核验是元数据检查，不冒称又做一轮全文件SHA；本审查没有连接Linux，因此所有远端实物陈述以这些记录为边界。

权重清单中“拟清理”属于新指令之前的历史，不是当前执行许可。提交包不含权重符合上传边界，不能据其缺席断言远端未保留。161份非权重元数据不变也只能按回执范围解释，不凭空升级为网页远端内容SHA重读。

**判断与最小处置：** 未发现预算或权重身份链新缺陷。18推理权重继续保留，删除0；不能自动选s1／B／E3，不能把保留候选改写成通过验收。自动删除或借本次保留启动校准／训练／test／持续学习均超出当前授权。

## 4. 未验证范围及执行纪律

本次没有重新载入任何实际权重、执行正式GPU前后向、恢复完整Adam、核验远端实时文件状态或读正式真值。依赖真值的逐群指标仍是本轮一次正式评价的输出；网页独立验证的是其身份、聚合、保存计数及统计。不能将来源SHA一致与自写policy一致当作数据真实性或模型泛化的充分证明。

本次未跑训练前微型训练用例，因为冻结实现不变且本次禁止新增训练；旧微型测试只作已关闭范围的背景，不当作本轮GPU事实。读取相关代码及原始记录不意味着重审所有历史数据生成、市场代表性或模型部署行为。

归档建议：

> 本轮九模型训练与valid评价完成，提交记录及当前重新执行的独立复算支持结果有效。E6 C−A的候选排序及群宏账号对AP／AUC改善，但均值与固定s0的Brier／log_loss退化，冻结13项9通过、4未通过。按用户最新明确指令保留全部18份Linux推理权重，C保留为后续候选。保持原验收、固定s0、主C−A与E6，不自动校准、调阈值、重训或选臂／种子／轮次；test与持续学习设计继续暂缓。

外审为建议，主执行者仍需逐项核对后决定处置。未确认缺陷为0不是对所有未访问资产的保证；不需要为了得到新的缺陷标签而添加无关整改。

## 5. 本次证据清单与语义参考

证据包包含实际执行脚本、来源检查、提交入口的新输出、独立数值输出、stdout／stderr、GNU time、命令／版本／退出码、脚本来源说明、错误展示片段说明以及逐文件SHA清单。`independent_result_audit.py --root <本次解压根> --out <新的输出目录>`只用标准库／NumPy读取保存证据；不导入模型库或项目训练入口。打包后的SHA清单不递归包含自身。原附件来源与新外审证据分开，不覆盖原analysis。

以下外部资料只用于解释通用指标定义，不用于替代包内实验数值；查阅日2026-09-28：

[R1] scikit-learn，`average_precision_score`官方文档：非插值AP不同于梯形PR面积。https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html

[R2] scikit-learn，Probability calibration，§1.16：Brier／log_loss不只是校准误差，涉及概率预测的可靠性、区分及数据不确定性。https://scikit-learn.org/stable/modules/calibration.html
'''
REPORT.write_text(text,encoding='utf-8');shutil.copy2(REPORT,EV/'review_report.zh.md')

# Preserve this execution's submitted-entry outputs separately from original analysis.
new=ROOT/'reports/seller_alias_continual/20260928/ranking_result/reviewer_analysis'
dest=EV/'submitted_entry_outputs';dest.mkdir(exist_ok=True)
for path in new.iterdir():
    if path.is_file():shutil.copy2(path,dest/path.name)

failure={
 'category':'reviewer_metadata_display_only','scientific_check_failure':False,
 'description_zh':'早先交互式显示sync元数据时，files为整数却调用len()；修正isinstance(list)后用归档inspect_run_records.py重新执行退出0。原提交来源和科学计算未变。',
 'raw_tool_output_excerpt':'Traceback (most recent call last):\n  File "<stdin>", line 7, in <module>\nTypeError: object of type \'int\' has no len()',
 'complete_raw_log_available':False,
 'scope_note':'这是保留的工具输出错误片段，不冒称为完整原始stdout/stderr或重新执行该错误。',
 'resolved_execution':'inspect_records.execution.json'
}
(EV/'metadata_display_error_note.json').write_text(json.dumps(failure,ensure_ascii=False,indent=2)+'\n')
(EV/'external_semantic_sources.json').write_text(json.dumps([
 {'id':'R1','access_date':'2026-09-28','url':'https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html','scope':'AP is non-interpolated and differs from trapezoidal PR AUC'},
 {'id':'R2','access_date':'2026-09-28','url':'https://scikit-learn.org/stable/modules/calibration.html','scope':'Brier/log loss are not pure calibration errors'}],ensure_ascii=False,indent=2)+'\n')
(EV/'findings.json').write_text(json.dumps({
 'attachment_sha256':'6d01ada467bc705044d31008302fb6a96aa6fd9fbb7817db4166dd06254c9474',
 'result_validity':'PASS_WITH_EXPLICIT_SAVED_EVIDENCE_SCOPE',
 'predefined_acceptance':'FAIL_9_OF_13_PASSED',
 'weight_retention':'RETAIN_ALL_18_LINUX_INFERENCE_WEIGHTS_PER_EXPLICIT_USER_INSTRUCTION',
 'confirmed_scientific_blockers':[], 'confirmed_reproducibility_defects':[],
 'mandatory_fixes':[],
 'out_of_scope_actions_not_required':['new_seeds_or_training','new_label_reads','model_loading','recalibration_or_threshold_search','new_test_or_continual_learning','expanded_candidate_or_open_query_scenarios','automatic_weight_deletion','unrelated_hardening'],
 'findings':[{'id':f'F{i}','confirmed_defect':False,'details':'review_report.zh.md'} for i in range(1,11)]
},ensure_ascii=False,indent=2)+'\n')
(EV/'README.zh.md').write_text('''# 本条重新上传结果包：独立审查证据

这不是旧PASS的拷贝；所有`*.execution.json`对应本条附件新解压目录上的实际执行。独立主脚本复用并检查审查者既有实现，来源与路径修改见`script_provenance.json`。保存原脚本与原日志，不为使路径美观而事后改写。

先读`review_report.zh.md`和`findings.json`，再核对`submission_verification.json`、`independent_analysis/independent_results.json`、`additional_saved_checks.json`、`report_integrity_results.json`。

`submitted_entry_outputs/`为本次复跑执行者入口的新输出，原提交analysis不变；`independent_analysis/`为不导入项目模块的独立实现输出。`*.stdout.log`、`*.stderr.log`、`*.resource.log`、`*.exit.txt`、`*.execution.json`保留命令、版本与退出状态。交互式元数据显示的一次错误单独说明，不能据此称科学复算失败或声称所有交互调用从未报错。

复核独立结果只需在已有NumPy环境调用：

```text
python independent_result_audit.py --root <ranking_result.zip解压根目录> --out <新输出目录>
```

该入口仅使用保存分数／矩阵／计数／清单及标准库／NumPy，不加载模型、不解析正式标签、不执行训练。`run_logged.py`保留网页实际绝对路径，重现时可直接运行上述带参数入口；不需要复建项目GPU环境。源ZIP未重复打包，权重、正式原始文本和标签不在本证据包。

原附件SHA和全部成员记录见`submission_verification.json`；本包各文件大小及SHA见`evidence_inventory.json`（不包含清单自身，避免递归）。CSV为普通审计数据，不含公式或可执行内容。主要验收只看固定E6 C−A，不从额外比较选择最佳配置。
''',encoding='utf-8')

print(json.dumps({'status':'REPORT_BUILT','report':str(REPORT),'report_bytes':REPORT.stat().st_size,'primary_csv':str(PUBLIC_CSV),'table_rows':len(primary),'acceptance_passed':False,'checks_passed':9,'mandatory_fixes':0},ensure_ascii=False,indent=2))
