#!/usr/bin/env python3
"""Format the already completed audit as Chinese Markdown. No scientific payload reads."""
import json
from pathlib import Path
E=Path('/mnt/data/pooling_result_reviewer_evidence')
X=json.loads((E/'independent_output_pass/independent_audit.json').read_text())
P=X['paired']; T={(r['run_id'],r['epoch'],r['split']):r for r in X['trajectory']}
F=json.loads((E/'final_scope_check.json').read_text())

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def f(x):return f'{x:.6f}'
def delta(x):return f'{x:+.6f}'
def ci(v):return '['+', '.join(delta(x) for x in v)+']'
metric_table=table(['指标','D（三种子均值）','加权（三种子均值）','加权−D','差值条件95%区间'],[[m,f(v['d_mean']),f(v['weighted_mean']),delta(v['mean']),ci(v['conditional_95pct_interval'])] for m,v in P.items()])
gates=[['MAP平均增益≥0.01',delta(P['map']['mean']),'未通过'],['MAP条件区间下界>0',f(P['map']['conditional_95pct_interval'][0]),'未通过'],['三个配对各自ΔMAP>0',' / '.join(delta(v) for v in P['map']['per_seed']),'未通过']]
for m,sgn in [('recall_at_5',1),('average_precision',1),('roc_auc',1),('brier',-1),('log_loss',-1)]:
    for scope,val in [('三配对均值',P[m]['mean']),('固定s0',P[m]['per_seed'][0])]:
        gates.append([f'{scope} {m} '+('不下降' if sgn==1 else '不增加'),delta(val),'通过' if sgn*val>=0 else '未通过'])
gate_table=table(['冻结条件','独立复算观察值','结论'],gates)
seed_table=table(['配对','D MAP','加权MAP','ΔMAP','ΔRecall@5','ΔAP','ΔROC-AUC','ΔBrier','Δlog_loss'],[[s,f(T[s+'_d','6','valid']['map']),f(T[s+'_weighted','6','valid']['map'])]+[delta(P[m]['per_seed'][i]) for m in ('map','recall_at_5','average_precision','roc_auc','brier','log_loss')] for i,s in enumerate(('s0','s1','s2'))])
domain_table=table(['域','ΔMAP','ΔRecall@5','ΔAP','ΔROC-AUC','ΔBrier','Δlog_loss'],[[d]+[delta(P[m]['by_domain'][d]) for m in ('map','recall_at_5','average_precision','roc_auc','brier','log_loss')] for d in 'ABC'])
traj_table=table(['配置','拟合MAP E3→E6','训练内校准MAP E3→E6','valid MAP E3→E6'],[[r]+[f(T[r,'3',role]['map'])+' → '+f(T[r,'6',role]['map']) for role in ('fit','calibration','valid')] for r in X['training']])
auto_table=table(['配置','阈值（logit，显示截断）','TP / FP / FN / TN','合并Precision','合并Recall','合并FPR'],[[r['run'],f'{r["threshold"]:.9f}',' / '.join(str(r['pooled'][k]) for k in ('tp','fp','fn','tn')),f(r['pooled']['precision']),f(r['pooled']['recall']),f(r['pooled']['fpr'])] for r in F['automatic_thresholds']])
num_table=table(['网页独立核查类别','被比较数值元素数','最大绝对差','使用容限'],[[k,v['elements'],f'{v["maximum_absolute_difference"]:.17g}',f'{v["tolerance"]:.10g}'] for k,v in X['numerical_checks'].items()])
report=r'''# 商品加权汇总：正式训练与valid结果独立外审

**审查日期：2026-09-26。审查对象：本次已完成正式训练及valid结果，不是训练前计划／实现审查的续签。**

## 0. 明确判定

**本轮在提交证据可覆盖的范围内，通过“结果有效性外审”；方法改进验收未通过，结论为有效负结果。**

新运行证据、实际源码与本次网页单CPU独立数值核查相互一致：五次新训练各864更新、合计4320；三配对E6平均ΔMAP仅+0.0020580330，条件95%区间[−0.0031217984,+0.0071048193]；固定s0排序及若干基础识别护栏退化。13项冻结检查5项通过、8项未通过。因此不应替换D，不进入本轮test验收，不据此开始持续学习设计。

“保留D”现在仅指保留原D的配置和保存结果作为比较基线。按用户后续扩大清理指令，原D训练权重也已删除；不存在“D仍可直接加载”或“候选仍可加载”的本次核验结论。后续重建需要新的阶段授权。

| 类别 | 本次数量／判定 | 处置 |
|---|---|---|
| SCIENTIFIC_BLOCKER | 未发现经证实、足以使本次负结果失效的新阻断缺陷 | 无必须先修复后才可记录结果的代码或实验问题。验收失败本身不是实验有效性缺陷。 |
| REPRODUCIBILITY_DEFECT | 未发现经证实、影响本次保存结果复算的新缺陷 | 不要求补训、标签重读或补回已授权删除的权重。权重删除限制将来的模型级重放，须如实披露。 |
| OUT_OF_SCOPE_OVERDESIGN | 下述新增要求不作为整改 | 额外种子／消融／训练、重读正式标签、事后挑E3或s2、为了外审重建权重、无关系统加固或重复打开已关闭R1／R2／R3。 |

这不是对代码“绝无错误”的保证，也不是独立重算正式真值后的端到端认证。正式GPU更新、远端权重实物和清理动作只能依据提交的运行记录及相互绑定检查；本次网页实际做的是源码审查、文件身份核验和保存证据的数值复算。外审提供建议，本地主执行者仍应逐项核对后决定如何归档和表述。

## 1. 附件身份、冻结来源与实际阅读范围

实际ZIP为`pooling_result_review.zip`，4,495,283字节，SHA-256：

```text
8c4115ce1d3ddf3c008df89fe56eb40794179b37c186984a7613080c4219b682
```

ZIP完整性检查通过。包含165份来源文件及清单自身，共166个成员；清单列明的165份文件解压后合计6,376,611字节，所有大小及SHA逐项匹配。95份回传小型运行／评价文件合计4,680,563字节，也逐项与回传清单匹配。这些数值分别是压缩包大小、来源文件解压大小、回传子集大小，不应混用。

完整阅读了本次`request.zh.md`、结果文档、冻结合同、政策、新分析入口、方法与执行主路径，并追读直接继承的输入、损失、优化器、模型保存恢复和评价实现；结合相关旧网页外审原文及本轮来源绑定核对。没有声称重新审计所有历史研究，也没有重新运行旧CPU真实BGE测试。

与上次训练前附件比对，共同的16份代码／政策／测试文件未变。合同仅有授权和阶段状态文字变化；启动时冻结合同SHA为`385dfb45257cf9d40101342b11dc32529158da9ad269abc80aec16786a76676c`，与本次实际合同及18份启动来源记录一致。合同的历史状态标题不需要为本次结果修改字节；当前状态由结果文档与新运行证据表达。

审查结束后再次核验，165份提交来源文件全部未变；原`analysis`未覆盖。网页重跑入口只写入新目录`Q/reviewer_analysis`，审查者补测只写入独立证据目录。

### 文件位置缩写

下文行号对应本次冻结附件，JSON路径写在文件名后。

| 缩写 | 实际路径 |
|---|---|
| J | `reports/seller_alias_continual/20260925/pooling_execution/20260925_125203/job` |
| Q | `reports/seller_alias_continual/20260926/pooling_result` |
| H | `reports/seller_alias_continual/20260924/chinese_execution/20260924_152642/job/run/split_rank` |
| M | `scripts/step28_alias_pooling.py` |
| R | `scripts/step28_alias_pooling_run.py` |
| A | `scripts/step28_alias_pooling_result.py` |
| B | `scripts/step28_chinese_base.py` |
| G | `scripts/step28_continual_population_evaluate.py` |
| P | `scripts/step28_continual_population.py` |
| I | `scripts/step28_continual_expression_run.py` |
| C | `docs/SELLER_ALIAS_POOLING.zh.md` |
| RESULT | `docs/SELLER_ALIAS_POOLING_RESULT.zh.md` |

## 2. 本次网页实测与证据边界

### 2.1 实际运行

使用网页既有Linux、Python 3.13.5、NumPy 2.3.5。没有安装包、改变项目环境、导入模型框架、加载模型、启动训练或读取正式文本／标签／owners。该环境不是远端Linux py310，二者已明确区分。

先静态阅读A，再设置单线程环境变量实际执行原分析入口，输出到新目录。退出0，约0.93秒，GNU time最大RSS104,476KiB。复现其12,420个数值比较、3,240组盲预测正例总数、131份小型来源／结果文件校验和13项判据；原Linux和网页入口输出的`metrics.csv`、`trajectory.csv`逐字节相同。

随后运行审查者独立脚本`independent_result_audit.py`。该脚本只使用标准库和NumPy，不调用生产指标、bootstrap或提交分析模块；固定进程CPU亲和性为一个CPU，实际`Threads: 1`。最终退出0，约1.19秒，最大RSS119,288KiB。另运行终末来源不变性及阈值边界核查，退出0。

独立脚本读取12份60×22 valid矩阵，以及24份拟合／校准矩阵，共36份保存指标矩阵；比对盲预测阈值计数3,456组，其中多出的216组来自6模型×36群E6校准阈值核对。独立复算所有群的六项固定阈值分类比率、整体／分域合并计数、主配对及逐种子22指标区间，以及自动分类配对区间。所有模型相关检查均只读记录、摘要和盲分数，没有访问任何权重载荷。

比较的数值元素是程序核对工作量，不是独立科研样本数、单元测试数或训练次数。重复出现在不同摘要中的同一统计量可能被多次核对。

### 2.2 容限没有混用

保存指标／归并／bootstrap使用绝对容限1e−12；实际差异远低于该容限。保存float32训练损失的BCE+rank与total之间，最大差1.2728075482471013e−8，单列在训练目标检查中使用1e−6。正文六位小数采用半个末位单位的舍入核对，不把6位打印误差混作全精度统计误差。整数计数和身份相等性采用精确比较。

{{NUM_TABLE}}

### 2.3 审查者自身一次失败与修正

独立脚本第一次运行在“仅用公开群ID重建分区”处失败。原因是我初稿的canonical JSON序列化漏掉项目既有的结尾换行，导致SHA派生的排序不同；项目`step28_continual_population_data.py:28–32`明确包含该换行。我只修正审查脚本，没有修改项目源码、政策、分区或结果。修正后在新输出目录重跑通过。

首次脚本、原始失败日志和退出码1均保留，修正后脚本、成功日志及退出码0也保留。此事属于审查工具纠错，不能列为项目的REPRODUCIBILITY_DEFECT，也没有被隐去或算成正式训练失败。

## 3. 用户要求→输入→训练→保存→评价：逐项发现

### F1. 干预确实是完整商品加权汇总配置，没有替换研究输入和评价任务

**代码／证据：** I:58–128；B:134–193；M:57–118、121–165；`J/run/partition.json`；各臂`manifest.json`的`parameter_count`、`fit_group_ids`、`calibration_group_ids`；RESULT:51–61。

模型仍只使用屏蔽后的商品标题和描述，同商品两通道保持配对，不按通道独立重排；先编码再按账号拆分，没有新增账号／群／域ID特征。ID仅用于分区、顺序和来源绑定。账号内全部商品参与；学习权重与均匀权重各占0.5，两通道共享商品权重；均值和总体标准差加权计算，数值稳定项和最终归一化保持。基础编码器、对称头、BCE＋排序损失及每臂864更新相同。

每域48群拟合、12群训练内校准、20群valid，共144／36／60群。审查者用已保存公开群ID独立重建了分区和三组日程，核对角色互斥和每域群数，不读取训练文本或真值。每群28账号、每查询27候选是当前评价边界，不是全库检索或开放集拒识。

D有326,571,265个参数，候选有326,702,465个参数，增加131,200，记录中均可训练。该对比检验的是完整加权配置的增量，包含新参数和其优化设置；不提供“纯注意力机制、排除所有容量因素”的独立因果结论。

**受影响结论／最小修复：** 输入与干预范围支持本轮候选排序比较，无必须修复项。为把完整配置效果拆成纯机制贡献而新增容量消融，属于本轮之外的研究，不是使负结果有效所必需。

### F2. 历史s0 D复用、配对共同初态和随机日程有实质来源绑定

**代码／证据：** M:45–54；R:35–75、117–141、230–382；B:52–86、110–131；H/manifest.json；`J/run/manifest.json`；`J/evaluation/historical_alignment.json`；`J/evaluation/preparse_verification.json`。

s0 D不是本轮第六次新训练；两份E3／E6指标矩阵与历史评价逐元素一致，固定0混淆计数也相同。审查者复算历史差异矩阵，所有差为0。共同初态、数据分区、864步日程、dropout流及预训练档案来源逐项绑定，不是仅凭配置名相同复用。

三组配对分别具有相同共同初态摘要、日程SHA和dropout流；我独立用政策种子和公开群ID重建全部三组日程。s0日程SHA为`16758c902e2d370492fcfd47b059f784fcf6a8cc0cbfbb52ef73674b1e4a8e82`，与历史D一致。新模块CPU局部初始化保留共同随机流的实现未变；正式记录的共同状态摘要相等。

**受影响结论／最小修复：** 历史复用和配对差有证据支持，无必须修复项。摘要、固定随机策略和共同初态并不等于本次独立重现了CUDA每一个随机算子或保证跨平台逐位复现；本次没有远端权重实物可供再验。

### F3. 五次新训练及新模块实际更新有正式运行证据

**代码／证据：** M:107–118、183–221；P:32–40；R:144–216、230–287；`J/train.log`；五臂`manifest.json`的`updates`、`training[*].observations`、损失与资源段。

五臂记录各864更新，总4320。日志每臂具有24、48、…、864的36个进度点，共180个里程碑，无该提交日志中的重复进度序列。各臂分别从同一个预训练档案构建模型和新优化器，不是把旧D继续训练成候选。

新参数在训练开始前进入第三AdamW组。三候选均记录第1、2、433步的梯度／状态；第1步隐藏W和b任务梯度为0符合输出层零初始化，不能把隐藏W的权重衰减变化当作监督梯度。第2和433步隐藏W、隐藏b、输出w均有正的有限任务梯度，并有更新状态记录；编码器、评分头也有非零任务梯度／变化记录。

例如s0候选第2步三项范数约为4.1624e−6、8.7137e−9、1.2870e−4；第433步约为5.9875e−3、1.5194e−4、3.2237e−2。它们支持新模块真正接入任务优化，而不只是“声明requires_grad”。六轮保存的BCE、排序损失和总损失逐轮下降，但这不自动保证泛化提高。

**受影响结论／最小修复：** 未见漏训练、汇总参数未入优化器、把首步衰减冒充学习等缺陷。正式梯度和Adam连续性的事实依赖运行记录及被绑定源码，不是网页再次跑了GPU更新；不声称每个参数每一步都有非零梯度。

### F4. E3／E6完整恢复和valid前12份权重检查的运行记录闭合

**代码／证据：** R:78–114、150–216、290–382、479–505；P:157–210；各臂`points.3/6.full_model_and_adam_reloaded`、`model.actual_reload_verified`、模型／盲分数SHA；`J/evaluation/preparse_verification.json`。

十个新E3／E6点均有完整model＋Adam保存恢复、恢复后完整盲分数相等以及推理状态另存恢复的标记和摘要。正式源码是在实际恢复与分数比较成功后才写出成功记录，新汇总模块通过注册状态包含在保存中。E3后继续原Adam状态到E6，E3不用于早停或挑主终点。

valid解析之前，记录核验了十份新权重和两份历史D权重的实际大小／SHA，来源18份、训练manifest、分区、更新与校准绑定也检查。审查者独立确认这些12条记录恰好对应六模型×两轮，且其路径、大小、SHA与后续用户删除回执相符。

**受影响结论／最小修复：** 支持运行时保存／评价身份链，未发现模型／epoch错配，无必须修复项。网页本次只核对记录，不把记录相符称为重新读取了12个权重文件；文件现已删除，不能再复做模型推理层核验。

### F5. 标签一次边界与先采集完整12矩阵、后统计的执行路径没有回归

**代码／证据：** I:58–128；R:230–287、385–505、541–559；`J/run/access.json`；`J/evaluation/access.json`；`J/completion.json.label_parses`；`J/evaluation/collected.json`及所指12份矩阵。

train父入口解析一次，供五次新训练共享已加载对象；valid父入口在五臂完成和12权重验证后解析一次。记录为train=1、development=1、heldout=0、owners=0；源码入口不加载owners或test文本／真值。训练内校准群不进入梯度。该结论是受检查入口和记录范围，不是远端全系统访问取证。

完整12份60×22 valid矩阵、群顺序、域映射、固定0混淆计数和六份E6自动分类计数先保存为collected，之后才运行历史对齐和bootstrap。评价中的点数据与collected一致。历史D差异完整记录为0，未出现为强求历史对齐而删除结果的证据。

**受影响结论／最小修复：** R1不回归，无必要重新读取正式标签；不得以增加“独立性”为由重读标签。本次网页分析也只使用保存记录，没有重建或推断单个账号对的真值。

### F6. 全22指标归并、分类分母和bootstrap一致，负结论不是统计拼接错误

**代码／证据：** G:21–100、225–228；B:680–750、775–784；M:224–275；R:415–476；`J/evaluation/evaluation.json`的`paired_primary`、`per_seed_comparisons`、`acceptance`；本审查独立输出。

审查者用逐重复抽样的第三条实现，重现了5000次域内整群抽样，并与提交分析脚本的频数加权实现相互核对。所有22个差值均值、各配对差、域切片及条件区间一致；固定／校准阈值的合并结果先合并TP、FP、FN、TN再算比率，没有把群precision均值当成合并precision。

AP按非插值平均精确率计算，不是梯形PR-AUC；两者分列。AP／ROC-AUC为先逐群计算再宏平均，不是把全部账号对堆起来的曲线。Recall@1按找回相关账号数／相关账号总数计算，不能当首位是否命中；本数据二元相关性下NDCG@1才相当于该命中指标。静态公式与既有手写验证支持这些定义；本次没有正式真值，不声称从真值重新计算AP、ROC-AUC、Brier、log_loss或检索指标。AP的非插值定义也与官方文档[外1]一致。

**受影响结论／最小修复：** 数值汇总和判据成立，R2不回归，无必须修复项。六位小数表不能用于判定临界是否通过，程序使用完整精度；本轮失败幅度亦不是舍入造成。

### F7. 新分析入口无已证实的本轮数值错误，但必须按其实际覆盖范围表述

**代码／证据：** A:36–115、118–150、153–273、274–311；`Q/analysis_execution.json`；`Q/analysis/analysis.json`；网页`Q/reviewer_analysis`；本审查脚本及独立输出。

提交入口只需NumPy。其频数矩阵bootstrap和手写线性分位数不是直接调用生产bootstrap；分类分母、配对均值、13条件均正确。损失1e−6与指标1e−12分开，原文12,420／3,240／131的数量在网页得以复现。`--out`目录不覆盖既有结果。CLI错误信息提到py310，但本次Linux现有Python 3.13.5实际运行退出0，报告不伪称在网页使用了py310。

原入口侧重valid，不独立复算每一项已存自动分类“配对差区间”和所有训练内群分类比率；本审查独立脚本补齐这两部分，均一致。原入口也没有也不应由盲分数反推完整真值或重新生成校准负例排序。它的status文字专用于当前负实验，不是经过验证的任意未来实验通用结果分类器；当前13个布尔量与状态一致，没有因此误判当前结果。

**受影响结论／最小修复：** 本轮入口未发现需修复的数值或结论错误。保持“保存结果核对，不是正式真值重算”的范围即可；不能把通过入口等同于证明全部科研行为正确。本次也没有把报出的0读取计数当成独立操作系统审计，而是结合源码读路径、附件内容和本次实际操作界定范围。

### F8. 正式耗时与资源结果有记录，未发现R3回归

**代码／证据：** R:144–216、508–538；`scripts/run_step28_alias_pooling_linux_20260925.sh:20–30`；`J/started.txt`、`finished.txt`、`exit_status.txt`、`resource_usage.log`、`completion.json.budget`；各臂资源记录。

2026-09-25 12:57:45—20:35:20 CST，退出0；时间戳差27,455秒（7小时37分35秒），GNU time为7:37:34，外围记录差1秒不构成矛盾。预算记录约27,452秒；新增产物观察峰值16,984,521,516字节，低于32GiB。实际运行记录为py310、单GPU RTX5090、单CPU线程。

各新臂耗时约90.48—92.71分钟，纯更新约73.95—75.60分钟；历史s0 D约94.26分钟不计入五次新训练预算。跨日历史D耗时不是严格性能基准，不能据此建立精确速度优劣结论。

正式每臂在加载前重置、至校准结束读取CUDA分配器峰值，allocated约7.94—7.96GiB。它不是整卡全部进程峰值；单列preflight／清理不在臂内峰值中。产物预算字段是观察到的路径大小峰值，不是对所有瞬时文件及整个文件系统作全量测量。

**受影响结论／最小修复：** 当前记录支持在冻结资源范围内完成，R3不回归。不存在由本轮小幅参数增加必然显著拖慢或加速的可靠因果结论，无必须修复项。

### F9. 两次授权清理不使保存结果失效，但现在没有可加载的原D

**证据：** `Q/weight_cleanup_inventory.json`、`weight_cleanup.json`、`windows_cleanup.json`；`reports/maintenance/20260926/trained_weights/cleanup.json`；RESULT:7、25–37、168。

第一次10文件共13,067,720,890字节，清理回执与删除前清单绑定；当时保留原D及预训练模型是历史事实。第二次105文件共111,917,493,162字节，明确覆盖原D；两次合计115个不同路径、124,985,214,052字节。当前回执记录原D已删、训练权重剩余0、预训练删除0。

我逐项核对本轮被评价的12个权重记录与两轮删除路径／大小／SHA一致，重算两轮数量与字节合计。95份小型证据在当前附件中可逐文件核验；“676份相关非权重文件元数据不变”只属于清理回执的元数据层声明，不是本次逐个读了676文件并比较内容SHA。完整全量候选盘点没有随包上传，正是用户明确的范围，不要求补发它或其他历史数据索引。

Windows检查仅覆盖第一次本轮权重副本，回执删除0；第二次指令是Linux全部训练所得权重，不应擅自扩大为所有Windows历史载荷也清理。

**受影响结论／最小修复：** 现有矩阵、计数和统计仍可复算；用户选择删除模型不让负结果无效，也不构成需要强制补回载荷的复现缺陷。但是模型级推理重放和未来test已无当前可加载载荷，原D只能作为配置／结果比较基线。结果文档已正确更新，无必须修复项；不能承诺重建可恢复原字节。

### F10. 结果解释基本准确；应补充区分“排序泛化下降”和“所有指标恶化”

**证据：** RESULT:63–124、126–151、166–170；所有36份保存矩阵；`Q/analysis/trajectory.csv`；本审查`trajectories.csv`。

s0候选训练内MAP明显上升而valid和校准MAP下降，构成排序泛化问题的信号；但s1 D也有valid MAP下降，s1／s2候选valid MAP反而上升。s0候选自身E3→E6的valid Brier和log_loss均改善，不能将其“相对D的E6护栏失败”混作“它自己所有指标随训练轮数都恶化”。

结果原文已称过拟合“信号”并拒绝唯一机制解释，因此没有构成已证实的科学误判。建议在当前结果解释中明确加上上述度量维度区别，防止后续摘要把结论泛化。无需改冻结合同、追加运行或追选E3。

**受影响结论／最小修复：** 这是解释精度补充，不是SCIENTIFIC_BLOCKER或REPRODUCIBILITY_DEFECT。不能从现有数据断定注意力塌缩、错商品高权重、容量不足、特定学习率／损失权重是唯一原因；当前不具备逐商品注意力分布或机制实验。

## 4. 主终点、13项判据和逐配对结果

全精度三配对平均MAP：D=0.48645312967175725，加权=0.4885111627056887，差值+0.0020580330339314604，约+0.2058个百分点。条件95%区间[−0.0031217984466988635,+0.00710481929384226]。

本轮同时失败于效应幅度、区间下界、配对方向和若干识别护栏，不是只因为业务K或人工复核预算未确定而无法结论。已有冻结科研验收足以判为本轮改进不合格，但不等于已证明完整方法普遍无效。

{{GATE_TABLE}}

固定s0的MAP差为负，Recall@5／AP下降，Brier／log_loss增加；ROC-AUC观察值略升。三配对均值虽然Recall@5、ROC-AUC、Brier和log_loss有局部改善，均值AP仍下降，且均值不能抵消固定s0失败。

{{SEED_TABLE}}

s1自身的ΔMAP条件区间约[+0.000367,+0.017715]，呈现局部积极结果；s2 MAP也上升但AP下降。不能因此替换事前固定s0，也不能用s2候选对历史s0 D作非配对“胜者比较”。辅助比较很多，未进行多重比较修正，不能把其中一个区间不跨0升级为整体成功。

s0 ΔMAP区间约[−0.017632,+0.000282]仍含0。因此可以说固定s0观察下降并触发事前护栏失败，不能说已在训练种子总体上显著证明s0 MAP必然降低。观察值护栏与总体统计显著性是两件事。

## 5. 完整22指标和区间

以下为三种子E6模型群宏平均，差值均为加权−D。Brier、log_loss越小越好，其余通常越大越好。precision至MCC为固定logit=0的逐群宏平均，不是合并计数结果。完整精度与所有域／轮／配置见证据包CSV。

{{METRIC_TABLE}}

其中平均log_loss的条件区间全部小于0，是允许报告的概率损失局部改善；其余主表区间均跨0。这不能替代MAP主终点及固定s0护栏。平均ROC-AUC和Brier的正向观察值也不能表述为总体非劣效已得到证明。

### 5.1 区间真正抽样了什么

令d(s,g,m)=加权模型指标−对应D指标。先对同一群g的三个配对取平均d̄(g,m)，再在A、B、C各20群内有放回抽样20群，三域等权，重复5000次。每次保留同群内查询／账号对及三个种子之间的依赖结构；不把3×60个种子群、1680个查询或22,680个账号对当独立样本。

固定bootstrap生成器种子20260925，取2.5%与97.5%线性分位数；提交分析使用频数矩阵乘法及手写线性分位数，审查者使用逐重复抽样及NumPy线性分位数。两者对应q(n−1)位置的线性插值[外2]，数值相符。

这些是条件于固定训练种子、已训练模型、生成数据与已校准阈值的群重采样区间。它没有纳入重新训练、重新生成数据、重新拟合阈值或多轮valid开发选择的全部不确定性。固定三个种子不是种子总体样本量保证；反复使用valid不会因本次bootstrap而变成独立test。

### 5.2 分域不是三域一致提高

{{DOMAIN_TABLE}}

A／B平均MAP上升、C下降。C的AP、ROC-AUC、Recall@5及Brier观察值也更差；但C log_loss略改善。可以描述本次域切片差异，不能事后删C、改单域成功线，或声称已经定位了某种域机制。

## 6. 分类计数、校准阈值与旧自动判定门

固定0与自动校准结果在原始评价分别保留。审查者从已保存TP／FP／FN／TN重算合并比率，并用盲logit仅核对预测正例总数，不使用真值重算TP／FP，也不推断单个标签。

下面是E6校准阈值下的合并结果；显示阈值仅供阅读，实际核对保留float64全精度：

{{AUTO_TABLE}}

s1两模型预测正例确实为0，不是遗漏分母或计数归并造成。D的最大valid盲logit为0.12313077598810196，小于阈值0.12313077598810197；加权最大值0.12620165944099426，小于阈值0.1262016594409943。float64比较保留这个边界，不能把显示舍入后的值再比较而凭空增加预测正例。

这样的0预测只说明当前模型、阈值和有限样本下没有正预测；零宽区间不代表部署风险为0，也不代表自动判定可用。六模型三域都没有同时满足旧FPR≤0.001且Recall≥0.5的绝对门，但它是诊断，不是本轮候选排序改进的主验收门。当前负结论不需要靠该旧门兜底。

在固定0下，例如s0 D的合并TP/FP/FN/TN为102/26/1098/21454，合并Precision=0.796875；这与表中跨种子、逐群宏平均precision=0.722855不是同一统计对象。所有域与合并计数结果均在`all_fixed_confusion_summaries.csv`和`all_automatic_confusion_summaries.csv`中保留。

## 7. 损失、E3→E6与“过拟合信号”的正确解释

{{TRAJ_TABLE}}

s0候选拟合MAP从0.569023974上升到0.699114178，增加约0.130090；valid从0.492321179下降到0.482364498，下降约0.009957；训练内校准从0.486798745下降到0.482127150。与“拟合继续改善、留出排序未同步改善”相符，支持将排序泛化作为诊断重点。

但s0候选自身valid Brier从0.047061397改善到0.043947711，log_loss从0.201937289改善到0.184437544。它的概率损失随轮数改善，与它E6相对D的Brier／log_loss仍较差可以同时成立，不能把比较对象混在一起。

s1 D的valid MAP也从0.493236635降到0.478125621；s1和s2候选从E3到E6的valid MAP分别提高。因此“全部加权候选都随训练退化”不成立。s1／s2候选E6训练目标值反而高于其配对D，但valid MAP较好；优化目标更低和留出检索更好并非同义。

六轮损失是训练过程中的均值，涉及模型状态不断变化和训练模式；E3／E6矩阵是固定保存点的重新评价。不要把在线epoch平均损失与保存点eval指标当成同一次前向的等价结果，也不要以两者趋势差异直接判为实现错误。

只能提出多种可能的解释，不能确定唯一因果：新增商品权重可能改变拟合分布与相似度排序，但当前没有保存逐商品权重分布，也没有能区分权重集中、文本噪声、容量、优化超参数等机制的实验。本轮不要求为补因果解释而读取新文本、重建模型或追加训练。

E3只是诊断轨迹，不成为事后替换E6的终点；s1或s2的局部改善也不成为事后更换固定s0的依据。

## 8. 真实问题分类、最小处置和不应新增的要求

本轮核对未发现需要以SCIENTIFIC_BLOCKER或REPRODUCIBILITY_DEFECT标记的新增真实问题。判据失败是研究结果，不应为得到“通过”而修改它。解释补充F10不是将实现缺陷降格，而是现有正确结论的更精确表述。

| 事项 | 分类／性质 | 影响与最小处置 |
|---|---|---|
| 13项8项失败 | 有效负科研结果，不是实现缺陷 | 保留负结果，不替换D配置／结果，不送本轮test，不开始持续学习设计。 |
| s0 E3→E6排序下降而概率损失改善 | 解释精度补充，非阻断缺陷 | 当前结果说明可补一句区分，不改冻结合同。 |
| 权重按用户指令全部删除 | 明示的可重放范围限制，非强制整改项 | 不再宣称D／候选可加载；保留结果级复算，重建另行授权。 |
| 更多种子、容量消融或改变损失／轮数 | OUT_OF_SCOPE_OVERDESIGN，若将其强加为本轮审查门槛 | 不追加；未来新问题需单独前瞻定义，不能覆盖本轮。 |
| 用正式真值重读／模型重载替代已有结果审核 | OUT_OF_SCOPE_OVERDESIGN，且超出本次授权范围 | 不执行；明确本审查不是从正式标签重算所有指标。 |
| 改E3、换s2、删域C、放宽MAP门 | 会改变已冻结问题，而非修复 | 禁止用作本轮成功解释；当前包没有这样做。 |
| 重复审R1／R2／R3、无关安全加固／密钥流程 | OUT_OF_SCOPE_OVERDESIGN | 只核对本轮不回归，不新增仪式。 |

## 9. 未验证范围

本次未独立接触远端正式文本、标签CSV、owners、封存真值、预训练或训练权重；未执行任何正式GPU更新、训练、模型重载、test或新的标签解析。正式AP／排序／概率损失等真值依赖指标只能在已有矩阵层核对归并和身份，不能冒称全部从真值重算。

正式梯度、Adam连续性、权重实际SHA、保存恢复以及删除动作依赖运行记录和对应源码；本次没有访问远端文件系统，不证明远端所有程序或所有时刻都没有其他行为。676份非权重元数据不变和全部训练权重删除也应按回执范围表述，而不是网页实物认证。

本次不验证真实市场、跨平台全库候选、没有相关候选的开放查询、生产人工复核成本、训练种子总体性能、后续新数据或持续学习性能。独立test没有执行，也不存在test通过结论。当前valid已经多次用于开发，当前三个种子固定，每个查询只有同群27个候选；这些限制不能被bootstrap消除。

## 10. 最终处置建议

将本轮归档为“正式运行完成、结果有效性外审通过、冻结改进验收失败”的有效负实验。保留原D配置／历史结果和本轮所有小型记录，不把本轮候选替换为D，不把外审通过改写为方法成功。

目前没有必须先修复的实现或统计问题，也没有理由为这次结果外审再训练、重读监督、复活已清理模型或新增种子。后续模型重建、另一项有针对性的改进或test属于新的阶段决定；本次外审不自动授权它们。

## 附录A. 实际命令与交付文件

### A1. 提交分析入口（已执行，退出0）

在解压项目根目录，使用既有环境：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python scripts/step28_alias_pooling_result.py \
  --job reports/seller_alias_continual/20260925/pooling_execution/20260925_125203/job \
  --out reports/seller_alias_continual/20260926/pooling_result/reviewer_analysis
```

`reviewer_analysis`必须不存在；不能用此命令覆盖已经有结果的同名目录。原始输出与GNU time日志保留为`submitted_entry.*`。

### A2. 审查者独立脚本（修正后已执行，退出0）

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python independent_result_audit.py \
  --root /path/to/extracted/project \
  --out /path/to/new/reviewer/output
```

本脚本核对原`Q/analysis`及A1生成的`Q/reviewer_analysis`。输出目录必须全新；不依赖PyTorch、BGE或正式标签。所需运行证据已经在用户附件中。

证据包保留`verify_submission.py`、`independent_result_audit.py`、首次失败版本、`final_scope_check.py`、报告格式化脚本、全部运行原始日志和退出码、来源核验清单、两条分析的实际输出。完整22指标、36矩阵整体／分域摘要、固定和校准合并计数、轨迹及5000×22 bootstrap结果也已导出。证据包有逐文件字节／SHA清单，便于主执行者回读。

所有统计脚本仅复算已有证据。`build_report.py`只将独立复算输出排版成本文，不是新科研计算。

## 附录B. 外部定义核对

[外1] scikit-learn官方`average_precision_score`文档：AP为Σ(R_n−R_{n−1})P_n，非插值，并与梯形PR-AUC区分。查阅2026-09-26。`https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html`

[外2] NumPy官方`quantile`文档：默认／指定`linear`方法按q(n−1)对应的相邻顺序统计量线性插值。查阅2026-09-26。`https://numpy.org/doc/stable/reference/generated/numpy.quantile.html`

两项外部资料只核对定义，不是本实验数字、正式运行或删除行为的来源。本实验事实依据附件源码／记录及本次独立运行输出。
'''
for key,value in [('NUM_TABLE',num_table),('GATE_TABLE',gate_table),('SEED_TABLE',seed_table),('METRIC_TABLE',metric_table),('DOMAIN_TABLE',domain_table),('AUTO_TABLE',auto_table),('TRAJ_TABLE',traj_table)]:
    report=report.replace('{{'+key+'}}',value)
target=Path('/mnt/data/pooling_result_review_report.zh.md')
target.write_text(report,encoding='utf-8')
print(f'Report: {target}; UTF8 bytes={target.stat().st_size}; characters={len(report)}; lines={len(report.splitlines())}')
