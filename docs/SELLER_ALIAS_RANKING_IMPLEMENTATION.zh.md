# 前排错误候选约束：实现准备与核验记录

日期：2026-09-27。当前状态：Linux既有py310单CPU核验完成，15项合同通过、0失败／跳过，4次实际BGE手工更新通过；13份小证据130,994字节回传且实际大小／SHA一致。10:54:50—11:01:44 CST，完整GNU time为414.34秒（6分54.34秒），退出0。**本轮网页GPT6 Pro外审及主执行者独立处置已完成，没有确认必须修复的新科学阻断或复现缺陷。用户随后明确“核验完毕后，无需向我申请，直接开始正式训练”，已按已汇报的九运行／7776更新范围正式启动，无须重复申请。** 实际启动、标签消费和进度以[正式阶段](../reports/seller_alias_continual/20260927/ranking_execution/authorization.json)及[交接](AI_RESEARCH_HANDOFF.zh.md)为准；此处核验不是性能提升证明。

## 要求与实际实现的对应

| 用户要求／前瞻约定 | 实现位置和已核对的计算 | 核验责任与证据边界（实测结果见下） |
|---|---|---|
| 提升中文合成马甲候选排序 | `step28_alias_ranking.objectives`：保留全378对BCE及原查询rank，C再加0.5×每查询全正例对最高分5个已知负例的softplus均值 | 独立标量值、解析梯度、实际BGE梯度及更新 |
| 基础识别不下降 | `acceptance`：AP／ROC-AUC不降及Brier／log_loss不升，分别保护三配对均值和s0；总计13条件 | 边界用例；正式valid尚未授权 |
| Recall@5确实提高 | 均值和s0差值都严格>0；相等不通过 | 严格边界用例 |
| 有针对性的固定比较 | D、较小编码器lr加预热衰减、再加hard；全部模型从同一BGE档案开始；无额外模型参数 | 原D包装逐位等价、三配置共同初态／首次前向、实际学习率 |
| 不挑种子或轮次 | 三配对种子，九个新模型，6轮／864更新各；E6 C−A主要，B−A与C−B解释性；s0固定 | 成对日程、学习率全序列、正式更新账 |
| 保存后能正确接续 | `checkpoint`保存恢复完整模型与Adam，核对步数、lr及完整三角色盲分数；E3后继续原Adam | 微型模型跨预热边界的新实例恢复；正式E3/E6实际恢复留待正式作业 |
| 合成数据边界 | 沿用原公开加载与标签对齐，只取train/development；训练校准群不参与梯度；模型仅输入标题描述 | CPU用例禁用正式加载器；本次准备不读取正式数据 |
| valid一次且先留完整结果 | `validate_run`先核验9运行、18实物权重、54分数；`collect`先保存18矩阵与计数，再`finalize` | 18份手工假文件逐个损坏均须在标签前拒绝；后处理故障保全和零标签恢复 |
| 不夸大统计范围 | 三固定种子逐群差先平均，三域整群5000次bootstrap；AP和梯形PR-AUC分列，全22指标 | 非恒定数据与独立频数加权实现比对 |

## 源码与依赖审读

新文件：[方法](../scripts/step28_alias_ranking.py)、[运行和评价](../scripts/step28_alias_ranking_run.py)、[CPU核验](../scripts/step28_alias_ranking_audit.py)、[合同用例](../tests/test_step28_alias_ranking_contracts.py)、[政策](../schema/step28_alias_ranking_policy.json)、[Linux入口](../scripts/run_step28_alias_ranking_linux_20260927.sh)及[研究合同](SELLER_ALIAS_RANKING.zh.md)。新阶段独立文件；旧加权汇总18份冻结来源不改写。

主执行者已从“排序提高且基础识别不下降”的要求出发，审读新增公式、top5排除自身／全部正例、每查询归约、学习率实际赋值、优化器连续性、来源和保存结果绑定。继承链重点核对 `step28_chinese_base` 的输入、均值／总体标准差、BCE和全候选rank、更新和盲评分，`step28_continual_expression_run` 的窄范围公开输入和标签对齐，`step28_continual_population` 的优化器及完整保存恢复，`step28_continual_population_evaluate` 的22指标，及原Budget。原生加载、CUDA／BF16、实际模块梯度和运行文件恢复仍须用运行记录证明，不能以静态对应替代。

直接相关的9月25日GPT6 Pro训练前审查原文 `fd06efe3-a548-4c85-894a-9379ab8b78c6` 已全文回读；9月26日结果审查 `43a26d10-3b68-4755-934b-053d9e2b2c16` 的原文、附件及主审处置在前一步已完整核对。原R1采集保全、R2计数与宏／合并口径、R3资源范围保持；本次相应受影响路径重新核查，未把旧外审通过冒充新实现通过。参见[旧实现审查原文](../reports/seller_alias_continual/20260925/pooling_implementation/review/response.zh.md)及[旧结果](SELLER_ALIAS_POOLING_RESULT.zh.md)。

静态审读中已修正手工目标用例的标签张量浮点类型；这是新测试准备过程中的修正，不是实际训练失败或运行通过。学习率末步明确为0：B／C有864次optimizer调用、863个编码器正学习率步，头保持1e-3；不能把零lr末步写成编码器参数变化。

## Linux CPU实测与主执行者核查

用户已明确“恢复上述 CPU 核验（推荐）”，见[授权](../reports/seller_alias_continual/20260927/ranking_implementation/authorization.json)。19份部署文件282,183字节实物核对一致，10份原依赖复用、9份新文件同步；首次校验误在SFTP队列完成前发出，因部分文件尚未完成而拒绝，入口当时尚不存在，未开始科研核验。等待全部传输完成后重新核对和检查两个入口`--help`及Bash语法，再启动唯一CPU核验。部署时序说明和成功回执均保留，没有正式文本或标签读取。

实际目录：`reports/seller_alias_continual/20260927/ranking_implementation/cpu/20260927_105450`。既有Python3.10、PyTorch和预训练BGE，CPU线程1、GPU不可见、离线模式；没有安装或修改环境。

- 15项合同全部通过，用时2.653秒；涵盖独立标量目标／解析梯度、查询等权、已知负例与并列规则、原D实际Adam等价、864步lr序列、零lr末步、微型新实例完整Adam恢复后87／88步接续、配对日程、独立频数bootstrap、13条件边界、18份手工假模型文件逐个等长损坏在标签前拒绝、全部18矩阵保全和零标签后处理恢复。
- 4次实际BGE手工更新：原D83.79秒、包装D80.17秒、schedule80.78秒、hard162.57秒；hard额外执行三个目标的独立梯度诊断，所以核验耗时不能直接当作正式C每步训练时间。四者均为326,571,265参数、24层，没有新模块。
- 原D与包装D的模型、Adam摘要、完整更新前后分数及训练前向logit逐位相等；B／C共同初态和首次前向与D逐位相等。B／C实际首步lr为1e-5/87，头为1e-3。
- 116个实际参数探针（每模型的24层query权重、词嵌入和4个头参数）均有有限非零任务梯度、参数变化及Adam步数1。hard中的BCE／rank／hard三项分别对编码器首层query权重和头hidden权重有非零梯度；这不声称每个参数每步梯度都非零。
- 实际logit对应目标与独立标量参考最大差2.338321e−7；已核对全部原始日志和四份原生记录。RSS峰值8,847,056KiB，低于可用主存；256MiB限制的是小型证据。没有保留原生模型文件，也没有正式数据、标签或GPU执行。

原始stderr的`test`命令拒绝是预定边界用例；张量转标量警告发生在手工数值断言，不改变后续梯度图；继承编码器维数API有弃用提示，当前调用成功。这些不是测试失败，未为消除提示改写冻结旧源或重复运行原生核验。

入口命令使用上述目录的`audit.json`为输出：`python -u -B scripts/step28_alias_ranking_audit.py --out <该目录>/audit.json`，外层`timeout --signal=TERM --kill-after=30s 1h`及GNU time，完整实际命令保存在资源日志。证据：[总回执](../reports/seller_alias_continual/20260927/ranking_implementation/cpu/20260927_105450/audit.json)、[主执行者独立记录复核](../reports/seller_alias_continual/20260927/ranking_implementation/cpu/20260927_105450/primary_check.json)、[完整日志](../reports/seller_alias_continual/20260927/ranking_implementation/cpu/20260927_105450/stderr.log)、[回传](../reports/seller_alias_continual/20260927/ranking_implementation/cpu_sync.json)。

## 原定CPU核验范围（已执行）

原批准范围：现有py310、单CPU、`CUDA_VISIBLE_DEVICES`为空，15项手工用例和原D／包装D／schedule／hard各一次实际BGE手工样本更新；hard另检查三个目标分别通向编码器和头的梯度。原预训练BGE仅从Linux加载，无需新下载。原估计约8—15分钟，1小时强制上限、256MiB小型证据上限；该字节上限不是主存上限。启动前实际可用主存263,201,423,360字节，磁盘696,030,126,080字节，48CPU且负载低；另一个同用户Python进程属于其他项目，没有修改或终止它。微型模型临时保存文件随用例清理，不保留原生模型权重。

核验只读代码、政策、预训练档案和手工样本；正式文本、train／valid／test标签、owners均不读，不用GPU。适用测试经审计器运行一次，没有跑历史全套；回传和独立记录核查已完成。CPU／FP32手工证据不能替代正式CUDA／BF16的7776更新、正式规模E3/E6保存重放或正式valid验收，这些仍未授权执行。

## 后续顺序与限制

60文件591,625字节包于11:12:43 CST提交网页GPT6 Pro；原自动审批拒绝、用户明确授权和提交均保留。实际回复`ed10c5f4-24d8-40e5-9d14-b0761cf601aa`、完整报告F1—F10、10项独立补测源码及全部原始日志已亲自读完并逐项对照当前计算。网页端15项当前合同＋12项继承合同＋10项独立测试共37项通过、0失败／错误／跳过，使用其Python3.13.5／Torch2.10 CPU环境；本次没有在项目Linux重复执行这些测试。60份提交来源、26份附件清单成员、当前18份CPU来源及13份CPU证据实物核对通过。完整[外审报告](../reports/seller_alias_continual/20260927/ranking_implementation/review/external/ranking_review_report.zh.md)、[独立测试](../reports/seller_alias_continual/20260927/ranking_implementation/review/external/independent_tests.py)、[主审处置](../reports/seller_alias_continual/20260927/ranking_implementation/review/disposition.json)和[核对回执](../reports/seller_alias_continual/20260927/ranking_implementation/review/primary_check.json)已归档。

外审中需要精确解释的内容已经核实：B／C的864步编码器学习率之和为0.00432，A为0.01728，比例1/4；这不是实际参数位移之比。`collect`在各矩阵落盘后仍计算轻量均值和固定分类；完整18矩阵及`collected.json`之后才做全部比较、bootstrap和验收。完整model／Adam重载后实际重放三角色全部分数；随后推理文件重载通过模型状态摘要逐位一致核验，网页微型补测另验证新实例推理分数，不能将其说成正式入口又执行了一遍推理文件后的全量评分。18权重和54分数的前置校验按各检查点交错执行，全部完成后才打开valid。上述均是证据措辞澄清，方法、代码、冻结合同及13判据未改。

用户最新启动指令已直接恢复所报正式阶段：九次新训练／7776更新、新train／valid各一次、单GPU单CPU，预计总计14—18小时，强制24小时／32GiB。先只读检查共享资源与进程归属、同步必要证据及正式授权，再启动唯一正式作业；不得自动重试或重新解析。原冻结政策中的“尚未授权”字段属于准备时点，运行器使用独立[正式授权回执](../reports/seller_alias_continual/20260927/ranking_execution/authorization.json)接续，不改写CPU已核验的18份来源。正式作业已于2026-09-27 11:50:06 CST在GPU0 RTX5090启动，单CPU核0；11:53:30实查s0_d至少24更新，训练PID3052330、启动PID3052316均属yongpeng，无失败记录。九个监督前预检完成，新train解析一次已消费，valid尚未开始，test／owners为0。预计总14—18小时，此观测时剩余约13小时57分—17小时57分，北京时间9月28日01:50—05:50完成；24小时截止9月28日11:50是强制上限，非预计耗时。依据历史BGE每模型约90—94分钟及本轮首24步，尚无完整臂／困难项／保存恢复耗时。 [实际启动证据](../reports/seller_alias_continual/20260927/ranking_execution/20260927_114646/training_started.json)。

方案只能检验固定中文合成数据、同群27候选、固定三种子下这套训练干预是否改善；不能保证通过，不能证明困难目标单独的全局因果贡献或真实市场效果。B对照支持区分学习率日程与额外困难项的观察增量，不能事后改选B为主要候选。valid反复开发的限制仍在，独立test及持续学习设计均按既定条件暂缓。
