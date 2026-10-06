# 本次审查上下文与阅读索引

2026-10-06。用户最新明确指令：“确定，一定要上传完整的上下文给网页端模型审查”。本包在此前228文件结果包上补充直接相关的研究背景、数据身份、设计、原始审查与基线资格，授权对象是本次结果审查及必要完整上下文，目的地仍为既有ChatGPT审查会话。包内旧“待许可／尚未训练”按当时时点解释，不阻断现已授权的上传，也不能重置冻结实验的权限或失败账。

## 先读这六组材料

1. **问题、边界与优先级**：docs/AI_RESEARCH_HANDOFF.zh.md、docs/RESEARCH_DISCIPLINE.zh.md及docs/RESEARCH_DISCUSSION.zh.md。判断适用对象、时点和范围，再判断优先级。最新明确用户决定在其范围优先，冻结合同和实际事实不追改。实质阻断、复现缺陷、未来建议与过度扩展分开判断；不要把外审建议自动升级为新实验许可。
2. **数据和基线为何如此**：docs/SELLER_ALIAS_EXPRESSION_DATA_RESULT.zh.md、docs/SELLER_ALIAS_BGE_CONTINUAL.zh.md、docs/SELLER_ALIAS_LOGIT_LOW.zh.md和docs/SELLER_ALIAS_LOGIT_RISK_RESULT.zh.md。LOGIT0.1原结果外审全文及主审在reports/seller_alias_continual/20261004/result_review/。本包的旧基线completion、before_valid、access、运行时记录为直接资格证据。
3. **算法要解决什么、已知什么风险**：docs/SELLER_ALIAS_RELATION_MEMORY.zh.md、其EXECUTION和PILOT合同的冻结副本；设计审查完整页面记录reports/documentation/20261004/relation_memory/external/page.json、设计主审report.zh.md及网页手算／反例原件。docs/SELLER_ALIAS_LITERATURE.zh.md是既有文献定位，按其日期使用，不能作为穷尽当前相关工作的证明。
4. **实现是否真实、修了哪些问题**：implementation中的原始网页审查、disposition、cpu_repair与gpu原生结果；pilot中的原始接入审查及disposition；20261005/relation_memory/report.zh.md及CPU实际输出、deployment/preflight/gate。P1恢复资格与继承预算、P2资源快照与停止、P3路径问题已按用户最新指令作最小修复并主审关闭；未声称网页重新执行过修后代码。
5. **本次实际执行和所有结果**：reports/seller_alias_continual/20261006/relation_result/job/、job.console.txt、job.wrapper.txt、inventory.json。冻结23来源在同目录source/；本次完成资格来自completion.json，不能仅凭evaluation.json的统计状态宣布有效完成。
6. **独立复算与主执行解释**：同目录verification.json、verification.console.txt、metrics.zh.md，scripts/step28_relation_memory_result_verify.py及docs/SELLER_ALIAS_RELATION_MEMORY_RESULT.zh.md。请独立判断，不将主执行者的“有效负结果”措辞当作审查前提。

## 研究历史中与本次判断直接相关的事实

研究目标是历史原始训练记忆有限、新群体依次到来时，保持旧分布未见卖家的候选排序与基础识别，同时学习新群体。只用屏蔽后的标题／描述。当前是中文合成主题混合与表达机制共同变化；不是纯表达变化因果实验、真实市场验证、跨语言方法或同一控制者跨阶段别名追踪。每群28账号、378无向对、20正358负，每查询27候选且有正候选。

每域48 fit／12 calibration／20 valid。valid已经开发，既有120群基础test已经使用；本次没有新最终留出。Audit A/B真值、owners及private_custody未开放。训练阶段看到当前fit与存活历史记忆，完整盲门后才解析valid标签，各一次技术访问不等于反复按结果搜索的许可。

此前BGE顺序训练存在遗忘，简单重放过强会影响新域学习，LOGIT0.1在旧域上有明显收益但并非对SEQ所有旧判据全过。risk固定点有效负结果已关闭；group-meta因当前完整二阶实现资源不可行被用户明确关闭，正式效果未知，不能借本次审查重启。更早历史只用于理解主线与证据边界，不把历史局部收益移作本候选收益。

## 固定关系目标记忆的科学对象

本候选d32、epsilon=.001，当前与历史目标权重各1，完整查询平方识别＋排序，关闭Dropout。H/b/c/N为累计监督二次目标摘要，六群缓存的旧X与当前Y拟合岭迁移，历史目标经可微求解对当前模型产生梯度；仅有一阶反传，不是完整二阶元学习。前两阶段固化与共同参考坐标刷新，第三阶段无后继用途而不新固化。原始记忆六群Algorithm R，全部附属状态合计≤1MiB，不作域均衡修正。

必须区分：固定表示下的二次目标精确归约，不能推出变化表示下的历史目标完全精确。既有审查确认输出投影盲区、近似迁移误差、逐步单群映射与阶段末联合映射的差异；这些是保证的边界，不是已证实本次效果下降的因果解释。平方排序目标也会约束类内分数离散和超过目标间隔的差异，其与MAP目标的取舍尚未被单独归因。

头、损失和Dropout与LOGIT0.1不同，因此候选必须重训自己的首阶段，不复用旧首域权重。比较只复用基线已开放的矩阵与计数；基线末端权重之前已获准删除，不可要求假装加载已不存在权重。本次s0三顺序共2592更新；三个顺序不是三个种子。

## 当前结果与解释边界

正式10月5日15:06:32—18:57:22，完整2592更新、九端点、28套结果，访问账及资源均通过。六项继续判据0/6，O/N/Z MAP与AP差区间均低于零；本配置不进入匹配重放对照，不调系数／种子／主端点翻转结果。还需保留：G本身为正；相对基线F_first/F/G的MAP差区间跨零；CAB首域相对自身获得时表现上升；旧域固定0.5工作点召回与合并F1略高，但精确率降低、误报显著增多。核心贡献尚未成立，配置失败不等于整个方法家族无效。

## 如何使用附件，哪些边界不是证据缺失

- 文件以项目相对路径打包。source/内是实际训练用23份冻结来源；根scripts/中的新核验脚本是训练完成后的独立分析，不属于当时训练来源。不要拿新脚本加入旧来源集合后判旧manifest不一致。
- 固定比较只需要基线九个端点的三角色矩阵／计数，共54个文件，以及两份collected.json。旧collected列有更宽历史点，未附无关旧数组不影响本次比较。当前候选28套为56个矩阵／计数文件。
- 正式文本、逐对真值、记忆二进制和权重不上传；本地Linux核验只哈希缓存／权重，不重新读取真值或载模。本包保留其身份和运行证据。因此网页可以独立重算保存矩阵统计，却不能在网页完整重跑包含缓存哈希的Linux核验脚本，亦不能从盲分数重新计算需要标签的MAP。这是权限和证据范围，不应伪称完整原生重跑。
- 核验脚本的workspace参数指原独立工作区，reference.linux_root在网页环境不天然存在；网页复算请明确映射到包内reference.local_root，不修改冻结产物。若构建临时路径视图或只运行矩阵统计部分，保留修改及实际覆盖说明。
- 包内原数据与旧外审文件均按历史身份保留；主报告/sync中的“结果外审尚未提交／待授权”是本包制作时快照。本次用户授权原话已在本页说明，提交状态由包外submission记录实际发生时间，不要求为该历史措辞改写科学原件。
- 审查时可核对封装清单及哈希；这些只能证明材料身份，不能替代对公式、输入、实际更新、配对和结论的判断。不要为不影响本次结果的建议追加训练、全面回归或新管理框架。
