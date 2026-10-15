# 当前实验设计入口

本页索引合同及当前待审静态提案，执行状态与授权补充见[当前交接](AI_RESEARCH_HANDOFF.zh.md)。合同记录该轮形成时的设定；其中“待批准”“保留权重”等措辞须结合后续明确决定解释，科学公式、访问、预算和原判据不追改。

[C_q实现与执行合同](SELLER_ALIAS_QUERY_REPLAY.zh.md)及获准的一次原生补核已执行完毕，固定四臂于2026-10-13 22:55:47正常完成。[原静态方案](../reports/documentation/20261013/query_replay/proposal.zh.txt)冻结，[主审补正](../reports/documentation/20261013/query_replay/review/primary_disposition.zh.txt)中的信息损失、偏置、条件控制与共享状态要求不变。10月14日[保存结果分析](../reports/seller_alias_continual/20261014/query_replay_result/report.zh.txt)未建立相对完整S的排序收益；四比较八个A/B MAP区间均跨零，概率代价和训练—valid差距分别保留，不推成等价或整个方法家族失败。结果外审已完整回收并经[独立主审](../reports/seller_alias_continual/20261014/query_replay_result/review/primary_disposition.zh.txt)关闭；24个未定义导出零值按N/A解释，宏／汇总分类及证据边界已补明，正式数值和原判据不变。五权重已按本次具体批准留证删除。原生累计11步／469秒与失败账不清零，旧C诊断不恢复；实际状态及授权只看[当前交接](AI_RESEARCH_HANDOFF.zh.md)。

已结束的[C_cf合同](SELLER_ALIAS_CROSSFIT_RETENTION.zh.md)固定完整S／C_cf／C_perm、s0 A→B于10月12日19:53:39正常完成；[结果](../reports/seller_alias_continual/20261013/crossfit_retention_result/report.zh.txt)未建立C_cf相对完整S的排序提升，也未建立对一次可行错配的优势。[独立主审](../reports/seller_alias_continual/20261013/crossfit_retention_result/review/primary_disposition.zh.txt)已关闭，补齐训练—valid差距及原判据解释，无当前科学／复现阻断。冻结原稿、实施审查、资格及失败账保持；已批准运行范围不转成新候选许可。

| 合同 | 研究内容及角色 |
|---|---|
| [完整S／G／F合同](SELLER_ALIAS_BALANCED_POOLING.zh.md)与[已批准修复补充](SELLER_ALIAS_BALANCED_POOLING_AMENDMENT.zh.md) | 原正式F/B求解失败，失败账与valid未开保留；F失败静态外审／独立主审后用户批准补充，固定4096次终端Q验收且原输出精度不变。最小实现及Linux CPU增量已完成，第二轮四项通过；具体包已获许可并提交合同与实际实现合并外审，等待完整回收／独立主审，见[实现报告](../reports/documentation/20261015/balanced_pooling_repair/report.zh.txt)。旧八次原生来源继承与本次固定三次F的入口已落实，后续仍须合并审查、原生、新投影和资源前提，不据CPU通过启动正式；费用与权限只看当前交接。 |
| [双边覆盖的记录匹配评分](../reports/documentation/20261014/balanced_pooling/proposal.zh.txt) | 2026-10-14新静态性能候选，静态外审及独立主审已关闭，建议进入实施合同准备；尚未实现／执行，不是合同。均匀双边际熵正则最优值替代均值，完整S与去边际G为拟议对照；数学作用和错误匹配反例已列，OT先例明确，效果及独立贡献未知。H按[主审补正](../reports/documentation/20261014/balanced_pooling/review/primary_disposition.zh.txt)解释为相对G的集中限制与相对S的差异化奖励；数值／接线／资源细节现由上行已批准合同和当前实现承接；本行保留静态审查时点，不用旧包许可覆盖新上传。 |
| [C_q实现与执行合同](SELLER_ALIAS_QUERY_REPLAY.zh.md) | 已完成固定运行及结果外审／独立主审的执行依据，无未关闭科学／复现阻断。完整S／Q／C_q／C_b，共同A＋四支B，1440更新／2592梯度群呈现、5端点／30诊断；CPU300秒、原生600秒／原6步（后获一次补核、累计最多11步）、正式16小时／18GiB、标签各一次及统一盲门。原生累计11步／469秒，原失败不清零；[实现主审](../reports/documentation/20261013/query_replay_implementation/review/primary_disposition.zh.txt)按未变范围复用。原合同不授权重训或根据结果调参。 |
| [自然查询锚定的历史重放C_q](../reports/documentation/20261013/query_replay/proposal.zh.txt) | 静态外审／[独立主审](../reports/documentation/20261013/query_replay/review/primary_disposition.zh.txt)已关闭，当时推荐合同准备，原稿不是执行合同。只改候选历史分支的查询条件分配及额外中心化响应MSE，保留完整S对照、当前S及0.5D0。整个历史目标的异来源交互盲区、R偏置／幅度和bootstrap条件控制按补正解释；Q/C_q共享增强规则键及完整共同A。稿内四臂为形成时未执行提议，后续实际执行／结果以本表上行和交接为准，不追改冻结原稿。 |
| [完整S贡献定位与机制缺口](../reports/documentation/20261013/s_contribution/report.zh.txt) | 当前静态研究报告，不是执行合同。核对实际S、归因实验和直接近邻；判断基础系统价值与独立创新是否充分，明确自然账号内部结构与同控重组的待研究问题。不新增训练、数据读取或旧C诊断，当前审查状态见交接 |
| [C_cf执行合同](SELLER_ALIAS_CROSSFIT_RETENTION.zh.md) | 已结束固定实验的执行依据。完整S／C_cf／C_perm、A→B、1440更新／2304梯度群呈现；CPU300秒、原生600秒、正式16小时／16GiB，标签各一次及统一盲门。原生失败及一次补核例外、完整结果审查与权重清理按交接和回执解释，旧静态稿中的待批准草案不恢复成当前阻断，也不重授训练权限 |
| [C折外支持的历史排序间隔保持](../reports/documentation/20261011/retention_design/crossfit.zh.txt) | 静态外审／[独立主审补正](../reports/documentation/20261011/retention_design/crossfit_review/primary_disposition.zh.txt)已完成，按有限增量策略推荐继续准备。完整群折外来源与全域来源共同支持的单边正负间隔，替换候选自己的D0，完整S不变。S／C_cf／可行错配C_perm能比较系统替换及原赋值相对本次错配，不能证明折外信息的独立必要性；已知cross-fitting／排序KD构件不冒称首创。拟议A→B、1,440更新／2,304梯度群呈现；新目标接线、Memory诊断、A/B评价、共享A和完整预算投影待落实，访问与12小时／16GiB等未获具体批准。尚未实现／运行或建立效果；[完整回收状态](../reports/documentation/20261011/retention_design/crossfit_review/submission.json) |
| [C跨账号条件配对保持候选](../reports/documentation/20261011/retention_design/matching.zh.txt) | 静态外审／[独立主审及补正](../reports/documentation/20261011/retention_design/matching_review/primary_disposition.zh.txt)已完成，本版本不推荐进入实现或四臂训练；数学主体成立，但结构化KD及U-centering已有、任务信息选择和独立贡献不足，效果未知。原稿保留，单合格视图9维、二阶式、反例及完整S更新边界以主审补正为准；不是新实验负结果，完整S仍为对照，旧C单独诊断不恢复 |
| [C联合排序响应收缩保持方案](../reports/documentation/20261011/retention_design/contraction.zh.txt) | 静态外审／[独立主审与补正](../reports/documentation/20261011/retention_design/review/primary_disposition.zh.txt)已完成，本版本不推荐实施；有限数学成立，但弱OT适配的任务信息依据与实质贡献不足，效果未知。四臂仅为未执行的固定策略比较；原稿保留，低维软约束、子包粒度、归因及文献勘误以主审为准。不是执行合同，未关闭整个C研究，完整S仍为主要内部对照 |
| [C保持约束局部更新诊断](../reports/documentation/20261011/retention_design/diagnostic.zh.txt) | 曾获准的完整S单路径及四条件撤回干预；仅完成手写CPU首轮。按用户最新条件及主执行者判断停止继续推进；原范围、源码与证据保留，不自动启动GPU／正式流程。处置见第10节及交接顶部 |
| [记录结构与重组监督三臂归因](SELLER_ALIAS_RECORD_ATTRIBUTION.zh.md) | LOGIT0.1／R0／S分别判读O/N MAP收益与代价，无新总通过门。三臂独立训练首阶段，保留共同过拟合诊断；原时间上限已有[明确补充](../reports/documentation/20261009/record_attribution/qualification/authorization_72h.json)，有效预算、资格与执行状态只看当前交接及对应回执 |
| [五臂改进与共同过拟合诊断](SELLER_ALIAS_REPLAY_IMPROVEMENT.zh.md) | 已结束并完成结果外审／主审；C_plus本固定配置负结果、原继续条件false，LOGIT历史复现缺陷R1及比较边界保留。用户已明确撤下两端候选／五臂活动实现，保留原C/S、LOGIT、通用诊断、冻结方案和证据；[关闭记录](../reports/maintenance/20261009/replay_improvement_closure/closure.zh.txt)，实际状态见当前交接 |
| [记录作用表与同控重组 C/S](SELLER_ALIAS_RECORD_REPLAY_PILOT.zh.md) | 已结束的开发实验；记录对头、同架构 C/S 对照及保存 LOGIT0.1 比较。训练、保存结果独立核验、结果外审及主审已完成；整体方案相对LOGIT有正向证据，C−S原主要条件未过，证据见交接 |
| [函数目标累计关系记忆](SELLER_ALIAS_FUNCTION_MEMORY_PILOT.zh.md) | 已结束的固定配置开发实验；[结果](SELLER_ALIAS_FUNCTION_MEMORY_RESULT.zh.md) |
| [关系目标记忆修改版](SELLER_ALIAS_RELATION_REVISION_PILOT.zh.md) | 已关闭，活动修改版已按用户决定清理；[结果](SELLER_ALIAS_RELATION_REVISION_RESULT.zh.md) |
| [关系目标记忆原版](SELLER_ALIAS_RELATION_MEMORY_PILOT.zh.md) | 已结束的固定配置开发实验；[结果](SELLER_ALIAS_RELATION_MEMORY_RESULT.zh.md) |
| [群体风险重放](SELLER_ALIAS_RISK_PILOT.zh.md)与[匹配 LOGIT0.1](SELLER_ALIAS_LOGIT_LOW.zh.md) | 已完成训练及结果审查；LOGIT0.1为后续保存结果比较基线，[联合结果](SELLER_ALIAS_LOGIT_RISK_RESULT.zh.md) |
| [BGE首轮持续学习](SELLER_ALIAS_BGE_CONTINUAL.zh.md) | 已完成问题诊断及SEQ／ER1／LOGIT1；后续仅继承各自合同明确适用的数据、训练和评价设置 |
| [ER0.5／0.25](SELLER_ALIAS_ER_WEIGHT.zh.md) | 已完成有限权重比较；[结果](SELLER_ALIAS_ER_WEIGHT_RESULT.zh.md) |
| [ER0.1单点](SELLER_ALIAS_ER_LOW.zh.md) | 已完成，主比较对0.25，完整报告对SEQ；[结果](SELLER_ALIAS_ER_LOW_RESULT.zh.md) |
| [匹配LOGIT0.25](SELLER_ALIAS_LOGIT_WEIGHT.zh.md) | 已完成，主比较匹配ER0.25，另报SEQ；[结果](SELLER_ALIAS_LOGIT_WEIGHT_RESULT.zh.md) |

新方法、追加对照或种子、新计算／数据权限及独立最终留出须另定具体范围。已完成合同不自动授权重训，保存结果比较不等于仍有旧权重。常设边界见[科研纪律](RESEARCH_DISCIPLINE.zh.md)，历史设计及资源未准入的群体元学习路线见[历史入口](AI_RESEARCH_HANDOFF.zh.md#历史与证据)。
