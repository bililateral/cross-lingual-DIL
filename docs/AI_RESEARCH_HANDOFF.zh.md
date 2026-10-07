# 当前科研交接

**最新研究候选（2026-10-07 16:06，方案外审已提交，等待审查；未实施）**：用户要求“重新开始吧，如果还不行就继续找下一个直至可行”后，主执行者形成[历史风险摘要与独立残差重放完整稿](../reports/documentation/20261007/residual_replay/proposal.zh.txt)：来源阶段压缩完整旧目标的局部二次响应，以独立原始群重放真实目标减近似目标，固定来源坐标、活跃锚点解码、不用BGE二阶计算。拟改变全局Algorithm R为每来源锚点/残差随机保留，仍六群/1MiB；每步三群对原LOGIT两群，因此要求同缓存同计算的LOGIT强对照。主执行者只推荐先作严格方案外审，创新、原生资源和效果均未确立；固定θ抽样恒等式不是适应性训练无偏保证。用户先答复“上传外审”，随后对具体包与目的地明确“授权将此review_input.zip上传到chatgpt.com进行方案外审”。16:05:51已通过本地Playwright发送原806,573字节／114成员包，网页可见6／Pro，16:06:18实见正在思考；[审查会话](https://chatgpt.com/c/6ac5fd5a-2e6c-83eb-b374-3628d1a65e26)、[实际提交记录](../reports/documentation/20261007/residual_replay/submission.json)及原请求／页面证据已保存。包内旧“尚未授权／待上传”为制作时快照，不重问本次许可；新许可仅用于本次静态方案外审，实现与新计算仍未授权。专用页保持打开，独立记录工作完成后等待，不轮询或新增监控。旧已关闭候选／负结果保留；若新候选不成立，继续原范围研究，不以候选关闭结束任务。本轮未连接Linux、未读新原始数据／标签／Memory、未执行新科研数值或修改科学代码。

**当前方案审查（2026-10-07，外审与主审已关闭，不推荐推进）**：用户告知“外审已结束”。本地Playwright已回收“同控组保持的组合重放，并对完整历史目标作精确边际化”的[完整外审](../reports/documentation/20261007/compositional_replay_review/external/review.zh.txt)及阅读范围，专用页已关闭。主执行者接受“现阶段不值得推进”，撤回此前“值得进入受限验证”的推荐：主要数学成立，但结构化组合风险／条件期望／排序计数的既有基础明确，精确积分相对同池MC32的必要性与未见身份收益未建立。方案未实施，不记作实验负结果或资源失败。原包96成员／95载荷身份一致，但遗漏当前表达生成的直接合同、policy与源码绑定，原“完整背景”表述过强；主审已补读并核对冻结来源，确认当前覆盖按账号／轴抽取，外审群级共因反例不能当作已发现事实。教师跨臂比较定义及JSON／FP32字节差额两项勘误接受；补齐三处仍不足以翻转创新与投入价值判断。详见[独立主审处置](../reports/documentation/20261007/compositional_replay_review/report.zh.txt)、[回收与来源核对](../reports/documentation/20261007/compositional_replay_review/recovery.json)、[实际状态](../reports/documentation/20261007/compositional_replay_review/submission.json)。原送审稿、外审原件和旧结果不追改；本轮无新科研代码／数值实验／数据标签读取／Linux连接。用户持续寻找合格方案的要求有效，论文算法仍未确定；允许继续原范围文献与数学研究，不自动实施MC32或申请本候选训练，后续具体新方案仍须用户与外审双审及新合同。

**最新方向纠正（2026-10-07）**：用户明确“我要的不是你在logit 0.1这个基线上简单堆砌的算法，你懂我意思吗？不懂就问？”主执行者撤回完整LOGIT0.1+βR候选，停止其原设计审查并向既有网页会话发送纠正；原草案/包仅保留历史，不进入实现或训练。用户此前允许修改核心仍有效，但要求围绕有限历史信息、新群体学习和旧分布未见卖家保持形成有实质贡献的完整机制。LOGIT0.1是比较对手，不把基线加常规损失或模块组合作为创新；共同编码器/优化器等可合理复用。具体新方案仍须用户与外审双审，新计算/数据权限未授予。见[纠正原文](../reports/documentation/20261007/rank_protection/correction.zh.txt)；网页仅收尾保存已有材料，不继续为已否决候选论证准入。
**当前研究（2026-10-07，函数目标累计关系记忆完成，结果负向）**：20261006_231800于10月6日23:19:15—10月7日02:45:15完成，exit0、12360秒，剩余0；09:45:59实查原作业进程已退出。1728新更新/3456群梯度呈现/9逻辑点/28指标计数集，完整盲门，train/valid各1、heldout/owners0，无正式失败账。171份非权重结果及23冻结来源已回传逐项大小SHA匹配；Linux保存矩阵独立核验通过（1.26秒，无载模/标签重读），本次分析脚本首次括号语法错误0.02秒的日志保留。五条件0/5，O/N/Z MAP .425195/.451009/.450481，相对LOGIT0.1差−.035606/−.016022/−.018508，区间均负；final_all MAP .433623、差−.029906。共享首域矩阵逐值相同，A2已落后且后续遗忘扩大，不能归因为起点弱。详见[完整分析](SELLER_ALIAS_FUNCTION_MEMORY_RESULT.zh.md)、[全指标表](../reports/seller_alias_continual/20261007/function_memory_result/tables.zh.md)和[回传清单](../reports/seller_alias_continual/20261007/function_memory_result/inventory.json)。用户随后明确“训练权重一起删掉”，九权重已于10:04:38核对后删除，逻辑释放11758264524字节，171份非权重SHA及23项保留例外元数据未变，见[回执](../reports/maintenance/20261007/function_weights/receipt.json)；Memory完整904979—925109字节。结果外审10:30回收，39成员／38载荷大小SHA一致；10:32主审核对外审与原评价9,504数值，最大差4.44e-16，另从保存counts独立确认O空预测群贡献10/120→64/120（N28/120→53/120，Z7/60→29/60）。外审及主审接收有效配置级开发负结果，无当前科学阻断或必须修复的复现缺陷，五条件仍0/5；[主审处置](../reports/seller_alias_continual/20261007/function_memory_result/review/report.zh.md)及[实际状态](../reports/seller_alias_continual/20261007/function_memory_result/review/submission.json)已保存，网页专用页关闭。本点科研收尾完成，不自动调参/重训/恢复，不建周期任务；新方案另立具体范围。当前活动源码保留，原版和旧负结果/清理事实不改。
**当前决定（2026-10-06 19:53，修改版已关闭并清理）**：用户明确要求删除修改版代码、保留原版和记录，并删除此次训练权重。Windows活动scripts/tests/schema恢复至修改前75e2fc22：删除10个修改版文件、恢复1个共用结果核验脚本，9份原版相关来源SHA一致；Linux删除34个修改版活动／部署文件，原版保留。20261006_135937九份阶段权重已于19:53:30—19:53:31删除，实际逻辑释放11758415337字节；315项科研证据SHA不变，23项保留例外元数据不变，九份Memory留作历史记录。详见[关闭记录](../reports/maintenance/20261006/relation_revision_closure/closure.zh.md)、[逐文件回执](../reports/maintenance/20261006/relation_revision_closure/receipt.json)。冻结源码和审查包仅为历史记录，不作为活动实现。原版关系目标记忆及LOGIT0.1基线代码、原结论保留；不重训、不恢复关闭路线。修改版正常完成13808秒、2592更新／9端点／28套结果，对LOGIT0.1六条件0/6，详见[原分析](SELLER_ALIAS_RELATION_REVISION_RESULT.zh.md)。结果外审于20:21回收，66文件／65载荷大小SHA全部匹配；外审接受有效配置级开发负结果、无当前阻断，见[回收说明](../reports/seller_alias_continual/20261006/relation_revision_result/review/recovery.zh.md)。随后已完成[独立主审处置](../reports/seller_alias_continual/20261006/relation_revision_result/review/report.zh.md)：复用原生保存矩阵核验，并比较原统计与外审独立统计8316数值、最大差4.44e-16，结果审查关闭；用户最新清理指令结束权重加载／恢复依赖，不再以等待外审作为保管条件。旧后台跟进保持停用。此前保留权重和推进修改版的文字仅是历史状态。

**原版历史状态（2026-10-06，关系目标记忆已完成，固定点负结果）**：10月6日09:35:38实查正式进程已退出；运行实际于10月5日15:06:32—18:57:22完成，退出0、耗时3小时50分50秒、剩余0。2592更新／9端点／28套完整指标计数，盲门前valid0、最终train/valid各1，heldout/owners0，无失败账。165份结果／源码回传大小与SHA全部一致；Linux单CPU禁GPU的保存矩阵独立核验通过，均值／分顺序值／配对区间与原统计一致，没有重读标签或载模。见[完整分析](SELLER_ALIAS_RELATION_MEMORY_RESULT.zh.md)、[同步状态](../reports/seller_alias_continual/20261006/relation_result/sync.json)。

独立workspace为`reports/seller_alias_continual/20261005/relation_memory_execution/20261005_150520/workspace`，job为其`reports/job`。旧PID3359214/3359221为历史身份，禁止重复启动。最终reserved峰值约8.725GiB、RSS约6.426GiB、产物观察峰值约14.602GiB，均低于合同上限；原12—18小时预估偏保守。九份权重已于10月6日10:39:18按授权清理，详见下段；09:35时点保管记录为历史。此前入口P1/P2/P3关闭及原生证据仍按[修复处置](../reports/documentation/20261005/relation_memory/report.zh.md)复用。

**结果审查关闭、九份权重已清理（10月6日）**：10:32:49实见外审结束，46成员证据包45载荷全部匹配；[主审处置与勘误](../reports/seller_alias_continual/20261006/relation_result/review/report.zh.md)已关闭三项展示／措辞问题，无科研阻断。保留原送审报告及冻结结果。按用户“完成后及时按照科研纪律删除此次试验的训练权重并做记录”和“授权后台跟进及条件满足后的清理”，在收尾及依赖核对后于10:39:18删除独立job/run/models下九份本次权重，合计11,758,382,199字节；165份非权重证据SHA不变，23项保留例外元数据不变。见[逐文件删除回执](../reports/maintenance/20261006/relation_weights/receipt.json)。预训练、共同首域完整状态、ER0.1和两份排序例外继续保留；不改负结果，不追加训练。后台跟进将在本轮同步／Git交付完成后停用。

沿用[固定合同](SELLER_ALIAS_RELATION_MEMORY_PILOT.zh.md)，对LOGIT0.1六项继续条件为0/6。O/N/Z MAP分别0.421179/0.439300/0.443666，差值-0.039621/-0.027731/-0.025323，三者条件95%区间均低于零；O/N/Z AP也有负向证据，Brier/log-loss均退化。保留CAB首域局部改善与旧域固定0.5工作点召回略升、误报增加等细节，不能写成所有单项都变差。按规则不启动匹配重放、调参或新训练；不将配置负结果升级成整个方法家族无效。用户10月6日明确“确定，一定要上传完整的上下文给网页端模型审查”。已补为272成员／5,077,487字节完整上下文包，10:04:52通过本地Playwright提交既有会话，选择器6／Pro，10:05:03实见正在回应；[提交证据](../reports/seller_alias_continual/20261006/relation_result/review/submission.json)。本轮外审及主审现已关闭，裁决接收有效配置级开发负结果；包内旧待许可／待提交／待审文字保留为制作时快照，不重问已确认许可。当前LOGIT0.1仍是强基线，原冻结输出不改。
**最近关闭路线（2026-10-04，用户明确回退）**：完整群隔离泛化重放（group-meta）整条候选路线已关闭，标记为 **资源受限不可行**。回退范围包括原始完整二阶、GPU直接／CPU暂存、分阶段HVP和JVP路径；取消候选先导、四条件机制验证及尚未提交的JVP外审，不继续修补、试跑或正式训练。具体边界与回退核对见[关闭记录](../reports/documentation/20261004/group_meta/closure.zh.md)。历史合同、验证计划、源码快照及已发生审查／失败原件仅供回溯，旧“下一步／待授权”均不再生效。

活动 `scripts/`、`tests/` 恢复到首次引入该候选之前的 `fd9ac726` 状态。当前科研起点恢复为既有基线及其结果：**LOGIT0.1保留为强基线，risk固定点的有效负结果保留，论文最终算法尚未确定**。不恢复已删除探索草稿或权重，不重跑既有实验；后续设计仍须围绕有限历史记忆下的新群体学习，以及旧分布未见卖家的候选排序和基础识别。

资源结论限定为当前服务器、已批准上限与本次完整二阶实现。小型CPU手写BGE曾成功完成元更新，但更大手写GPU形状的直接、CPU暂存、分阶段及JVP路径均未通过资源准入；正式训练从未启动，算法效果未知，不能记为效果负结果或否定整个元学习家族。CPU核验累计551.47秒；后续GPU核验与尝试累计151/3600秒，失败账保留，剩余额度不继续使用。最新关闭、同步与无活动作业核对见关闭记录。

LOGIT0.1／risk的结果外审和主审均已关闭，详见下方“最新结果状态”。以下运行、授权及旧“当前”文字按其日期作为历史记录阅读；本页此处的最新用户决定优先。

## 研究问题与目标

目标为具有实质方法贡献的CCF B／C期刊或会议投稿；刊会与期限未指定，不阻塞当前已批准工作。研究前提已确认：历史训练数据可用量有限，新卖家群体按分布分批到达，研究旧分布未见卖家的候选排序／基础识别保持和新群体学习。当前是中文合成受控任务，输入只含屏蔽标题／描述，无身份特征；不研究同一控制者跨阶段新开别名追踪，不转向一般作者识别或AI文本检测。

当前每群28账号，每查询27候选且有同控候选；结论不能直接外推到全市场、无正候选查询或真实中文市场。先完成本主线，再决定扩展范围。基础排序改进、已知LOGIT适配和调λ均不自动成立方法创新。

用户10月3日在否决先前方案并继续讨论后，明确要求实现[群体风险分布保持重放](SELLER_ALIAS_RISK_REPLAY.zh.md)，完成后交网页端审核。该新指令取代本方案此前暂不上传的限制；当前范围为算法实现、必要Linux手写CPU核验和网页外审。新工作不修改LOGIT0.1活动合同，不授予正式数据／标签访问或新GPU训练，创新性和未见卖家效果仍待审查／实验。算法及完整阶段API已实现；Linux CPU46于17:15:00—17:20:27完成7项核验和3次原生BGE更新探针，全部通过，17份证据回传匹配。见[实现与核验](../reports/documentation/20261003/risk_replay/implementation.zh.md)。用户随后明确“那现在让网页端审核吧”；17:59:11已通过本地Playwright上传并发送同一51成员包，网页[审查会话](https://chatgpt.com/c/6ac0d1ea-764c-83e9-a768-254fa9aed2ad)已经完成。18:50:27回收的291成员原包全部290载荷大小／SHA匹配；[外审全文](../reports/documentation/20261003/risk_replay/external/risk_audit/deliverables/report.zh.txt)与[主审处置](../reports/documentation/20261003/risk_replay/report.zh.md)已保存。当前生产实现未发现未关闭计算错误，但单向分位平方泛函已有直接先例，创新尚不充分；合法反例证明三代理风险均改善仍可MAP下降、误报上升。原7项测试有阶段γ路由和同文本群参考错配两处盲点，网页独立检查已覆盖当前代码，项目回归尚待纳入。建议保留核心、先补最小检查与正式装配，再作待确认的单点配对开发实验；不是新训练许可。本轮没有修改算法或新启训练，详见[审查状态](../reports/documentation/20261003/risk_replay/submission.json)。

## 接手顺序与信息地图

**最新结果状态（10月4日）**：用户要求同步两个实验结果并按科研纪律分析。LOGIT0.1于10月3日16:01:03—21:23:48完成，risk于10月3日21:23:46—10月4日02:43:55完成，正式退出均0；各1728更新、六终点、完整盲门、train/valid各一次，heldout/owners为0。10月4日10:02:41实查无项目活动训练。risk监听21:29:31确认真实更新后自删，02:43:55退出0；禁止重复启动。完整小结果及37/32份冻结来源已回传并逐项匹配，Linux保存结果独立核验均通过，权重12份仍留Linux。本轮未新增训练、标签解析或载模。

[联合分析](SELLER_ALIAS_LOGIT_RISK_RESULT.zh.md)：LOGIT0.1对ER0.1和LOGIT0.25均23/23，对SEQ20/23；旧域MAP有正向证据，N AP、Z MAP/AP观察保护未满足。risk对LOGIT0.1、ER0.1、SEQ、LOGIT0.25分别4/23、5/23、7/23、4/23；相对LOGIT0.1的O/N/Z MAP均有负向证据，此固定点未达目标。有效性、具体效果、原验收与创新边界分开。用户10月4日回复“可以”，专项授权9,639,622字节、729成员的联合结果包。10:50:59已通过本地Playwright上传并发送，网页可见6/Pro，本次外审现已结束且主审接受；见[实际提交](../reports/seller_alias_continual/20261004/result_review/submission.json)及[审查会话](https://chatgpt.com/c/6ac1bf0d-9950-83ea-8edd-0836a7ee7486)。打包时的“待授权”保留为历史快照，当前无需再次询问本包许可。直接证据见[LOGIT回传](../reports/seller_alias_continual/20261004/logit_low_result/inventory.json)、[risk回传](../reports/seller_alias_continual/20261004/risk_result/inventory.json)及各verification/audit。10月4日通过本地Playwright回收116成员外审证据包，115载荷逐项匹配；[外审全文](../reports/seller_alias_continual/20261004/result_review/external/unpacked/external_review.zh.txt)及[主审处置](../reports/seller_alias_continual/20261004/result_review/report.zh.md)已保存，无未关闭科学阻断或交付复现缺陷，两固定点科研收尾完成。主审从保存计数核对15行工作点及6组诊断UID，采纳误报／召回取舍、探针通常不同群和权重核验边界三项补充；主要结论及原判据不变。网页已关闭，本轮未新增训练。新增方法／系数／种子／数据另立范围；同步与Git仍须分别核验。

原执行授权、监听修正及原生CPU核验继续见[固定合同](SELLER_ALIAS_RISK_PILOT.zh.md)、[接入主审](../reports/documentation/20261003/risk_pilot/report.zh.md)。LOGIT0.25六权重已于10月3日19:04:45按用户指令删除，7,838,884,668字节，179份非权重证据不变，见[删除回执](../reports/maintenance/20261003/logit_quarter_weights/receipt.json)。10月4日用户在了解具体范围和依赖后明确“那就删”；11:13:39已删除本次LOGIT0.1／risk共12份末端推理权重，15,677,742,102字节，511份非权重证据SHA不变，保留例外元数据不变。此更新取代上文10:02时点的12份保管状态；清理依据是已结束且无当前依赖，不改变实验有效性、正负结论或外审状态。共同首域完整状态、BGE预训练、ER0.1六权重和两份排序例外保留。详见[本次清理回执](../reports/maintenance/20261004/logit_risk_weights/receipt.json)及[存储记录](STORAGE.zh.md)。

先读本页和[科研纪律](RESEARCH_DISCIPLINE.zh.md)，建立问题、当前授权、历史结论与未决事项的全局认识；再沿下表进入本次任务的合同、实际代码、直接依赖和原始审查。按本轮变化与证据缺口确定[最小核验及完成条件](RESEARCH_DISCIPLINE.zh.md#5-实现核验与证据复用)，证据充分即推进交付。历史结论按需回到原件，不因换Agent重读全部历史、重跑已验证工作或扩大审查范围。任何读取仍受数据与结果权限限制。

| 要掌握的信息 | 原始入口与使用边界 |
|---|---|
| 为什么研究、与已有方法重叠何处 | [研究前提和用户决定](RESEARCH_DISCUSSION.zh.md)、[文献定位及原审查](SELLER_ALIAS_LITERATURE.zh.md)；文献整理有日期，新主张需补最新相关证据 |
| 早期真实来源、身份线索和拆分如何形成 | [schema](STEP1_SCHEMA.md)、[隔离](STEP2_SPLIT_AND_LEAKAGE.md)、[画像](STEP3_SELLER_PROFILE.md)、[silver候选](STEP4_SILVER_CANDIDATES.md)、[复核与冻结](STEP5_REVIEW_AND_FREEZE.md)；属于历史来源链，不等于当前合成数据或真值读取许可 |
| 当前数据是什么、域到底改变什么 | [9月10日生成结果](SELLER_ALIAS_EXPRESSION_DATA_RESULT.zh.md)、[首轮合同第2节](SELLER_ALIAS_BGE_CONTINUAL.zh.md#2-数据和逐阶段可见范围)；中文合成主题混合与表达机制同时变化，不能称纯表达因果实验 |
| 当前数学定义、运行配置和评价 | [合同索引](CURRENT_EXPERIMENT_DESIGN.md)、[首轮模型与评价合同](SELLER_ALIAS_BGE_CONTINUAL.zh.md)、[ER0.1 policy](../schema/step28_er_low_policy.json)、[LOGIT0.25 policy](../schema/step28_logit_weight_policy.json) |
| 实现如何走到更新和结果 | [首轮实现说明](SELLER_ALIAS_BGE_IMPLEMENTATION.zh.md)、[共享模型／记忆](../scripts/step28_bge_continual.py)、[加权更新](../scripts/step28_er_weight.py)、[续训编排](../scripts/step28_er_weight_run.py)、[保存结果评价](../scripts/step28_er_weight_evaluate.py)；本地后继版本不能冒充ER0.1活动源码 |
| 审查真正说了什么、哪些已关闭 | 最新联合结果[外审原文](../reports/seller_alias_continual/20261003/result_review/external/REVIEW.zh.md)及[主审](../reports/seller_alias_continual/20261003/result_review/report.zh.md)；实现阶段[ER0.1原文](../reports/documentation/20261002/er_low/review/external/REVIEW.zh.md)及[主审](../reports/documentation/20261002/er_low/review/report.zh.md)、[LOGIT原文](../reports/documentation/20261002/logit_weight/review/external/REVIEW.zh.md)及[主审](../reports/documentation/20261002/logit_weight/review/report.zh.md)继续复用；附件和失败修订从各主审记录回溯 |
| 当前到底能执行什么、实际用了多少 | 下方状态表、[ER0.1授权及31来源](../reports/seller_alias_continual/20261002/er_low_execution/20261002_124018/authorization.json)、[LOGIT授权](../reports/seller_alias_continual/20261002/logit_weight_execution/20261002_153200/authorization.json)及[36项部署清单](../reports/seller_alias_continual/20261002/logit_weight_execution/20261002_153200/source_inventory.json)；授权上限不是实际消费数，后者查运行账和状态回执 |
| 已经得到哪些正负结果、什么仍未知 | [ER0.1结果](SELLER_ALIAS_ER_LOW_RESULT.zh.md)、[LOGIT0.25结果](SELLER_ALIAS_LOGIT_WEIGHT_RESULT.zh.md)、下方历史判读表、[早期分类](RESEARCH_DISCUSSION.zh.md#早期结果及后续限制)、[旧LaBSE阶段](RESEARCH_PROGRESS_20260923.zh.md)；新结果已完成回传、独立数值核验、结果外审与主审，全部正负判定保持 |
| 文件在哪里、缺失能否恢复、接下来做什么 | [Linux与活动文件](#linux与活动文件)、[存储与删除事实](STORAGE.zh.md)、[已授权计划与新合同边界](RESEARCH_PLAN.md) |

当前主模型是`BAAI/bge-large-zh-v1.5`，标题／描述分别编码并以均值和总体标准差汇总，4096维账号表示进入对称8192→128→1头；编码器和头均训练。基础损失为BCE＋query-rank＋0.5 hard，ER的λ乘完整历史基础损失，LOGIT再加独立0.5 MSE。省略历史损失是λ=0／SEQ的科学对照，省略乘数而保留历史项是λ=1；不能据此宣称任意实现都逐位等价。

数据根为`reports/seller_alias_continual/20260910/expression_generation/20260910_150500/data`：每域60个train群分48拟合／12校准，另20 valid／40已用基础test群。O为最终两个旧域能力，N为后两阶段刚学完的新域能力，Z为最终最新域能力；F_first为首域获得后至最终的损失，G为同一路径的新域学习增益。精确公式、22指标、输出角色和条件区间见[首轮第7节](SELLER_ALIAS_BGE_CONTINUAL.zh.md#7-评价对象和统计)，不从简称自行重写评价。

## 已授权工作与最近保存的状态

| 工作 | 最近已保存的观测 | 合同及实际状态入口 |
|---|---|---|
| ER λ=0.1单点 | 10月3日：正式退出0，1728更新、完整盲门及访问账符合；270份结果回传，独立核验、结果外审与主审通过。对0.25为23/23并按规则选0.1，对SEQ为10/23；本点关闭 | [合同](SELLER_ALIAS_ER_LOW.zh.md)、[完整分析](SELLER_ALIAS_ER_LOW_RESULT.zh.md)、[状态与实际耗时](../reports/seller_alias_continual/20261002/er_low_execution/current_status.json) |
| 匹配LOGIT0.25 | 10月3日：正式退出0，1728更新、完整盲门及访问账符合；198份结果回传，独立核验、结果外审与主审通过。对ER0.25为23/23，对SEQ为16/23；监听已退出／自删，本点关闭 | [合同](SELLER_ALIAS_LOGIT_WEIGHT.zh.md)、[完整分析](SELLER_ALIAS_LOGIT_WEIGHT_RESULT.zh.md)、[状态与实际耗时](../reports/seller_alias_continual/20261002/logit_weight_execution/current_status.json) |

ER0.1于10月2日12:55:35—18:16:57运行，实际5小时21分22秒，原估5—7小时；LOGIT于16:31:29—21:53:13运行，实际5小时21分44秒，原估6—8小时。两项剩余训练时间均为0，均在各自12小时／16GiB上限内。上述程序完成及来源核对证据见[最新观测](../reports/seller_alias_continual/20261002/er_low_execution/20261002_124018/observation/20261003_095536/remote_status.json)；ER31份、LOGIT隔离目录34份科学来源匹配。下一步使用现有输出，不重复启动。

两项均只新增1728更新／6终点，s0及ABC/BCA/CAB，复用首域完整model/Adam/RNG和指定历史证据；各自一次train/valid许可，valid仅在完整盲门后开放，不读test／owners。ER0.1相对0.25原23项全过才替换，否则保留0.25；完整报告对SEQ，单点好坏均结束。匹配LOGIT为L当前＋0.25 L历史＋0.5 MSE，MSE不再乘0.25；复用SEQ和ER0.25，分别报告对ER与对SEQ，不随0.1结果自动改系数。

**当前执行补充**：用户明确“不用等λ=0.1，资源够了就开始”，LOGIT实际已于ER结束前启动，资源门和预算没有改变。监听在10月2日16:37:14确认真实更新后自删Linux脚本，随正式作业于21:53:13退出0；最新实查确认脚本不存在。[具体决定](../reports/seller_alias_continual/20261002/logit_weight_execution/20261002_153200/decision.json)、[自删原记录](../reports/seller_alias_continual/20261002/logit_weight_execution/20261002_153200/observation/20261003_095536/listener/script_deleted_after_updates.txt)。监听任务已结束，不重建。

**本次harness调整的边界**：已批准范围内连续执行；未来可批准手写CPU先于外审及条件故障恢复。两项实验按冻结来源、原访问账和判据完成，没有新增λ、种子、训练、标签或成功线。结果审查和主审已关闭，既有实现／原生核验证据继续复用；不以外审未直接读取旧逐步日志或未认证后台身份另开实验、重发审查或补读受限材料。[联合审查请求](../reports/seller_alias_continual/20261003/result_review/request.zh.md)、[输入包身份](../reports/seller_alias_continual/20261003/result_review/package.json)、[专项上传许可](../reports/seller_alias_continual/20261003/result_review/upload_authorization.json)及[实际完成记录](../reports/seller_alias_continual/20261003/result_review/submission.json)均保留。网页外审使用本地Playwright MCP，专用页已关闭；此前CUA失败不计为审查。

## Linux与活动文件

- LOGIT0.1独立workspace：`reports/seller_alias_continual/20261003/logit_low_execution/20261003_150000/workspace`；采用自身相对入口，旧两项冻结来源不覆盖。正式job位于该workspace下`reports/job`，10月3日21:23:48已退出0；旧PID仅为历史身份，禁止重复启动。完整结果已回传，沿用既有SSH认证。
- 现有服务器：yongpeng@10.201.109.111；项目根：/home/yongpeng/cross-lingual；Python：/home/yongpeng/miniconda3/envs/py310/bin/python。沿用既有SSH认证，凭据不写入项目。
- ER0.1作业：reports/seller_alias_continual/20261002/er_low_execution/20261002_124018/job；活动来源留主目录，不能用本地后继代码覆盖。
- LOGIT隔离工作区：reports/seller_alias_continual/20261002/logit_weight_execution/20261002_153200/workspace；其中reports/authorization.json是Linux实际授权文件，reports/listener存退出／自删记录，reports/job为已完成正式输出。Linux一次性监听脚本已删除；本地复现源码继续保留。历史PID只标识当时进程，不证明当前存活或身份。
- 当前部署来源为LOGIT34份科学文件及必要包装／冻结记录；已部署ER0.1使用其原31份。通用文档维护不进入这些冻结来源，不复制大模型，不清理活动状态。

## 数据、保管和未授权范围

当前数据、划分、六群Algorithm R／1MiB历史及附属状态、随机性、校准、22指标和条件区间均以[首轮合同](SELLER_ALIAS_BGE_CONTINUAL.zh.md)及相应扩展合同为准。三个顺序不是三个训练种子，已开发valid不是独立确认集。

既有120群基础test已使用，不能再次当作未开发最终留出。风险算法的单点开发运行已按上文新合同授权；新最终留出、多种子确认、大候选库和无正候选查询只可先提出方案，尚无本轮执行许可。Audit A/B真值、Audit B预测、owners和私有身份资产维持既定封存边界。

当前保留（10月3日清理后）：排序s0候选及对照两份（ranking_execution/20260927_114646/job/run下s0_hard和s0_d的models/epoch6.pt）；首轮三个共享首域完整状态（bge_continual_execution/20260930_143700/job/run/branches下ABC_shared.pt、BCA_shared.pt、CAB_shared.pt）；ER0.1六份新权重、必要状态、缓存、预训练档案及证据。LOGIT0.25六份末端权重已按用户最新指令于19:04:45删除，释放7,838,884,668字节；原小结果和历史保管清单保留，当前事实以[删除回执](../reports/maintenance/20261003/logit_quarter_weights/receipt.json)为准。路径所属日期及完整清单见[存储](STORAGE.zh.md)和[依赖／删除记录](../reports/maintenance/20261002/unused_weights/report.zh.md)。已删49份旧推理权重的事实不变，旧18份全保留措辞不再适用；保留或后续清理由实际依赖与恢复需要决定。

## 已有结论与历史边界

| 实验／证据 | 应保留的判读 |
|---|---|
| 中文BGE基础与加权汇总 | 候选排序收益与严格自动识别失败分别保留；加权汇总改进失败，不把旧严格识别门变成全部研究的门槛。[基础结果](SELLER_ALIAS_CHINESE_RESULT.zh.md)、[汇总结果](SELLER_ALIAS_POOLING_RESULT.zh.md) |
| 困难排序C／校准／基础test | 原排序9/13，排序正增益与概率退化并存；校准17/17；固定s0 test原8/9，用户事后接受实际提升，原passed=false不改。[排序](SELLER_ALIAS_RANKING_RESULT.zh.md)、[校准](SELLER_ALIAS_CALIBRATION_RESULT.zh.md)、[test](SELLER_ALIAS_TEST_RESULT.zh.md) |
| BGE首轮持续学习 | SEQ遗忘门通过；ER1−SEQ 4/23；LOGIT1−ER1 23/23，但相对SEQ旧域R5、新域和最终最新域仍有退化。有效的相对正结果不等于整体成功。[结果与审查](SELLER_ALIAS_BGE_RESULT.zh.md) |
| ER0.5／0.25 | 两者对ER1均23/23、对SEQ均4/23，按原规则选0.25；原结果有效、对SEQ的不足保留。[结果](SELLER_ALIAS_ER_WEIGHT_RESULT.zh.md) |
| ER0.1（结果外审／主审已关闭） | 对0.25为23/23，对SEQ为10/23；O MAP和首域遗忘有正向证据，N MAP/R5有负向证据。从0.1到0时O MAP下降，不能宣称λ越小全面越好或0最优。[结果与审查](SELLER_ALIAS_ER_LOW_RESULT.zh.md) |
| 匹配LOGIT0.25（结果外审／主审已关闭） | 对匹配ER为23/23，对SEQ为16/23；相对SEQ的O MAP/R5、首域遗忘和final_all MAP正向，新域若干观察值保护未满足，跨零不等于无退化；不换ER0.1作主要对手。[结果与审查](SELLER_ALIAS_LOGIT_WEIGHT_RESULT.zh.md) |
| 更早的有效正／零／负与无效运行 | [历史分类与原报告](RESEARCH_DISCUSSION.zh.md#早期结果及后续限制)逐项区分Step7—28；不能把小样本、旧输入或内部开发的局部正结果外推到当前任务 |

特别保留：Step27-v1工程无效，而v1.1已实际完成并未通过OOF门；Step28-v12的固定合成家族内正结果不同于Step28-v13 v1.12；V9.3-R2为AUDITOR_DESIGN_INVALID_NO_DATASET_CONCLUSION，不能称有效数据集失败；v12.1应用弃权及Audit A未开真值的盲预测不是准确率负结果。

Step28-v13 v1.3—v1.12、V8、V9、V9.1、V9.2、V9.3-R2关闭路线不重启。合格V9.4.1投影、训练／开发V2及Audit A盲预测的有效身份保留。旧风格迁移25世界主端点失败及停止边界不变，1世界辅助正结果不替代它。[迁移结果](STYLE_TRANSFER_TARGET_RESULT_20260906.zh.md)。旧LaBSE持续学习的有效负结果、后继数据的独立身份见[阶段汇总](RESEARCH_PROGRESS_20260923.zh.md)。

## 历史与证据

历史只在与当前问题相关时读取，不作为现行授权。首选上述实际结果／合同和[用户原话](RESEARCH_DISCUSSION.zh.md)中的原始审查链接。按用户明确要求，不保留重构前八份文档的额外副本，临时归档及清单已删除；现行文档原位维护，已有Git历史和原始科研报告继续提供回溯。

也可直接阅读已交付提交eecee09f的[原交接](https://github.com/bililateral/cross-lingual-DIL/blob/eecee09f4d5c25f88d054cab359ba6fb97dcb8fa/docs/AI_RESEARCH_HANDOFF.zh.md)、[原进展](https://github.com/bililateral/cross-lingual-DIL/blob/eecee09f4d5c25f88d054cab359ba6fb97dcb8fa/docs/PROJECT_PROGRESS.md)、[原设计](https://github.com/bililateral/cross-lingual-DIL/blob/eecee09f4d5c25f88d054cab359ba6fb97dcb8fa/docs/CURRENT_EXPERIMENT_DESIGN.md)、[原计划](https://github.com/bililateral/cross-lingual-DIL/blob/eecee09f4d5c25f88d054cab359ba6fb97dcb8fa/docs/RESEARCH_PLAN.md)。这些是固定历史版本，旧“当前／下一步／待批准”不生效。

常设规则见[科研纪律](RESEARCH_DISCIPLINE.zh.md)，当前合同索引见[设计入口](CURRENT_EXPERIMENT_DESIGN.md)，随后工作见[计划](RESEARCH_PLAN.md)，文件位置和删除历史见[存储](STORAGE.zh.md)。今后更新当前状态只改本页相关位置与所属status，不向其他入口追加复制段落。
