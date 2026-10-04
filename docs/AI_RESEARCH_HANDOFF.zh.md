# 当前科研交接

**当前工作（2026-10-04）**：LOGIT0.1／risk结果外审及主审已关闭，详见下方“最新结果状态”。用户随后要求实现完整群隔离泛化重放，并明确批准Linux单CPU、无GPU、手写输入加既有BGE、总2小时/64GiB/128MiB（包含失败修复）的核验。本轮[方案与实现定义](SELLER_ALIAS_GROUP_META.zh.md)和完整四群可微更新已实现，父算法源码未改；截至14:06:38（Asia/Shanghai），最终五项微型/解析/阶段核验及一次原生BGE完整元更新通过。三次原生技术失败均留证，累计七次程序墙钟548.31秒，成功原生峰值RSS45.90GiB；阶段288步是编排spy，未做288次正式训练。详见[实现与完整核验](../reports/documentation/20261004/group_meta/implementation.zh.md)。当前不读取正式数据/标签、不启GPU或新增效果比较；用户已专项授权286,285字节/103成员新包；14:16:21通过本地Playwright实际上传并提交，网页可见6/Pro，审查者已开始读取附件及运行检查。外审及独立[主审](../reports/documentation/20261004/group_meta/report.zh.md)已完成；204成员原包的203载荷核对通过。beta0呈现计数已修正，原实现报告head梯度范数由0.99251316勘误为0.92513164；15:10:47—15:10:50在Linux完成六项微型回归，全部通过，原生正beta证据复用，累计计算551.47秒。当前无未关闭实现缺陷；创新不足与效果未验证的主张限制继续保留，正式GPU运行及比较尚未授权。见[提交记录](../reports/documentation/20261004/group_meta/submission.json)及[网页会话](https://chatgpt.com/c/6ac1ef2f-dee0-83e9-87c8-dc7778dc82f4)。包内“待授权”是提交前历史快照，原包不改。新实现不能被写成已成立的论文创新或未见卖家收益。

**下一步顺序（用户最新要求）**：已将四条件写入[验证计划](SELLER_ALIAS_GROUP_META_VALIDATION.zh.md)。先准备候选的固定点先导运行，观察完整效果，再按前瞻规则决定必要机制对照；不要求先把四个条件全部训练。先导结果不替代机制归因或创新证明。用户随后明确开始执行；已新增[GPU手写核验方案](SELLER_ALIAS_GROUP_META_GPU.zh.md)及独立入口。用户已确认GPU0/单CPU、总1小时、分配器28GiB/RSS警戒128GiB/产物128MiB、仅手写输入与BGE预训练档案，并允许先核验后增量外审。16:09:35—16:12:31已完成三次手写核验：Tiny GPU完整梯度/更新一致性通过；BGE直接路径CUDA OOM；CPU暂存路径RSS采样128.62GiB超警戒，由包装终止，均未完成原生元更新。合计包装墙钟101秒，正式数据/旧权重访问为0；资源准入未通过，正式训练未启动。16:23:17实查GPU已释放，无本项目活动计算。详见[实际结果](../reports/documentation/20261004/group_meta_gpu/implementation.zh.md)。按本轮明确授权，16:22:45已通过本地Playwright上传427942字节/73成员包并提交[增量外审](https://chatgpt.com/c/6ac20ccf-de48-83e9-87fb-88a19382292c)，可见6/Pro，已开始核对附件；[提交记录](../reports/documentation/20261004/group_meta_gpu/submission.json)。等待审查后独立处置资源实现问题，不将资源失败当效果负结果，不默改一阶/head-only。

研究问题仍是有限历史记忆下学习新群体，并保持旧分布未见卖家的马甲候选排序／基础识别；不能以缓存拟合或代理损失改善替代主任务收益。LOGIT0.1保留为强基线，未确定论文最终算法。此前按用户指令删除七份未采用探索草稿，原实验及审查证据保留；本轮文档仅记录实际实现、授权及原生证据。Linux独立实现目录为reports/documentation/20261004/group_meta/workspace，所有手写作业已结束；以下10月3日运行叙述为历史观测，不表示仍有活动训练。
维护日期：2026-10-03。本页只维护一个当前入口；运行状态以带观测时点的实际回执为准。两项作业均已完成，468份16,506,607字节正式小结果完整回传并逐文件核对；10月3日10:23:58（Asia/Shanghai）Linux独立保存结果核验通过。联合结果已通过本地Playwright MCP完成实际网页外审，原报告及223成员证据包已保存核对，最终[主审处置](../reports/seller_alias_continual/20261003/result_review/report.zh.md)接受限定开发结论。两项科研收尾均完成；ER0.1按原规则入选，匹配LOGIT0.25相对ER0.25增量通过，对SEQ完整保护仍分别10/23、16/23。同步与Git交付独立记录于[本轮交付](../reports/seller_alias_continual/20261003/result_review/delivery.json)，不能把科研完成等同于远端交付。

**当前新增工作（10月3日）**：LOGIT0.1单点的实际网页外审及独立[主审](../reports/documentation/20261003/logit_low/report.zh.md)已完成，无未关闭阻断；原37份科学来源和31项/4次原生CPU证据保持，未改代码或重复核验。用户明确要求主审后开始正式训练。16:01:03（Asia/Shanghai）已在独立workspace启动唯一正式job，训练PID3330435，GPU0/CPU47；16:08:07实查ABC第二阶段已记录24次新更新、Adam312，确认该路径首域完整恢复/盲回放通过并实际训练；train访问1次、valid/heldout/owners均0，无退出状态，不预记正式终点或效果。预计启动后5—7小时，即10月3日21:01—23:01完成，硬上限10月4日04:01、16GiB。详见[实际更新观测](../reports/seller_alias_continual/20261003/logit_low_execution/20261003_150000/observation.json)及[当前状态](../reports/seller_alias_continual/20261003/logit_low_execution/current_status.json)。不得重复启动；后续按合同监控、完整回传、保存结果核验及结果外审。

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
