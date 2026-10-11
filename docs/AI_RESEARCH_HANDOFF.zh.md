# 当前科研交接

本页顶部表格是唯一当前工作入口。先读本页与[科研纪律](RESEARCH_DISCIPLINE.zh.md)，再按任务进入合同、实际实现、直接依赖及相关原始审查；历史不重新授予执行权限。当前为用户已确认的三臂归因实验准备：LOGIT0.1、无重组记录模型R0、原S，先明确原C/S系统收益来自记录结构还是重组监督。旧五臂已结束，C_plus负结果及撤下边界不变；本次数值修复免审不适用于新研究。

## 当前工作

| 项目 | 当前范围、证据与执行前提 |
|---|---|
| 最新用户决定 | 已批准三臂、读取范围及归因诊断判读；10月9日明确要求“手写GPU核验通过后自动开始GPU正式训练”，真实更新后探测自删。按同条后半句和既有合同执行单CPU线程／单GPU训练，首句“CPU训练”不另开全CPU方案。新原生投影超48小时后，用户明确“同意将正式上限改为72小时”，其他范围不变。见[72小时补充授权](../reports/documentation/20261009/record_attribution/qualification/authorization_72h.json)、[原三臂合同](SELLER_ALIAS_RECORD_ATTRIBUTION.zh.md)、[自动启动及资格处置](../reports/documentation/20261009/record_attribution/qualification/formal_autostart.zh.txt)。旧主审“不自动创建正式门”和原48小时均按对应历史时点解释。 |
| 研究与工作量 | LOGIT0.1、R0、S；s0及ABC/BCA/CAB；各自从预训练起点训练三个阶段。7,776物理更新、12,960梯度群呈现、27阶段状态、162逐epoch诊断及两个未训练架构盲起点。R0/S只切换重组监督，首阶段不共享训练状态；R0−LOGIT是系统结构比较。主要分别报告O/N MAP及区间，无新总通过门，不自动追加候选。 |
| 预算与三种权限 | CPU手写累计≤300秒，原生GPU核验≤600秒；正式单CPU／单GPU全过程现为≤72小时，输出峰值仍≤48GiB、CUDA reserved≤28GiB、RSS≤64GiB。train／valid各一次新解析尝试，全部训练／恢复／盲门后统一开放valid；test／owners／Audit／private为0，失败记账且不自动重启GPU／正式运行。72小时补充只改变时间界限，不扩计算或读取范围。 |
| 实现与CPU | [正式入口](../scripts/step28_record_attribution_run.py)的计算及原11项检查证据复用；原CPU失败与28.73秒保留。新增编排7项／6.78秒，预算衔接6项／5.14秒全部通过，CPU保守累计43／300秒、剩余257秒。预算衔接仅允许policy与timeout改72小时、validate_gate引用原资格；26来源逐字节不变，训练／评价AST不变，预算监控实际接受49小时并在72小时停止。见[编排](../reports/documentation/20261009/record_attribution/qualification/formal_listener_check/attempt01/result.json)、[预算核验](../reports/documentation/20261009/record_attribution/qualification/budget_extension_check/attempt01/result.json)。不把旧CPU／GPU回执改成新来源，不重复原生训练。 |
| 审查与处置 | 完整外审266成员包已回收，265载荷及原上传包核对一致；[外审正文](../reports/documentation/20261009/record_attribution/review/external/review.zh.txt)、[独立主审](../reports/documentation/20261009/record_attribution/review/primary_disposition.zh.txt)关闭，NO_OPEN_BLOCKERS。唯一非阻断RD-P1为旧规划说明的算术差异：按浮点余量及完整公式应为44.279656小时／42.308229GiB，原44.2764小时少约11.72秒；已补充更正，不改冻结原文或科研代码。网页已于19:27关闭；身份和完整包在[提交回执](../reports/documentation/20261009/record_attribution/review/submission.json)。 |
| 原生资格与自动串接 | 原生唯一attempt01与72小时资格衔接均已完成，原失败／48小时拒绝现场保留。正式作业于10月9日20:59:05启动，10月11日00:24:33正常退出，实际27小时25分28秒、剩余0，未触及72小时强制上限。10月11日09:48:51核对原监听、wrapper、timeout与训练PID均已退出，无failure；7,776更新、12,960梯度群呈现、27阶段完整恢复、162逐epoch诊断和两个盲起点全部完成。完整盲门先通过，再统一开放valid；最终train=1、valid=1、heldout=0、owners=0。29项来源与正式门及执行／评价一致；峰值CUDA reserved约8.82GiB、RSS约6.62GiB、输出约35.28GiB，均未超预算。见[完成原件](../reports/seller_alias_continual/20261011/record_attribution_result/job/completion.json)、[实际退出](../reports/seller_alias_continual/20261011/record_attribution_result/job.wrapper.txt)、[回传实物清单](../reports/seller_alias_continual/20261011/record_attribution_result/inventory.json)。历史估时与最大形状投影按其时点保留，不再作为当前进度。 |
| 当前部署与恢复入口 | 已结束的独立workspace为reports/seller_alias_continual/20261009/record_attribution_execution/20261009_200600/workspace；原20261009_182606保留原生通过及48小时停止现场。完成结果在前者reports/job，资格与已退出监听回执在reports/qualification；活动监听源码已自删，不重建、不重启。1,234份小结果／冻结来源／资格证据共38,049,242字节已SFTP回传并逐文件核对；结果归档为reports/seller_alias_continual/20261011/record_attribution_result，两端保存完整小结果与Linux分析输出。27权重共35,237,126,610字节、27份Memory共8,389,549字节仅留Linux，未载入或上传；见[接收核对](../reports/seller_alias_continual/20261011/record_attribution_result/receive_verification.json)。本轮Harness工作区改动不覆盖冻结workspace。 |
| 本轮结果与待办 | [详细主分析](../reports/seller_alias_continual/20261011/record_attribution_result/report.zh.txt)已完成，Linux保存矩阵独立参考核对25,344项统计量，最大差4.44e-16。R0−LOGIT的O MAP为+0.010473（条件95%区间全正）、N为+0.000984（跨零）；S−R0的O/N分别+0.016436／+0.015999（均全正）。结构提供部分旧域收益，重组在同一记录结构上增加旧域与后续新域收益；N增量主要在第三阶段，CAB首域、原始概率与固定阈值取舍保留。单seed、重复开发valid和无新总通过门的边界不变；历史C／C_plus负结果与R1不改写。用户已批准本次28,532,122字节、1,266成员、SHA256为9aaf0f2a82de854741313ee1b948432d335132085f5bc92e21e3ae9f9a2ff29a的结果包外审；网络恢复后仍有连接重置，当前附件上传尚未确认完成，尚未提交外审消息。包许可持续有效；恢复先查看本地Playwright现有页面与[提交状态](../reports/seller_alias_continual/20261011/record_attribution_result/review/submission.json)，避免重复上传／提交。结果提交9e3aa425已形成，Git传输待完成及远端独立核对；不把本地提交当远端交付，不自动新开实验。 |
| 已完成数值修复与边界 | 保留的BGE／LOGIT、C/S入口已显式统一数值设置并记录有效值；CPU与用户批准的C/S手写原生补核完成，后者4次更新逐字节一致，监听已退出／自删。见[修复](../reports/maintenance/20261009/runtime_numerics/report.zh.txt)、[原生完成](../reports/maintenance/20261009/runtime_numerics_gpu/completion.zh.txt)。历史LOGIT R1根因仍未隔离，不能追改旧结果或宣称完整历史复现。 |
| 已关闭五臂 | 10,368更新、36状态、216物理诊断及结果外审／主审已结束；C_plus原继续条件false。C_plus/S_strong及五臂活动入口／测试已依许可撤下，原C/S、LOGIT、通用诊断和冻结证据保留。36权重已留证删除，36份Memory仍保留。见[结果与主审](../reports/seller_alias_continual/20261009/replay_improvement_result/review/primary_disposition.zh.txt)、[两端撤下](../reports/maintenance/20261009/replay_improvement_closure/closure.zh.txt)。旧72小时及旧标签许可仅属已结束五臂。 |

## 研究问题与目标

目标为具有实质方法贡献的CCF B／C期刊或会议投稿；刊会与期限未指定，不阻塞已批准工作。历史训练数据可用量有限，新卖家群体按分布分批到达，研究旧分布未见卖家的候选排序／基础识别保持和新群体学习。当前是中文合成受控任务，输入只含屏蔽标题／描述，无身份特征；不研究同一控制者跨阶段新开别名追踪，不转向一般作者识别或AI文本检测。

每群28账号，每查询27候选且有同控候选；结论不能直接外推到全市场、无正候选查询或真实中文市场。先完成中文合成主线，再决定扩展。编码器与评分模型继续训练；LOGIT0.1作为比较对手，基础排序提升、已知方法适配、调λ或常规模块叠加均不自动构成创新。用户允许修改算法核心，但具体新方案仍须用户与外审双审及相应范围确认。

## 接手顺序与信息地图

当前执行状态只看顶部表格；本节是证据导航。按本轮实际变化确定最小核验，未变且适用的证据可复用，不因换Agent递归重读或重跑全部历史。

| 要掌握的信息 | 入口与使用边界 |
|---|---|
| 用户问题、研究前提、后续决定 | [讨论记录](RESEARCH_DISCUSSION.zh.md)、[计划](RESEARCH_PLAN.md)；辨认每项决定的对象、时点和范围。 |
| 当前计算、真实更新及资格 | 顶部三臂合同与实际入口；旧五臂只作已审参考，[来源清单](../reports/documentation/20261008/record_replay_improvement/sources_to_sync.json)、CPU原件与实现说明。相关历史实现原审查从本轮送审背景及原C/S主审回溯，不把旧通过当作新实现已审。 |
| 数据究竟改变什么 | [表达生成结果](SELLER_ALIAS_EXPRESSION_DATA_RESULT.zh.md)、[首轮数据合同](SELLER_ALIAS_BGE_CONTINUAL.zh.md#2-数据和逐阶段可见范围)；主题混合与表达机制同时变化，不称纯表达因果实验。 |
| 早期来源与身份资格 | [schema](STEP1_SCHEMA.md)、[隔离](STEP2_SPLIT_AND_LEAKAGE.md)、[画像](STEP3_SELLER_PROFILE.md)、[silver候选](STEP4_SILVER_CANDIDATES.md)、[复核冻结](STEP5_REVIEW_AND_FREEZE.md)；历史来源链不授予真值或当前正式数据读取许可。 |
| 评价定义与不确定性 | 本轮合同明确继承的[首轮第7节](SELLER_ALIAS_BGE_CONTINUAL.zh.md#7-评价对象和统计)及实际评价函数。区分AP、逐查询MAP、raw与校准，不从简称重写公式。 |
| 方法近邻与创新边界 | [文献定位](SELLER_ALIAS_LITERATURE.zh.md)、下方方案审查原件；文献有检索时点，新主张补相关一手证据。 |
| 已完成结果、失败与关闭路线 | [近期历史进展](#近期历史进展)、[已有结论与历史边界](#已有结论与历史边界)；读取原报告及原始审查，不用摘要代替支撑当前主张的证据。 |
| 文件位置、删除与清理提醒 | [Linux与活动文件](#linux与活动文件)、[存储记录](STORAGE.zh.md)；保管状态与历史身份分开，恢复前核对实物。 |

## Linux与活动文件

- 本次数值补核已无活动监听／核验作业（10月9日16:54回查及随后核对）。原监听PID3505392已退出，scripts/run_step28_numerics_linux_20261009.sh已在16:39:43实际更新后自删；不重建或重复启动。完整启动／退出账在reports/maintenance/20261009/runtime_numerics_gpu/listener，唯一attempt01已PASS。核验脚本scripts/step28_record_replay_numerics.py、Windows监听源码及source_snapshot.zip作为复现证据保留。该作业是已完成手写原生补核，不是历史正式训练重跑。
- 服务器：yongpeng@10.201.109.111；项目根：/home/yongpeng/cross-lingual；Python：/home/yongpeng/miniconda3/envs/py310/bin/python。沿用既有SSH认证，密码只作认证输入，不写入项目。同步用SFTP／SCP，Git仅在Windows执行。
- 已结束的72小时五臂workspace：/home/yongpeng/cross-lingual/reports/seller_alias_continual/20261008/replay_improvement_execution/20261008_151407/workspace，28项来源见[来源清单](../reports/documentation/20261008/record_replay_improvement/budget_72h/sources.json)。原48小时workspace（同一上级下20261008_131300/workspace）和首次原生证据保留；均为历史证据，不作为现行启动入口。
- 已完成五臂作业：上述72小时workspace下reports/job；资格与启动记录在reports/qualification，历史正式入口scripts/run_step28_replay_improvement_linux_20261008.sh。原单CPU0／GPU0训练PID3407207及wrapper3407201已于10月9日13:38:39正常退出，不重启。完整小结果及分析归档为reports/seller_alias_continual/20261009/replay_improvement_result；原job/run/models的36权重已于14:48:54按许可留证删除，36份Memory继续保留job/run/memory。旧workspace的attempt01亦已结束。
- 主目录的规则与交接是可维护文件；本轮已撤下主目录五臂活动入口。独立workspace及结果归档中的冻结源码、合同、来源身份和运行证据不得被同步或代码清理覆盖，已送审28来源仍保持原字节。
- 旧运行路径只用于追溯：C/S为reports/seller_alias_continual/20261007/record_replay_execution/20261007_225409/workspace，完整小结果为reports/seller_alias_continual/20261008/record_replay_result；LOGIT0.1为reports/seller_alias_continual/20261003/logit_low_execution/20261003_150000/workspace；ER0.1为reports/seller_alias_continual/20261002/er_low_execution/20261002_124018/job；LOGIT0.25为reports/seller_alias_continual/20261002/logit_weight_execution/20261002_153200/workspace。这些作业已结束，旧监听已退出／自删，不重建或重启。

## 数据、保管和未授权范围

数据根为reports/seller_alias_continual/20260910/expression_generation/20260910_150500/data。每域60个train群分48拟合／12校准，另20 valid／40已用基础test群。历史与附属状态采用六群Algorithm R及1MiB上限，具体权限以本轮合同为准；调度器可见全表不等于学习器可见未来域，研究归档不等于可用记忆。

记录表模型将分别L2归一化的标题／描述表示拼接并除以√2，得到2048维记录表示；记录对头4096→128→1，再按账号汇总。LOGIT账号汇总为4096维、头输入8192维。四个记录表臂的A0/A1监督与teacher系数、LOGIT目标分别见五臂合同，不把跨架构比较当作单一机制归因。

10月8—9日五臂均使用已选中文BAAI/bge-large-zh-v1.5同一预训练归档（revision 79e7739b6ab944e86d6171e44d24c997fc1e0116）；四个记录表臂网络相同，LOGIT的聚合／头不同，各自训练后参数分化。36份Memory是27个记录表与9个LOGIT物理阶段的历史缓存快照，共11,479,595字节；每条实时路径最多六群／完整状态1MiB，包含允许的训练文本、标签、教师目标及附属状态。归档快照不合并成学习器可用历史，不等于模型权重；具体说明见[关闭记录](../reports/maintenance/20261009/replay_improvement_closure/closure.zh.txt)。

O为最终两个旧域能力，N为后两阶段刚学完的新域能力，Z为最终最新域能力；F_first为首域获得后至最终的损失，G为同一路径的新域学习增益。三个顺序不是三个训练种子，开发valid不是独立确认集，既有120群基础test已使用。新最终留出、多seed确认、大候选库和无正候选查询尚无执行许可。Audit A/B真值、Audit B预测、owners与私有身份资产保持既定封存边界。

保管按最新[存储记录](STORAGE.zh.md)及实际依赖核对：BGE预训练、首轮三个共享首域完整状态、ER0.1六权重、排序s0_hard与s0_d两份例外及必要科研证据继续保留。10月9日本轮36权重、此前C/S十五权重、旧LOGIT0.1／risk十二权重、LOGIT0.25六权重及关系／函数记忆相关已清理权重，不按旧清单假定仍存在。SHA不能恢复已删模型，重训仍需相应授权。

两端项目大清理已于10月8日完成；最近完成日期、下一到期日期和独立提醒任务统一见[清理周期](STORAGE.zh.md#项目大清理周期)。提醒仅提示到期，实际清理仍须先说明理由、范围、空间收益及保留依赖并征得许可。局部清理不重置周期，普通删除提交不等于改写Git历史。

## 近期历史进展

以下均为已结束阶段或历史决定，按所属日期解释。详细过程、原始审查、计数、来源及删除记录留在所引原件中；本表不授予新运行权限。

| 阶段 | 已发生事实、结论边界与证据 |
|---|---|
| 2026-10-09：五臂改进训练结束 | 实际22小时12分42秒，原继续条件false；相对C/S的新域损失、相对S_strong的概率局部收益、相对本轮LOGIT系统收益并列保留。统一22指标、原23项、全部六轮过拟合曲线与历史复现差异见[完整分析](../reports/seller_alias_continual/20261009/replay_improvement_result/report.zh.txt)。结果外审实际状态只看顶部；旧LOGIT值和旧原5项结果不追改。 |
| 2026-10-08：统一评价增量 | 仅读已开放保存矩阵，C−LOGIT0.1、S−LOGIT0.1、C−S原23项分别23/23、23/23、16/23；原5项与整体false不改。这是事后同口径开发补充。旧三算法缺同口径过程曲线，过拟合直接证据不足，多轮valid选择偏差保留。七条失败对应六个端点—指标组合、五种指标类型。外审29成员包完整回收，外审与主审接收有限比较、页面关闭；审查会话与状态见顶部。见[分析](../reports/seller_alias_continual/20261008/record_replay_result/common_evaluation/report.zh.txt)、[完整表](../reports/seller_alias_continual/20261008/record_replay_result/common_evaluation/calculation/tables.zh.txt)、[主审及外审原件入口](../reports/seller_alias_continual/20261008/record_replay_result/common_evaluation/review/report.zh.txt)。 |
| 2026-10-08：记录作用表C/S完整实验 | 10月7日23:55:07—10月8日07:39:49正常退出0，7小时44分42秒，4320更新／7776梯度群呈现／15状态，完整恢复与盲门，train／valid各1、heldout／owners0、无failure。有效开发实验；原C−LOGIT五条件5/5、C−S4/5，整体false。C−S O MAP差−0.000138641479，条件95%区间跨零，新增teacher分配收益与创新未建立。见[完整分析](../reports/seller_alias_continual/20261008/record_replay_result/report.zh.txt)、[指标表](../reports/seller_alias_continual/20261008/record_replay_result/verification/attempt01/tables.zh.txt)、[完成账](../reports/seller_alias_continual/20261008/record_replay_result/job/completion.json)。 |
| C/S审查与保管 | [实现主审](../reports/documentation/20261007/record_replay_implementation/20261007_225409/report.zh.txt)、[原生处置](../reports/documentation/20261007/record_replay_implementation/20261007_225409/native_disposition.zh.txt)及[结果主审](../reports/seller_alias_continual/20261008/record_replay_result/review/report.zh.txt)均已关闭；[完整结果外审](../reports/seller_alias_continual/20261008/record_replay_result/review/external/unpacked/RECORD_REPLAY_REVIEW_PACKAGE/RECORD_REPLAY_RESULT_REVIEW.zh.txt)接收有效结果，无该轮未关闭阻断。raw O/N/Z log-loss对LOGIT退化、校准后更好；C−S群宏F1局部负向与A2/G log-loss辅助正向并列；CAB弱获得点主导遗忘差，不能把低遗忘直接当最终旧域更强。15权重09:43:08按许可清理，释放19,565,561,724字节，224非权重SHA及23保留例外元数据不变，见[回执](../reports/maintenance/20261008/record_weights/receipt.json)。Memory、21冻结来源与完整科研证据保留。 |
| 2026-10-07：记录作用表方案审查 | 静态方案外审／主审不推荐，有限表重建和条件一阶VJP成立，但重组对未见身份的必要性、表达样式混合及额外成本收益依据不足；Q期望测度与双通道归一化勘误保留。用户随后另行批准完整C/S实验，因此旧“不实施”不阻断后来的已批准运行。见[静态主审](../reports/documentation/20261007/record_replay/report.zh.txt)、[外审原文](../reports/documentation/20261007/record_replay/external/review.zh.txt)、[实际阅读范围](../reports/documentation/20261007/record_replay/external/reading_scope.zh.txt)。外审自报使用3辅助任务的流程偏差保留，不把辅助数量当独立证据。 |
| 2026-10-07：候选筛选与方向纠正 | 群风险算子拟固定首域BGE、只求关系矩阵，主执行者在用户质询后撤回推荐；未实施，不是性能负结果。后续回到编码器／评分头实际继续学习。固定缓存上的粗细视图差分风险反例说明无偏性不保证训练可行或泛化，不能称项目已发生该故障。见[算子原稿](../reports/documentation/20261007/operator_memory/proposal.zh.txt)、[后续筛选](../reports/documentation/20261007/trainable_screening/report.zh.txt)。 |
| 2026-10-07：历史风险摘要／残差重放 | 方案外审／主审不推荐；来源二次代理由单锚点解码未必在来源点匹配网络一阶梯度，直接近邻及额外计算价值未建立。两项数学／舍入勘误保留；未实施，不是实验负结果或整类方法无效。见[主审](../reports/documentation/20261007/residual_replay/report.zh.txt)、[外审原文](../reports/documentation/20261007/residual_replay/external/review.zh.txt)。 |
| 2026-10-07：组合重放 | 方案外审／主审撤回推荐；主要数学成立，精确积分相对同池MC32的必要性、创新与未见身份收益未建立。原96成员包遗漏生成直接来源的背景覆盖问题、后续补核及两项勘误保留；外审群级共因反例不当作已发现的生成器事实。未实施。见[主审](../reports/documentation/20261007/compositional_replay_review/report.zh.txt)、[外审原文](../reports/documentation/20261007/compositional_replay_review/external/review.zh.txt)。 |
| 2026-10-07：不接受简单基线堆砌 | 用户否决完整LOGIT0.1+βR作为创新方案，相关候选撤回且停止审查。允许改核心不等于允许任意新运行；共同模型与优化器可合理复用，实质贡献仍须证明。见[纠正原文](../reports/documentation/20261007/rank_protection/correction.zh.txt)、[完整用户研究要求](../reports/documentation/20261007/compositional_replay_review/user_brief.zh.txt)。 |
| 2026-10-07：函数目标累计关系记忆 | 10月6日23:19:15—10月7日02:45:15完成，1728新更新，完整盲门，train／valid各1；五条件0/5，O/N/Z MAP相对LOGIT差−.035606／−.016022／−.018508，区间均负。共享首域相同，不能归因为起点弱。外审／主审接收配置级有效负结果；九权重按许可删除，活动源码和证据保留。见[结果](SELLER_ALIAS_FUNCTION_MEMORY_RESULT.zh.md)、[主审](../reports/seller_alias_continual/20261007/function_memory_result/review/report.zh.md)、[删除](../reports/maintenance/20261007/function_weights/receipt.json)。 |
| 2026-10-06：关系目标记忆修改版 | 完成2592更新／9端点，对LOGIT六条件0/6。用户决定保留原版、删除修改版活动代码与九权重；冻结来源、九份Memory及315项科研证据保留。主审接受有效配置级负结果；旧后台跟进已停用。见[结果](SELLER_ALIAS_RELATION_REVISION_RESULT.zh.md)、[主审](../reports/seller_alias_continual/20261006/relation_revision_result/review/report.zh.md)、[关闭与清理](../reports/maintenance/20261006/relation_revision_closure/closure.zh.md)。 |
| 2026-10-06：关系目标记忆原版 | 实际10月5日15:06:32—18:57:22完成，3小时50分50秒、2592更新／9端点；六条件0/6，O/N/Z MAP对LOGIT均有负向条件区间，保留CAB首域局部改善及召回／误报取舍。原入口问题与核验已关闭，结果外审／主审接收有效负结果，九权重按依赖与许可清理。见[结果](SELLER_ALIAS_RELATION_MEMORY_RESULT.zh.md)、[接入处置](../reports/documentation/20261005/relation_memory/report.zh.md)、[结果主审](../reports/seller_alias_continual/20261006/relation_result/review/report.zh.md)、[删除](../reports/maintenance/20261006/relation_weights/receipt.json)。 |
| 2026-10-04：完整群隔离泛化重放关闭 | 用户明确回退整条group-meta候选路线，包括完整二阶、GPU直接／CPU暂存、分阶段HVP及JVP，取消先导、机制验证与未提交审查，不继续使用剩余预算。当前服务器与批准上限下资源受限；正式训练未启动，效果未知，不否定整个元学习家族。CPU551.47秒及GPU151/3600秒尝试账保留。见[关闭与原始证据](../reports/documentation/20261004/group_meta/closure.zh.md)。 |
| 2026-10-04：LOGIT0.1与risk | 各1728更新／六终点，正式退出0、完整盲门与访问账；LOGIT0.1对ER0.1／LOGIT0.25为23/23、对SEQ20/23，risk对LOGIT0.1为4/23且O/N/Z MAP负向。联合外审／主审已关闭，保留误报／召回取舍、探针不同群及权重核验边界。12权重于10月4日按许可清理，不再据旧清单加载。见[联合结果](SELLER_ALIAS_LOGIT_RISK_RESULT.zh.md)、[结果主审](../reports/seller_alias_continual/20261004/result_review/report.zh.md)、[删除](../reports/maintenance/20261004/logit_risk_weights/receipt.json)。 |
| 2026-10-02至03：ER0.1与匹配LOGIT0.25 | 两项各1728更新，实际5小时21分22秒／5小时21分44秒，完整盲门与访问账，结果外审／主审已关闭。用户允许资源足够时LOGIT不等ER结束；旧监听已退出／自删。ER对0.25为23/23、对SEQ10/23；LOGIT对匹配ER为23/23、对SEQ16/23。ER六权重保留，LOGIT六权重已清理。见[ER结果](SELLER_ALIAS_ER_LOW_RESULT.zh.md)、[LOGIT结果](SELLER_ALIAS_LOGIT_WEIGHT_RESULT.zh.md)、[联合审查状态](../reports/seller_alias_continual/20261003/result_review/submission.json)、[旧监听回执](../reports/seller_alias_continual/20261002/logit_weight_execution/20261002_153200/observation/20261003_095536/listener/script_deleted_after_updates.txt)。 |

## 已有结论与历史边界

| 更早实验／证据 | 应保留的判读 |
|---|---|
| 中文BGE基础与加权汇总 | 候选排序收益与严格自动识别失败分别保留；加权汇总改进失败，不把旧严格识别门变成全部研究的门槛。[基础结果](SELLER_ALIAS_CHINESE_RESULT.zh.md)、[汇总结果](SELLER_ALIAS_POOLING_RESULT.zh.md)。 |
| 困难排序C／校准／基础test | 原排序9/13，排序正增益与概率退化并存；校准17/17；固定s0 test原8/9，用户事后接受实际提升，原passed=false不改。[排序](SELLER_ALIAS_RANKING_RESULT.zh.md)、[校准](SELLER_ALIAS_CALIBRATION_RESULT.zh.md)、[test](SELLER_ALIAS_TEST_RESULT.zh.md)。 |
| BGE首轮持续学习 | SEQ遗忘门通过；ER1−SEQ 4/23；LOGIT1−ER1 23/23，但相对SEQ旧域R5、新域与最终最新域仍有退化。有效相对正结果不等于整体成功。[结果与审查](SELLER_ALIAS_BGE_RESULT.zh.md)。 |
| ER0.5／0.25 | 两者对ER1均23/23、对SEQ均4/23，按原规则选0.25；有效结果与对SEQ不足均保留。[结果](SELLER_ALIAS_ER_WEIGHT_RESULT.zh.md)。 |
| ER0.1、LOGIT0.25的外推限制 | ER0.1的O MAP／首域遗忘正向与N MAP/R5负向并列，不称λ越小全面越好或0最优。LOGIT0.25对SEQ的新域观察保护未全满足，跨零不证明无退化，不换ER0.1作其主要对手。完整结果见上表。 |
| 更早正／零／负与无效运行 | [历史分类与原报告](RESEARCH_DISCUSSION.zh.md#早期结果及后续限制)逐项区分Step7—28；不能将小样本、旧输入或内部开发局部正结果外推到当前任务。 |

特别保留：Step27-v1工程无效，v1.1实际完成但未过OOF门；Step28-v12固定合成家族内正结果不同于Step28-v13 v1.12；V9.3-R2为AUDITOR_DESIGN_INVALID_NO_DATASET_CONCLUSION，不能称有效数据集失败；v12.1应用弃权及Audit A未开真值的盲预测不是准确率负结果。

Step28-v13 v1.3—v1.12、V8、V9、V9.1、V9.2、V9.3-R2关闭路线不重启。合格V9.4.1投影、训练／开发V2及Audit A盲预测的有效身份保留。旧风格迁移25世界主端点失败与停止边界不变，1世界辅助正结果不替代它，见[迁移结果](STYLE_TRANSFER_TARGET_RESULT_20260906.zh.md)。旧LaBSE持续学习的有效负结果与后继数据的独立身份见[阶段汇总](RESEARCH_PROGRESS_20260923.zh.md)。

## 历史与证据

历史仅在与任务相关时读取，不作为现行授权。优先按上表进入实际合同、结果和所属主审，再读直接相关的外审原文与实现证据。详细用户决定见[讨论记录](RESEARCH_DISCUSSION.zh.md)，保管与删除事实见[存储](STORAGE.zh.md)。不称全部背景已经语义全审，不以文件数、哈希或外审自述代替主执行者判断。

本页精简前的详细接手流水可通过已交付提交9bf7ed58的[交接原文](https://github.com/bililateral/cross-lingual-DIL/blob/9bf7ed58dd8e7f3bad58d6ac1420a4d89f02f07b/docs/AI_RESEARCH_HANDOFF.zh.md)回溯。更早重构前的[原交接](https://github.com/bililateral/cross-lingual-DIL/blob/eecee09f4d5c25f88d054cab359ba6fb97dcb8fa/docs/AI_RESEARCH_HANDOFF.zh.md)、[原进展](https://github.com/bililateral/cross-lingual-DIL/blob/eecee09f4d5c25f88d054cab359ba6fb97dcb8fa/docs/PROJECT_PROGRESS.md)、[原设计](https://github.com/bililateral/cross-lingual-DIL/blob/eecee09f4d5c25f88d054cab359ba6fb97dcb8fa/docs/CURRENT_EXPERIMENT_DESIGN.md)、[原计划](https://github.com/bililateral/cross-lingual-DIL/blob/eecee09f4d5c25f88d054cab359ba6fb97dcb8fa/docs/RESEARCH_PLAN.md)继续可追溯。这些是固定历史版本；原合同、报告、审查及运行证据保持原字节，不另存重构前文档副本。

当前状态只原位更新顶部表格与所属实际status；通用规则见[科研纪律](RESEARCH_DISCIPLINE.zh.md)，合同索引见[设计入口](CURRENT_EXPERIMENT_DESIGN.md)，后续动作见[计划](RESEARCH_PLAN.md)。新进展不再向历史索引或其他入口追加第二份“当前”。
