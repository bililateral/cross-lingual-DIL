# 科研计划

状态与观测只维护在[当前交接](AI_RESEARCH_HANDOFF.zh.md#当前工作)，本页仅列后续动作和授权边界。有效合同及增量修复从[设计入口](CURRENT_EXPERIMENT_DESIGN.md)进入，不复制PID、计数或过期工期。

L²R／ER+GCR按[822小时原执行范围](../reports/documentation/20261026/continual_comparison/execution.zh.txt)、[恢复补充](../reports/documentation/20261030/comparison_resume/scope.zh.txt)和[成功槽接续纠正](../reports/documentation/20261031/comparison_continue/scope.zh.txt)继续。统一控制器接管三条已在运行的L²R，任一健康槽成功后可接唯一原待派CAB_s6_GCR；不重启三条训练，不把“单独启动补跑”限制成成功槽不再接任务。原14成功及所有失败费用保留，原启动时刻/截止时间/资源限额不重置；失败槽退休，健康槽继续补位，无自动再试。全部18条按精确来源合并并过108状态完整门后，再执行原新120群生成、确认和统计；缺项不发整批成功。未变算法/校准/恢复资格复用既有证据，不为调度修正新增模型资格或科学尝试；主审闭合见[执行报告](../reports/documentation/20261031/comparison_continue/report.zh.txt)。

S_default＋历史H（G_half）继续用[失败隔离补充](../reports/documentation/20261030/defaults_repair/scope.zh.txt)的原科学workspace及[重新绑定的唯一监听](../reports/documentation/20261031/comparison_continue/defaults/formal_gate.json)，等待comparison_continue完整训练、108状态、生成、确认、分析全部成功和相关进程退出，再检查共享资源。原18条/54端点、108h+2h、单条6h和最多三路保持，只训练并保存旧dev盲分数；自身失败槽退休、健康槽接续，不放宽完整性、不自动再试或重复监听。补跑单条成功不等于前序整批成功。

VendorLink和FOREVER正式范围已分别获明确接受：[VendorLink四臂36条/每市场40轮/单路296h](../reports/documentation/20261030/vendorlink/formal/plan.zh.txt)，[FOREVER新增9条/复用H27/单路116h及新120群/4320前向](../reports/documentation/20261030/forever/formal/plan.zh.txt)。算法、已完成的单路GPU资格及适用旧证据复用，不新增GPU校验；按[本轮定向复审及主审](../reports/documentation/20261030/forever/formal/review/closure/primary_disposition.zh.txt)保留适用闭合结论和证据；未变且无具体疑点的部分不重复核验或再审，实际状态只看交接。用户最新明确先别训练、等开始指令，因此当前不启用这两项训练或自动资源等待。收到新开始指令后，核对实际部署、H27身份与共享资源，再先后启用两个新队列，不改活动L²R及已有H/S监听，不重复G1/G2或FOREVER资格。

VendorLink依赖扫描发现57群中48群相连且包含Dream全部8群，用户接受固定数据描述比较：保存完整排序/遗忘差值，不计算整体置信区间、不作统计验收，不宣称真实潜在同控识别或统计确认抗遗忘。不得为获得区间重分群、选账号、加数据或改统计模型。FOREVER保留额外全参数锚点、原O/F主比较及N/Z点值保护，不重训H、不追加HPO；不将两线预算或评价权限互借。

原五方法45条及其新确认/统计、结果外审和[主审](../reports/seller_alias_continual/20261026/continual_result/review/primary_disposition.zh.txt)均已结束。H相对S最终MAP正向，但原O/F共同主门未过、额外抗遗忘未确认，不重复旧恢复或为翻转结果重训。H/S54端点继续服务新对照；其余三方法81份独立权重已按用户选择删除，非权重证据和原490h/失败账保留，见[存储记录](STORAGE.zh.md#2026-10-26-五臂中的三臂权重清理)。
## 已结束Optuna的授权与恢复历史

已按[S/S_pos/H_pos有限Optuna合同](SELLER_ALIAS_OPTUNA.zh.md)、[272小时预算补充](../reports/documentation/20261019/optuna_budget/proposal.zh.txt)及[自动执行授权](../reports/documentation/20261019/optuna_budget/authorization.json)完成三臂36次搜索、9次新seed复验与集中评价/独立分析；审查与实际执行状态只看[当前交接](AI_RESEARCH_HANDOFF.zh.md#当前工作)。既有18次原生调用复用，不增加资格尝试，不重复启动已派发的控制器或已自删监听。已发生资源停止的恢复仅按[具体补充](../reports/documentation/20261019/optuna_recovery/scope.zh.txt)及最新用户决定：9GiB空闲/允许其他CUDA、从S_pos第2次搜索原参数继续，三个已完成实验不重跑；保留失败收费，本轮明确免审、独立主审及恢复门通过后已于10月19日18:58恢复，实际状态只看交接。

10月21日服务器重启后，用户明确要求“从中断的那次搜索开始重新运行”，并确认“此次依旧免审”。按[本次特定恢复补充](../reports/documentation/20261021/optuna_reboot/scope.zh.txt)及[独立主审](../reports/documentation/20261021/optuna_reboot/primary.zh.txt)保留25次完成结果，从S第9次搜索原参数/种子完整重跑，再接续原36+9；保留中断费用与额外train1，最终实际读取46/37，科学轨迹数不变。该次恢复使用optuna_reboot/workspace；Optuna现已结束，三次workspace均只按历史身份追溯，不据旧路径另起作业。

搜索按已固定双域U和精确tie规则选参；九次复验全部盲完成后统一开放，不依据中途复验结果增改候选。除上述本次明确授权的服务器重启恢复外，正式失败、超时或非有限值保留已用预算和访问账后停止，无自动重试。完整结果按合同回传并独立分析；未完成的日常进度核对按用户要求简报，不另造流水。新包外发仍按具体包权限。

H_cons、固定S_pos/H_pos及本轮Optuna均已完成。Optuna结果外审和独立主审已结案，两候选六格门均未通过；H_pos旧A最终排序有条件优势，新B及较少遗忘未获确定证据。当前只覆盖A→B及重复开发集，三固定新seed不替代独立最终确认。其后续三阶段和三域独立留出范围现已由本页顶部获批五方法方案确定；原AB80群设想不再作为当前确认方案。Linux遗留19份Optuna权重已按用户要求删除，Memory和证据保留；不能直接续用已删除端点，不自动重训。当前结论、审查和保管从[交接顶部](AI_RESEARCH_HANDOFF.zh.md#当前工作)进入。
## 已完成路线与后续研究边界

已批准的[C_q具体实现与执行合同](SELLER_ALIAS_QUERY_REPLAY.zh.md)固定四臂已正常完成，保存结果已回传并在Linux独立分析；原资格、失败及[实现主审](../reports/documentation/20261013/query_replay_implementation/review/primary_disposition.zh.txt)证据按适用范围复用。[原方案](../reports/documentation/20261013/query_replay/proposal.zh.txt)、原判据和上传包冻结。本次获准的结果外审已完整回收并经[独立主审](../reports/seller_alias_continual/20261014/query_replay_result/review/primary_disposition.zh.txt)关闭：补明未定义字段、分类汇总和证据范围，未修改正式数值，也没有需要重训的科学阻断；本固定实验按原范围结案，实际收尾见[交接顶部](AI_RESEARCH_HANDOFF.zh.md)。不得据已见结果自行调参、重训、补种子或恢复旧C诊断。

此前已完成[静态外审与独立处置](../reports/documentation/20261013/s_contribution/review/primary_disposition.zh.txt)，区分“有效基础系统”“值得保留的任务化候选”和“贡献充分的最终方法”。S符合前两项，第三项尚未建立；不因构件已有就否定组合价值，也不把独立历史增益设成所有增量贡献的硬门槛。若主张历史专属作用，仍须相应归因。新增近邻C²Aug及FDR覆盖已补足。具体机制缺口收窄为：处理重组引入的共同原账号来源证据，同时保留有价值的非退化重组；严格禁止所有来源重叠会退化为整账号置换，简单屏蔽未成为推荐方案。该静态S贡献研究可在已有范围内继续形成明确运算与证伪逻辑，本段不授权其新训练或恢复旧C诊断，也不恢复上方已结束的Optuna作业；新包外发、新执行及最终方法选择分别遵守对应许可，外审不自动授权。

C_q四臂窄比较保留逐自然查询排除来源、有向监督、D0外解析响应MSE、Q/C_q共同增强键及完整共同A状态；整个历史目标的零行列和交互盲区、K2正例盲区、R的查询无关偏置／幅度代价和bootstrap条件控制继续限制归因。当前[结果](../reports/seller_alias_continual/20261014/query_replay_result/report.zh.txt)未建立相对完整S的排序提升，不能用更低缓存R替代自然查询收益，也不能自动要求提高系数。原valid仍是开发评价，单配置区间不覆盖多seed或研究选择。已按原范围结束固定实验，保留完整S和有效未达收益证据；若形成下一候选，先说明新问题、可区分比较和预算，按相应范围另行取得许可。

既有C研究证据与失败边界继续保留，不能把S收益追认为C贡献。第一轮见[工作稿](../reports/documentation/20261011/retention_design/proposal.zh.txt)、[均值／方差裁决](../reports/documentation/20261011/retention_design/candidate.zh.txt)、[后续筛查](../reports/documentation/20261011/retention_design/search.zh.txt)及[信息采集工作稿](../reports/documentation/20261011/retention_design/acquisition.zh.txt)；收缩、跨账号配对的静态否决和C_cf固定配置未建立主要改进，均按原范围解释。先前旧C诊断已停止，原批准与CPU证据保留，不补测或转用其预算。后续若形成候选，仍须说明具体信息、实际运算／更新、近邻区别和可推翻的比较，不要求实验前保证胜出，也不以候选名称为继续理由。

[三臂归因合同](SELLER_ALIAS_RECORD_ATTRIBUTION.zh.md)及[72小时补充](../reports/documentation/20261009/record_attribution/qualification/authorization_72h.json)保留已结束实验的计算、预算及权限来源，不转为新方案许可。已完成的核验和审查按适用范围复用，新的实质机制再明确受影响核验、数据权限、预算和结果判据；不为下一阶段恢复已退出监听或已删权重。

原C/S、ER、LOGIT、风险重放及关系／函数记忆等已结束运行的结论与证据，从[交接历史入口](AI_RESEARCH_HANDOFF.zh.md#近期历史进展)回溯，不在本页重复状态。权重清理不构成重训许可，实际保管与清理周期见[存储记录](STORAGE.zh.md)。可继续已授权文献与数学分析，具体新方案形成后按范围汇报，不凭已开放结果自行开启新实验。

## 已关闭五臂与后续研究

用户2026-10-08要求提高C相对S的收益，并加入LOGIT0.1及本算法的过拟合检查。[原建议与诊断规格](../reports/documentation/20261008/record_replay_improvement/proposal.zh.txt)及[冻结合同](SELLER_ALIAS_REPLAY_IMPROVEMENT.zh.md)对应的五臂10,368更新、216物理逐epoch诊断已全部完成，结果外审／主审已收尾。C_plus未改善原C/S主要表现，原三组23项同时全过条件false，登记为本固定配置负结果；局部正结果、原C/S结果和LOGIT历史复现缺陷R1分别保留。用户已明确撤下C_plus、S_strong和五臂专用活动入口／测试，保留原C/S、LOGIT及通用诊断；方案、policy、冻结源码和证据保留，见[关闭记录](../reports/maintenance/20261009/replay_improvement_closure/closure.zh.txt)。该轮已结束，不自动调参、复跑或扩种子。

10月8日用户要求后续涉及这些算法的实验实际纳入过拟合诊断：同一A0/eval模式，当前fit和存活缓存分别与相应域未见群比较，区分raw与校准、缓存拟合与旧域遗忘、模型拟合与跨轮开发选择偏差。逐轮valid预测先盲存，完整训练后按合同一次开放；诊断不得改变训练状态/随机流或取回淘汰群。本轮六轮曲线已经保留，原更早LOGIT/C/S运行缺失的曲线不倒填。当前45条按后续已批准的[五方法范围](../reports/documentation/20261023/continual_baselines/proposal.zh.txt)及[主审B02/B11补充](../reports/documentation/20261023/continual_baselines/review/primary_disposition.zh.txt)执行：明确不保留全epoch诊断、不做未训练初始全量评分，保留阶段端点、训练/Memory记录及实际恢复证据；因此本批不能定位某个epoch开始过拟合，也不能把R11解释为相对预训练的净首学增益，不据本段旧要求追加计算。通用诊断模块保留不等于已接入每个入口；未来合同按当时明确范围实际接线并核验，不把规格或文件存在写成已执行。

围绕有限历史信息的代表性、重复强度、排序保持与新学习取舍提出可检验机制，先对照最近方法，明确主张、公式、比较与最小消融。可以整理方案和已有证据；创新算法、追加候选／种子、训练预算、数据权限和新最终留出未经具体确认不执行。

未来开发合同分别写技术读取、结果开放、适应性选择范围，可批准有限候选及总预算；最终确认事前固定方法、对手、端点和一次开放，不能用已经开发过的valid或已用基础test冒充独立确认。详见[科研纪律](RESEARCH_DISCIPLINE.zh.md#4-三种访问权限)。

最终目标是CCF B／C投稿及实质方法贡献；已知方法适配或基线改善不等于创新已经成立。中文合成主线的有效负结果保留，获准的新对照和原版补充按各自合同推进；VendorLink已有独立的实现与审查范围，不能再用旧“全部扩展都等中文主线结束”解释为暂停该范围。大候选库、无正候选查询及其他未批准研究仍需具体方案。

## 已确认的下一步：保留D基线，先做针对性改进

此标题仅兼容旧文档的历史锚点，指2026-09-24时的决定，相关基础改进和验收已完成。它不重新暂停当前持续学习。原D权重后续已按明确清理要求删除，旧“保留”不作为当前加载许可。[原决定](../reports/documentation/20260924/next_plan/decision.json)、[完整历史计划](https://github.com/bililateral/cross-lingual-DIL/blob/eecee09f4d5c25f88d054cab359ba6fb97dcb8fa/docs/RESEARCH_PLAN.md)、[历史入口](AI_RESEARCH_HANDOFF.zh.md#历史与证据)。
