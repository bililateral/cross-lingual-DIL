# 科研计划

状态与观测只维护在[当前交接](AI_RESEARCH_HANDOFF.zh.md)，本页说明后续动作及其授权边界，不复制执行流水。

## 已确认范围内

按用户2026-10-13最新决定，先完成完整S的贡献定位，核查直接近邻，判断它是否值得作为最终方法基础；不足时再明确具体机制缺口。暂不开新训练，不恢复旧C单独诊断，也不因较早“没找到前不要停下来”而越过本轮静态范围。完整结论、来源与后续最小证伪要求见[贡献定位报告](../reports/documentation/20261013/s_contribution/report.zh.txt)；当前外审及交付状态只看[交接顶部](AI_RESEARCH_HANDOFF.zh.md)。

本轮已完成[静态外审与独立处置](../reports/documentation/20261013/s_contribution/review/primary_disposition.zh.txt)，区分“有效基础系统”“值得保留的任务化候选”和“贡献充分的最终方法”。S符合前两项，第三项尚未建立；不因构件已有就否定组合价值，也不把独立历史增益设成所有增量贡献的硬门槛。若主张历史专属作用，仍须相应归因。新增近邻C²Aug及FDR覆盖已补足。具体机制缺口收窄为：处理重组引入的共同原账号来源证据，同时保留有价值的非退化重组；严格禁止所有来源重叠会退化为整账号置换，简单屏蔽未成为推荐方案。可在已有静态范围内继续形成明确运算与证伪逻辑，暂不开训练或旧C诊断；新包外发、新执行及最终方法选择分别遵守对应许可，外审不自动授权。

既有C研究证据与失败边界继续保留，不能把S收益追认为C贡献。第一轮见[工作稿](../reports/documentation/20261011/retention_design/proposal.zh.txt)、[均值／方差裁决](../reports/documentation/20261011/retention_design/candidate.zh.txt)、[后续筛查](../reports/documentation/20261011/retention_design/search.zh.txt)及[信息采集工作稿](../reports/documentation/20261011/retention_design/acquisition.zh.txt)；收缩、跨账号配对的静态否决和C_cf固定配置未建立主要改进，均按原范围解释。先前旧C诊断已停止，原批准与CPU证据保留，不补测或转用其预算。后续若形成候选，仍须说明具体信息、实际运算／更新、近邻区别和可推翻的比较，不要求实验前保证胜出，也不以候选名称为继续理由。

[三臂归因合同](SELLER_ALIAS_RECORD_ATTRIBUTION.zh.md)及[72小时补充](../reports/documentation/20261009/record_attribution/qualification/authorization_72h.json)保留已结束实验的计算、预算及权限来源，不转为新方案许可。已完成的核验和审查按适用范围复用，新的实质机制再明确受影响核验、数据权限、预算和结果判据；不为下一阶段恢复已退出监听或已删权重。

原C/S、ER、LOGIT、风险重放及关系／函数记忆等已结束运行的结论与证据，从[交接历史入口](AI_RESEARCH_HANDOFF.zh.md#近期历史进展)回溯，不在本页重复状态。权重清理不构成重训许可，实际保管与清理周期见[存储记录](STORAGE.zh.md)。可继续已授权文献与数学分析，具体新方案形成后按范围汇报，不凭已开放结果自行开启新实验。

## 已关闭五臂与后续研究

用户2026-10-08要求提高C相对S的收益，并加入LOGIT0.1及本算法的过拟合检查。[原建议与诊断规格](../reports/documentation/20261008/record_replay_improvement/proposal.zh.txt)及[冻结合同](SELLER_ALIAS_REPLAY_IMPROVEMENT.zh.md)对应的五臂10,368更新、216物理逐epoch诊断已全部完成，结果外审／主审已收尾。C_plus未改善原C/S主要表现，原三组23项同时全过条件false，登记为本固定配置负结果；局部正结果、原C/S结果和LOGIT历史复现缺陷R1分别保留。用户已明确撤下C_plus、S_strong和五臂专用活动入口／测试，保留原C/S、LOGIT及通用诊断；方案、policy、冻结源码和证据保留，见[关闭记录](../reports/maintenance/20261009/replay_improvement_closure/closure.zh.txt)。该轮已结束，不自动调参、复跑或扩种子。

后续涉及这些算法的实验必须实际纳入用户要求的过拟合诊断：同一A0/eval模式，当前fit和存活缓存分别与相应域未见群比较，区分raw与校准、缓存拟合与旧域遗忘、模型拟合与跨轮开发选择偏差。逐轮valid预测先盲存，完整训练后按合同一次开放；诊断不得改变训练状态/随机流或取回淘汰群。本轮六轮曲线已经保留，原更早LOGIT/C/S运行缺失的曲线不倒填。通用诊断模块保留不等于已接入历史入口；未来新合同须实际连接并做受影响核验，不把规格或文件存在写成已执行。

围绕有限历史信息的代表性、重复强度、排序保持与新学习取舍提出可检验机制，先对照最近方法，明确主张、公式、比较与最小消融。可以整理方案和已有证据；创新算法、追加候选／种子、训练预算、数据权限和新最终留出未经具体确认不执行。

未来开发合同分别写技术读取、结果开放、适应性选择范围，可批准有限候选及总预算；最终确认事前固定方法、对手、端点和一次开放，不能用已经开发过的valid或已用基础test冒充独立确认。详见[科研纪律](RESEARCH_DISCIPLINE.zh.md#4-三种访问权限)。

最终目标是CCF B／C投稿及实质方法贡献；当前已知方法适配或基线改善不等于创新已经成立。先完成当前中文合成主线，再决定大候选库、无正候选查询或其他扩展。

## 已确认的下一步：保留D基线，先做针对性改进

此标题仅兼容旧文档的历史锚点，指2026-09-24时的决定，相关基础改进和验收已完成。它不重新暂停当前持续学习。原D权重后续已按明确清理要求删除，旧“保留”不作为当前加载许可。[原决定](../reports/documentation/20260924/next_plan/decision.json)、[完整历史计划](https://github.com/bililateral/cross-lingual-DIL/blob/eecee09f4d5c25f88d054cab359ba6fb97dcb8fa/docs/RESEARCH_PLAN.md)、[历史入口](AI_RESEARCH_HANDOFF.zh.md#历史与证据)。
