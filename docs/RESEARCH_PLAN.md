# 科研计划

状态与观测只维护在[当前交接](AI_RESEARCH_HANDOFF.zh.md)，本页说明后续动作及其授权边界，不复制执行流水。

VendorLink扩展已明确选择完整H迁移及原论文方法对照，四臂为VendorLink_SEQ、H_SEQ、H_ER、H_full。AlphaBay/Dream/Agora来源、固定划分导出与实施审查已经落实，10月30日单路G1通过且已按用户决定结案，G2三路未通过，见[单路主审](../reports/documentation/20261030/vendorlink/native/primary_disposition.zh.txt)；不再等待旧监听或要求补三路资格。下一步是依据现有单路实测补正式合同：共同当前域epoch E及选择依据、拟36轨迹/108阶段/9252E当前更新、主比较与Q/O/F/N公式、query及域权重/依赖处理、完整盲评分后标签开放、单路累计训练/评分/保存/失败预算与资源门。现有单轨迹execute、盲评分和query AP可复用，整批派发/累计账、216格完整门、4104次留出群前向及固定统计还须接成正式入口并作受影响核验。代理任务与原方法掩码平均补全假设保持；不增加第五臂或HPO，不借490/822/110小时预算，不把观察账号视图当真实潜在同控真值。首阶段共享未实现，按108阶段估算。正式36条、E、预算和评价开放尚未获准，不把资格剩余额度转作训练。

FOREVER的最小方法/阶段/安全点实现、增量外审主审及单路39调用原生资格也已完成，见[后续单路补充](../reports/documentation/20261030/forever/implementation/qualification/scope_update.zh.txt)及[最终资格](../reports/documentation/20261030/forever/implementation/qualification/final/gpu_summary.json)。现有[修订方案](../reports/documentation/20261030/forever/revision.zh.txt)拟9条FOREVER、复用H27端点、新120群/4320群前向和H−FOREVER固定O/F主比较，零新增HPO；这些定义不需重新选方法或超参数。正式CLI目前显式关闭，尚需在获准范围内接通train-only Supply、九轨迹派发/累计账/完整门、H端点与原校准绑定、新确认生成/盲评分/集中开放及统计、成功后full/anchor依赖清理。116h（108训练+6确认+2辅助）仍为提议，须按已测单路更新工期/资源和失败账继承；不能继续引用旧三路16—30h预计工期。额外全P锚点在资格中已批准，正式使用仍应在正式合同中明确接受，不宣称等总历史状态/算力/HPO。阶段末校准与失败留证沿用当前必要修复证据，不预先改变校准公式/门槛，也不因L²R失败要求FOREVER重跑完整阶段资格。用户“先不要启动正式训练”仍有效。

原版S_default＋历史H（G_half）的补充范围也已批准：三顺序×三种子×两方法18条、最多三路、108h训练＋2h辅助=110h，单条6h；只训练、保存54端点及旧dev盲分数，不开放评价标签、不新增确认或HPO。[等待增量主审](../reports/documentation/20261026/continual_defaults/waiting/review/primary_disposition.zh.txt)及必要修复核验已完成，[唯一监听](../reports/documentation/20261026/continual_defaults/waiting/launch.json)已启用。下一步由现有监听等指定L²R／ER+GCR整批训练、生成、确认和分析全部成功、相关进程退出，再按实验所需剩余显存和RAM/disk门逐路派发；不要求整卡没有CUDA进程，禁止重复启动。失败停止且不自动重试，原失败和费用均承接。

两新对照的现行执行范围已于2026-10-26明确批准：[822小时执行范围](../reports/documentation/20261026/continual_comparison/execution.zh.txt)覆盖L²R／ER+GCR各9条、H/S54阶段权重复用及原定新120群确认；新增编排明确免外审，既有算法／资格结论复用。按固定队列完成18条盲训练、全108端点门后连续进入生成／确认／统计；失败保留费用并停派，不自动重试。下方关于新正式预算尚待授权的旧段落不再是当前状态，原算法及结果审查边界保持。

原五方法45条及冻结新确认／统计已经完成，结果外审与独立主审已结案，见[结果主审](../reports/seller_alias_continual/20261026/continual_result/review/primary_disposition.zh.txt)。原O/F共同主要门未过：H相对S最终MAP正向，但额外抗遗忘尚未确认。原490累计小时及恢复账保留，不再据旧启动计划重复训练或开放标签。H/S54端点继续服务已批准新对照确认；其余三方法的81份独立权重已按用户选择清理，非权重证据保留，见[存储记录](STORAGE.zh.md#2026-10-26-五臂中的三臂权重清理)。

原45条的资源故障与恢复按[当前明确重试范围](../reports/documentation/20261024/continual_retry/scope.zh.txt)及[独立处置](../reports/documentation/20261024/continual_retry/primary.zh.txt)接续：保留24成功轨迹，未完成21条按原身份运行；旧费用和读取不清零。资源不足只暂停补入一个空槽，其他worker继续；首次启动和其他故障条件保持。已有控制器／监听的身份、真实更新和完成状态统一看交接，不因旧待运行文字重复建队列。此恢复不是任意中断的自动断点续训授权。

L²R／ER+GCR的设计、实现和有限BGE资格复用[合并结案](../reports/documentation/20261024/continual_recovery/review/primary_disposition.zh.txt)，不重复资格；当前18条及新确认按上述822小时授权执行。原“新比较及独立预算尚待授权”是批准前历史，不再构成待办；资格及顺利运行仍不能代替最终效果或确认结论。

VendorLink三市场数据与原生方案的[外审主审](../reports/documentation/20261029/vendorlink/three_market/review/primary_disposition.zh.txt)、[实施修复闭合](../reports/documentation/20261029/vendorlink/native/review/primary_disposition.zh.txt)及后续[单路结案](../reports/documentation/20261030/vendorlink/native/primary_disposition.zh.txt)共同构成已有证据。10月29日10GiB/9GiB allocator一次尝试、旧等待门及G2提议为历史；后续用户取消allocator限制并接受单路范围，旧失败与读取账保留。后续正式合同及新增编排/评价只核验实际增量，按科研纪律第6节处理需要外审的新增范围；已闭合核心方法、数据导出和单路GPU资格不重复测试或再审，不将本次进展梳理当成新的实施/预算/正式运行许可。

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
