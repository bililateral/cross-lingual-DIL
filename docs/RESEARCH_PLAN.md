# 科研计划

状态与观测只维护在[当前交接](AI_RESEARCH_HANDOFF.zh.md#当前工作)，本页仅列后续动作和授权边界。有效合同及增量修复从[设计入口](CURRENT_EXPERIMENT_DESIGN.md)进入，不复制PID、计数或过期工期。

## VendorLink账号监督：按已审阶段链推进CE与遗忘检验

[实现及必要核验](../reports/documentation/20261110/vendorlink_redesign/implementation/report.zh.md)、完整外审和[主审](../reports/documentation/20261110/vendorlink_redesign/implementation/review/primary_disposition.zh.md)已闭合，无新增代码修复或CPU/GPU资格。按用户最新“没问题了就正式开始实验”及[现行范围补充](../reports/documentation/20261110/vendorlink_redesign/implementation/scope.zh.md)连续执行，不再恢复旧暂停或重复询问开始许可；实际状态只在交接对应项维护。

继续既有唯一控制器的D八个A/B冷启动、条件P六条及M六条、C完整端点确认；固定51/52/53，最多两路H＋V/H＋H、最多一条V并行，88h=80训练/4GPU/4辅助，32GiB输出帽。完整阶段后按现有门开放结果，逐域/逐seed/配对差、均值/样本SD/范围分别解释学习、遗忘、Full系统及Full−ER组件；阶段前提失败正常停止，不追选候选、顺序或seed。不要重复启动，不重建已合格12/4/2/2数据，不恢复旧层叠控制器或已删权重；当前运行的来源/gate保持冻结，按实际并发与速度更新工期。

已定标准保持：H相对V_CLS逐域初学均值最多低2pp/逐seed最多低5pp，明显遗忘均值3.5pp及跨seed保护；Full同时满足旧域最终、遗忘、新域和外部基线门。取消人为GPU配额及固定GiB整PID终止，资源预留仍执行；单worker OOM退出，健康worker继续当前阶段剩余独立任务，失败不重跑，缺项不开放该批效果或下游。工程核验不证明科学目标已实现。

原20261109完整结果的5pp门、STOP_LEARNING和71450秒费用保持；后设3.5pp不追改验收。原关系候选已删，跨账号发现仍为无真值应用输出，不报已验证马甲准确率。旧ABC/Evolution结案及中文暂停不动。状态只在交接唯一当前项维护；根docs仅Windows，本地提交、不推送。

## Evolution：已结束路线与保留边界

以下记录已结束路线的既有范围，不是当前待启动任务。用户当时接受方向咨询建议，明确“继续推进，采用双路同时运行”，并批准开发数据构建与固定96条fit语言判断。历史来源为[六轨迹执行范围](../reports/documentation/20261108/evolution_registered/execution/scope.zh.md)、[许可](../reports/documentation/20261108/evolution_registered/execution/authorization.json)及[实现证据](../reports/documentation/20261108/evolution_registered/execution/implementation.zh.md)；已核实结束状态见交接唯一Evolution项。原[30轨迹草案](../reports/documentation/20261108/evolution_registered/proposal.zh.md)只作来源，后续24条及24小时预算未获准。

已获准的一次Linux CPU资格及“允许旧test作为新确认候选”补充已执行，随后已按许可导出288开发账号并保留288确认身份，确认正文未导出或开封。旧fit/dev可用于开发；旧test与未使用账号可作为新确认候选，后者必须未参与旧训练/开发或效果选择。新任务同账号身份跨fit/gallery/query，文本与角色日期严格隔离；开发与确认账号仍互斥。旧test不是科学上永久不可用，使用前须保留已明确的用途转换和旧访问历史，不能以新合同重置已见结果。

本次定向实现外审及R01/R02主审闭合、开发导出、语言门、正式六条与完整评分分析均已完成，来源从[正式门](../reports/documentation/20261108/evolution_registered/execution/formal_gate.json)进入。[本轮结果](../reports/documentation/20261108/evolution_registered/result/report.zh.md)为有效的STOP_LEARNING：两SEQ的新学习增量通过，但绝对能力保护与共同明显遗忘均失败；冻结对照显示更新收益，不能笼统写为无学习或任何地方都不遗忘。用户明确本次结果不提交外审，已完成保存字段主审核对。按既定停止条款结束本有限链及当前Evolution资料上寻找明显遗忘的协议扩展，不启动H_ER/Full、V_ER或独立确认，不改门、换seed或追加HPO翻转结果。控制器没有M/C路径；未来若另立研究目标或数据，保留本轮已见开发结果与停止事实。

当前只导出96×3开发，确认身份保留、不导出正文；严格4/2/2、固定H1/V1、原P技术投入门、均值与逐格报告保持。用户后续明确删除本次六轨迹训练权重，已按[保管记录](STORAGE.zh.md#2026-11-08-registered六轨迹训练权重定向删除)执行；已有评分/指标/日志与诊断保留，不能再把这些端点列为可载模复用输入，重训仍需另行授权。冻结评分列各批刚学后和最终差值，不新增自动价值门、不以最终冻结更好否定曾获得后丢失的能力。Full主要效果/代价口径须在未来M结果前固定。确认候选余量极小，后续若不足不得自动借旧dev填确认。

费用按新5小时桶承接既有资格352秒及本轮全部失败/核验/准备，不借旧实验额度。必要手写CPU子帽180秒在辅助1小时内；不追加GPU资格。已闭合方向、数学与并发证据按适用范围复用，只审本轮实际接线变化；新具体审查包需外发许可，外审及主审不自动扩大新数据或后续轨迹权限。

原季度路线仍结束，不续原C/D、不重训已删权重、不改门搜索顺序/种子/池规模。原[科学方案](../reports/documentation/20261107/evolution_pilot/proposal.zh.md)、[固定数据](../reports/documentation/20261107/evolution_pilot/data_preparation/report.zh.md)、[双路补充](../reports/documentation/20261107/evolution_pilot/execution/scope.zh.md)、[原实现主审](../reports/documentation/20261107/evolution_pilot/implementation/review/primary_disposition.zh.md)与[GPU证据](../reports/documentation/20261107/evolution_pilot/gpu_single/report.zh.md)是已发生事实的来源。A/B未揭示共同明显遗忘，127开发诊断同样没有；原VendorLink有效负结果、ABC前置门停止且无新候选训练权重、全部费用保持。已完成删除的对象和Git恢复界限见[存储记录](STORAGE.zh.md)，不可恢复的模型及127原始分数不冒称仍可复用。

主张仍为同观察账号文本检索，不扩为真实跨账号同控。全局未来文本参与离线清洗、观察日非创建日、启发式语言和近重复近似边界保持，H演示改写行不得作效果/选参依据。中文H消融等候用户明确恢复，本轮不干预。按已有要求维护唯一交接、计划与决定记录，本地提交，不将旧批推送许可扩至新材料。

## 其他研究线的既有后续安排

H历史两系数[本轮结果](../reports/seller_alias_continual/20261107/history_result/report.zh.md)的网页审查及[7项独立处置](../reports/seller_alias_continual/20261107/history_result/review/primary_disposition.zh.md)已闭合，本范围不追加候选、重训、标签读取或重采样。原执行与结果状态只见交接；保留新H27和原H27及完整原件。用户已选定历史开发版H作消融母体，并确认按[已审方案](../reports/documentation/20261106/h_history_ablation/review/primary_disposition.zh.md)实施；[实现及主审处置](../reports/documentation/20261107/h_ablation_prepare/review/primary_disposition.zh.md)现为下一阶段入口。用户随后明确“开始正式训练”，以[本次正式许可](../reports/documentation/20261107/h_ablation_start/authorization.json)及源绑定正式门执行唯一63条队列；随后用户确认暂停派发并明确核对后直接部署；[已核验暂停交接](../reports/documentation/20261107/h_ablation_pause/review/primary_disposition.zh.md)按当前交接的实际状态执行，在跑任务完成、空闲不续领，新后处理同样等待。仅用户明确恢复后续接原清单，完整成功后才自动进入既定新确认生成/评分/分析，缺项或运行失败不进入下游。待完整结果后按原范围回传、独立分析和具体包结果外审，再按实际用途处理权重；不追加GPU资格、候选或失败重试。英文ABC若前置条件未成立，按原约定停止；后续数据/目标/候选变更另行明确，不自动寻找遗忘更明显的子集。

L²R／ER+GCR的[结果分析与审查处置](../reports/seller_alias_continual/20261101/comparison_result/review/primary_disposition.zh.txt)已完成，[本批符合条件的权重清理](../reports/seller_alias_continual/20261101/comparison_result/cleanup_receipt.json)已实际执行；本范围无剩余科研动作。保留完整指标/分数/日志/实际来源、逐文件清单、旧失败与费用；H−S额外抗遗忘仍未确认，不为翻转原结论追加训练、调参、标签解析或重采样。H/S54基线、供H_pos_history使用的原H27及预训练模型继续保留；已删除的新对照权重不能由SHA恢复，若需新实验另按范围确认。

S_default＋历史H（G_half）的[方案A](../reports/seller_alias_continual/20261102/defaults_result/scope.zh.txt)结果从[分析及审查](../reports/seller_alias_continual/20261102/defaults_result/report.zh.txt)使用：四固定配置共同60旧dev群、全22指标与回顾条件区间、未定义覆盖及合并分类均保留，不把开发集结论变成独立确认或依据其追加训练/选参。代码/评价/分析审查按本轮证据结案，无必要补跑；按用户后续具体确认，[默认54份权重已留证删除](../reports/seller_alias_continual/20261102/defaults_result/cleanup_receipt.json)，结果、分数、日志和Memory继续保留；旧S_tuned/H_pos_tuned的54份基线保留，其中H27继续供H_pos_history使用。已删模型不能凭SHA恢复，未来重新训练或载模评价须另定范围；旧训练及失败账不改，后续仍原位维护唯一交接项。

VendorLink和FOREVER沿用[VendorLink四臂36条/每市场40轮/单路296h](../reports/documentation/20261030/vendorlink/formal/plan.zh.txt)及[FOREVER新增9条/复用H27/116h和新120群/4320前向](../reports/documentation/20261030/forever/formal/plan.zh.txt)的科学范围，调度以[11月1日完整确认的补充](../reports/documentation/20261101/formal_chain/scope.zh.txt)为准：H/S18条54端点完整有效且相关进程退出后，既有唯一下游链自动进入FOREVER；FOREVER九条及生成、确认、分析完整有效且相关进程退出后，自动接VendorLink。该具体决定已替代10月30日“等新开始指令、不建自动等待”的限制，不再重复索要同范围开始许可或创建第二份监听。FOREVER最多三槽、allocator9GiB、整个PID监控阈值及每槽预留10GiB，可先少槽后补足，扣除已承诺而尚未反映的占用；成功槽续领，失败槽退出、健康槽继续，无自动重试。VendorLink保持单槽17GiB准入。缺项或运行失败阻断下游，完整有效的科学负结果允许交接；原116h/296h及单条12h/8h不变，累计任务费用不能当作实际工期。FOREVER对照仍为原H_pos_tuned的27模型，本次H/S默认模型只是前序执行条件。启动时由实际入口核对原H27、部署及共享资源。

复用既有单路GPU资格、[11月1日显存短测](../reports/documentation/20261101/forever_memory/report.zh.txt)及适用旧审查；新增并发/交接按[两轮定向外审与主审](../reports/documentation/20261101/formal_chain/review/primary_disposition.zh.txt)结案。三槽是已批准的调度范围，不能写成三路大模型资格已通过，也不能把旧单路工期直接除3。不追加GPU试跑、G1/G2或无具体缺口的重复审查；实际进度和下一步只看[当前交接](AI_RESEARCH_HANDOFF.zh.md#当前工作)。

FOREVER的[结果回传/分析及清理范围](../reports/seller_alias_continual/20261103/forever_result/scope.zh.txt)已由[完整报告](../reports/seller_alias_continual/20261103/forever_result/report.zh.txt)、[外审逐项处置](../reports/seller_alias_continual/20261103/forever_result/review/primary_disposition.zh.txt)及[实际删除回执](../reports/seller_alias_continual/20261103/forever_result/cleanup_receipt.json)闭合；本范围无剩余科研或清理动作。保留全部分数、指标、日志、Memory和原有效结论，原H_pos_tuned27端点继续作为H_pos_history未来输入。已删FOREVER权重不可凭SHA恢复，不从本轮正结果派生新候选、消融或重采样；既有VendorLink链按原范围继续。

VendorLink的[本轮结果分析范围](../reports/seller_alias_continual/20261104/vendorlink_result/scope.zh.txt)承接完整回传、逐项指标分析与结果外审，现按[完整分析](../reports/seller_alias_continual/20261104/vendorlink_result/report.zh.txt)及[外审逐项处置](../reports/seller_alias_continual/20261104/vendorlink_result/review/primary_disposition.zh.txt)结案；两项必要解释补全已关闭，无需代码修复、补实验或再审。保留当前固定负结果，原结果收尾不追加训练、标签读取或重采样。用户后续固定ABC方向，要求先准备表示改进和英文训练适配方案，重点减少学完Agora后对Dream的遗忘并保护新市场学习，明确先不启动训练；用户随后明确两worker安排并要求先完成方案、必要核验和外审，现从[独立准备方案](../reports/documentation/20261104/english_abc/plan.zh.txt)及[实现/核验边界](../reports/documentation/20261104/english_abc/implementation.zh.txt)承接；随后用户已批准HPO整批后的自动接续；该接续已执行dev前置评价并按原门正常停止，实际依据见[本轮状态附注](../reports/seller_alias_continual/20261107/history_result/report.zh.md#7-实际耗时保管和自动接续)。后续科学范围变化另行明确，有限GPU短测许可不扩成失败自动重试；运行状态只见交接唯一当前项。ABC四方法/三种子/三阶段36权重保留，供后续开发集前置评价；按随后明确清理许可，[BCA/CAB72权重已定向删除](../reports/seller_alias_continual/20261104/vendorlink_result/cleanup_receipt.json)，完整结果证据保留，ABC前置开发评价结果按原停止规则保管。依赖扫描发现57群中48群相连且包含Dream全部8群，用户接受固定数据描述比较：保存完整排序/遗忘差值，不计算整体置信区间、不作统计验收，不宣称真实潜在同控识别或统计确认抗遗忘。不得为获得区间重分群、选账号、加数据或改统计模型。FOREVER保留额外全参数锚点、原O/F主比较及N/Z点值保护，不重训H、不追加HPO；不将两线预算或评价权限互借。

原五方法45条及其新确认/统计、结果外审和[主审](../reports/seller_alias_continual/20261026/continual_result/review/primary_disposition.zh.txt)均已结束。H相对S最终MAP正向，但原O/F共同主门未过、额外抗遗忘未确认，不重复旧恢复或为翻转结果重训。H/S54端点继续保留，其中H27已供FOREVER比较并继续供H_pos_history使用；其余三方法81份独立权重已按用户选择删除，非权重证据和原490h/失败账保留，见[存储记录](STORAGE.zh.md#2026-10-26-五臂中的三臂权重清理)。
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
