# Evolution 预实验实现

后续增量：完整实现外审的五项局部缺陷已按[独立主审](implementation/review/primary_disposition.zh.md)和CPU04/05关闭；新增来源、语言样本绑定、CSV字段及资源检查进入真实入口，生产源码为2ba0f685。CPU累计27秒，原CPU01整轮失败及原通过子项均保留。用户随后明确“把数据集准备好”，由[数据准备许可](data_preparation/authorization.json)开放的既定一次构建、分区及固定fit语言复核已[完成交付](data_preparation/report.zh.md)。下文首轮“正式数据关闭”是当时时点；最新动态状态仅见交接。训练、评分、监听仍关闭。

2026-11-07。用户在确认净学习修法、保留双SEQ遗忘门及其余安排后，明确“确认好方案后就开始具体的实现吧，考虑清楚各种边界，不要出错”。本轮据此实施[现行方案](proposal.zh.md)§7列明的实际接线和Linux py310手写CPU核验（累计不超过1200秒，计入后续辅助费用），准备实现审查。正式数据构建/划分、人工样本查看、预训练载模、GPU训练/评分、自动等待与推送仍关闭；本轮只用随机微型模型和手写输入。具体新审查包外发许可另核，不沿用旧方案包许可。

## 实际入口与对应行为

| 文件 | 实际用途 |
|---|---|
| `scripts/step30_evolution_data.py` | 保管端四表构建；调用原audit.scan与followup近重复原语；记录同一原始行，保留大小写；固定增广匹配、160/64/96分区、4条/账号、8账号群；盲文本与qrels分开，固定fit语言抽样。已按本次数据许可完成一次正式调用，不授予重建。 |
| `scripts/step30_evolution_run.py` | 同一cased工厂供训练/教师/恢复/评分；H复用原真实VJP和目标；V_ER联合类别、两次分支反传、一次clip/step；单域或三域正常完成；每任务一个起点小head引用，阶段末各一个encoder/head推理端点，无每轮latest/optimizer/Memory整包。 |
| `scripts/step30_evolution_policy.py` | 有限A候选和固定种子；B双基线门、C/D各比较和质量保护；修正后的同域同对照净学习双路径；固定5000次配对群bootstrap及内外部支持分层。 |
| `scripts/step30_evolution_evaluate.py` | 所有任务完整后九格盲评分；全部分数有限且ID/形状完整后才记一次qrels开放尝试；十指标与逐query保存。 |
| `scripts/step30_evolution_io.py` | 来源核验、不同角色访问记录、1200秒CPU子预算、48小时三桶、真实资源/保存回调；失败费用不退回。 |
| `scripts/step30_evolution_control.py` | 独立目录单worker；A→B→C→D逐门派发；失败不再派发，科学门false正常结束；无监听/自动等待/重试，正式CLI必须具备明确许可和已关闭实现审查。 |

旧 `step29_vendorlink_data.py`、`step29_vendorlink_methods.py` 及 `step29_vendorlink_run.set_rates` 只复用不改。旧run.execute/save_state和旧英文小写工厂不进入本轮执行。中文活动部署及原英文失败/结果不变。新系统名为H0_SEQ/H_SEQ/H_ER/H_full/V_SEQ/V_ER；内部Settings保留VendorLink_SEQ只表示复用的分类/余弦计算家族，不表示数据来自旧ABC。每个任务和端点都绑定Evolution时期、方案、来源与数据清单。

## 必要核验、失败与范围

CPU01—03证据由Linux既有py310在随机4层/8维微型BERT及手写输入上取得，保留正式32记录/16视图/120视图对形状，但缩短epoch及群数。生产CLI不提供这些夹具覆盖参数，仍固定768维骨干、20群×40轮和完整任务清单。这些核验未读取官方包、正式文本/标签、既有预训练权重或GPU，不能证明Evolution效果，也不能认证完整真实语料构建已经成功。CPU03资源证据只覆盖其正常准入、真实闭门失败和停止派发；后续CPU04/05才补局部原子改名、目录帽及构建中断，不泛称全部资源越界或原生GPU已经验证。source_row表示TSV解析记录序号，source_sha256表示解码行字典的JSON摘要，不冒称物理文本行及其原始字节哈希。

| 证据 | 真实发生的检查与结果 |
|---|---|
| [CPU01原回执](implementation/cpu01/receipt.json) | 数据与更新检查先通过：4条准入而非旧20条、确定匹配与分区、大小写绑定最早同一解析记录、29/32语言界限；H_SEQ/H_ER/H_full/V_SEQ/V_ER各走完3阶段×2更新，逐阶段紧凑保存并恢复九格；单域独立正常结束。随后判据夹具的局部变量覆盖A结果列表，出现TypeError，整轮记FAIL，未掩盖。失败源码可由Git 628e7311精确恢复。 |
| [CPU02原回执](implementation/cpu02/receipt.json) | 修复夹具后只补判据及运行边界：高起点正学习放行、负/零学习及混域拒绝、双基线遗忘、C不通过不派D、完整种子、5000配对群bootstrap、真实子进程失败后禁止下一派发、失败费不退款与正式门关闭。源码Git 594c726e。 |
| [CPU03原回执](implementation/cpu03/receipt.json) | 对最后新增整批评价入口作定向核验：H_full与V_ER手写微型轨迹实际训练，实际score_batch恢复0/1/2/3状态、生成18格盲分数，再由实际analyze_batch开放标签；缺一分数时标签未读，完整后一次开放，第二次拒绝。内部门通过而外部V_ER区间跨0时，总体强目标仍false。正常资源准入及真实闭门失败派发分支补核，不代表覆盖全部越界。最终科研代码来源Git d1949200。 |

V_ER累积梯度与独立“一张图上CE_current+.1CE_history”的参考在该确定性夹具中最大差为0；删除历史项后encoder梯度最大差0.001957843080163002，说明历史监督确实参与了参数更新。H_full训练日志记录非零出生保持损失；原H VJP与四层反传的数学/梯度资格复用既有已审证据，不以非零总梯度替代旧资格。三种回放的历史群抽取序列一致；H三臂当前序列相同、V两臂当前序列相同。独立20群/域reservoir夹具累计9次新教师调用、0次刷新；V_ER没有教师调用，完整序列化Memory超帽会失败。

每条微型完整轨迹实际只有3个阶段推理文件及1个起点小文件，单域只有1个阶段文件及1个起点小文件；保存内容检查不含optimizer/RNG/Memory，无latest.pt。起点恢复分数逐值相同，包括Vendor无分类头起点。正式规模124个阶段模型/52个起点小head或空head引用仍是最大计划，未产生任何正式Evolution权重，也未实测完整BERT全程资源。

[全生命周期计费](implementation/budget.json)采用外层启动至退出计时：CPU01为8秒、CPU02为6秒、CPU03为8秒，合计22/1200秒，包含导入及失败。各原receipt内部另保存测试函数耗时，它们不是全生命周期收费，不以内部较小值替代本账。小型连接/同步/文件身份核对/打包准备预记600秒保守辅助额，明确包含后续本轮收尾、排除人工作业和纯等待，不冒称实测；辅助合计622秒，训练/GPU仍0。后续构建或正式启动必须继承此账及其后新增费用，48h不重置。费用不授予未批准动作。

[回传清单](implementation/receive_manifest.json)的18份来源/原回执/日志均已双端按大小和SHA核对；最终远端12份必要代码/核验来源匹配，CPU作业已经退出。源码仅部署到`/home/yongpeng/cross-lingual/reports/documentation/20261107/evolution_pilot/implementation/workspace`，没有覆盖根目录旧科研入口或中文活动来源。

## 前提的后续处置

1. 本次实现审查已完成：[完整原文及独立主审](implementation/review/primary_disposition.zh.md)记录五项最小修复与CPU04/05定向证据。生产修复来源Git 2ba0f685，CPU05只改测试，来源Git 8ff15015；不冒称修后版本又经过网页复审，不重复GPU或全历史测试。
2. 用户随后明确“把数据集准备好”，其[固定数据准备交付](data_preparation/report.zh.md)已完成一次正式构建、960账号/3840记录及96条fit语言门。该事实由真实产物及实际入口证明，区别于上文CPU手写证据。CPU01—03的622秒、修复后627秒是各原阶段费用；后续承接[data_preparation/budget.json](data_preparation/budget.json)累计890秒，不重置旧账。
3. 正式模型/数据/许可/费用与资源绑定：后续收到实际训练指令、核对资源及其他活动合同尚未体现的显存承诺后，才可生成开放campaign的运行门。当前build/language专门许可不开放训练、评分或监听，不能沿用旧PID或旧空闲量。

一次性CPU核验脚本已在五个执行版本均从Git实际读回并核大小/SHA后退役，两端删除见[定向清理与恢复入口](implementation/cleanup.json)及同目录两端回执；九生产源码与资格来源未删，原失败/费用/通过证据和冻结来源清单不改。原Chinese/ABC/VendorLink事实不改，其他会话的并行修改未纳入本次Evolution提交或同步。

## 正式双路执行增量

用户后续正式执行指令及资源/故障补充见[执行补充](execution/scope.zh.md)。[单路GPU核验](gpu_single/report.zh.md)已实际通过并保留前置导入失败；它不认证双路或正式效果。当前只补实际控制器并发、整PID14GiB/allocator13GiB、资源承诺、费用和失败槽退出/健康槽继续领任务，部署新execution/workspace；旧数据及单路证据的实际来源保持。必要CPU增量和具体新包外审完成后沿已有授权运行，原文中的未开训/单路条件按各原时点保留，不能当作当前禁止指令。
