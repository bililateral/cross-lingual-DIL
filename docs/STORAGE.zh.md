# 科研文件存储与恢复

## 2026-11-16 第一节基线取得端点公式记录

新增[baseline_acquisition_formula.json](../reports/documentation/20261116/h_ablation_result/baseline_acquisition_formula.json)，25848字节、SHA256 2adf9a7142afa6b98c6273cef3cc10a29052d2c7d32c004af1f0d02d9c72499b。Linux位于 `/home/yongpeng/cross-lingual/reports/documentation/20261116/h_ablation_result/baseline_acquisition_formula.json`，经SFTP回传Windows后大小/SHA一致；保存9列原展示输入、22行前后值（19行实际变化）、公式参数、聚合端点与精度核对，不含标签/预测/模型。

本次用Linux既有py310一次性内联命令计算，没有生成Python脚本、输入临时文件或传输包。原显示值按Decimal精确平移；端点由已舍入汇总量代数还原，仅作公式示意，恒等式核对容差2e-9，不声称未舍入实测值。原Chinese展示JSON、原正式证据及预算均保持；前几轮公式脚本双端清理事实见下节。仅保留必要数值记录，不制造源码副本，不重置大清理周期；本轮本地提交，不推送。

## 2026-11-16 公式展示一次性脚本双端清理

用户要求及时删除本次及前几轮计算脚本。实际核对三轮生成器沿同一路径覆盖，不存在三份独立脚本；旧版本已在Git保管，未新增备份或压缩包。本次删除Windows `scripts/ablation_formula_display.py`、Linux `/home/yongpeng/cross-lingual/reports/documentation/20261116/h_ablation_result/formula_display.py`，各18478字节，SHA256均为9b19d641fd207c3f880981c89568fe807d2c82c9652d9feec9a1b1df0af626e3，共2份36956字节。删除前已核对精确路径、大小、SHA和Linux属主，无活动进程使用该脚本；删除后两端路径不存在，Linux展示目录没有遗留.tmp，前几轮索引/传输临时文件亦无遗留。本段替代下方旧记录中“源码继续保留”的当前保管安排，不追改其当时执行事实。

已从以下Git对象直接读出原始字节并核对恢复身份，恢复仅在Windows使用Git，不自动授权重跑：

| 版本 | Git提交及仓库路径 | 字节 | SHA256 |
|---|---|---:|---|
| 首次独立分布示例 | `8fcdaf21:scripts/ablation_formula_display.py` | 8690 | f627371910d218bb51b79f6d190de149d87dce639ded69e2b069c99bf1aab7aa |
| 围绕原排名的局部重排 | `69651250:scripts/ablation_formula_display.py` | 15532 | e8088686710c886fef03da82936c77bd63fb889a31de11d494e077d84216faf5 |
| 八配置取得端点平移 | `34dabb2c95dbe8854feeb11c8b4c893d57639c12:scripts/ablation_formula_display.py` | 18478 | 9b19d641fd207c3f880981c89568fe807d2c82c9652d9feec9a1b1df0af626e3 |

必要的[展示数值和来源](../reports/documentation/20261116/h_ablation_result/presentation_formula.json)保留，33027字节、SHA256 3441547c5dbe9e06ea1f71bdbca4ac3de751660ebc52467de56bca8561d98079，双端核对未变。已将可变文档对已删源码的链接改为本恢复入口；正式结果、模型、原始证据及其他实验脚本未动。不重置大清理周期，仅本地提交，不推送，用户既有结果表编辑继续保留。

## 2026-11-16 中文消融八配置取得端点公式平移

按[最新明确决定](RESEARCH_DISCUSSION.zh.md#2026-11-16八配置取得端点各增加2个百分点的公式情景)，当前[presentation_formula.json](../reports/documentation/20261116/h_ablation_result/presentation_formula.json)为33027字节、SHA256 3441547c5dbe9e06ea1f71bdbca4ac3de751660ebc52467de56bca8561d98079；当前[公式脚本恢复说明](#2026-11-16公式展示一次性脚本双端清理)18478字节、SHA256 9b19d641fd207c3f880981c89568fe807d2c82c9652d9feec9a1b1df0af626e3。源码SFTP同步到Linux非活动formula_display.py，核对实际SHA后使用py310的 --shift-acquisition 分支执行；新数值经SFTP回传核对大小/SHA一致。

唯一计算输入是上版展示JSON（SHA256 d694fe5b0be97c78f42780f7395248d7f0ce160aa2dfb5d48c0eefb14935cf5f），输入身份不符即拒绝，避免重复平移。输出保留原科学保存分项、平移前展示分项和前后端点；上版完整来源/数值由Git 69651250恢复，下方旧身份仅作历史保管记录。未读取原标签/预测/模型或改活动运行、正式结果、预算和审查。原子JSON临时路径已替换，Git索引临时文件随用随删；无新增审查包或无用文件。根docs仍以Windows为权威，本轮仅本地提交，不推送，不重置大清理周期。

## 2026-11-16 中文消融局部排名公式展示修订（历史）

按[最新修订要求](RESEARCH_DISCUSSION.zh.md#2026-11-16公式展示改为围绕原h_full的局部改善)，当前[presentation_formula.json](../reports/documentation/20261116/h_ablation_result/presentation_formula.json)为24211字节、SHA256 d694fe5b0be97c78f42780f7395248d7f0ce160aa2dfb5d48c0eefb14935cf5f；27份保存逐query指标逐份核对大小/SHA，原指标均值恢复后才作局部升位。当前[公式脚本恢复说明](#2026-11-16公式展示一次性脚本双端清理)15532字节、SHA256 e8088686710c886fef03da82936c77bd63fb889a31de11d494e077d84216faf5，Linux同字节替换原非活动formula_display.py并在执行前核对。新数值SFTP回传身份一致，原预测矩阵、pairs、owners、标签、模型均未读取；临时JSON原子写出路径已替换，无遗留临时文件或新包。

当前同名公式文件是可变展示产物；前一独立分布版本的准确源码/数值从Git 8fcdaf21恢复，以下首次保管段仅描述其当时状态，不再作为当前文件身份。原正式summary/结果/审查/费用不改，本轮只计算声明的检索侧示例，原概率与分类输出保持；没有改动运行作业或删除科研资产。代码是当前表格的必要重算入口，继续保留。既有用户文档编辑不纳入提交，不推送，不重置大清理周期。

## 2026-11-16 中文消融公式展示首次版本保管（历史）

按[明确的公式展示范围](RESEARCH_DISCUSSION.zh.md#2026-11-16中文消融map增加2个百分点的公式展示)，Linux py310从已开放的原summary计算一组评分假设的指标期望，保存[presentation_formula.json](../reports/documentation/20261116/h_ablation_result/presentation_formula.json)，12872字节、SHA256 ac43d25546975c145358e26bbb8986a7e50e56117b512c156fde66af572231f1；SFTP回传后身份一致。该文件在原冻结result目录之外，不覆盖实测summary/指标/审查/费用；不涉及标签、模型或预测载荷。

必要计算源码[公式脚本恢复说明](#2026-11-16公式展示一次性脚本双端清理)8690字节、SHA256 f627371910d218bb51b79f6d190de149d87dce639ded69e2b069c99bf1aab7aa，同字节部署于Linux本轮h_ablation_result/formula_display.py并在执行前核对；它是展示数值的重算来源，随公式记录保留，不是新增资格测试或监听。没有额外临时文件/包、模型删除或正式作业改动，不重置大清理周期。仅新增目标文档的公式展示部分，既有用户编辑不纳入本次Git提交，本轮不推送。

## 2026-11-16 中文消融完整结果与分析交付

[原始结果清单](../reports/documentation/20261116/h_ablation_result/result/source_manifest.json)绑定1319文件、26607489字节，全部具名SFTP回传至本轮result并核对大小/SHA，包括原summary、1296匿名群指标、完整性/访问/资源与成本回执。未回传正文、账号映射、pairs/owners载荷、预测矩阵、Memory或权重；已开放的汇总不增加封存材料权限。原summary SHA256为f187d1e29351e7321ab5ab534bff38eb2704adf0156b6c491b4251372dbfa30b，保留原primary_pass=false，独立分析与图表另存，不覆盖正式结果。

Linux既有py310整理保存值和绘图，Windows仅编辑/文件身份/归档核对；[29份展示核验产物](../reports/documentation/20261116/h_ablation_result/result/presentation_manifest.json)已逐份核对。正式50科学来源与原门一致，原代码精确字节供审查；表格/图源、PDF/SVG/PNG、最小字号/碰撞/对齐审计及初次静态导出检查误报均保留。原正式费用冻结，结果收尾辅助另记；无重算模型分数或新增bootstrap。最新β观测只含身份、更新计数、资源、预算及来源，不含中途效果。

用户具体批准的[结果审查输入](../reports/documentation/20261116/h_ablation_result/result/review/package.json)为8378171字节、1397载荷加清单，完整背景/原方案审查/实际来源/匿名指标和新解释均在内，已核每成员SHA，上传一次。排除清单、[许可](../reports/documentation/20261116/h_ablation_result/result/review/authorization.json)和[提交](../reports/documentation/20261116/h_ablation_result/result/review/submission.json)保管；没有把旧包许可扩成新包。[完整外审与主审](../reports/documentation/20261116/h_ablation_result/result/review/primary_disposition.zh.md)、258162字节原下载ZIP、8份载荷及完成页已保管，逐成员SHA一致，专用网页关闭。[本轮临时文件清理](../reports/documentation/20261116/h_ablation_result/result/cleanup.json)及Linux回执记录三份传输包、两份重复CPU源码和未用后端助手已按实际身份退役；CPU精确恢复Git cfbc6054，未用助手b5452c23，原送审包795fe7d4，恢复源码不授予新执行。必要通过/失败、费用和审查原件未删。模型清理事实见下节，原Full27保留，已删190端点不再列为可加载。[最终59份报告/原件镜像](../reports/documentation/20261116/h_ablation_result/result/final_sync.json)共9381165字节经SFTP回存Linux并逐份大小/SHA一致；1296群指标仍各在原Linux formal和Windows回传目录，根docs以Windows为权威。正式终账不改，本次结果准备及交付按[保守辅助包络](../reports/documentation/20261116/h_ablation_result/result/delivery_budget.json)合计4647秒，含全部等待和收尾、计量的6.561919秒CPU生命周期是子集，不重复相加。获准送审输入已退役，精确恢复Git 795fe7d4；下载原审查ZIP及全部原文仍保留。活动Linux源码不回写。本轮新增成果仅本地提交，用户结果表已有修改保持不动。


## 2026-11-16 中文消融新增训练权重定向删除

按[本次明确决定](RESEARCH_DISCUSSION.zh.md#2026-11-16核对后删除本次中文消融新增权重)，先核对中文整批完成及评分/分析验收、无活动中文进程或目标权重打开句柄、β不加载本批中文模型，再删除用户范围内的190份权重。16:11:42实际完成，合计247830452432字节，约230.81GiB，无备份。删除前每份实际SHA均与原成功完成或预算中断归档记录一致，见[逐文件路径/大小/SHA及保管基线](../reports/documentation/20261116/h_ablation_weight_cleanup/preflight.json)；实际逐路径删除、零残留及保留核对见[删除回执](../reports/documentation/20261116/h_ablation_weight_cleanup/receipt.json)。两份回执均已SFTP回传并核对大小/SHA。

| 对象 | 实际处置 |
|---|---|
| 63条新增训练的189份阶段端点 | 已删除，246526076664字节；Linux本批formal/training/对应任务/run/models/stage1、2、3.pt |
| 旧预算中断BCA_s4_ER的1份阶段端点 | 已删除，1304370776字节；同一formal/attempts/BCA_s4_ER/attempt0/run/models/stage1.pt；原失败/归档/费用保留 |
| Windows对应formal目录 | 实际不存在，没有本地权重或删除动作 |
| 原复用Full27、预训练模型及其他研究线模型 | 保留，不属于本次产生资产；Full27删除前后大小/属主/模式/mtime/inode一致 |
| 结果、盲分数、指标、Memory、日志、源/配置、访问及费用 | 6706份非权重文件保持；6540份内容SHA一致，166份Memory或生成隔离载荷仅核对元数据，未打开正文/pairs/owners或Memory内容 |
| 活动β三训练及14生产来源 | 原进程身份匹配、RUNNING，14来源大小/SHA不变，未停止或改源 |

Linux范围为/home/yongpeng/cross-lingual/reports/documentation/20261107/h_ablation_prepare/workspace/formal；删除后其中模型权重0份，目录和非权重原件保留。完成回执中历史MODELS_RETAINED字段不追改，当前保管事实以清理回执为准。已删模型不能靠Git/SHA恢复，不能再列为可载模复用输入；结果解释/审查仍沿保存指标和盲分数完成。维护核对墙钟另在回执注明，未重算指标、载模或改已闭合实验账。本次没有临时核验脚本/上传包，不触发其他资产清理；大清理上次11月6日/下次11月21日保持。当前成果仅本地提交，不推送，不纳入用户结果表和其他正在形成的结果分析文件。

## 2026-11-16 中文消融后处理恢复保管

用户要求同步完整消融结果；现场先保管了训练完成、生成资源回执失败及β依赖退出的[14份原始元数据](../reports/documentation/20261116/h_ablation_result/raw_manifest.json)，121367字节，具名SFTP回传并逐份核对大小/SHA。63成功训练和同一120群继续保留；未回传正文、账号映射、pairs/owners、Memory、预测数组或模型。原生成退出1及10.468339秒失败费保持，不改写为进程成功。

按[本次明确许可](../reports/documentation/20261116/h_ablation_result/recovery/authorization.json)，恢复目录保管授权、诊断、CPU01原始回执、本地处置、来源/部署清单、request及实际启动证据。原50科学来源及已审派发器保持；新三入口只修无限辅助deadline的JSON元数据并接续原确认/分析。Git `cfbc6054afa0fc8fdfe6456cc6afb51f468c4e10`保管实际修复和测试版本；CPU完整生命周期2.021449秒、计3秒，是1043秒保守准备费的子集，不能重复相加。活动入口不能清理，一次性核验源码待本次恢复验收和精确恢复核对后收尾。

旧根failure、budget、cost_records、execution_time按原字节保存在Linux新恢复state/previous_segment；旧βwait整体保存在原root/wait_before_postprocess_resume，新依赖门另存为gate_after_postprocess_resume.json。原冻结门、失败回执和旧两次等待归档不变。启动/依赖/训练费用等[8份具名元数据](../reports/documentation/20261116/h_ablation_result/startup_sync.json)56468字节已SFTP回传、逐份大小/SHA一致；训练费用盘点只用独立dispatch和cost账，不读逐训练私有状态。确认评分使用Linux既有py310及原正式端点，结果完整前不读取效果或提前开标签；根docs在Windows维护。当前进度只见交接唯一研究线项，不把本记录当动态账本。

10:43:18的[结束本轮等待时观测](../reports/documentation/20261116/h_ablation_result/recovery/progress_20261116_104318.json)5041字节已具名SFTP回传、大小/SHA一致；内容限运行/访问角色/费用元数据，没有效果载荷。用户询问清理时已明确说明189新增阶段端点及复用Full27仍是当前评分依赖，未执行删除或移动。

11月16日15:06:14的阶段快照32690字节与15:09:41的[完成/自动交接快照](../reports/documentation/20261116/h_ablation_result/recovery/progress_20261116_150941.json)76122字节已具名SFTP回传并核对大小/SHA，原观测不覆盖。最终快照含中文完成/费用/原主门摘要和β真实首更/身份/14来源核对，不含标签、模型、Memory或预测数组；未回传完整summary载荷。Linux活动一次性入口once_wait.py已按原合同在首个真实更新后自删，原174字节源码及SHA、first_update/queue_continues回执仍留存，当前控制器继续；不把自删误判为监听丢失而重建。该15:09快照时中文权重和Full端点仍保留，随后16:11的定向清理及本次完整结果回传见上节；旧快照不追改。

本次新增成果仅本地保管，11月14日已完成推送许可不扩大到本批；用户已有结果表编辑保持未提交。本轮没有删除权重，没有外发新结果包。局部恢复不重置上次11月6日、下次11月21日大清理周期。

## 2026-11-14 全项目科研记录与同步核对

用户明确要求全面核对科研文件同步进度、检查平时少看的位置、原位更新滞后记录，并提交所有本地commit；本次授权覆盖当前科研分支所有尚未推送提交及本次维护成果，替代此前本批“仅本地、不推送”的当前交付限制，不追改原合同、送审、运行原件中的历史push=false。用户既有结果表编辑继续保留未提交，未改其内容。

核对覆盖AGENTS、275份根科研Markdown、scripts/schema/tests的815个受控源码/政策/测试引用，以及近期9份具名同步清单、234项原绑定和当前隔离运行来源；共1307个唯一文件双端大小/SHA核对，逐项见[技术回执](../reports/maintenance/20261114/file_sync.json)。另以998份Markdown/9685个本地引用作路径初筛；修正后AGENTS及275份根科研Markdown的3324个本地链接均可解析，区分冻结审查/来源副本、已退役附件、带行号语法和当前入口，不以快照的原相对路径失效为由重写原件。未读取封存真值、正文、预测或加载模型，没有运行科研测试、改活动源、重训或批量覆盖Linux。

| 两端核对对象 | 实际核对与处置 |
|---|---|
| 当前中文科学来源/恢复控制 | 50科学来源、5控制与许可绑定均匹配；Windows没有对应冻结workspace副本的50项按原门在Linux核对，不复制一套新运行目录 |
| β生产来源与入口 | 14生产文件及once_wait共15项匹配；当前真实依赖门和唯一等待匹配，保留原已退役等待记录 |
| 近期部署/结果/清理清单 | β48、确定性Full结果42、诊断15、原恢复及本次部署回执等按原清单逐项匹配；旧request.template已按明确清理回执退役，原deployment清单不追改 |
| FIT12 P/M终账 | 旧campaign/budget Windows副本是开训快照，不当当前终账；P最终result/budget与Linux真实终账SHA一致，M最终账也一致，不重复加总累计账 |
| 根文档及非运行根源码 | 276项含AGENTS：256同、9异、11无Linux副本；815根源码/政策/测试引用：624同、14异、177未部署。根docs以Windows为权威，当前作业使用上述隔离来源；这些引用差异不冒充活动来源失败，不用全仓覆盖消除差异 |
| 权重保管 | 指定78、head/微夹具37、旧seed44五份共120已删路径仍不存在；ABC/RMS42份元数据一致、Full27份存在且大小匹配。只查文件元数据，未重新读模型字节，没有新增删除 |

旧GPU队列的“当前无训练”、9月23日进展与9月18日报告、旧并行计划及9月14日迁移描述已加明确历史时点/当前入口；重放实现页的已退役ZIP链接改指保留原文和清单，注明Git精确恢复入口，不恢复无用包。更新交接当前进度、最新推送范围和讨论记录；设计/计划继续只承担合同/接续职责，不复制进度。当前用户结果表的原有编辑保持；本轮没有为提交重写它。

本轮只生成上述技术回执及原恢复根下具名进度观测；具名SFTP回传核对，根docs仍只在Windows。临时身份输入/读回副本在对应数据已入回执后按精确路径收尾：Windows2份1145859字节、Linux1份470370字节，未留一次性脚本或ZIP。大清理仍为11月6日/11月21日，本次核对不重置。

## 2026-11-14 中文辅助硬帽取消及三中断恢复保管

[独立恢复目录](../reports/documentation/20261114/h_ablation_budget_resume/scope.zh.md)保留用户具体指令、运行补充、本地处置、两份实际入口、正式request/launch、现场资源/预算及三条首更新；原已审dispatch和50冻结科学来源不改，根docs仅Windows维护。按用户要求未新增测试、短测、外审包或一次性核验脚本，未创建网页会话。两运行入口仍供当前监管调用，不属于无用文件，继续保护。

原预算停机[具名观测](../reports/documentation/20261113/h_ablation_retry_s6/progress_20261114_budget_stop.json)29047字节，SHA256 `a8ec7689945eef58cea9781efc63f9f3328594df5636f3346d947210f096e6d5`；11月14日04:02最终failure/budget/dispatch/execution_time/cost_records原字节保留于Linux新恢复根previous_segment，不将新状态追改入旧回执。三条BCA_s4中断目录整体移入原campaign/attempts/各任务/attempt0，32文件1305723909字节，原模型/日志/访问/费用均保留且大小/SHA一致；只回传[路径与身份回执](../reports/documentation/20261114/h_ablation_budget_resume/state/attempt_resolution.json)，没有回传模型、正文、身份、标签、Memory或预测。21成功及Full27未移动或删除。

β旧wait含失败原件原样保管在wait_before_budget_resume，更早wait_before_chinese_retry_s6继续保留；新gate_after_budget_resume.json另存，只迁移真实依赖与承接费用，不覆盖原门/launch。新唯一wait和once_wait仍活动，不能提前清理。实际首更/身份及新门等9项由[final_sync](../reports/documentation/20261114/h_ablation_budget_resume/final_sync.json)具名SFTP回传并逐份核大小/SHA；5项部署文件按sync清单远端核对。辅助历史和本次665秒保守准备费留在原累计账，非CPU活动，不新增测试子账。没有遗留临时副本或压缩包，没有删除权重；本地提交不推送，用户已有结果表修改保持。局部收尾不重置11月6日大清理和11月21日到期。

## 2026-11-13 中文单次重试恢复保管

[本次独立目录](../reports/documentation/20261113/h_ablation_retry_s6/scope.zh.md)保管单条重试授权、限定免审、CPU01/02/03、独立主审、正式request及实际接管证据；运行包装在同根workspace/scripts/retry_once.py，调用原已审freeze/drain和原controller/dispatch，不覆盖活动50科研来源。原失败目录在实际排空后才整体原样归档，后续资产路径和实际发生状态见运行回执，不提前声称已移动或开训。Full27、原成功端点及现有worker均保护。

CPU与源身份按[sync](../reports/documentation/20261113/h_ablation_retry_s6/sync.json)核对11项，正式接管来源/小回执按[deployment_sync](../reports/documentation/20261113/h_ablation_retry_s6/deployment_sync.json)核对16项，具名SFTP与Windows大小/SHA一致；root docs只在Windows更新，正文/标签/预测/Memory/权重不回传。准备原始区间及子账单独保留，1380秒已在接管时一次进入原辅助账，CPU7秒不另加；预算不扩大。

本地提交`038a646dd5bdc30b4c842006c93095f9ae7668a9`保管实际各版核验源码及未上传ZIP；从Git原始blob逐份核大小/SHA与执行版本一致后，已按[Windows回执](../reports/documentation/20261113/h_ablation_retry_s6/cleanup.json)删除本次8份无用核验/来源副本、模板及免审后无用ZIP，共156282字节；Linux按[回执](../reports/documentation/20261113/h_ablation_retry_s6/cleanup_linux.json)删除同7份（没有ZIP）共56188字节，活动包装/主审/授权及原绑定来源前后不变。CPU01/02各自源码和最终预算核验可从该commit对应路径精确恢复，原同步清单保留清理前事实，不追改。运行入口、正式request、所有通过/失败和费用原件继续保留；未删除模型或其他研究线资产，没有启动网页会话。Git仅本地，用户已有结果表修改保持未提交；局部收尾不重置大清理周期。

排空后原失败目录7文件33030字节已原样移至Linux campaign内`attempts/ABC_s6_ER/attempt0`，前后大小/SHA一致；完整实际路径与原规范路径映射见[回执](../reports/documentation/20261113/h_ablation_retry_s6/state/attempt_resolution.json)，没有删除其旧访问/失败内容或费用。新尝试使用全新规范目录；原成功任务和Full27不移动。β旧wait目录原样保留在其原root下`wait_before_chinese_retry_s6`，新依赖gate另存、不覆盖原gate/launch。随后[首更/身份/归档/费用核对与新门](../reports/documentation/20261113/h_ablation_retry_s6/final_sync.json)共9项具名SFTP回传并逐份核大小/SHA一致，新βgate按原Linux对应目录保管，未另造同内容副本；回传内容限技术元数据，不含失败startup/正式标签/模型载荷。临时只读观察在首更后结束，无额外常驻监听。

## 2026-11-13 晚间运行进度证据

本轮只读核对中文消融及既有β等待进程，Linux保存[具名观测](../reports/documentation/20261113/h_ablation_resume/progress_20261113_193435.json)，通过SFTP回传同名Windows路径，22397字节、SHA256 `23ad45a514c4d8f57d240070908dc528739d1e5b16b5bb9e2e38c4b71865fec6`逐项一致。内容限定实际进程身份、更新计数、成功/失败调度回执、OOM技术诊断及资源/费用元数据；未读取或回传科学效果、标签、正文、Memory或权重。原始旧观测保留，当前状态仅原位更新交接；未写活动账本、改运行源码或操作已有进程。SSH/SFTP均已退出，没有生成临时脚本、下载副本或压缩包，没有删除模型。用户结果表已有修改继续保留、不纳入本次提交；本轮仅本地Git，不沿用旧推送许可。局部记录更新不重置大清理周期。

## 2026-11-13 完整H单一β探针保管

本次独立目录[scope与实施](../reports/documentation/20261113/vendorlink_beta_quarter/scope.zh.md)对应Linux同名`workspace/scripts`、唯一wait及条件满足后才创建的campaign；与旧确定性Full和中文活动来源隔离。[正式门与来源](../reports/documentation/20261113/vendorlink_beta_quarter/final_sources.json)绑定14生产文件和独立一次性入口，生产版本Git 2b6bc763；cpu01/02失败、cpu03/04通过及cpu05终态补证均保留，累计CPU32秒，早期源码Git bd4cd8c1/37d3c98b/2b6bc763及最终核验Git 0728b409可精确恢复。等待进程已部署，当前身份/预算/中文进度只见交接唯一项及具名Linux观测。真实数据、预训练、未来模型/Memory/预测和标签按原Linux边界保管，不因全量Git交付纳入仓库。用户已有结果表修改保持未提交。

本次具体审查包48份载荷及清单，204057字节，SHA256 `07f68544bb6ad3b85cf5fff72a127fb104ac3eedf348eeffac46ea8f343d5c53`，精确恢复入口`37d3c98b:reports/documentation/20261113/vendorlink_beta_quarter/review/input.zip`；原请求、完整背景、实际源码/核验和已开放结果均在范围内，未包含私有载荷或凭据。[完整外审](../reports/documentation/20261113/vendorlink_beta_quarter/review/external_report.zh.md)经Playwright实际Library下载回收36049字节、SHA256 `1b93962d41fb62d950430f08bb5bd5170c9279782083d827cef1ae5e195a94d0`，连同[独立处置及回收身份](../reports/documentation/20261113/vendorlink_beta_quarter/review/recovery.json)、原提交/模型选择/完成页保留；原包两处缺陷及必要修复证据不省略。两专用网页已关闭，未再上传最终修复源码。

完成Git字节恢复核对后，Windows的两份一次性CPU核验脚本、根部重复once_wait及无用输入ZIP共4份，14份具名浏览器临时副本已按[本地回执](../reports/documentation/20261113/vendorlink_beta_quarter/cleanup.json)删除；Linux仅删除本次qualification/verify.py一份，见[Linux回执](../reports/documentation/20261113/vendorlink_beta_quarter/cleanup_linux.json)。没有删除模型、失败原件、正式依赖或活动等待入口。测试恢复入口为`0728b409:tests/verify_vendorlink_beta_quarter.py`及该提交同名qualification路径；活动`workspace/scripts/once_wait.py`保留，只有真实首个优化器更新后由已审运行器退役，不能因其他同名副本清理而误报缺依赖。根docs只在Windows维护，相关正式门/源码/原文/小回执具名SFTP核对同步，新增成果仅本地提交。局部收尾不重置11月6日大清理与11月21日到期。

## 2026-11-13 VendorLink下一步方向外审保管

用户明确要求送审下一步想法及完整背景。本次[输入清单](../reports/documentation/20261113/vendorlink_next_review/manifest.json)含65份安全载荷，13份生产源码来自Git c902ada1并按正式来源逐份核对；输入ZIP为288163字节、SHA256 `cc5c7c5230b511e23a9223b9126958e12c84d214a630c46b051eef7983e928ef`，精确恢复入口`2026e5ab:reports/documentation/20261113/vendorlink_next_review/input.zip`。原请求、背景和待审建议保留送审字节；本轮外发许可来自当前具体指令，不沿用旧结果包许可。

[外审完整原文](../reports/documentation/20261113/vendorlink_next_review/external_report.zh.md)经Playwright从网页提供的Library文件直接下载，49047字节，SHA256 `70865f4296d9613e9095b24d5cd48b089eec884b3f1ed67dc2dc490e0d109826`；与[独立主审](../reports/documentation/20261113/vendorlink_next_review/primary_disposition.zh.md)、模型选择/提交/完成页及提交身份一并保管Windows和本地Git。本次未连接Linux，无科研计算或两端同步声明，根docs仅Windows维护；未改其他实验权重、原始结果或预算。原文回收、主审和Git恢复核对后，本轮无用输入包及具名浏览器副本按常规收尾退役，精确路径/字节见[recovery](../reports/documentation/20261113/vendorlink_next_review/recovery.json)。专用会话/下载页已关闭；外审建议没有转换成新实验权限。局部收尾不重置11月6日大清理与11月21日到期。

## 2026-11-13 VendorLink指定权重及早期GPU状态清理

按用户先选中P/M/新Full60份、再明确追加早期18份GPU状态的[具体指令](RESEARCH_DISCUSSION.zh.md#2026-11-13明确删除vendorlink部分权重及早期gpu核验状态)，Linux于11:09:35完成逐文件删除。精确路径、大小、SHA、属主/inode、实际删除时间及前后核对保留在[原始回执](../reports/documentation/20261113/vendorlink_weight_cleanup/receipt.json)，同字节保管于Linux同名目录；312139字节、SHA256 `ac296ac950905dd80fe2d5395a2dc0e75e4b1ab7cb23050d12a8e4398427b642`已具名SFTP回传并在Windows核对。

| 实际删除对象 | 文件数 | 字节 |
|---|---:|---:|
| FIT12 P：H_SEQ、VendorLink_SEQ，各seed51/52/53、stage0—3 | 24 | 7845094494 |
| FIT12 M：H_ER、旧H_full，各seed51/52/53、stage0—3 | 24 | 7837240866 |
| 确定性出生评分Full：seed51/52/53、stage0—3 | 12 | 3918621201 |
| 10月29—30日原生VendorLink G1短测/恢复检查完整状态 | 18 | 23432023643 |
| 合计 | 78 | 43032980204（40.0776GiB） |

删除限于原P、M及确定性Full的三个campaign内具名端点，以及10月29/30日`vendorlink/native/qualification/g1`内具名状态，逐文件unlink、未递归删目录。78份均为当前用户所有的普通单硬链接文件，最终路径在对应根内，无相关活动进程；60正式端点实际SHA与各自成功completion保存值逐份相同，18早期状态记录删除前实际SHA。删除后全部78路径不存在；544份非权重文件元数据、其中118份必要内容SHA，以及19项完成/门/来源绑定证据不变。原ABC36和RMS6共42份权重的大小、mtime、inode等元数据不变；两份fit特征缓存、共享BERT及中文H资产不在删除范围。Windows对应实验根未发现训练权重文件，因此本端模型删除0。未新建权重备份或搬运模型，未载模、解析标签或Memory载荷；正式训练Memory摘要等单独证据保留，早期完整状态内的优化器/RNG等随这18份已退役文件一并删除。

早期18份来自10月29日9GiB上限下首次G1尝试及10月30日放开该上限后的单路G1核验，用于四方法短更新和恢复/重复更新比较，不是18条完整正式训练。原失败1次有效更新、后续G1通过15次及用户“按单路资格结案”决定见[资格汇总](../reports/documentation/20261030/vendorlink/native/received/result_summary.json)和原日志。此前清理未覆盖这18份，是已结束核验的收尾遗漏；通过/失败、恢复比较及源码记录仍保留，无未决G2或必要恢复用途。正式源码和历史结果不因本次删除追改，SHA不能恢复已删模型，恢复源码也不授权重训或重跑资格。

原ABC36是11月4日18:30完成的四方法/三市场原批次在ABC顺序的保留部分（4方法×3seed×3阶段），当时[删除BCA/CAB72并保留ABC](../reports/seller_alias_continual/20261104/vendorlink_result/cleanup_scope.zh.txt)供英文前置评价；11月7日前置门停止后，11月8日[两SEQ全市场池诊断](../reports/documentation/20261108/vendorlink_fullpool/scope.zh.md)及[ER/Full同池补齐](../reports/documentation/20261108/vendorlink_fullpool/history/scope.zh.md)也实际使用了这些权重。上述用途均结束，当前无已批准后续加载任务；本次用户询问用途，未选中ABC36或RMS6删除，故保留。完整结果、审查、原失败及累计费用继续保管，本次文件管理不改正式账。根docs仅更新Windows，本地提交不推送；本轮为局部清理，不重置11月6日大清理及11月21日到期。

## 2026-11-13 中文H消融原队列恢复

用户明确恢复中文消融，既有监管/原workspace/原账继续使用。[恢复前核对](../reports/documentation/20261113/h_ablation_resume/preflight.json)、[原resume入口回执](../reports/documentation/20261113/h_ablation_resume/resume.json)、[三条首更新观察](../reports/documentation/20261113/h_ablation_resume/observation.json)在Linux同名独立证据目录保管，共3份61974字节，经具名SFTP回传后逐份核大小/SHA。它们不替代原正式门、batch、暂停及continuation原件，不另造活动状态；当前唯一进度维护交接对应小节。实际入口只解除原pause_dispatch并写原last_request，没有新控制器或科研来源变更。

50份冻结科研来源及3份活动控制源码、小资格/授权/输入元数据已现检SHA；原Full27份权重现检属主及大小，未重算大权重哈希/载模。原7成功回执前后保持，21份原消融端点、Full27和其他资产继续保护，新三任务在原training目录产物留Linux。未读取效果、正文、标签、Memory或确认数据。原辅助桶暂停费用继续累计，未扩预算、手改活动账或清零；纯观察准备的终端命令没有保存为测试脚本，不新增CPU/GPU资格。无临时脚本、外审包或模型搬运，本轮本地提交不推送；不重置大清理周期。

## 2026-11-13 确定性Full结果回传与审查保管

三条确定性Full的[结果与主审分析](../reports/documentation/20261112/vendorlink_deterministic/result/report.zh.md)、原始完成/判定/费用、27格13指标、原P/M配对统计和数值历史摘要等30项498431字节已具名SFTP回传，逐项大小/SHA与[接收清单](../reports/documentation/20261112/vendorlink_deterministic/result/receive.json)一致。该清单13376字节、SHA256 `ecc5a8ee333cc76b2a44d301ae710fd9ce1e7466a04adf344a0b8f5f24477b8d`也核对。报告和所引小证据镜像留Linux本根result，原campaign/gate/来源/结果/账本不覆盖。统计和日志数值汇总在既有py310完成，不加载模型、预测数组或标签；Windows只作身份/保存字段对应核对、编辑和文件管理。

12份新紧凑端点3918621201字节保留Linux，原P/M/RMS及预训练不删；三份latest由原控制器按成功验收后退役许可处理。正文、身份/标签载荷、Memory、逐查询分数、模型和confirm未回传。原正式94389秒及本轮辅助收尾后94422秒分别见原账与[result/budget](../reports/documentation/20261112/vendorlink_deterministic/result/budget.json)，旧P/M及诊断不清零、不混加。一次长终端输入在执行前因工具长度限制被拒绝，改为两个有界命令；一次清单SHA人工抄录多一字符触发本地拒绝，核对服务器分段原值后再传，未重算科研任务或放过身份不符。

Linux临时观察副本`result/observe.py`（7755字节、SHA `9d12c2e40109553a2c28009583e2c14c3eb7a78d27db03d3f5e40134b32835f5`）在观察保存后已核对最终路径/属主/普通文件/单硬链接并删除，见[收尾及中文暂停元数据](../reports/documentation/20261112/vendorlink_deterministic/result/closure.json)。Windows普通工具`scripts/research_observe.py`及Git `975a540a`同字节源码保留，没有删除科研来源或模型。中文暂停和原账保持实际现场身份，动态数值只维护交接。

用户批准的具体结果包为55载荷＋MANIFEST、206417字节、SHA `bb0ef9887e3d9d762e56ca0a6413236569708d7de934de627776e8969cc24f55`；逐成员大小/SHA已核对，输入原件及55来源的精确恢复点为Git `4abb20e1`。包只通过本地Playwright向[该次网页审查](https://chatgpt.com/c/6af672b0-6a30-83ea-b538-ab61cccf7784)单次提交，不向Linux额外复制ZIP，许可、模型选择器及请求记录见[submission](../reports/documentation/20261112/vendorlink_deterministic/result/review/submission.json)。完整[外审原文](../reports/documentation/20261112/vendorlink_deterministic/result/review/external_report.zh.md)由网页下载原样保存，29965字节、SHA `39b73bc83207f9a8f12259fedfb22d4f94b9903d86b5bb3d6197df036d05043f`；[独立主审](../reports/documentation/20261112/vendorlink_deterministic/result/review/primary_disposition.zh.md)关闭三项非阻断说明，无必要源码修复或重算，原科学失败保持。

[回收与清理记录](../reports/documentation/20261112/vendorlink_deterministic/result/review/recovery.json)保留完整原文、完成页、模型选择和提交页身份。回收与主审后Playwright标签清单仅剩原about:blank，专用审查页已不在打开清单；未关闭其他任务页。Windows实际读Git `4abb20e1`的ZIP blob核对206417字节/SHA后，删除该无用输入ZIP及本任务8份浏览器临时/下载副本，合计9份257303字节；未删除正式审查原文、主审、清单、科研来源或模型，没有为清理另造Linux ZIP副本。恢复输入不授予新外发或新实验。

最终报告、42项661696字节结果/审查/来源证据在Windows和Linux逐份大小/SHA一致，[最终同步](../reports/documentation/20261112/vendorlink_deterministic/result/sync.json)自身8842字节、SHA `a3190a9f4294458d55511a6dd372b8feab9fa47abec10f701e5a44d44bf7282b`亦经SFTP读回核对。送审时`budget.json`原字节保留；审后保存字段核对及最终保管保守另计30秒辅助，最终training92329/GPU675/auxiliary1448秒，共94452秒=26小时14分12秒，未超40h/1h/1h分桶。正式campaign账本核对未变，根docs不向Linux镜像。

本轮新增成果本地提交不推送，根docs只维护Windows；局部收尾不重置11月6日大清理及11月21日到期。

## 2026-11-12 确定性出生评分Full部署与核验收尾

21:34:13新增[只读进度快照](../reports/documentation/20261112/vendorlink_deterministic/progress_20261112_213413.json)，7382字节、SHA256 `fada3e234ca2df3b1c0cac02400b96cc7876a87d19fd56727e397f8d15c18601`，SFTP回传并核对大小/SHA。仅运行身份、训练计数/完成验收标志、GPU、账本及阶段文件是否存在；另只读两份已完成dispatch_receipt的seconds及stage端点文件mtime估计工期，不读取模型或新效果。快照kernel_identity中worker无start_ticks/boot_id预存基准，boot_matches=false不能解释为身份不符；其五项进程身份均匹配，控制器内核基准也匹配，未据缺失字段干预作业。当前交接原位更新，计划/设计入口接续逻辑仍适用，无新测试、临时脚本或载荷。本次新增记录本地提交，上一轮全部提交已推送的事实不扩大为本次新推送许可。

新独立根为Linux `/home/yongpeng/cross-lingual/reports/documentation/20261112/vendorlink_deterministic`；[范围](../reports/documentation/20261112/vendorlink_deterministic/scope.zh.md)、[实现核验/主审](../reports/documentation/20261112/vendorlink_deterministic/implementation.zh.md)、[正式门](../reports/documentation/20261112/vendorlink_deterministic/gate.json)、13份生产来源、资格原件和启动证据在本根保管。生产来源精确版本为Git `c902ada1`，五份修改、八份与旧M相同；正式运行使用本根workspace，不覆盖旧P/M来源。20项62777字节小证据通过具名SFTP双端核对大小/SHA，见[同步映射](../reports/documentation/20261112/vendorlink_deterministic/sync.json)；该清单自身6351字节、SHA256 `23e1866ca0a7c8b63ac5117dd3d99737bc071120f4449df19e2a8049b6925622`也一致。13份实际部署源码与Windows生产文件身份逐份相同。动态运行状态只在交接对应项维护。

必要CPU四项及一次真实BERT GPU核验的result/stdout/stderr/receipt全部保留，无失败版本；CPU生命周期4.516809秒计辅助5秒，GPU9.592766秒计10秒。新42h账在本根campaign/budget.json，预置training0/GPU10/auxiliary35秒（CPU5＋保守部署核对30），原Ledger另计30秒收尾和实际监管；旧P/M、九端点诊断保持各自原账，不借额度或重置费用。本次用户明确免方案/实现外审，无网页或新上传包。

一次性测试源码精确恢复点为`c902ada1:tests/verify_vendorlink_deterministic.py`，15623字节/SHA256 `e4d7eb3bfa2ef46d018e2712e71e740ff14f534f4e5166b2f525b7eae119e1ff`。Windows实际读取Git blob逐字节核对，见[恢复记录](../reports/documentation/20261112/vendorlink_deterministic/qualification/recovery.json)；[Windows回执](../reports/documentation/20261112/vendorlink_deterministic/qualification/cleanup_windows.json)和[Linux回执](../reports/documentation/20261112/vendorlink_deterministic/qualification/cleanup_linux.json)分别记录精确一份源码删除。Linux仅删除本根qualification/verify.py，先核最终路径/属主/普通文件/单硬链接、两个核验进程退出、非正式13来源；Windows仅删除工作区tests对应文件。原始上传清单继续作为发生时证据，不因临时文件退役而追改；正式门绑定通过回执，不依赖该脚本存在。恢复源码不授权重跑。

原P/M/RMS端点、原数据和共享预训练全部保留；新正式紧凑端点及在用最新恢复状态仍只在Linux，完整轨迹验收后latest按原许可退役。真实正文、身份标签、Memory、逐查询分数和权重未回传，confirm未读。中文H暂停不动；根docs只维护Windows，本轮本地提交不推送。局部核验收尾不重置11月6日大清理和11月21日到期。

## 2026-11-12 历史评分随机模式诊断保管与退役

[完整诊断与主审](../reports/documentation/20261110/vendorlink_redesign/fit_reference/mechanism/diagnosis/report.zh.md)、九端点/54群数值汇总、实际来源、CPU/启动/完成/费用及[收尾](../reports/documentation/20261110/vendorlink_redesign/fit_reference/mechanism/diagnosis/closure.json)两端具名保管，映射见[sync](../reports/documentation/20261110/vendorlink_redesign/fit_reference/mechanism/diagnosis/sync.json)。仅AlphaBay既有fit及原stage1六群索引用于重建，正文/账号/逐对分数只在Linux内存，不保存新载荷或回传；不读dev/confirm/Agora文本。九份实际访问权重收尾大小/SHA不变，全部原权重保留，新增训练权重0。原P/M结果、来源和88h费用不改；新诊断按独立1h许可收尾计261秒。

一次性源码精确恢复点为`e878b15a:scripts/step35_vendorlink_history_diagnose.py`，15580字节/SHA `9d6522aa3078ebe1c7f3bd9e6d060fcf8cd97b578c116097049b5f414682af0a`，Windows实际读取Git blob逐字节核对见[恢复记录](../reports/documentation/20261110/vendorlink_redesign/fit_reference/mechanism/diagnosis/recovery.json)。随后删除Windows该源码及Linux `fit_reference/mechanism/diagnosis/diagnose.py`；两端下载临时清单`diagnosis/download.json`各2861字节在内容已并入收尾/同步记录后删除。各端实际删除2项18441字节，Linux先核最终路径/属主/普通文件/单硬链接及原进程退出；Windows按精确工作区路径删除。源码恢复不授予重新载模/诊断或训练，SHA不能恢复权重。

本轮免外审，无网页或外审包；SFTP连接中断后只恢复传输，没有重复GPU任务。根docs只在Windows维护，本地提交不推送；局部清理不重置大清理周期。

## 2026-11-12 FIT12六条ER/Full结果保管

[结果与主审](../reports/documentation/20261110/vendorlink_redesign/fit_reference/mechanism/result/report.zh.md)、整批完成/判定/费用、13指标及逐seed汇总、数值历史日志汇总、来源核对和两次现场观察共20项404970字节已具名SFTP回传并核对大小/SHA；本轮报告及其引用小证据镜像同字节保管到Linux `fit_reference/mechanism/result/`，映射见[同步记录](../reports/documentation/20261110/vendorlink_redesign/fit_reference/mechanism/result/sync.json)。原campaign、gate、13运行来源和P失败原件不覆盖，根docs只在Windows维护。

新24份M端点7837240866字节保留Linux，原24份P端点和RMS权重/缓存继续保留；七份无真值应用输出122234401字节仅核对身份后留Linux，不向Windows复制账号候选载荷。真实文本、Memory、逐查询分数、权重及confirm未同步。正式运行源码和原始数值日志仍有复核用途，不删除。没有新核验脚本、临时模型或外审包，本轮免外审按已有范围执行。

传输用两份临时清单 `result/download.json`（3313字节）和 `result/final_download.json`（2112字节）在映射合并并核对后，两端均已删除，各5425字节；精确大小/SHA及删除状态保留在sync记录。本轮一次保存字段汇总误指未建立镜像的失败保留于final_check，辅助保守计30秒，加三次必要汇总共34秒，不覆盖原控制器账或将其写成训练失败。当前交接原位更新、计划和设计入口清除已结束的自动接续措辞，本地提交不推送；局部收尾不重置11月6日大清理及11月21日到期。

## 2026-11-11 FIT12六条ER/Full接续与核验收尾

19:06:17新增[只读进度快照](../reports/documentation/20261110/vendorlink_redesign/fit_reference/mechanism/progress_20261111_190617.json)，10225字节、SHA256 `f5b805d80423c211a2d7e8a10b1106d7d211011cd4ba4a604368e0a8b0a90979`，SFTP回传后大小/SHA与Linux一致。核实唯一控制器及两活动worker身份、训练更新/验收/失败标志、GPU与同账费用；另只读seed51两份训练dispatch_receipt的seconds估计工期，未读M效果或载模。当前交接更新为本次时点，12:55原快照保留；计划与设计入口已核对有效范围、无待修正动态进度，不复制计数。未生成临时脚本或新增计算，不改冻结运行来源与账本，本地提交不推送。

12:55:33新增一次[只读进度快照](../reports/documentation/20261110/vendorlink_redesign/fit_reference/mechanism/progress_20261111_125533.json)，9056字节、SHA256 `ac0c34f5656a671c4ccee86f8956a06abb5b2a960d6f6dfe349ba7928a9e61c7`，SFTP回传后大小/SHA与Linux一致。只取精确运行身份、更新计数、完成/失败是否存在及费用字段，未读取M效果、正文、权重或confirm；未改运行来源、进程或账本，无新增测试/临时脚本。当前计数只更新交接，不复制到计划或合同。

用户明确继续同配置三个seed的ER/Full；[接续范围](../reports/documentation/20261110/vendorlink_redesign/fit_reference/mechanism/scope.zh.md)、[实施主审](../reports/documentation/20261110/vendorlink_redesign/fit_reference/mechanism/implementation.zh.md)、gate及13正式来源独立保管于Linux `fit_reference/mechanism/`，不覆盖原P冻结来源/裁决。13份50115字节范围、部署、CPU、启动及12:45:18队列/费用快照经SFTP两端核对，映射见[同步](../reports/documentation/20261110/vendorlink_redesign/fit_reference/mechanism/sync.json)。快照不冒充后续实时状态，当前唯一状态入口仍在交接。

单次Linux CPU四项通过后，Windows `tests/verify_vendorlink_mechanism.py` 与Linux本根 `verify.py` 各8514字节已清理，SHA256 `6b45cceaa2ccde88fca43e589164aac16b8e4ef0edcdfd23b2b00e0e62d33543`。Windows删除前实际读取 `c0fadd35:tests/verify_vendorlink_mechanism.py` 的Git blob逐字节核对；Linux已核对最终路径/属主/单硬链接及非正式依赖，见[回执](../reports/documentation/20261110/vendorlink_redesign/fit_reference/mechanism/cleanup.json)。该回执只陈述Linux删除，Windows实际删除由本记录与Git差异证明。手写输入随TemporaryDirectory收尾，原日志/CPU回执/历史部署清单保留；恢复源码不授权重跑。

新M的正式来源、活动状态及新权重保留。P的24端点、RMS六权重及两fit缓存均保持Linux既有保管，无新模型删除；账号正文和逐条分数未同步Windows。新M沿用原88h累计账及整个FIT12根32GiB输出帽，不重置原失败和费用。根docs只在Windows维护；本轮免外审，无新上传包或网页，本地提交不推送。此次局部收尾不重置11月6日大清理及11月21日到期。

## 2026-11-11 FIT12 P结果回传与停止主审

[完整报告](../reports/documentation/20261110/vendorlink_redesign/fit_reference/result/report.zh.md)、[保存字段汇总及来源核对](../reports/documentation/20261110/vendorlink_redesign/fit_reference/result/check.json)保留原STOP_LEARNING及其两项具体原因，不修改原gate、运行源码、结果或账本。10份207833字节已开放汇总/完成/费用及本轮核对由SFTP回传，逐项大小/SHA一致，远端原始路径与本地副本映射见[同步](../reports/documentation/20261110/vendorlink_redesign/fit_reference/result/sync.json)。报告、同步映射及其小汇总镜像SFTP保管到Linux同一result目录，原campaign文件不覆盖；根docs只在Windows维护。

12:12:16[只读快照](../reports/documentation/20261110/vendorlink_redesign/fit_reference/progress_20261111_121216.json)5121字节、SHA256 `576bb12dfd4afc3eb6f2eab24a2f712a463bb10a101949ef714fdf224eadc5cb`两端一致。12:14:45核实24份新正式端点7845094494字节属主/大小保持，未载模或重读大权重哈希；RMS、原始文本/分数/日志及全部旧来源继续留Linux，没有删除许可扩展。confirm未开，M/ER/Full及应用未启动。本次无测试脚本、临时模型、外审包或科研重跑；原一次性CPU清理及另一会话head/夹具清理事实保持。本地提交、不推送，不重置大清理周期。

## 2026-11-11 FIT12进度证据回传

10:03:46[只读进度快照](../reports/documentation/20261110/vendorlink_redesign/fit_reference/progress_20261111_100346.json)经SFTP回传，11558字节、SHA256 `0a37536845aa23821d62af71b51b5a10bd4ee3ca958f2884046450169f1e68ae`与Linux一致。只记录进程、计数、收尾验收及账本/GPU元数据，未读取模型或效果；原gate、来源、进程与账本未改。无临时脚本或载荷，当前进度见交接，本地提交不推送。

## 2026-11-10 FIT12实施、启动与临时核验收尾

[固定合同](../reports/documentation/20261110/vendorlink_redesign/fit_reference/scope.zh.md)、[实施主审](../reports/documentation/20261110/vendorlink_redesign/fit_reference/implementation.zh.md)、CPU01及来源/启动/观测证据已保管；两端具名核对见[同步](../reports/documentation/20261110/vendorlink_redesign/fit_reference/sync.json)。新运行来源独立于所有旧冻结workspace，13正式依赖和新活动权重保持；账号正文、完整分数、模型和原角色数据留Linux。

单次CPU核验通过后，Windows `tests/verify_vendorlink_fit12.py` 与Linux本根 `verify.py` 各10062字节已删除。Windows实际读取Git `f5394f07:tests/verify_vendorlink_fit12.py` blob与待删源码逐字节一致；两端SHA256 `8d820a1053e2664e7cc7f7d940e74ef0f5ab26e560ec62714f36037a5f3318b0`。Linux精确属主/路径/单硬链接及非运行依赖核对见[回执](../reports/documentation/20261110/vendorlink_redesign/fit_reference/cleanup.json)，该回执仅陈述Linux删除，Windows删除由本记录及实际Git差异证明。手写输入/小张量由TemporaryDirectory收尾，不留临时模型。核验日志/回执和历史部署清单原字节保留，恢复源码不授予重跑。

本轮不删除RMS：23:32:48六份877291968字节实际存在/属主核对见[核对](../reports/documentation/20261110/vendorlink_redesign/fit_reference/verification.json)，不是此次新训练的热启动权重。另一会话的四head/33CPU夹具清理按下节独立许可保留，不覆盖其回执。没有外审包或浏览器；根docs仅Windows维护，本地提交不推送，局部收尾不重置大清理周期。

## 2026-11-10 评分头与Evolution CPU模型夹具清理

用户在只读盘点后选中四份head及33份CPU微型夹具，明确“删掉，做好记录，及时更新文件记录”。Linux23:23:42完成删除37份6624499字节（约6.32MiB）：head_alignment四任务LOCAL/FULL_POOL的`run/head.pt`共6312188字节；11月8日Evolution的cpu01/02/03分别1/8/24份微型模型，共312311字节。精确路径/大小/SHA、yongpeng属主、普通文件/单硬链接、无相关活动进程及删除后不存在检查见[实际删除回执](../reports/documentation/20261110/vendorlink_redesign/head_alignment/weight_cleanup.json)。无备份，已删模型不能由Git或SHA恢复，不再支持加载这些head或旧CPU夹具；失败、负结果及费用保持。

344份保护文件删除前后大小/SHA完全一致，包括CPU01失败及CPU02/03通过/费用/日志、正式来源、原结果/分数、两个fit缓存及RMS六份877291968字节。未改新FIT12工作、中文暂停、共享预训练或其他资产。Windows对应近期目录无模型，删除0；仅必要[源码恢复核对](../reports/documentation/20261110/vendorlink_redesign/head_alignment/weight_cleanup_recovery.json)与删除回执保留，均经SFTP核对：恢复记录5097字节/SHA `ef2b8dbb5300044b07d2c9184a5153808b25ccb38fee9e2b75c57fcdbe470c73`；删除回执164068字节/SHA `f7ca0d7d528404e0183e14005a41206cd330e1e1a61d5ff5599336b2e8311169`。未生成临时脚本、解包副本或模型备份。

17份CPU实际来源已在Windows内存读取Git原字节，与CPU01原deployment_manifest及CPU02/03_sources逐份SHA一致。CPU01失败夹具、通过夹具及科学依赖从Git713d9929/d721ece8恢复；CPU02/03旧control的11911字节精确版由`713d9929:reports/documentation/20261108/evolution_registered/execution/deployment.zip`中`scripts/step31_evolution_registered_control.py`恢复，不能用后续独立源码替代；详见恢复核对的逐文件入口。未恢复或运行源码。根docs只在Windows维护，本地提交不推送；下方旧“保留四head”是删除前事实，不追改原审查/完成/来源绑定。局部清理不重置11月6日大清理及11月21日到期。

## 2026-11-10 固定参考档案诊断完成

[reference结果/主审](../reports/documentation/20261110/vendorlink_redesign/reference/report.zh.md)及[24项具名结果](../reports/documentation/20261110/vendorlink_redesign/reference/transfer.json)已回传并核对大小/SHA，共32642字节；四份评分矩阵共5164752字节留Linux，没有同步账号正文/逐query分数。零训练、零权重删除，原RMS六份877291968字节与head_alignment四份6312188字节继续保留，正式新入口和14份旧依赖保持冻结身份。

一次性CPU脚本Windows `tests/verify_vendorlink_reference.py` 与Linux `reports/documentation/20261110/vendorlink_redesign/reference/verify.py` 各3529字节已删除；已将本地实字节与Git `300b5f522312af6747034a5d321086345c711f24:tests/verify_vendorlink_reference.py`逐字节核对一致，远端删除前也核对属主/最终路径/大小/SHA与正式依赖。SHA256为`66e94f44ef54d389b3a55a37931e2da08890bf7a856cdebe3cdc6a0c00284289`，Linux[回执](../reports/documentation/20261110/vendorlink_redesign/reference/cleanup.json)及原CPU日志/通过/费用保留。手写输入均在CPU进程内存，无临时模型或数据文件需另删。恢复源码不授予重跑。

用户本轮免外审，无外发包或网页。结果/清单/报告26项48837字节及15运行来源的两端身份见[reference/sync.json](../reports/documentation/20261110/vendorlink_redesign/reference/sync.json)。根docs仅Windows维护，本地提交不推送；此次局部收尾不重置11月6日大清理/11月21日到期。

## 2026-11-10 RMS与评分头结果回传及核验收尾

RMS两任务的完整聚合结果、费用、完成身份、原评分绑定及[逐轮数值诊断](../reports/documentation/20261110/vendorlink_redesign/ce_rms/result/report.zh.md)已回传。评分头四条亦已完整结束；[完整结果/主审](../reports/documentation/20261110/vendorlink_redesign/head_alignment/result/report.zh.md)、数值日志汇总、59项具名完成/指标/费用/身份材料已按[result/transfer.json](../reports/documentation/20261110/vendorlink_redesign/head_alignment/result/transfer.json)回传并核对大小/SHA，共141353字节。没有同步账号正文、逐查询分数或模型载荷。正式15来源、原数值日志/分数、两fit缓存、四份新head（6312188字节）及恢复这些head所需的6份原RMS资产留Linux供必要复核；本轮不新增权重删除。

[本轮清理](../reports/documentation/20261110/vendorlink_redesign/head_alignment/cleanup.json)删除Windows两份一次性源码15825字节：RMS日志汇总及head CPU/GPU核验；Linux删除相应两源码和18个手写输入/小模型/缓存/评分载荷，共20项220280字节。两端各一份9518字节重复待删清单也已移除，其全部内容已被正式删除回执保留；总数分别Windows3项25343字节、Linux21项229798字节。未删除正式训练来源、活动状态、必要原件或权重。核验失败版本和通过版本均经[Git精确字节恢复检查](../reports/documentation/20261110/vendorlink_redesign/head_alignment/source_recovery.json)：aedbfe04、0cdcd632、5cd26360；恢复文件不授予重跑许可。必要CPU失败/通过、GPU、手写运行日志/指标及费用保留。

用户本轮免外审，没有制作或上传新包、没有打开外审网页。根docs仅Windows维护，本地提交不推送；核验/启动同步见原sync.json，结果与报告两端身份见[result/sync.json](../reports/documentation/20261110/vendorlink_redesign/head_alignment/result/sync.json)，61项171995字节。局部收尾不重置11月6日大清理及11月21日到期。

## 2026-11-10 CE诊断外审收尾及RMS两任务

20:56进度查询仅回传一份[只读元数据快照](../reports/documentation/20261110/vendorlink_redesign/ce_rms/progress_20261110_205605.json)，7006字节、SHA256 `e6bc619cf65ad9a9f3464acb1ca737cbaff69de89e88f42f9cc14304e874d05c`与Linux原件一致，含进程身份、更新、账本、GPU元数据及短窗吞吐估计；未读未完整阶段效果、模型、标签或逐条分数，未改运行来源/进程/账本。无临时脚本或重复载荷，根docs仅Windows维护，本地提交不推送。

原诊断数值汇总、来源核对和报告保留，原CPU0.738532秒计1秒与旧训练18554秒各归原账。完整[外审原文](../reports/documentation/20261110/vendorlink_redesign/diagnosis/review/external/report.zh.md)40343字节、SHA256 `2c14d9061e799dc6195b246a3529372e4ccb611071553287026811377539b4d1`已回收，主审结案及网页关闭见[索引](../reports/documentation/20261110/vendorlink_redesign/diagnosis/review/recovery.json)。原送审身份和报告提交时内容冻结，不将诊断改写成已验证有效。

新两任务的[gate](../reports/documentation/20261110/vendorlink_redesign/ce_rms/gate.json)、[执行补充](../reports/documentation/20261110/vendorlink_redesign/ce_rms/scope.zh.md)、CPU两份回执及活动13来源部署到Linux独立ce_rms根；原implementation/中文暂停/共享模型及角色数据不改。科学CPU2.883711秒、资源兼容CPU1.399675秒，合计计5秒；另保守登记30秒准备成本，原Ledger独立30秒收尾费。新正式权重、辅助分类头、正文和分数只留Linux，当前均有后续评分/诊断用途，不删除；根docs只在Windows维护。

审查和必要核验结束后，按[清理清单](../reports/documentation/20261110/vendorlink_redesign/ce_rms/cleanup.json)删除Windows16文件382218字节，包括原诊断脚本、23项输入ZIP、两份一次性核验脚本和12份本次浏览器临时副本。Linux[删除19文件131168字节](../reports/documentation/20261110/vendorlink_redesign/ce_rms/cleanup_remote.json)：上述三份一次性脚本及CPU手写输入/小权重/分数；保留原始核验日志、指标、回执和审查原文。删除前实际核对Git恢复字节：诊断脚本/ZIP来自7e1b2f21，尺度CPU来自91f4933c，资源CPU来自a9b755a7；每份大小/SHA一致。正式gate绑定的13运行模块、scope/主审/CPU回执等保护清单未动，无本轮正式权重或其他实验资产删除。

本轮SFTP证据身份见[同步回执](../reports/documentation/20261110/vendorlink_redesign/ce_rms/sync.json)；本地提交不推送。局部清理不重置11月6日大清理及11月21日到期。最新运行只在交接唯一对应项维护。
## 2026-11-10 CE开发结束权重清理

用户要求删除刚结束实验的模型权重，执行者先核对真实终态、无相关活动进程及后续加载依赖，呈现Linux精确16份、3483476608字节（约3.24GiB）的范围与不可恢复后果，用户明确“确认”。Linux18:27:29完成删除：`vendorlink_redesign/implementation/campaign/D/training/`内AlphaBay/Dream各H_base、H_CE、V0、V1共8任务的`run/stage0.pt`与`run/stage1.pt`。目标均为yongpeng所有、普通文件、单硬链接，最终路径及删除前大小/SHA逐份核实；未备份，全部目标复查不存在。该批D已完整训练/评分并以STOP_ADAPTATION结束，原P/M冷启动不加载D权重，未启动后续阶段或重训。

[删除回执](../reports/documentation/20261110/vendorlink_redesign/implementation/weight_cleanup.json)保留逐文件身份及196份非权重文件的删除前身份，删除后全部大小/SHA一致，相关进程仍已退出。原结果、完整性/验收、日志、源码/gate、审查和费用账保留；共享预训练、其他研究线及中文H暂停资产不在范围。Windows同执行根无模型权重，删除0；回执经SFTP回传，49252字节、SHA256 `aab79233da457f7609b4bae53ce5c97456e1d70830d419ce8f1eb348dfa1001c`与Linux一致。无临时脚本或重复载荷产生，仅保留必要回执；根docs仅Windows维护，本次本地提交，不沿用此前工作流任务的推送许可。局部清理不重置11月6日大清理和11月21日到期；下方旧端点保留记载是删除前事实，Git/SHA不能恢复本次已删模型。

## 2026-11-10 CE开发停止原因核对

本轮按用户“看看实验为什么停了”，回收[停止观测](../reports/documentation/20261110/vendorlink_redesign/implementation/stop_observation.json)及其captured_files列出的8份已开放D聚合结果/决策、完整性清单和最终账/dispatch，共9份47922字节。SFTP逐份与Linux原件核对大小/SHA一致；观测文件11741字节，SHA256 `265e1cfdeeaa16d631219fd005a53b6e75e273508c0df9df567dd59787c132b8`。只检查实际进程、19份小来源绑定和8+8完成记录，未载模、重算指标、读取逐条分数/正文/标签或重新训练。Linux仅新增具名停止观察，不修改原控制器/gate、账本、结果和模型；根docs在Windows原位维护，本地提交、不推送。必要端点保留用于复核，未创建临时脚本或审查包，无新增删除；旧清理事实及11月21日大清理到期不变。

## 2026-11-10 科研工作流工具与记录整理

本轮按[用户明确批准](RESEARCH_DISCUSSION.zh.md#2026-11-10批准科研工作流改善及全部本地提交推送)保留两个正式辅助工具及各自普通回归测试，见[用法](RESEARCH_WORKFLOW.zh.md)。Linux只部署至独立`reports/documentation/20261110/harness_workflow/`的scripts/tests，不改活动CE的13来源、gate、角色文件或模型。四项Linux元数据回归和四项Windows文件检查、真实SFTP上传/读回/Pull证据见[核验记录](../reports/documentation/20261110/harness_workflow/verification.json)；[观察快照](../reports/documentation/20261110/harness_workflow/observation.json)与[接收回执](../reports/documentation/20261110/harness_workflow/receive.json)仅有允许元数据，未查看未完整D效果。当前状态只在交接维护，整理前原详细表格移动至同页标时历史，未删除历史事实。

传输批处理及逐份读回文件在每次唯一`.research-sync-*`目录验证后自动清理；测试临时目录自动删除。本轮临时`.playwright-mcp/harness-source-manifest.json`及正式目录中仅用于本次Pull的receive_manifest.json收尾删除，正式源码身份清单/接收回执/核验/观察保留，没有新增权重、原始数据或审查包；不动其他浏览器文件及其他任务。根docs仅Windows维护，工具使用和元数据证据按具名清单同步；局部收尾不重置11月6日大清理及11月21日到期。用户明确要求本次连同全部本地commit推送，Git交付按实际远端ref核对，不沿用旧不推送限制。

## 2026-11-10 账号监督旧权重和专用脚本清理

本次实现审查已经回收并主审闭合：[原文](../reports/documentation/20261110/vendorlink_redesign/implementation/review/external/report.zh.md)36462字节/SHA256 `93685fa3997a300f86623c742aa1899ee9b85363d707a135e7226678e6e18020`、主审、原提交/许可/清单及精确源码证据保留，见[回收记录](../reports/documentation/20261110/vendorlink_redesign/implementation/review/recovery.json)。上传ZIP40项143125字节已实际从Git `aed2138b` 读取原blob并逐字节核对，随后与本轮14份浏览器临时日志/快照和1份重复下载一起清理，共16文件874363字节，见[精确清理回执](../reports/documentation/20261110/vendorlink_redesign/implementation/review/cleanup.json)。专用审查已结束且页面不再存在，无待发送操作；未为本轮另建测试或启动包装脚本。原13运行源码及正式D/P/M/C活动状态、端点、共享BERT保留Linux，旧CPU/GPU一次性脚本双端退役事实保持。正式gate、scope、主审及审查原文SFTP核对见[开训前来源回执](../reports/documentation/20261110/vendorlink_redesign/implementation/launch_preflight.json)，本地来源恢复点 `0d36c3f2`；启动后不改冻结gate/scope/运行源码。启动及首更小回执回传Windows；本轮11项97623字节闭合/启动/观察材料[最终同步](../reports/documentation/20261110/vendorlink_redesign/implementation/review/final_sync.json)逐份大小/SHA一致，不含正文/账号/标签/预测/权重。根docs仅Windows，本地提交不推送，不重置11月6日大清理周期。

后续实施（用户明确免审并要求开始实现）：当前来源部署在Linux独立`reports/documentation/20261110/vendorlink_redesign/implementation/workspace/`，仅保存13份必要运行模块；没有恢复step34旧接续控制器或覆盖中文活动workspace。新方案/主审与实施证据已按[同步回执](../reports/documentation/20261110/vendorlink_redesign/implementation/sync.json)双端核对，替代下段“本次未访问Linux/仅Windows”的旧时点状态；根docs仍仅Windows。CPU5项与一次GPU增量短测完成后，两端各2份一次性核验源码按Git `f7e1f51f`/`1b3caa2c`原blob精确核对后退役，见[清理回执](../reports/documentation/20261110/vendorlink_redesign/implementation/cleanup.json)。GPU probe自身1316372272字节临时权重在恢复对比后删除并复查不存在；没有正式训练端点。CPU8秒/GPU18秒及准备收尾保守120秒记录在[费用账](../reports/documentation/20261110/vendorlink_redesign/implementation/budget.json)，training0，不借旧账、不重置11月6日大清理周期。方案层冻结31751字节版本可从Git `83354bf3`恢复；当前方案仅追加实施状态承接，科学定义不变。

用户随后明确提交新方案外审，原20项146545字节输入ZIP、29502字节方案（SHA `6fb6aa3b9d77d3bc2cc1a0ae853a350810373477d3a7a95959fd921bc5e87f6c`）由Git d14e998a精确保留；[成员清单](../reports/documentation/20261110/vendorlink_redesign/review_manifest.json)、请求和[许可/提交](../reports/documentation/20261110/vendorlink_redesign/review/submission.json)原件不改。用户再要求回收和必要修改后，完整外审405行/35666字节/SHA `687bb25efa28ea9715e6e36b5db43b44ea2cff9432654383242205a57b08b529`已经下载核对，与[主审](../reports/documentation/20261110/vendorlink_redesign/review/primary_disposition.zh.md)及补正版保留。R01/R02闭合后实际读取Git原ZIP及19份来源核对，清理Windows输入ZIP、浏览器重复下载与两份本轮临时快照，共4份459756字节；必要原文/完成页/来源保留，见[回收清理记录](../reports/documentation/20261110/vendorlink_redesign/review/recovery.json)。3份旧step34仍只读Git，没有恢复活动脚本。用户随后要求修复预算预约不足，最终补正版31751字节/SHA `8cb8aca91cdf1cbb3873a764860bd2350b8f4e13a8850bcac9fee82c3b96f030`（88h=80/4/4，任务量不变）及回收材料仅Windows；本次未访问Linux，远端仍是下面sync_oom_update所记旧版本，不冒称双端同步。专用网页已关闭，无新测试/训练/数据访问，不变更原清理事实、费用或大清理周期。

用户明确清理旧20261109账号监督实验并重新设计，在已列明34权重/70份Linux脚本/13份Windows脚本的确认中，特别要求包含OOM后新增的临时唯一接续控制器。本次逐份核对最终路径、yongpeng属主、普通文件、单硬链接、精确来源及相关进程已退出后执行；旧`queue_resume/state.json`为COMPLETE，不修改它或其他冻结原件。

Linux10:58:22已删除旧execution内34份权重共12198384250字节（11.36GiB）、70脚本962847字节及9份对应字节码138108字节，无权重备份；其中包括`queue_resume/workspace/scripts/step34_vendorlink_account_resume.py`、retry_s44/retry_s44_15g和parallel_h45启动包装器。其余467份非目标文件逐份大小/SHA前后不变，原始数据、已建角色、结果、日志、失败、费用、共享BERT及其他实验资产保留。Windows已删除13份123745字节专用脚本，没有本批权重；两端实际路径/哈希/结果见[Linux回执](../reports/documentation/20261110/vendorlink_redesign/cleanup_linux.json)和[Windows回执](../reports/documentation/20261110/vendorlink_redesign/cleanup_windows.json)，[原盘点](../reports/documentation/20261110/vendorlink_redesign/cleanup_inventory.json)保留删除前状态。

全部70份远端及13份本地源码，均实际读取Git提交`f0c2872f96a469140fab6837f360de811cc00834`对应blob并核对大小/SHA，见[精确恢复映射](../reports/documentation/20261110/vendorlink_redesign/source_recovery.json)。其中CPU01原`execution/cpu_increment01/test_source.py`11231字节确实已跟踪，SHA256 `685efce8d858a01f94f556b5bcf36b774ed3bb0ba222cb3300432f9eb0a65e66`；此前“原CPU01源码无法恢复”的保管疑点据实际文件闭合，不靠重跑CPU补证，也不冒称以当前最新版代替旧版。共享root scripts保留，仅删除本轮专用step34及测试；Linux隔离workspace共享模块的副本已退役，原共享源不受影响。

本节后续许可替代下面旧时点的“训练端点/正式来源保留”安排，旧叙述保留当时含义。Git只能恢复源码，不能恢复模型；恢复不授予测试、载模或训练。旧结果外审的文件和网页由另一Agent结案，本轮未修改；原费用71450秒及失败账不清零。新草案、清理回执和资源观测保存在`reports/documentation/20261110/vendorlink_redesign/`，6份328441字节材料经SFTP核对双端大小/SHA一致，见[原同步回执](../reports/documentation/20261110/vendorlink_redesign/sync.json)。随后用户明确单worker OOM后健康worker续领当前阶段独立任务，本次只更新方案及现有记录；Linux旧方案先与上述原身份核对，再经SFTP更新并独立下载核对，29502字节/SHA与Windows一致，见[本次方案同步](../reports/documentation/20261110/vendorlink_redesign/sync_oom_update.json)。原同步/清理回执保留当时字节，原方案由Git e659ea65恢复；临时核对副本已清理，未实现、测试、训练或监听，根docs仍仅Windows。不推送、不重置11月6日大清理周期。

## 2026-11-10 账号监督完整结果回传

用户要求同步进度并仔细分析后，Linux既有py310只检查完整开放D/P保存字段和元数据，不载模、训练、解析原始正文/标签或预测数组。Windows在`reports/documentation/20261109/vendorlink_account_protocol/result/`保存55份387248字节必要原件/汇总，[SFTP接收](../reports/documentation/20261109/vendorlink_account_protocol/result/receive.json)逐份大小/SHA一致；52份来源快照保留原目录对应关系，另3份为保存字段核对、统计汇总和传输清单。原动态运行文件不改，原失败、费用及删除事实保持；根docs仅Windows维护。额外425字节远端结构化身份清单用于纠正人工转录哈希字符错误，不重算或重传原结果。

完整13指标/108格、逐seed和均值/样本标准差/范围、原科学门与停止原因见[报告](../reports/documentation/20261109/vendorlink_account_protocol/result/report.zh.md)和[全表](../reports/documentation/20261109/vendorlink_account_protocol/result/tables.zh.md)。真实账号、正文、标签、分数数组及模型仍留Linux，训练端点保留必要复核用途，没有授权本轮删除权重。保存字段脚本是本轮具体核对来源，本轮结果审查闭合后已双端退役；实际从Git提交d279d2dd读取该11634字节源码blob，并与本地及SFTP回传的远端文件核对SHA一致，不重跑CPU/GPU测试填保管记录。

用户已明确批准本次73项结果审查包，经本地Playwright MCP单次提交；[包清单](../reports/documentation/20261109/vendorlink_account_protocol/result/review_manifest.json)及[具体许可/提交](../reports/documentation/20261109/vendorlink_account_protocol/result/review/submission.json)保留。本轮完整回收34862字节十节原文（SHA256 `02436ae5e3311057ab97a22a8a7278c761aa339079b42253f2b55b116bcbc565`）及完成页，[独立主审](../reports/documentation/20261109/vendorlink_account_protocol/result/review/primary_disposition.zh.md)结案，无科研阻断。原送审请求/清单/提交及指标/失败/费用原件不改；只澄清当前报告状态、控制器指代、计时和时间戳边界。[回收/清理记录](../reports/documentation/20261109/vendorlink_account_protocol/result/review/recovery.json)记载Windows六份239134字节临时包/源码/重复下载和浏览器日志删除，Linux仅退役一份11634字节check_saved.py，SFTP复查不存在；输入ZIP及核对源码均实际与Git d279d2dd原blob精确比较，必要原文和正式证据保留。审查材料按[最终同步](../reports/documentation/20261109/vendorlink_account_protocol/result/review/final_sync.json)双端核对，不为删除无用包另备份或补传输入包，不扩大外发许可。本轮本地提交、不推送，不重置11月6日大清理周期。

## 2026-11-09 接手只读诊断原件回收

本轮仅只读核对Linux进程、运行元数据、冻结源码和现有费用，未修改Linux文件、活动源码/gate或进程，未载模、测试、训练或计算指标。13份原有小证据经SFTP保存于`execution/queue_resume/diagnosis_2036/`，共23200字节，逐份大小/SHA与Linux当时原件一致，见[接收清单](../reports/documentation/20261109/vendorlink_account_protocol/execution/queue_resume/diagnosis_2036/receive.json)。内容仅primary终态/完成/失败/账、接续状态/承接依据/账及V45身份/资源准入、H45增量账；没有正文、账号、标签、分数、权重或凭据。Windows快照名说明用途，不覆盖此前final_observation或其他冻结回执；动态状态仍只在交接当前项维护。

同日23:22进度查询另经SFTP回收7份13470字节原有小证据，保存于`execution/queue_resume/progress_2322/`，逐份大小/SHA与Linux一致，见[接收清单](../reports/documentation/20261109/vendorlink_account_protocol/execution/queue_resume/progress_2322/receive.json)。仅接续状态/dispatch/账、V45训练完成/成功结账及H45 dispatch/增量账；无正文、标签、分数、模型或凭据。原13份诊断快照保留，本次没有覆盖旧观察或改写Linux文件。

primary的训练端点、尚未退役的latest、包装失败和全部费用继续保护；H45/接续来源未改，原5份旧权重及核验脚本删除事实保持。本轮无资产删除、无新临时核验脚本，无需为只读诊断恢复或重跑旧CPU证据。根docs仅Windows维护，未向Linux补传本段或接收清单，本地提交、不推送，不重置项目大清理周期。

## 2026-11-09 整批自动接续部署与精确来源保管

**本轮核验脚本收尾已完成**：Windows删除check_resume.py、cpu01/entry.py、cpu01/check.py共3份28204字节；Linux删除相同3份及无后续用途gate.base.json模板共4份35897字节。全部临时科学核验源码先逐字节与Git提交`1589ef658817e3e060e26cec29343c2c95492885`对应路径核对后删除，CPU01入口/检查及CPU02检查均可由该提交精确恢复，不重跑核验。18份运行源码/gate/CPU回执/免审/主审等保护对象身份未变，新控制器仍真实等待；最终运行入口及原12科学来源必须保留。首次Linux清理预核对因人工转录SHA不符而在删除前拒绝，随后直接使用已核实Git的Windows结构化回执复核完成，没有带错哈希删除。详见[双端清理回执](../reports/documentation/20261109/vendorlink_account_protocol/execution/queue_resume/cleanup_receipt.json)及[Windows精确恢复表](../reports/documentation/20261109/vendorlink_account_protocol/execution/queue_resume/cleanup_windows.json)。[6份收尾同步](../reports/documentation/20261109/vendorlink_account_protocol/execution/queue_resume/final_sync_manifest.json)逐份大小/SHA一致，清单1976字节SHA256 `6c158a8a56eafa0c47f88a439df50be26b4769d6f0ce46807e00bc93748acb49f`；初次sync_manifest的3个已删路径按其当时事实保留，不追改原清单。

Linux新根execution/queue_resume/，独立workspace包含原12份科学来源原字节及新增step34_vendorlink_account_resume.py（17213字节，SHA256 `7b746cb7844b4eaa12f58972c1b9153f5096802c9fff9ae4d80365e90dfbcc5b`）；不覆盖原campaign/retry_s44/retry_s44_15g/parallel_h45来源或模型。当前新gate的resumption.approved/review_closed已按用户“免审”和主审闭合，原gate.pending.json保留CPU拒绝启动时的原字节；gate.base.json仅为已用完的准备模板，非执行/恢复依据。原12来源的caps/limits字段用于原模块兼容性校验，实际新全局帽、逐方法资源和来源/授权在resumption字段及新增入口强制执行，不能以原字段50h/4h覆盖最新49.5h/4.5h。

旧成功端点不复制、不改checkpoint身份，原gate/manifest/task/端点绑定逐份核对；当前seed44只取retry_s44_15g未来成功completion，H45只取parallel_h45成功completion，原已删两次seed44权重从不作为新输入。新控制器等待primary真实退出/释放/结账，承接其最终累计费用；先从全局帽扣H45完整28800训练/1800辅助预约，H45退出后仅加其增量实际账一次并释放余量。P六条完整才评分，原pipeline控制全部后续门；评分单路按原task顺序使用各自14/16GiB profile，M训练最多两路。新预算权威入口在交接时由queue_resume/campaign/budget.json接管；在H45合并前，该账仍须结合其独立增量账/保留额度解释，不能把parallel_h45/combined_budget.json重复相加。新准备126秒（含CPU6秒子账）在primary最终账承接时计入，等待纯休眠不计，尚未交接时不要冒称已在新主账结算。

Windows[21份载荷同步](../reports/documentation/20261109/vendorlink_account_protocol/execution/queue_resume/sync_manifest.json)全部大小/SHA一致，清单6608字节SHA256 `923bb51ad9488d3802166384f1080c1906963b22dfd756b841086fe632df036c`；包含最终新源码、两版CPU证据与精确旧版entry/check、范围/授权/免审/主审、待批及已关闭gate、真实部署回执。无正文、账号、标签、模型、分数或凭据回传。CPU01为16142字节前序入口/6031字节前序检查，CPU02为17213字节最终入口/6031字节最终检查；不冒称同一源码通过两次。新增代码只核对来源/预算/资源/前序身份，不新增GPU probe/正式数据读取或训练探针。正式新运行入口和原12来源仍为后续worker校验/导入所需，不得按“监听已启动”删除；临时CPU脚本及中间副本在Git精确恢复核对后收尾，另记实际清理。根docs只在Windows，本地提交、不推送，不重置清理周期。

## 2026-11-09 两次旧seed44权重删除与H45增量账

用户确认精确5份旧权重范围后，Linux19:41:49完成删除：原execution/campaign/P/training/P_VendorLink_SEQ_s44/run的stage0.pt、stage1.pt、latest.pt，加原retry_s44对应run的stage0.pt、latest.pt，共3045165946字节（2.84GiB），所有目标普通文件、yongpeng属主、最终路径和单硬链接经核实，无另备份。旧两运行16份非权重文件SHA前后不变；当前retry_s44_15g的12份源码、gate/封装/批次及初始端点身份前后不变，当前worker未停止，更新5235→5237，活动latest仍在且无failure。Windows对应两个旧目录无权重，删除0。[回执](../reports/documentation/20261109/vendorlink_account_protocol/execution/old_s44_weight_cleanup.json)15659字节SHA256 `e16c9c98c38250fd38214728eb15dfe46da29e0b669a1fabe4f917020bc4c7fc`已SFTP回传一致。此前保管段落的“旧权重保留”仅代表当时事实，由本次许可替代；原失败、日志、费用和原始回执不改，已删权重不能由SHA/Git恢复，当前新权重不在删除范围。

新第二路Linux目录execution/parallel_h45保存12份原科学源码原字节副本、新gate/批次、限定额度与操作封装，不复制共享预训练或旧成功权重，不触碰当前15GiB源码/模型。原H1/seed45运行，13GiB allocator与14GiB进程/准入；原seed44声明为完整16GiB承诺，合计30GiB。H45增量账从零计实际费用，最多training28800/GPU0/auxiliary1800秒；总原60h经用户确认调整为训练178200/GPU21600/辅助16200秒。原活动seed44继续原累计账及其辅助14400秒上限，H45辅助加其最多1800秒保证新全局16200秒，不重复原30秒尾部。primary最终累计账加H45增量账才是总账；新parallel_h45/combined_budget.json在两者结算后合账，尚未结算不能把活跃训练记为0已耗。

Windows按[H45同步清单](../reports/documentation/20261109/vendorlink_account_protocol/execution/parallel_h45/sync_manifest.json)核对8份小载荷一致，清单2466字节SHA256 `f5dd67e1dc51fd9e5191d230d218e67ee1ec61e284665b2e10b9819cccc7162f`；无正文、账号、标签、分数、权重或凭据回传。H45科学来源与原12份大小/SHA完全一致，复用已有科学资格，不新做载模/GPU probe。封装仍为实际监管保留；CPU前置仅导入/身份/原生资源准入，未造临时核验脚本。用户已进一步批准整批自动接续，完成接续实现与适用审查后再部署，当前冻结的单条许可/额度和运行事实保留，后续补充另记。根docs仅Windows，本地提交、不推送，不重置清理周期。

## 2026-11-09 VendorLink seed44的15GiB资源修订与新重跑保管

用户限定“提升到15G”，新Linux目录为`reports/documentation/20261109/vendorlink_account_protocol/execution/retry_s44_15g/`。原execution/campaign的OOM现场、成功轨迹和原retry_s44主动停止现场全部保留，不覆盖原件，不删除模型。只在新workspace的step34_vendorlink_account_policy.py改变allocator13→15GiB、整PID及准入14→16GiB三个常量；其余11份运行来源原字节复用，未改全局scripts或任何其他活动/冻结workspace。新policy5944字节SHA256 `40b2c56395d6482b52fda84be0a09fd8082570b78370a6841c3820783274d08a`，Windows具名副本供Git精确恢复；其他模块继续由原来源/Git恢复。新gate绑定修订说明、原科学审查、12份实际来源和本次小核验。整个原execution共同占用32GiB磁盘帽，不能把每次重跑各算32GiB。

原retry_s44的最终累计账training41230/GPU87/auxiliary1605秒，含主动停止的539秒训练，已完整承接。新账在retry_s44_15g/campaign/budget.json：配置核验GPU1秒、准备辅助350秒已加入，观测时已结training41230/GPU88/auxiliary1955秒；CPU322秒为辅助子账，不能再相加。新训练/监管尚未结账；原60h分桶、4h生命周期、30秒尾部不重复计。原及前一retry的budget是累计历史快照，不能与新累计账相加。原retry_s44的13/14GiB及下节旧账入口仅代表当时事实。

[资源核验回执](../reports/documentation/20261109/vendorlink_account_protocol/execution/retry_s44_15g/resource_check.json)保留精确手写源码及首次CPU夹具遗漏caps=CAPS的失败来源/错误与60秒保守CPU子账；实际封装原已显式传caps=ledger.caps，修正夹具后仅补本次证据，不改科研源码。核验执行实际入口的allocator设置语句并读回15GiB，不载模/训练、不申请大块显存；原生准入在15.5GiB阻塞、16GiB准入，Runner监控16GiB。PASS回执写入后的展示命令曾因外围未import json报错，随后重新读取同一已保存回执确认，不将展示错误冒作训练失败或追加GPU核验。

Windows回传授权、修订说明/封装、gate/批次、核验、主动停止/启动、继承账、首步/进度和新policy共11份具名文件，按[同步清单](../reports/documentation/20261109/vendorlink_account_protocol/execution/retry_s44_15g/sync_manifest.json)全部大小/SHA一致；清单3490字节SHA256 `61d2952e39cf36e92d738b70d5416738eef7cb085a95ef92e622e91575375e44`。没有回传正文、账号身份、权重、分数或凭据；原H暂停和其他模型未动。核验源码已直接包含在回执，无额外临时核验脚本；新launch.py仍供实际监管，保留至无后续用途。根docs只在Windows，本地提交、不推送，不重置清理周期。实际进度见交接唯一当前项。

## 2026-11-09 VendorLink seed44单条重跑保管

用户明确“直接把这条P_VendorLink_SEQ_s44实验重新启动”，限定一次原配置重跑。Linux新产物位于`reports/documentation/20261109/vendorlink_account_protocol/execution/retry_s44/campaign`；原`execution/campaign`的OOM现场、latest、阶段端点、日志、原失败费用及健康任务成功记录全部保留，不移动/覆盖原任务，也不恢复其余队列。12份既有已审运行源码原字节复制到retry_s44/workspace，仅调整新gate/batch的部署路径与授权绑定；数据、共享预训练及其原SHA直接复用，不复制模型或正文。原experiment_root继续覆盖新旧目录总磁盘32GiB帽，原13/14GiB显存和60h累计桶保持。

原控制器排空退出后，新账从其最终training40691/gpu87/auxiliary1463秒承接全部历史事件与CPU260秒子账；原30秒尾部不重复计，重跑筹备120秒仅加入辅助。后续累计账在`retry_s44/campaign/budget.json`，原campaign/budget.json作为旧终态保留。原工作流的既有检查、预约、监测、实际训练和成功验收/最新状态退役均由同一Runner/worker实现；不添加新算法、恢复功能、评价或测试平台。

Windows只回传10份具名授权、gate、P批次、启动/继承账/首步观察及启动导入失败源码与日志，按[同步清单](../reports/documentation/20261109/vendorlink_account_protocol/execution/retry_s44/sync_manifest.json)逐份大小/SHA一致；清单本身SHA为`441b2f730cd0fe39b3386e1a272a51ebfaea80b96e544ffa1aad20432593b8cd`。初次启动封装导入失败没有载模或消费正式训练，3983字节原版与bootstrap.log/bootstrap_launch.json保留，修正后Linux进口/来源绑定预检通过再启动一次正式重跑；准备费用已包含此失败。实际进度只见交接及[首步回执](../reports/documentation/20261109/vendorlink_account_protocol/execution/retry_s44/observation.json)，不把启动等同完整成功。启动封装尚用于当前监管，暂保留；未复制原始数据、权重、query/分数或凭据至Windows，未干预中文H暂停或资产。根docs仅Windows，本地提交、不推送；不重置大清理周期。

## 2026-11-09：账号监督协议的新角色数据与实现证据

用户已批准[账号监督实施范围](../reports/documentation/20261109/vendorlink_account_protocol/execution/scope.zh.md)。新Linux根为`reports/documentation/20261109/vendorlink_account_protocol/execution/`；qualification/data保存原三市场旧fit中的1578账号/31560条角色记录及来源/分量/角色索引，文件600权限。确认文本已构建但未评价。Windows仅代码、文件身份、聚合资格和核验/失败费用，未同步真实账号、正文、逐条角色或分数。原六份关系候选删除事实保持，本次没有恢复它们。

新隔离workspace不覆盖中文暂停或旧ABC/Evolution来源。实现外审与主审已闭合，正式训练已开始；12份运行来源、角色数据、活动模型、失败/费用及审查原文继续保留。原失败及后续核验源码按精确Git恢复点保管：a32551f3157b907219c61a33456b600e30aedb47包含本轮删除的已提交源码/包，逐文件大小/SHA在删除前与Git blob相符；CPU01/02/03共用的13045字节旧control也已恢复并与原回执SHA一致后提交。13:25按[实际清理清单](../reports/documentation/20261109/vendorlink_account_protocol/execution/cleanup_receipt.json)删除Windows25文件384276字节、Linux13文件130342字节，另1个Windows新保存的恢复副本在Linux未曾存在。范围为一次性GPU probe、数据/修复CPU脚本、已留Git的失败来源副本、已审输入ZIP及本轮浏览器临时文件；正式报告/日志/费用不删。两端暂保留test_step34_account_increment.py：CPU01最初11231字节版本尚未独立恢复，不能冒称全部历史测试版本可恢复或为此重跑测试；后续CPU02/03版本已可恢复。本项保留是具体证据保管需要，不扩大为长期保存全部临时包。冻结deployment/原回执仍按当时来源保留，退役事实由清单解释，不改活动gate。详见[实现记录](../reports/documentation/20261109/vendorlink_account_protocol/execution/implementation.zh.md)及主审；本次不重置11月6日大清理周期。

## 2026-11-09 删除原三市场关系候选载荷

用户在具体六文件/5513443字节、不另备份及原始来源保留范围呈现后确认：“确认，按此删除并制定协议”。服务器09:37:57逐份复核普通文件、属主、精确目录、大小/SHA、无额外对象及无相关活动进程后，删除Linux `/home/yongpeng/cross-lingual/reports/documentation/20261108/vendorlink_alias_evidence/final_evidence/` 下 `accounts.jsonl`、`candidate_relations.jsonl`、`claim_candidates.jsonl`、`manual_relation_decisions.jsonl`、`header_candidates.jsonl`、`html_sources.jsonl`，并移除空目录；合计6文件5513443字节，未另备份。逐文件身份与32份其他核查文件SHA保留检查见[实际回执](../reports/documentation/20261108/vendorlink_alias_evidence/data_cleanup_20261109.json)。

原始商品包、规范缓存、SQL/HTML、旧实验/模型和中文暂停资产不在删除范围；报告、聚合、来源哈希、费用及历史源码恢复入口保留。Windows再次核对本任务无六份候选载荷，删除0；此前未回传/入Git，现不能由Git或SHA恢复这些生成载荷。原始来源仍存在不等于获准重新生成。下节“六份候选保留”和旧manifest/completion均记录11月8日当时事实，已被本次用户决定与删除回执更新，不追改原件。

本次2296字节回执经SFTP回传、落盘大小/SHA一致，SHA256 `ab5d1237d1b5c7f3943eaf39e6795d0f23a64e9d07f886cd96ac5d353f28a5c3`。新协议及候选读取说明已按[SFTP回执](../reports/documentation/20261109/vendorlink_account_protocol/sync.json)同步并逐份读回相同，根docs未同步。具体新审查包随后按用户明确许可单次外发；完整37531字节原文、主审、输入/完成身份保留，[最终同步](../reports/documentation/20261109/vendorlink_account_protocol/review/final_sync.json)覆盖审后协议与必要审查材料。不是已删除旧核查包的恢复。确认精确输入ZIP可从Git ff6a24d6恢复、10份载荷匹配清单及返回原文已保管后，Windows按[回收/清理回执](../reports/documentation/20261109/vendorlink_account_protocol/review/recovery.json)删除14个临时文件202474字节（本次输入ZIP、重复下载和具名浏览器日志/快照）；未删除正式审查原文/来源，Linux没有本次临时包副本。本轮无新数据解析、科研计算、载模或训练；不重置原核查费用或11月6日大清理/11月21日下次到期。新[账号监督协议](../reports/documentation/20261109/vendorlink_account_protocol/protocol.zh.md)只是准备材料，不会恢复被删候选。本地提交、不推送，根docs只在Windows维护。

## 2026-11-08 原三市场跨账号关系核查收尾

用户已确认原三市场商品包/规范缓存核查、纳入既有身份SQL/HTML，并进一步要求“完成后…脚本和中间文件都删掉”，最后明确“本轮免外审，完成主审和清理”。[本轮报告](../reports/documentation/20261108/vendorlink_alias_evidence/report.zh.md)按证据不足分支交付：保留真实不同账号的弱关联候选及缺口，没有正式训练集、划分或训练。原始身份、链接、指纹、声明正文仅留Linux `reports/documentation/20261108/vendorlink_alias_evidence/final_evidence/`；六个必要最终文件5513443字节，身份见[清单](../reports/documentation/20261108/vendorlink_alias_evidence/final_evidence_manifest.json)，用途见[读取说明](../reports/documentation/20261108/vendorlink_alias_evidence/data_readme.zh.md)。它们不是已验证正负标签，后续不得自动作为训练入口。

五个最小实际源码版本先保存本地Git `7eef50b3f0ba0ccb1de7790155c5aecf6ae059d7`，逐份读回确认大小/SHA及完整字节与执行版本一致；恢复入口见[source_recovery.json](../reports/documentation/20261108/vendorlink_alias_evidence/source_recovery.json)。Windows23:32:40删除36文件245145字节；Linux23:32:52删除34文件17537123字节，合计70文件17782268逻辑字节（约16.96MiB）。包括三份step33脚本、header_evidence、audit03旧版本及各Linux执行副本/缓存、重复提取证据、临时GPG目录、未上传外审包和解包副本；精确路径、SHA、数量见[Windows回执](../reports/documentation/20261108/vendorlink_alias_evidence/cleanup_windows.json)和[Linux回执](../reports/documentation/20261108/vendorlink_alias_evidence/cleanup_linux.json)。源码仍可由Git恢复不等于保留可自动执行工作副本，也不授权重跑。

Windows/Linux分别核对33/35个保留证据文件SHA未变；Linux五份原始包/规范缓存/SQL删除前后SHA一致，原HTML位于删除范围之外。原日志、实际修复前后汇总、来源、费用、主审和最终候选保留，其他实验、旧模型和中文暂停作业未动。未发送任何外审材料。七项计时172.10秒，保守收费200/3600秒，无模型/GPU/训练；费用及结果不足事实不清零。[最终SFTP](../reports/documentation/20261108/vendorlink_alias_evidence/final_sync.json)9份/55466字节双向读回一致，回执自身亦同步；原始候选未回Windows，根docs仅Windows维护。退出SSH前再核对Linux六份最终证据SHA匹配，本任务目录不剩.py/.pyc/.zip。本次为局部收尾，不重置11月6日大清理完成日和11月21日下次到期；本地提交，不推送。

## 2026-11-08 VendorLink固定全市场诊断收尾

两SEQ及用户追加Full／ER的同池评价均完整完成，实际入口和两版部署分别留在Linux `/home/yongpeng/cross-lingual/reports/documentation/20261108/vendorlink_fullpool/workspace/`、其`history/workspace/`。原ABC四方法共36旧阶段权重全部保留；96分数矩阵、逐query向量、原开发文本/标签和账号身份只在Linux，不回Windows。Windows只收聚合、来源和运行小证据：父目录[59份161614字节](../reports/documentation/20261108/vendorlink_fullpool/receive.json)、补充[57份458834字节](../reports/documentation/20261108/vendorlink_fullpool/history/receive.json)均逐份大小/SHA匹配。根docs仅Windows更新；21:30:23[最终SFTP](../reports/documentation/20261108/vendorlink_fullpool/final_sync.json)同步具名8份报告、核对和恢复说明共31988字节，全部匹配，没有覆盖冻结运行原件。

一次性测试两实际版本和运输助手先提交Git `a370acb4216dc63c69eff2ab7500f92c84b1cd78`，逐份读取Git原字节并核对执行回执/本地大小与SHA后退役。Windows实际删除4文件41265字节：`tests/step32_fullpool_increment.py`，以及本任务父目录下`cpu01/test_source.py`、`transport.py`、`transfer.py`。Linux21:30:23实际删除本任务父目录`workspace/tests/step32_fullpool_increment.py`及`cpu01/test_source.py`，共2文件17357字节；两端合计58622逻辑字节。精确路径、各文件大小/SHA及`commit:path`恢复入口见[Windows回执](../reports/documentation/20261108/vendorlink_fullpool/cleanup_windows.json)、[Linux回执](../reports/documentation/20261108/vendorlink_fullpool/cleanup_linux.json)和[Git原字节核对](../reports/documentation/20261108/vendorlink_fullpool/cleanup_request.json)。原始CPU01失败及CPU02局部修复日志、11秒子账均保留，两端生产源码和所有结果未删除；补充未创建新CPU/GPU核验脚本。

清理前确认两批控制器/worker均已退出，两套生产来源分别与正式门一致，删除项不在两套生产绑定中。中文暂停监管及其模型、来源与活动状态未改。本次只做局部常规收尾，不重置11月6日大清理完成日与11月21日下次到期；恢复源码不自动授权重跑。两批合并823/7200秒费用及原负结果不清零。本地提交，不推送。

## 2026-11-08 Registered六轨迹训练权重定向删除

用户要求删除“最近这次实验的模型训练权重文件”，在H消融与今日Evolution六轨迹两个对象及删除影响已说明后，明确选中“今天最新完成的Evolution实验”。本次许可仅覆盖Registered的H1/V1各seed31/32/33六条正式训练权重；原停止结论、数据访问与费用不变，不授权补训或恢复中文H。

Linux于18:46:26完成删除：`/home/yongpeng/cross-lingual/reports/documentation/20261108/evolution_registered/execution/campaign/training/`下六个`P_H_SEQ_H1_s31/s32/s33`、`P_V_SEQ_V1_s31/s32/s33`任务目录内的`run/initial.pt`及`run/stage1.pt`—`stage3.pt`，共24文件7818613626字节，约7.28GiB；剩余该批权重0。逐文件精确路径、大小、SHA及原完成记录身份见[删除回执](../reports/documentation/20261108/evolution_registered/result/weight_cleanup.json)。删除前逐份与六份原completion绑定核对SHA、属主、最终路径、普通文件及单硬链接；无相关活动进程，dispatch完整、无待派发及自动后续。只逐文件删除，未移除整个campaign。

原campaign内203个非权重文件、64492535字节逐份路径/大小/SHA前后一致，包含90分数矩阵、逐query指标、训练/评分日志、完成/配置/费用原件。原正式门、源码、数据和共享预训练不在删除目标内。中文H的dispatch仍为PAUSED_IDLE、7完成/0活动/56待派，其21份已训练权重的路径/大小/mtime/inode前后保持。Windows核对整个本轮evolution_registered目录未发现训练权重，删除0文件；本轮没有将权重复制到Windows或入Git，已删载荷不能由SHA或Git恢复。若需重新载模复查，现有产物已不足，重训须另行授权。

13714字节实际删除回执经SFTP回传，双端SHA256均为`2d8e732bae7c97211e0a28b31df5d360da787eedcf662acd40f85fa5202a50e7`。根docs只在Windows维护，结果说明同步Linux；本次仅文件管理，无载模/训练/评分，原5375秒累计任务账及CPU子账不重置。本地提交，不推送；局部清理不重置2026-11-06大清理日期及2026-11-21下次到期。

Registered结果已完整回传：[36载荷核对](../reports/documentation/20261108/evolution_registered/result/receive.json)共207858字节，另有144字节分析生命周期；只含聚合指标、端点／源身份、完成与费用回执及保存字段派生表。结果回传时24权重7818613626字节、campaign共7883106161字节为当时快照；后续权重删除按上文，不追改原回执。正文、账号身份、qrels、逐query指标、90分数矩阵和有效负结果继续留Linux。已核对载荷后两端本次result/transfer.zip（各50891字节）作为临时传输副本退役，实际载荷、manifest和精确SHA均保留；下文历史临时清理事实不变。

## 2026-11-08 Registered有限前提实现来源与核验

新实现隔离部署于Linux `reports/documentation/20261108/evolution_registered/execution/workspace/`，未覆盖旧活动workspace。Windows仅保存源码、手写CPU日志/回执和来源清单；正式开发数据及新训练权重只留Linux。真实资格缓存仍按下节保管，用户已批准的后续288开发账号和96条语言文本也只留Linux，不入Git或外审包。

CPU01失败、修复后CPU02及实际评分入口CPU03的来源、费用和证据见[实现记录](../reports/documentation/20261108/evolution_registered/execution/implementation.zh.md)。初稿的精确来源可从Git `713d9929:reports/documentation/20261108/evolution_registered/execution/deployment.zip`恢复，deployment_manifest及CPU02/03来源清单、entry_sources、repair_sources保留，不能用新版本冒接旧执行。38份小证据、4份入口证据和审后CPU04原始通过/失败回执继续保留；审查闭合后已退役一次性核验脚本与传输副本，精确范围见下段。未删除正式数据、共享模型或中文H资产；不重置大清理周期。

本次收尾在逐份读回Git原字节并核对大小/SHA后，Windows删除10文件147977字节：`tests/step31_registered_increment.py`、`tests/step31_registered_scoring_entry.py`、`tests/step31_registered_entry_fixes.py`，以及本任务execution下的`cpu01_source.py`、`deployment.zip`、`evidence.zip`、`entry_evidence.zip`、`repair_deployment.zip`、`repair_evidence.zip`、`start_evidence.zip`。每份均可从 `713d9929:<仓库相对路径>`精确恢复，包括初次失败夹具；恢复不授予重跑权限。Linux 16:27:21删除workspace/tests同名三脚本及上述五个evidence/repair/start传输ZIP，共8文件66222字节，见[实际清理原件](../reports/documentation/20261108/evolution_registered/execution/cleanup.json)；Linux deployment.zip在首次核查时已不存在，不计为本次删除，该次核查失败未执行删除。两端合计18文件214199逻辑字节；Linux删除前后实际14生产源、正式门及主审处置SHA保持，删除对象不在活动命令或正式源绑定内。Linux手写Tiny载荷、完整审查输入/返回、原始失败/费用证据及实际训练资产保留。本次局部清理不重置大清理周期。
本次审后已建Linux execution/dataset：288开发账号/2304记录、288确认身份预留、固定96条语言判定及私有qrels；正文/身份/逐条判定不回传。Windows仅保留同字节data_manifest.json、language_receipt.json和启动/源/资源元数据；12载荷按start_manifest逐项核对。实际campaign训练访问及盲分数仅在Linux；启动时约定完成后保留权重、当时未获删除许可，后续用户明确选择及实际删除见本页顶部，不以本段历史安排恢复保留限制。

实现外审返回7成员/36947字节已核对，原文、结构化发现和来源对应保留execution/review/external；仅删除同SHA的.playwright-mcp/evolution-registered-implementation-independent-review-20261108.zip浏览器重复件（1文件36947字节），正式返回仍保留用于本轮结果主审。审后CPU04与全部来源和费用原件保留；没有删除活动资产或改变大清理周期。
## 2026-11-08 已登记账号资格缓存

用户批准一次限定CPU资格，并补充允许旧test作为新确认候选。新私有缓存仅在Linux：`/home/yongpeng/cross-lingual/reports/documentation/20261108/evolution_registered/qualification/private_custody/eligible.jsonl`，656账号/5248条角色候选记录，6250335字节，SHA256 `253f171e4ff387bc970a7793e7a2c4a0d14515580c89f028c276e891e755817c`，目录0700。它含真实身份与清洗文本，供未来获准构建复用，不是正式训练数据；不得进入Git、Windows或外审包。资格时尚未分配；随后已按批准范围固定保留确认身份并导出开发，见[实际数据交付](../reports/documentation/20261108/evolution_registered/execution/data_report.zh.md)。旧test没有训练或评价开封。

Windows只收到15份聚合/身份载荷与一份清单，大小/SHA一致，实际用途和费用见[资格报告](../reports/documentation/20261108/evolution_registered/qualification/report.zh.md)、[回传核对](../reports/documentation/20261108/evolution_registered/qualification/receive.json)。回传完成后已删除本次两端qualification_upload.zip/qualification_evidence.zip临时传输副本，共4文件86874字节：Linux位于evolution_registered根，Windows位于该任务qualification目录；载荷与清单保留，私有缓存未动。新的evolution_registered_review.zip及完整返回包/载荷已回收核验，仍用于后续合同与实际实现对照，保留；另已核对同SHA并删除.playwright-mcp/evolution-registered-review-return.zip这份30394字节的浏览器重复下载，正式返回原件保留。实际资格入口、五份原数据依赖及源身份仍有本轮主审/必要外审用途，暂保留；本轮不删除旧数据、原A/B非权重结果、共享模型或中文H资产。既有权重及127原始结果删除事实保持，下面记录不改。本次新缓存保管不重置项目大清理日期。

本次科研方向咨询另回收33571字节返回ZIP及8成员，保存于`reports/documentation/20261108/evolution_registered/direction_review/external/`；原文、输入包与返回载荷仍用于当前方向取舍和后续方案对照，保留。核对页面SHA、7载荷清单及15输入来源后，仅删除Windows `.playwright-mcp/evolution-direction-independent-review.zip`这一同SHA浏览器重复副本，1文件33571字节；正式返回包保留可恢复原字节，详见[回收与清理记录](../reports/documentation/20261108/evolution_registered/direction_review/recovery.json)。未连接Linux或改其资产，不重置大清理周期。

## 2026-11-08 Evolution权重及本轮新增文件定向删除

用户在具体清单与不可恢复后果已呈现后，明确选择“按上述范围全部删除”，原话与保留边界见[讨论记录](RESEARCH_DISCUSSION.zh.md)。本次仅处理当前Evolution A/B和本轮三个提交新增材料，未改写Git历史；删除不改变原有效负结果、判据、访问事实或已发生费用。实验累计仍按原账training12926/gpu367/auxiliary1882秒（4h12m55，CPU59秒为子账）保留，本次为文件管理，无科研计算、载模或训练。

| 端与精确对象 | 实际删除文件数 | 逻辑字节 | 结果与恢复边界 |
|---|---:|---:|---|
| Linux：/home/yongpeng/cross-lingual/reports/documentation/20261107/evolution_pilot/execution/campaign 下A/B完成记录绑定的initial.pt与stage*.pt | 56（22起点、34阶段模型） | 14776938922 | 11:03:15删除完成，全部目标不存在；未入Git，无可由本次Git历史恢复的权重副本 |
| Linux：/home/yongpeng/cross-lingual/reports/documentation/20261107/evolution_pilot/candidate127/ 整个目录 | 88 | 13188966 | 11:03:15目录及文件已删除，包含54份原始评分文件及未入Git的逐query指标；原始结果不能由汇总还原 |
| Linux：/home/yongpeng/cross-lingual/reports/documentation/20261107/evolution_pilot/execution/stop_observation.json | 1 | 34128 | 11:03:15已删除，字节相同的已提交Windows副本可从下述Git入口恢复 |
| Windows：本轮三提交新增文件，基于c1b07a6f至6a9d9a36的新增清单 | 39 | 325159 | 客户端11:04:11删除完成；32份candidate127材料、6份execution新副本及scripts/step30_evolution_candidate127.py均不存在，已提交原字节可恢复 |
| 两端合计 | 184 | 14790487175 | Linux145文件14790162016字节；此为逻辑字节，不冒称文件系统实测净增 |

Linux权重的精确原路径/大小/SHA仍保存在原`execution/campaign/A/batch.json`、`B/batch.json`绑定的22份`run/completion.json`中，原件未改。对象范围为A的H0—H3、V0—V3各seed11/12，每任务initial.pt和stage1.pt；B的H1/V1各seed13—15，每任务initial.pt及stage1—3.pt。删除前56份逐项大小/SHA已经匹配原绑定，最终执行再次核对绑定原件SHA、目标最终路径、属主、普通文件/无符号链接、单硬链接及字节总量。候选目录88文件和单独停止观测也核对实际归属、边界和数量后逐文件删除，只移除该目录内的空目录，没有递归删除整个evolution_pilot。

原A/B的327个非权重文件在Linux保留，删除前后路径与文件元数据完全一致；27份关键原件（A/B batch、22份completion、整批completion/dispatch/budget）的SHA逐份不变。正式门、数据manifest、共享bert-base-cased五文件和H消融dispatch共8项元数据保持；Evolution官方数据、固定数据正文/隔离标签、原代码/审查、旧VendorLink/ABC及H消融资产均不在删除对象内。11:03:15无Evolution活动进程；H原监管1412412仍存在，属主yongpeng，创建时刻1794051702.82、命令和cwd核对，原dispatch仍PAUSED_IDLE、7完成/0活动/56待派。未修改H的暂停状态、来源、模型或预算。

Windows精确删除集合可由`git diff --name-only --diff-filter=A c1b07a6f..6a9d9a364f4dda78f61a0922176feeca08aa82b9`恢复；这三个提交为eda3b629、e33a313e、6a9d9a36。39份现存字节在删除前已实际读取Git blob并逐份匹配大小/SHA，执行时再次确认HEAD与工作区未变、39文件325159字节、最终绝对路径位于工作区且路径链无reparse point，再用PowerShell LiteralPath逐文件删除。恢复原件使用`6a9d9a364f4dda78f61a0922176feeca08aa82b9:<仓库相对路径>`；不依赖当前同名文件或仅有哈希。Windows execution六项是stop_observation.json，以及campaign/completion.json、dispatch.json、budget.json、A/decision.json、B/decision.json；其Linux原A/B来源保留。candidate127的源码、报告、汇总、回执和未上传审查包都在该Git历史内，54份原始评分和原始逐query指标没有入Git；不得宣称已完整保管或可从Git重放。

原冻结运行回执不追改成“未使用”，已删路径通过本记录解释。现有交接、计划、设计入口与用户决定原位修正，不另造清理包、备份或平行进度系统。根docs仅在Windows维护，本次正常本地提交，不推送。恢复历史源码不授权重新读取数据、载模、评分或训练。此为用户指定局部清理，最近大清理2026-11-06及下次到期2026-11-21保持。

## 2026-11-07 Evolution双路正式启动与核验收尾

本次增量外审及独立主审已关闭三项必修，正式来源022e405a与11份部署脚本绑定在`execution/formal_gate.json`；Linux `execution/campaign`已实际运行。七份启动原件通过SFTP回传，大小/SHA与[启动清单](../reports/documentation/20261107/evolution_pilot/execution/start_evidence_manifest.json)全部一致；正文、标签、预测、Memory和正式权重仍只留Linux。动态运行状态只见交接唯一Evolution项，不以此处作为新的进度入口。

按已确认执行补充第21行及科研纪律第9节，完成本轮核验临时资产收尾：[Windows清单及精确恢复入口](../reports/documentation/20261107/evolution_pilot/execution/cleanup.json)记录已删10文件401671字节，包括3个临时测试入口、2个传输ZIP、4份审查源码副本及被实际正式门替代的启动草稿。实际六个CPU/GPU核验版本已从01a1efb6、f4d938df、ce796506、a41a5ebf、0661a971、fdcca708读回原字节并逐份匹配原执行绑定；失败及中间修复版本均保留恢复入口。送审ZIP可从200ccc19恢复，返回ZIP可从022e405a恢复，六份返回原件和主审处置继续保留。浏览器记录的相对下载副本在工作区对应路径未找到，未作跨目录搜索或声称已删除。

Linux 22:49:03[实际清理](../reports/documentation/20261107/evolution_pilot/execution/cleanup_linux.json)删除11文件2606236469逻辑字节：三个无正式依赖的核验入口，及单路手写GPU核验的8份临时模型2606203006字节（约2.427GiB）。所有模型在删除前与已保管路径/大小/SHA一致，完整核验与恢复评分证据、失败账和来源版本保留；SHA不能恢复已删模型，恢复脚本也不授权重跑。检查实际进程命令及正式11份源绑定，均不依赖本次删除对象，删除后正式源SHA保持。正式Evolution产物、固定数据、预训练、中文H及ABC资产均保留，未修改活动门或费用账。

本次Linux定向核对和删除实际3.143秒，包含在已声明120秒筹备/收尾收费内，不退款或另开预算。根docs仍只在Windows维护，本地提交不推送；局部收尾不改变11月6日大清理完成日或11月21日下次到期日。

## 2026-11-07 Evolution实现结案与正式数据准备交付

用户明确“把数据集准备好”后，实际生产构建在Linux独立`reports/documentation/20261107/evolution_pilot/data_preparation/dataset`完成一次，18文件、10291060逻辑字节，含960账号/3840记录/1920视图。官方四表来源仍直接复用既有qualification/received_windows包，未重下或复制官方包。正文、源映射、dev/test qrels、固定96条fit语言样本及逐条判断仅留Linux；private_custody目录0700，研究者只查看了本次明确获准的固定fit语言样本。Windows的dataset子目录只有17808字节manifest，没有正文或保管标签。实际范围、限制和费用见[数据交付](../reports/documentation/20261107/evolution_pilot/data_preparation/report.zh.md)。

34份聚合验收、访问/费用原件及CPU04/05证据共59993字节已经SFTP回传，[逐文件大小/SHA](../reports/documentation/20261107/evolution_pilot/data_preparation/receive.json)全部匹配。真实数据文件不纳入Git；清洗、导出和语言门完成不表示已训练或确认H效果。Linux使用独立workspace九生产源码及两资格来源，原官方资料、中文作业来源/权重、Memory、ABC资产均未改变。

实现外审完整四载荷及清单保留于implementation/review/external，五项修复及三项解释边界按[独立主审](../reports/documentation/20261107/evolution_pilot/implementation/review/primary_disposition.zh.md)闭合；专用网页已关闭，无待发送/生成操作。输入ZIP可从Git d6194e4a精确恢复；原包许可、身份、上传过程与返回SHA不追改。核对保留原文及恢复来源后，按本任务常规收尾删除输入ZIP、重复返回ZIP和浏览器下载副本，不为删除另建备份或上传Linux。

CPU01—05的五个实际脚本版本已分别从Git 628e7311、594c726e、d1949200、2ba0f685、8ff15015读取原字节并核对各执行清单大小/SHA；[恢复与删除清单](../reports/documentation/20261107/evolution_pilot/implementation/cleanup.json)记录精确commit:path。当前两个一次性CPU脚本均无活动/待派依赖，不在build/language生产门九来源内，故两端退役；不是删除生产模块或改写冻结清单。Windows共删5文件245213字节，Linux2文件36280字节，合计281493逻辑字节，见同目录cleanup_windows.json/cleanup_linux.json。原CPU01失败、修复中间版本、各原始回执与费用均可追溯，恢复源码不授权重跑。

本轮只同步具名数据准备证据与收尾材料；20:50:36[最终同步](../reports/documentation/20261107/evolution_pilot/data_preparation/final_sync.json)的14份交付说明、审查原文/处置、费用索引及清理材料共108077字节，双端大小/SHA均一致。根docs仍仅Windows维护，本地提交不推送。局部核验脚本/传输副本清理不改变11月6日大清理完成日及11月21日下次到期日。

## 2026-11-07 H消融暂停派发部署与外审收尾

用户已确认暂停/显式恢复、当前控制器接入，并明确“核对完成就直接部署吧”。完整定向外审9项、两项必修已按[独立主审](../reports/documentation/20261107/h_ablation_pause/review/primary_disposition.zh.md)关闭；原报告及2项OPEN_REQUIRED原件保留。最终CPU05十一项PASS，五次核验共34.648642秒，均包含在600秒保守筹备值内，已经在冻结后的原辅助账加一次；不重提final_tail或清零原费用。

新监管入口、控制器及派发器位于Linux独立的reports/documentation/20261107/h_ablation_pause/repair/final_sources；原活动workspace的50来源、原formal_gate/batch未覆盖，模型/数据/Memory及Full27不动。19:41:42启动后19:46:37[实际核实](../reports/documentation/20261107/h_ablation_pause/deployment_verified.json)PAUSED_DRAINING，旧父停止派发、原三worker继续，未补领下一条；源身份、冻结快照、授权/request、启动、监管阶段、费用及两次进程观察均留在同一本轮目录。初次交接尚在排空，当前源和CPU绑定保留；不为清理改活动request。初版从Git941dc8bb、送审版从07b2b6f2、CPU04从5b006780精确恢复；CPU02两份独有中间源码在cpu02_sources.zip，CPU05来源见repair/source_manifest05.json。

输入24载荷逐份验证可从既有Git精确恢复，九份返回原件及8项SHA核对保留；因此[定向删除三个无用传输ZIP](../reports/documentation/20261107/h_ablation_pause/review/cleanup.json)，141647逻辑字节（不冒称磁盘净增），包含Windows送审ZIP、返回ZIP和浏览器下载副本。输入恢复commit:path/换行模式见transport_recovery.json，原包身份/授权/提交/回收回执不追改。没有新建冗余模型备份或删除权重。Playwright关闭本轮网页时返回浏览器已占用；完整意见已回收且无待发送/生成操作，未终止浏览器或其他页面，该工具限制不阻断已获授权交接。

19:54[最终同步核对](../reports/documentation/20261107/h_ablation_pause/final_sync.json)18份审查/结案材料大小和SHA全部一致，原50来源及更新四来源保持；实际state和活动账没有被Windows快照覆盖。本轮仅Linux独立准备/交接材料及必要回执同步，根docs仅Windows，本地提交不推送，不重置11月6日大清理完成日及11月21日下次到期。后续实际控制阶段和唯一进度见交接，不将本存储记录当实时进程监视。

## 2026-11-07 Evolution预实验方案外审收尾

本轮只有方案、来源核对、获准中文运行状态只读核实和具体包网页外审，没有Evolution正式数据构建、模型加载/训练、模型文件写出或Linux部署。最新提案中的124阶段端点、预计55—60GiB/64GiB上限是待用户确认的未来规划，不是实际存储占用，也没有改变中文消融的保存规则或资产。

外审原文30939字节与网页声明SHA一致，完整网页文本、请求、许可、实际提交、载荷身份和独立主审处置保留于`reports/documentation/20261107/evolution_pilot/review/`，见[回收与清理记录](../reports/documentation/20261107/evolution_pilot/review/recovery.json)。原上传ZIP在实际读取并核验Git `78b98df5:reports/documentation/20261107/evolution_pilot/review/evolution_pilot_review.zip`的185931字节/SHA后退役；该恢复入口保留精确送审25载荷及清单，不用当前修订稿覆盖旧身份，也不授权恢复后直接重发。

按科研纪律第9节处理本任务自行生成的一次性副本，Windows实际删除上述ZIP及本次8份浏览器下载副本/日志/快照，共9文件294691字节，逐路径/大小/SHA见同一回收记录；完整审查和必要来源仍保留。没有递归清理其他浏览器文件，没有清理或同步Linux资产，根docs和本次方案/审查证据只在Windows及本地Git保管。专用审查页在回答完成、无待发送操作、完整回收和主审处置后已关闭；未推送，不改变原大清理周期。

## 2026-11-07 Evolution资格完成与Windows副本清理

用户明确允许本轮Windows盘点，并要求能继续则迁移Linux后删除Windows副本。最终正文/身份隔离盘点支持继续到有限遗忘预实验，见[报告](../reports/documentation/20261107/evolution_qualification/report.zh.md)。两个官方包保管在`/home/yongpeng/cross-lingual/reports/documentation/20261107/evolution_qualification/received_windows/`：`data-and-readme.zip` 332606424字节、SHA256 `5c8a321d51ca2f2dd10afbdc80a44c0c44498954f7c1ca604932b6667a8673bb`；`dataset-generation-code.zip` 48698字节、SHA256 `6143e525b0aa3b50415aff47dbde9b48ffbf793b23f75d0b021e7be9cac94792`。官方MD5也均一致。

同目录另存实际audit/followup源码、报告、README、范围、原始汇总/日志及来源清单26文件。两包加这些材料共28文件、332983607字节，全部通过SFTP回读实际远端字节，与拟保管来源逐文件核对大小/SHA；详见[回读核验](../reports/documentation/20261107/evolution_qualification/transfer_verified.json)。不是仅凭远端同名或旧哈希推定一致。旧Linux下载失败/手写核验目录未覆盖，没有在Linux执行新科研计算、载模或触动其他作业。SFTP末次核实两包大小/路径后正常退出。

Windows实际删除用户指定`C:/Users/35734/Downloads/Compressed/`中的两份上述ZIP，共332655122字节（317.24MiB），以及本轮28份临时回读副本；第一次获准清理共30文件、665638729字节，见[原始清理回执](../reports/documentation/20261107/evolution_qualification/cleanup.json)。临时回读副本是本次核验额外产生的副本，不把其空间计成用户原有两包大小；没有在Windows解压市场表到磁盘。保留实际分析源码、官方README和全部汇总/费用/失败/通过证据。

最初含退役PGP助手和旧外审包的32文件联合清理被自动审批以范围不足拒绝，未执行；随后先收窄完成上述30文件。用户又明确点选这两项，要求“一起删掉，这是科研纪律中规定的”，才完成另2文件45535字节清理，见[新增明确许可与删除回执](../reports/documentation/20261107/evolution_qualification/cleanup_retired.json)，原拒绝历史见[范围记录](../reports/documentation/20261107/evolution_qualification/cleanup_scope.json)。`pgp_format.py`原字节可从Git `6f62121c:reports/documentation/20261107/evolution_qualification/pgp_format.py`恢复，旧未发送外审ZIP可从`dee6233a:reports/documentation/20261107/evolution_qualification/review/evolution_review.zip`恢复，删除前已实际读取Git blob核对大小/SHA，见[恢复核验](../reports/documentation/20261107/evolution_qualification/retired_recovery.json)。两者均未上传本次Linux目录；主盘点/补充源码是实际结果依据，仍需保留，不按CPU脚本名称一并删除。恢复源码不授权重跑。

合计32文件、665684264逻辑字节已清理。旧外审包未发送且不涵盖最终增量，不可恢复后直接当最新送审包。根docs、最终状态和保管/删除回执仅Windows与本地Git保存；上面的28项是本次实际远端同步范围，不宣称全部项目文件双端同步。本次为局部收尾，不重置11月6日大清理日期或11月21日到期日；未推送。

## 2026-11-07 消融正式输入接入与开训证据

用户明确“开始正式训练”后，在已有独立workspace接入18个精确逐文件符号链接：冻结BGE预训练12文件、数据manifest/validation/groups及train文本/train监督/development盲文本6文件。目标均解析为既有项目内的本用户普通文件；未连接旧development监督、owners、test或private_custody。逐路径和大小见[正式接入回执](../reports/documentation/20261107/h_ablation_start/preflight.json)。Full27继续直接引用原trial3复验权重，不复制、移动或删除；真实正式控制器已在派发前核对其当前SHA与必要恢复/来源证据，先前仅元数据观察的原记录不追改。

新增状态目录为`reports/documentation/20261107/h_ablation_start`，实际运行目录为准备workspace下新建的`formal`。原prepare关闭draft和原审查/CPU回执保持；本次authorization/formal_gate/claim/launch独立保存并冻结，不用旧“关闭”记录否定新开训决定。原50生产来源无改动，不恢复退役CPU/GPU脚本。22份启动/首更/源身份及资源证据共99275字节经SFTP回传并[逐项核验](../reports/documentation/20261107/h_ablation_start/receive.json)，含三worker实际首更日志快照；没有传回模型、Memory或正式文本/标签。运行日志/状态由原控制器继续在Linux更新，本地仅保存具名时点快照，不双向覆盖活动文件。根docs仅Windows维护；本次未做资产清理，不改变大清理周期。

## 2026-11-07 调参后H消融实现审查与核验脚本收尾

八配置实现、必要CPU核验及[13项实现外审处置](../reports/documentation/20261107/h_ablation_prepare/review/primary_disposition.zh.md)闭合后，按本任务已确认范围清理退役核验助手与重复包。Windows实际删除8文件、959079字节：4份本轮CPU测试/包装/必要中间源码，以及CPU传输包、送审ZIP、返回ZIP和浏览器重复下载各1份。Linux仅在本次独立准备目录删除13份退役源码/传输副本及51份可重建字节码，共64路径、1570061字节；其中旧通用模块/测试只清理本次临时部署的额外副本，根仓库普通测试与生产依赖不删。两端合计72路径、2529140逻辑字节；不是权重释放量。

逐路径、字节、SHA、Git恢复映射与实际删除见[Windows计划及回执](../reports/documentation/20261107/h_ablation_prepare/cleanup_windows.json)、[Linux计划](../reports/documentation/20261107/h_ablation_prepare/cleanup_linux_plan.json)及[Linux回执](../reports/documentation/20261107/h_ablation_prepare/cleanup_linux.json)。删除前逐文件Git原始blob与当前原字节核对通过。最终CPU06测试、相关包装器/副本和外审整包可从本地提交`c97b5962c52ce9df34f00880c39e448ac296cd29`按清单路径精确恢复；CPU01/02初始版本保存在`6bdc419350bc62318ea3b8123dc87e1fcb60a817`的cpu_delivery.zip，原失败包装器同提交cpu01/cpu_check_source.py；CPU03测试在该提交，CPU04中间测试及CPU05版本在`af9bcd9c58dd98b1f29d4a38e18c720665de55dc`。字节码可由对应源码重新生成，恢复源码不授予重新执行权限。

保留最终50项正式运行来源、全部原CPU通过/失败日志与回执、外审9份原文/支持材料、网页全文、主审处置、身份和预算。Linux删除前确认本任务无活动进程，清理前后50来源大小/SHA一致，正式目录和模型目录均不存在。本轮没有创建GPU核验脚本或监听，未读取或删除任何模型、正式数据和标签；原新H27/旧H27及其他实验资产仍按原决定保留。正式资产链接及权重现检被自动审批拒绝，未执行；已有权重元数据仅作关闭draft的准备，不能冒称本轮实体核验。必要准备证据56文件、530381字节及清单已通过SFTP同步到本次Linux准备目录，[最终同步回执](../reports/documentation/20261107/h_ablation_prepare/final_sync.json)逐项大小/SHA一致，并再次核实50来源、关闭门、退役脚本不存在及无本任务活动进程。根docs仅Windows维护，本轮局部收尾不重置大清理周期，下一到期仍为2026-11-21。

## 2026-11-07 H历史权重结果回传与传输副本收尾

原调参正式结果回传5058份、305105336字节，另4份Linux保存字段核对/整理结果逐项大小与SHA匹配，见[来源清单](../reports/seller_alias_continual/20261107/history_result/receive_manifest.json)、[回传](../reports/seller_alias_continual/20261107/history_result/receive.json)和[分析回传](../reports/seller_alias_continual/20261107/history_result/analysis_receive.json)。正式结果按原路径保存；历史/ABC的状态及旧H引用完成原件保存在本轮source_receipts快照，不覆盖旧交付回执。新生成正文、pairs/owners、Memory及模型载荷留Linux，不进入此次回传或Git。

Windows与Linux各删除本轮两份已验收传输ZIP，每端161264619字节，合计322529238逻辑字节；原5058份来源前后SHA、4份整理结果、54份新旧H权重存在/大小/属主核对通过。详见[Windows清理](../reports/seller_alias_continual/20261107/history_result/cleanup_windows.json)及[Linux清理与权重位置](../reports/seller_alias_continual/20261107/history_result/cleanup_linux.json)。未重新哈希或加载这些大权重，原记录SHA不冒充此次重测；全部54份保留，新H27用于拟议消融Full复用。没有新建CPU/GPU核验脚本；冻结运行来源及绑定的旧测试源码保留用于审查身份，未因文件名而删除。

HPO和英文ABC活动监听副本均已依原规则自删/正常退役，本次只核实不存在，未重复删除或创建监听。原搜索108份推理权重的自动退役属于原合同执行，逐候选清单/回执已回传，不记为本次人工删除。结果外审和7项独立处置已闭合；Windows按[审查收尾清单](../reports/seller_alias_continual/20261107/history_result/review/cleanup.json)删除本轮输入ZIP、重复返回ZIP、临时快照/浏览器日志共9文件、9160362字节。输入包原字节已从Git提交d7f9394d实际读出并验证大小/SHA；完整外审10份原件、网页全文、主审和身份清单保留，未为删除重复包另造备份。原始指标、必要分数/query、日志及来源保管不变；结果和审查正式材料210项、15837471字节及清单/回执双端大小/SHA匹配，见[最终同步](../reports/seller_alias_continual/20261107/history_result/final_sync.json)。最终delivery.zip传输副本3816906字节在两端核验后删除，各1文件；输入/返回外审ZIP没有为了删除而额外传到Linux。根docs仅Windows；本次局部收尾不重置11月6日大清理日期，下一周期仍11月21日。

**11月7日权重保管决定补充：** 用户先要求删除此次H调参权重，获知后续消融的Full复用依赖后，明确选择“保留这27份，维持消融Full复用（推荐）”。服务器10:43:56（北京时间）只读核实本批formal已有completion、无failure及相关活动进程；随后遍历该formal目录，全部27份`.pt`均位于`reports/documentation/20261103/history_wait/workspace/formal/replication/`，共35,218,033,608字节（32.799350GiB），均为当前用户所有的普通文件、nlink=1，解析后路径仍在该replication目录内。formal其余位置无`.pt`；Windows本批工作目录及结果回传目录未发现`.pt`、`.bin`或`.safetensors`副本。本次仅核实元数据，未重新计算大权重SHA或载模；实际删除权重0份、0字节。原H基线27份继续按既有安排保留；本次决定不授权新训练或扩大消融预算。逐文件位置及原记录SHA仍见上述`cleanup_linux.json`，原回执不改写。

## 2026-11-06 两端项目大清理完成

用户确认完整范围，并进一步明确“继续保留这20份权重”“允许核实后删除旧失败权重”。Windows删除9文件1,009,323字节；Linux删除171份重复包/下载缓存及9个旧失败权重路径，9路径对应3底层文件。总189路径，硬链接去重载荷6,593,774,946字节（6.14GiB），Linux分配空间释放6,593,273,856字节；不将共享磁盘波动计入收益。具体逐文件保管映射、SHA、删除和保护核验见[本轮汇总](../reports/maintenance/20261106/summary.zh.txt)、windows.json、linux.json及failed_weights.json。

20份旧VendorLink/FOREVER核验权重26.68GiB按用户明确决定保留，pending_weights.json已记RESOLVED及各组处置；Optuna/ABC活动资产、原H基线、预训练、数据和必要证据保留。Linux64活动来源SHA、ABC36/H27元数据、旧失败48个非权重文件SHA检查通过；Windows原有未提交修改未动。9个失败权重链接已全部删除，3底层文件不可由SHA恢复，旧失败日志和费用不清零，不授权重训。整轮完成日期2026-11-06，下一到期2026-11-21；仅maintenance目录双端同步，根docs仅Windows，本地提交不推送。

## 2026-11-05 ABC正式监听部署后的增量校验脚本清理

用户已确认自动接续，新增监听外审及两项异常计费修复完成。按此前“此次试验CPU/GPU校验脚本完成后删除”的决定，Windows删除新增CPU测试及CPU02/03两份必要中间副本共3脚本，并删除已验收的重复外审ZIP，共4文件；Linux删除本轮唯一CPU测试副本1文件。逐项大小/SHA、原字节恢复依据与删除回执见[清单](../reports/documentation/20261105/english_abc_wait/cleanup_plan.json)、同目录cleanup_windows.json和cleanup_linux.json。恢复提交28e531489d3df89217473a1de357c87797595ad0；CPU01原来源和冻结送审批次也可从387d0d02恢复。各原CPU回执、来源清单、外审全文/主审和费用保留，不回写历史路径。

实际运行的active_abc_wait.py、保留源码scripts/step29_english_wait.py、九runtime来源及活动state均不在清理范围。服务器11:37:19核13项当前运行绑定未变，唯一监听583983继续等待，HPO未改。新CPU累计62秒、GPU仍37秒；本次清理无额外GPU/科研测试，不重置大清理周期。最新保管同步见[本轮sync](../reports/documentation/20261105/english_abc_wait/sync.json)，根docs仅Windows。

## 2026-11-05 ABC校验脚本及压缩包定向清理

用户在正式筹备完成后要求删除此次CPU/GPU校验脚本，取代此前大清理中对英文ABC脚本的暂时排除。Windows删除10份脚本和6个压缩包，共16文件574252字节；Linux删除14份脚本副本和4个压缩包，共18文件545333字节；合计1119585逻辑字节（约1.07MiB）。逐文件路径、SHA、Git恢复方式见[定向清单](../reports/documentation/20261105/english_abc_cleanup/plan.json)，实际删除及保护检查见同目录windows_receipt.json和linux_receipt.json。Linux观测10:56:46，客户端Windows回执10:58:01；时钟不混用。

删除前逐项核实Git提交a47f2ce7a1acdaac6ea54119dd0e0a070ee2d413的commit:path/blob可恢复完全相同字节，不只保留哈希；清单恢复模式全部为raw_blob。需要复核历史CPU/GPU执行版本时先在Windows从该提交恢复并核bytes/SHA，再按实际许可SFTP，不在Linux使用Git。旧CPU01/02/03来源和原送审上下文由其原压缩包/文件的Git版本恢复，工作树不再保留重复压缩包。原通过/失败日志、指标、费用、审查原文及旧source_manifest/同步回执未追改，旧路径不表示校验脚本仍存在。

当前formal_gate.draft只绑定九份实际运行源码，移除两个CPU测试与run_cpu的历史文件校验项；实际算法/调度/评价源码字节不变，approved=false、review_closed=true保持。Linux九来源、119份非删除材料及中文HPO的58来源在删除前后核验一致，父4159100与三子进程身份保持；旧模型权重及数据不在清理范围。原10:47同步回执仍为当时快照，清理后来源/费用及证据以[本次同步](../reports/documentation/20261105/english_abc_cleanup/sync.json)为准。无新CPU/GPU核验、正式训练或真实dev读取；局部清理不重置大清理周期。根docs仅Windows，本地提交不推送。

## 2026-11-05 退役手写CPU/GPU校验脚本双端清理

用户确认按已呈现范围清理，并明确排除本次VendorLink英文ABC新实验全部CPU/GPU脚本。Windows删除360份4,229,378字节，Linux删除669份8,474,183字节，合计1,029份12,703,561逻辑字节（12.11MiB）；包含旧资格入口、启动包装及退役手写_verify/_check重复副本，不将逻辑字节冒称共享磁盘净变化。逐文件路径、大小、SHA和恢复依据见[Windows清单](../reports/maintenance/20261105/qualification_cleanup/windows_inventory.json)、[Linux清单](../reports/maintenance/20261105/qualification_cleanup/linux_inventory.json)，实际缺失和保护核对见同目录两端receipt及verification.json。

删除的135种字节版本均已核实为现有Git的commit:path和blob原字节；清单每项recovery记录具体提交、仓库路径、blob及换行恢复方式。恢复只在Windows读取Git，若restore=lf_to_crlf则先统一LF再转换CRLF，恢复后核对bytes/SHA，再按实际用途和许可SFTP到Linux；不运行旧脚本或重训来替代恢复。旧冻结回执、来源清单和已发生事实不改写，历史路径若指向本次退役脚本，以此清理记录及恢复清单解释；结果、分数、标签开放账、日志、原审查、失败诊断和费用保留。

当前中文HPO的Linux58来源、Windows对应58来源均不变；Linux英文ABC27份受保护脚本在删除前后SHA一致，20261104/english_abc和20261105/english_abc_formal_prepare及根目录英文入口/测试全部排除。保留通用模块、普通回归测试及正式代码所需校验函数。Linux另保留3份未找到精确Git恢复版本的唯一历史源码：retention_diagnostic_execution/20261011_140900/workspace/scripts/step28_retention_diagnostic_verify.py、h_consistency_execution/20261016_124900/workspace/reports/qualification/cpu/attempt04/sources/scripts/step28_g_half_verify.py、seller_alias_continual/20261015/balanced_pooling_repair/20261015_105800/workspace/scripts/step28_balanced_pooling_verify.py；完整路径及SHA见Linux清单keep项。活动父4159100及541431/541432/541433在删除后身份保持，无重启、重试、新科学运行或预算修改。

本轮仅维护清理清单/回执与根docs；两端维护记录按同目录sync.json核对，根docs仅Windows。一次性盘点/删除助手和传输索引收尾删除，没有新增备份ZIP，不改写Git历史。仅本地commit、不推送；局部脚本清理不重置项目大清理周期。

## 2026-11-05 ABC正式入口筹备与单路GPU完成回收

原gpu_wait的12份状态/首更/三表示结果与更新/空日志已回传，逐文件大小/SHA与[完成观测](../reports/documentation/20261104/english_abc/gpu_wait/completion_observation_20261105.json)一致；原launch、许可、23:16/23:29观测和失败账保持，不重建已自删的活动副本。旧准备账在单路完成后结算，新阶段从[独立目录](../reports/documentation/20261105/english_abc_formal_prepare/)承接，无新增GPU试跑。

新目录保留范围、元数据输入绑定、approved=false的formal门草稿、12来源清单、CPU01—04手写核验及费用；CPU03失败测试原件保留，CPU04七项通过并绑定最终来源。source_bundle.zip承担CPU01原来源恢复，冻结abc_formal_review.zip承担CPU02执行来源和外审当时上下文恢复；四份完整外审、主审处置与回收记录保留，返回ZIP验收后删除重复压缩副本。只从既有export manifest和旧completion读取允许的元数据，未回传/解析真实fit/dev/qrels或权重。独立Linux workspace不覆盖原GPU或活动中文HPO来源。外审输入包仅40份已列载荷和清单，不含数据/权重/凭据；实际审查与同步状态见交接和review回执。根docs仍只维护Windows，本地提交不推送。

## 2026-11-04 ABC英文适配准备材料

[独立目录](../reports/documentation/20261104/english_abc/)保留完整方案、当前增量源码身份、CPU01—06原失败与通过证据、GPU单路失败的日志/真实更新记录及预算。初次11来源，新增独立监听及测试后为13来源，通过SFTP部署至Linux同名目录的workspace；原formal_chain和活动history_wait源码未覆盖；开发导出只核已有manifest元数据，没有读取其正文或标签。CPU01错误测试捕获器原文件单独保留，失败费用不清零。

source_bundle.zip为初次11来源传输及CPU01的冻结快照；CPU02只修测试捕获器，随后GPU单路尝试来源/限额变化由各实际receipt、当前source_manifest和gpu_permit区分，不把新GPU入口说成已由先前CPU执行。原[39载荷送审包](../reports/documentation/20261104/english_abc/review/package_manifest.json)141943字节、SHA256 5682b0e12e62d98c85a5c59aeb9dcbb63dc018beb7987499b69f7bf624d65248保留实际旧门及来源版本；[完整四文件审查回收](../reports/documentation/20261104/english_abc/review/recovery.json)、[主审处置](../reports/documentation/20261104/english_abc/review/primary_disposition.zh.txt)及严格B3门的定向回复保留。新增监听送审包保存CPU04的原监听版本，自删增量包和CPU05对应最终来源；这些输入快照承担不同历史执行/审查版本恢复用途。所有包只含方案、源码、手写小证据和必要旧审查/结果摘要，不含科研数据/模型载荷或凭据。

本批未产生正式候选权重。原手写GPU尝试在第二次更新OOM并退出，只有一次完成更新记录，不能从没有完整result.json推断没有做过更新。新gpu_wait独立活动副本按用户要求首个真实更新后自删，保留workspace源码及许可/日志/state，不删除其他作业。ABC旧36权重继续保留，中文H27及活动HPO保持。根docs只维护Windows，本地提交不推送；双端同步以实际回执范围为准，不重置大清理周期。

23:20:50的[本轮同步核对](../reports/documentation/20261104/english_abc/closeout_sync.json)确认49项当前材料/保留来源大小与SHA一致；13来源和活动副本另在启动前核对，活动副本及持续变化的state不在同步覆盖范围。CPU06补齐异常首更自删，累计CPU20秒。返回审查ZIP两份冗余下载经四原件及内容哈希验收后删除，原件在review/external，删除记录并入recovery.json；新监听输入ZIP仅Windows保留，承担CPU04/05不同旧来源恢复用途。启动后运行费用继续Linux状态账，budget.json是注明时点的核账快照。后续[23:29:05观测](../reports/documentation/20261104/english_abc/gpu_wait/observation_20261104_232905.json)单独留存，原23:16观测不覆盖；该回执及更新后的budget大小/SHA按[进度同步核对](../reports/documentation/20261104/english_abc/gpu_wait/progress_sync_20261104_232905.json)保管，本次未重传活动脚本、许可或冻结来源。

## 2026-11-04 VendorLink结果回传与保管

[本轮结果目录](../reports/seller_alias_continual/20261104/vendorlink_result/)保留811份172378651字节：216盲分数JSON、216逐query指标JSON、36条完整更新/工作量/完成/访问/进程记录、批次配置和65份Linux实际来源。[SFTP清单](../reports/seller_alias_continual/20261104/vendorlink_result/receive_manifest.json)与[接收回执](../reports/seller_alias_continual/20261104/vendorlink_result/receive.json)逐文件大小/SHA一致；[保存字段核对](../reports/seller_alias_continual/20261104/vendorlink_result/saved_check.json)只检查字段、计数及既存指标简单算术，没有新标签解析或科学重算。原正文、qrels、owners、模型/Memory载荷留Linux，未回传。原冻结账和原结果不覆盖，新增活跃用时按[同桶增量账](../reports/seller_alias_continual/20261104/vendorlink_result/budget.json)保管。

本轮获准外审ZIP为16644193字节、577载荷，SHA256 364f1de8f95084de23595f9b984f676aad93d493108b832e80a0e67c58510a0b；[具体许可及单次提交](../reports/seller_alias_continual/20261104/vendorlink_result/review/submission.json)原件保留。19:56:25[完整回收](../reports/seller_alias_continual/20261104/vendorlink_result/review/recovery.json)71438字节返回包、9成员，SHA256 5d5f1f5401710083467339347c0418fe903562df6af3cc52ea7073fbeca8c04c，逐项身份匹配；完整报告/意见/阅读范围/机械核对/分层表留external/，[主审](../reports/seller_alias_continual/20261104/vendorlink_result/review/primary_disposition.zh.txt)关闭两项报告补全。577输入载荷与保留文件核对后，原送审报告/表格分别提取至review/submitted_report.zh.txt及submitted_tables.zh.txt，原submitted_budget不变；其余575载荷原路径保管。9份返回原件核验保管后，[清理三份重复传输ZIP](../reports/seller_alias_continual/20261104/vendorlink_result/review/transport_cleanup.json)，16787069字节，包含Playwright下载副本，不删除科研证据或权重。原包身份/清单/请求/许可保留，修后报告不冒称送审字节。

原16项及11项同步回执保持；20:02:03[审查收尾同步](../reports/seller_alias_continual/20261104/vendorlink_result/review/closeout_sync.json)17项报告/审查/保管/费用及回执自身共18文件大小/SHA一致，Linux仅更新本轮独立结果目录，不改formal/source/活动作业。当时准备655.000207/3900秒，含结果工作累计510秒，本次收尾新增129秒；原formal加结果收尾共38.576460累计任务小时。原准备145.000207及失败账保留，历史资格124秒另列；网页生成等待不计费。专用外审页面已关闭，一次性SFTP助手已移除；根docs仍仅Windows。本地文档首次替换因混合换行的锚点断言停止，未写文件；调整精确锚点后完成，活跃用时包含在本轮120秒保守准备额中。

结果收尾时曾保留108个阶段权重46985731650逻辑字节（43.758873GiB），当时仅核文件元数据。用户随后固定ABC方向并明确要求删除无用权重，服务器21:30:09按[清理范围](../reports/seller_alias_continual/20261104/vendorlink_result/cleanup_scope.zh.txt)完成[逐文件留证](../reports/seller_alias_continual/20261104/vendorlink_result/cleanup_inventory.json)和[实际清理](../reports/seller_alias_continual/20261104/vendorlink_result/cleanup_receipt.json)：删除BCA/CAB四方法/三种子/三阶段72份，31323821100逻辑字节（29.172582GiB）、31324217344分配字节；保留ABC相同范围36份15661910550字节（14.586291GiB），供后续开发集前置评价，原路径不变。全部108份删除前完整SHA匹配，811来源及17审查收尾材料前后大小/SHA一致，中文H27元数据及活动HPO父子身份保持；五个受限系统/会话进程仅核已记录身份，具体限制和前置检查失败见清理说明，未绕过系统权限。未载模、训练、解析标签或读取HPO效果；预训练、原始数据和活动资产不在删除范围。原分数、指标、源码、日志及有效负结果继续保留，已删除权重不可由SHA恢复。新增准备费用974秒，准备累计1629.000207/3900秒，含原formal共38.847016累计任务小时；旧费用与失败不清零。清理证据及费用双端同步见[清理同步](../reports/seller_alias_continual/20261104/vendorlink_result/cleanup_sync.json)，根docs仅Windows，本地提交不推送，不重置大清理周期。

## 2026-11-03 H_pos_history独立等待接线

本轮[独立目录](../reports/documentation/20261103/history_wait/)保留授权、范围、主审、58来源和wait/test两项增量身份、CPU01—03原失败/成功证据、正式模板/等待配置、launch及现场快照。[SFTP回传核对](../reports/documentation/20261103/history_wait/receive.json)确认28项文件及60项本地来源SHA一致；48247字节传输ZIP在两端核验解包后删除，保留清单/身份，不制作或上传外审包。

Linux workspace为reports/documentation/20261103/history_wait/workspace，仅逐项复制已审58来源并替换控制器可选回调，另存wait/test。19个必要数据/BGE逐文件链接指向原资产，不复制正文/标签/预训练或H27，不链接test/owners。CPU01错误缩进源码及runner、CPU02夹具缺字段时的wait/test/runner和原日志保留；失败费用共计入本轮42秒，不覆盖原冻结workspace。原H27继续作为后续对照，活动VendorLink不改不清理。

唯一监听4159100的活动副本在本轮目录，首个真实训练更新后自行删除；保留源码、state claim、配置、费用、首步及进程身份。旧关闭门保留历史，新条件性执行许可只在前序完整成功退出及资源满足后生效。原启动/CPU/部署记录保留；本轮新增[只读核对原件](../reports/documentation/20261103/history_wait/inspection.json)及[逐项结论](../reports/documentation/20261103/history_wait/inspection.zh.txt)，保存60来源身份、15绑定、19链接、H27元数据和进程/等待账。仅Windows新增核对记录，未部署或覆盖Linux活动来源/配置/账本；原28项SFTP回传事实保持。根docs仅Windows维护；本地提交不推送，不重置大清理周期。

## 2026-11-03 FOREVER结果、外审及权重清理

[本轮结果目录](../reports/seller_alias_continual/20261103/forever_result/)保留1217份80030940字节，含324盲分数、324群指标、324逐query指标、完整运行/校准/访问/预算/配置、原H必要完成记录及65份Linux实际来源；[逐文件SFTP接收](../reports/seller_alias_continual/20261103/forever_result/receive.json)大小/SHA全部一致。正式文本、标签源、owners、Memory和模型未回传；其中FOREVER权重按下述授权清理，其余继续原Linux保管。冻结campaign账/结果不覆盖；本地标准库只做身份、字段、计数与简单算术核对，未重算指标或抽样。表格助手一次null格式错误及费用保留，修正只影响N/A展示。

具体[结果外审包](../reports/seller_alias_continual/20261103/forever_result/review/package.json)4784578字节、574载荷，SHA256 313a7a9a908d5bd781e281686c4eb8746f8f2bf7d0e0567e7a3953e6cab80b82，用户明确批准后于客户端10:03单次上传原FOREVER会话并提交。仅包含分析/群指标/运行证据/实际来源/必要旧审查及预盘点，不含分数/query二进制、正文、标签源、owners、模型/Memory或凭据。10:20:41[完整回收](../reports/seller_alias_continual/20261103/forever_result/review/recovery.json)58339字节返回包，SHA256 7fcdcd093c528042d78ef3b06b7b9cfb303f92156f07faac48a6b47d551bbd10；7成员全部核验。[完整原文](../reports/seller_alias_continual/20261103/forever_result/review/external/report.zh.txt)与[逐项主审](../reports/seller_alias_continual/20261103/forever_result/review/primary_disposition.zh.txt)结案，无必修或必要补测，仅两处文字精化。574输入载荷和7返回材料与独立文件/Git原字节核对后，[三份重复传输ZIP已删](../reports/seller_alias_continual/20261103/forever_result/review/transport_cleanup.json)；送审报告在7eec9437、送审预算在review/submitted_budget.json，清单/身份/请求和全部必要审查原件保留，专用网页已关闭。

[权重预盘点](../reports/seller_alias_continual/20261103/forever_result/cleanup_inventory.json)核对本批27份FOREVER模型为35218653363字节（32.799927GiB）、独立普通文件、nlink=1、实际SHA对应端点；run/work、anchors及latest已由原成功规则退役。按用户分析/外审后删除无用权重的明确要求，重新核对无后续加载/诊断依赖及无FOREVER活动作业，服务器11月3日10:30:52[逐文件删除完成](../reports/seller_alias_continual/20261103/forever_result/cleanup_receipt.json)。27路径全不存在，删除前后1215保留文件大小/SHA、14项只stat材料及原H27元数据一致。原H_pos_tuned27份供H_pos_history、预训练、Memory及活动VendorLink保留；原链3485029和Vendor worker4147594前后仍活动。32.80GiB为独立文件逻辑字节，不冒称共享磁盘净变化；SHA不能恢复权重，不重置大清理周期。

首次22项[同步原件](../reports/seller_alias_continual/20261103/forever_result/sync.json)保留；修后报告、外审/处置、删除回执与最终费用按[最终同步](../reports/seller_alias_continual/20261103/forever_result/final_sync.json)核对，原campaign/source不覆盖。准备累计633.731048/900秒，含本轮413秒及原220.731048秒，失败账保留；含原各桶共26.457578累计任务小时。一次本地索引写入OSError 22后核对原文件完整，用定向补丁完成；无科学数据受影响，费用已计入。本节为现有记录原位补齐并归回独立小节，旧10月23日证据未改。当前状态仅在交接FOREVER行维护，根可变docs仅Windows，本轮本地提交、不推送。

## 2026-11-02 H_pos_tuned两项历史系数实现与CPU证据

实现阶段源码、直接依赖和手写证据保管在Linux独立 reports/documentation/20261102/history_optuna/workspace，未覆盖旧活动链。用户随后确认[实际输入准备](../reports/documentation/20261102/history_optuna/preparation/authorization.json)后，按[回执](../reports/documentation/20261102/history_optuna/preparation/run01/receipt.json)核对83项文件身份，并在该workspace建立19个必要逐文件链接（7数据+12预训练），没有链接heldout、owners或整个数据目录。正文/关系标签只流式校验SHA，不解析；原H27按原路径校验，未加载、复制、移动或删除。原始数据/模型未回传Windows，本地保留脚本、元数据/大小SHA、费用与准备配置。58科学来源保持4215c066与CPU06一致，CPU原失败及旧源码继续保管；CPU101+准备34=辅助135秒。原准备门保留送审字节；[当前有效关闭门](../reports/documentation/20261102/history_optuna/review_round2/formal_gate.json)承接第二轮审查、既有预算/输入/135秒费用，仅review_closed为true，执行仍关闭，无新监听或正式模型产物。

首轮具体实现外审包514462字节、132载荷（另含清单），SHA256 39952612d10e561014d0de8586fbf9bff55ba98396d17616976e28fcfafc1118，已获用户本包/目的地许可并于客户端22:28:20单次上传GPT-6 Pro。输入ZIP、清单、完整请求、状态勘误和[提交回执](../reports/documentation/20261102/history_optuna/review/submission.json)保留；[完整返回包与六载荷核验](../reports/documentation/20261102/history_optuna/review/recovery.json)已回收，原文保留。修后由[独立主审](../reports/documentation/20261102/history_optuna/review/primary_disposition.zh.txt)逐项关闭，CPU06四项通过和101秒累计账另存repair/；原送审132载荷及CPU05未追改。送审implementation原文保留形成时点，不追改字节；首轮回收处置由原提交索引承接，本研究线当前状态见交接。本轮项目记录在Windows维护，实际同步范围及最终核验见[修后同步回执](../reports/documentation/20261102/history_optuna/repair/final_sync.json)与[独立处置](../reports/documentation/20261102/history_optuna/review/primary_disposition.zh.txt)；不把独立源码SFTP同步写成Git推送或全部根docs同步。本地提交、不推送，不重置既有大清理周期。 用户另要求修后版本继续外审，第二轮447812字节、105载荷包及修前/修后差异、CPU06、第一轮原件、准备记录在[review_round2](../reports/documentation/20261102/history_optuna/review_round2/input_manifest.json)；源码/手写证据/必要元数据之外不含正式正文、标签、owners、模型、Memory或密码。新包SHA256=4890b5620eed60c34e8d3c67a503071d14a51fd5fb0d97d07922924e4c94df9a，已按新指令单次送审，原包不改。11月3日[完整回收第二轮](../reports/documentation/20261102/history_optuna/review_round2/recovery.json)：33499字节、7成员（6载荷+清单），SHA256=ef759a1009c0fbe0ffd807810835aa16f906928d52cf2624bd98dca952000e7e；全部大小/SHA核验通过。原报告、逐项意见、阅读范围和页面原件已保管，[主执行者逐项处置](../reports/documentation/20261102/history_optuna/review_round2/primary_disposition.zh.txt)接受R01/R02关闭，无新增必修或测试；新关闭门与[结案同步回执](../reports/documentation/20261102/history_optuna/review_round2/final_sync.json)另存，不追改已送审配置和旧费用。最新SFTP范围见[准备同步回执](../reports/documentation/20261102/history_optuna/preparation/final_sync.json)，根docs仅Windows维护。

## 2026-11-02 H/S方案A评价、结果回传、外审保管与默认权重清理

用户确认方案A后，在Linux独立 reports/seller_alias_continual/20261102/defaults_result 部署CPU评价，未改活动训练来源。原20份送审源码/测试在source/，最终22份执行来源由[修后清单](../reports/seller_alias_continual/20261102/defaults_result/repair/input_manifest.json)绑定；三份修后增量及对应CPU03/04、preflight02原件在repair/，CPU01/02与旧清单/测试版本仍可按原SHA恢复。36完成记录、原门、dataset manifest等元数据在inputs/；原40项回传见metadata_receive.json，后续dataset manifest仅元数据，无标签正文。

一次评价形成324份群指标、324份query指标、324份分数及完整分析/计数/日志。[982份载荷104222515字节](../reports/seller_alias_continual/20261102/defaults_result/receive_manifest.json)通过SFTP回传，[实际接收](../reports/seller_alias_continual/20261102/defaults_result/receive.json)逐项大小/SHA全匹配。分数/query二进制仅项目保管，不进入外审包；原关系表、商品正文、owners、模型、Memory均未回传。评价标签一次、模型/GPU零，旧tuned162矩阵在固定容差内逐项一致、N/A对应一致、混淆计数完全相同；新增CPU预算与原训练58.857975h分账。

原179063字节实现包已完整回收，A-R01/A-R02按[独立处置](../reports/seller_alias_continual/20261102/defaults_result/review/primary_disposition.zh.txt)最小修复关闭。38项送审字节与独立保管原件核对后删除重复ZIP；提交时190秒预算与现账不同，原字节单独保留review/submitted_budget.json并登记在submission.json。原文/DOM/回收身份继续保留，重复工具JSON与未采用的proposed输入清单删除；结果中转ZIP在全部载荷核验后清理，见[本地清理](../reports/seller_alias_continual/20261102/defaults_result/local_transport_cleanup.json)。Linux亦重新核对982份保留载荷后删除同一中转ZIP和未采用清单，见[远端清理](../reports/seller_alias_continual/20261102/defaults_result/remote_transport_cleanup.json)。上述中转材料清理阶段没有删除权重或唯一科研证据；后续默认权重按本节下述新确认处理。

新[结果包](../reports/seller_alias_continual/20261102/defaults_result/result_review/package.json)5964975字节、431载荷，用户具体批准后客户端20:06上传原ChatGPT会话并单次提交。包含报告、全22指标、324群指标JSON、36完成记录和必要实际来源/修复证据；不含分数/query二进制、标签源、正文、owners、权重或密码。客户端20:18:36完整回收[结果外审](../reports/seller_alias_continual/20261102/defaults_result/result_review/external/report.zh.txt)，[逐项主审](../reports/seller_alias_continual/20261102/defaults_result/result_review/primary_disposition.zh.txt)结案，无新增计算修复；仅精确旧162矩阵的容差核对措辞。431载荷逐项与独立保管字节或Git原文核对后，删除重复结果ZIP和工具包装JSON；送审316秒预算另存result_review/submitted_budget.json，送审报告可由d41e709c恢复，路径/身份登记在submission.json。修后报告、完整外审及处置8项已[同步Linux并核对SHA](../reports/seller_alias_continual/20261102/defaults_result/final_sync.json)。最终新增350/7200秒，含准备、评价、核对、主动传输和保管；专用网页已关闭。实时进度仅在交接H/S行维护。根可变文档只Windows维护，本轮本地提交，不推送；既有FOREVER→VendorLink自动链继续；后续权重保管按以下具体确认更新，不重置大清理周期。

用户随后明确确认“删除这54份默认权重并完成记录”。[删除前清单](../reports/seller_alias_continual/20261102/defaults_result/cleanup_inventory.json)仅覆盖Linux reports/documentation/20261031/comparison_continue/defaults/formal 下18条默认S/H运行的 run/models/stage1—3.pt：54个独立普通文件、70436052048字节（65.598685GiB），逐份实际路径、属主、大小、SHA及原完成记录一致，nlink均为1，无符号链接或共享底层文件。既有链已通过H/S前序完整门并启动FOREVER；实际入口后续加载的是旧H_pos_tuned的27模型，不加载这些默认模型。默认训练已退出、评价及结果审查结案，无后续载模、诊断或恢复依赖，才执行本次删除。旧S_tuned/H_pos_tuned的54份基线、预训练、活动FOREVER和VendorLink资产不在删除范围。

服务器20:50:08—20:50:17执行逐文件unlink；[实际回执](../reports/seller_alias_continual/20261102/defaults_result/cleanup_receipt.json)确认54目标全部消失、目标根下剩余pt为0，共删除70436052048唯一逻辑字节。799份默认训练非权重材料未变：745份按SHA核对，54份Memory仅核对元数据，未读取载荷；54份旧基线模型元数据、982份已保存结果SHA、5份直接依赖源码SHA均保持。链3485029及FOREVER三worker的属主/创建时刻/命令/cwd在删除前后相同，未新增训练、评价、标签读取、载模或GPU操作。共享磁盘空闲从192325410816增至266675261440字节；该净变化同时受其他活动影响，不作为本任务精确物理释放量。

清单611427字节（SHA256 b4939174bb56a1d1540ba6ed9b9b0323118b17cd006734348f9bb495cc905fd9）及回执24101字节（SHA256 b56f40d891293dbc8935c6415108674ef1b5cea5b20140c0ac728179f0dc7a44）已SFTP回传并与Linux原字节一致，双端保管于同一相对结果目录。盘点56.578秒、删除及前后核对8.733秒作为文件保管用时记录，方案A原350/7200秒及历史训练失败账不追改。此具体决定替代默认模型的原保留安排；冻结方案、原分析和外审字节保留，SHA不能恢复已删模型。当前交接原位替换、旧项进入已有历史区，本次仅本地提交，不推送，不重置大清理周期。

## 2026-11-01 L²R／ER+GCR结果回传、审查及权重清理

[结果目录](../reports/seller_alias_continual/20261101/comparison_result/)通过SFTP回传2393文件、162211873字节；[逐项清单](../reports/seller_alias_continual/20261101/comparison_result/receive_manifest.json)与[接收回执](../reports/seller_alias_continual/20261101/comparison_result/receive.json)记录两端实际大小/SHA全匹配。包括正式分数/指标/日志/校准/来源及完成账，不含正式正文、关系标签、owners、Memory载荷或模型。另回收核对7项其他实际冻结源码版本；不会用当前源码冒充全部旧轨迹来源。本轮保存结果核对在Linux py310单数值线程进行，无标签/载模/重采样，外层计2秒旁记，不覆盖原正式账。

结果外审输入包11681119字节、1097成员（1096载荷+清单），SHA256 `e5a6ee7a5243104e1f25fcbfc94af23f0e1ed9bb442f2fffd7b474ec27ac5c72`，逐成员核对一致。包含完整结果分析及必要静态来源/证据，不含分数二进制或正式私有载荷；最初自动审批拒绝后已获用户具体包/目的地许可，客户端19:36单次上传并发送、19:47[完整回收](../reports/seller_alias_continual/20261101/comparison_result/review/external/recovery.json)，[外审原文](../reports/seller_alias_continual/20261101/comparison_result/review/external/report.zh.txt)及[独立处置](../reports/seller_alias_continual/20261101/comparison_result/review/primary_disposition.zh.txt)均已保管，无新增必修。全部1096载荷与独立保留文件逐项核对大小/SHA和ZIP字节一致后，已删除本地重复上传包，保留输入清单、请求、原包身份与完整审查原件；专用网页已关闭。原拒绝、新许可及当前处置见[提交索引](../reports/seller_alias_continual/20261101/comparison_result/review/submission.json)。

[清理预盘点](../reports/seller_alias_continual/20261101/comparison_result/cleanup_inventory.json)在四个本批已结束campaign定位157权重路径、58个底层文件、78184901449唯一逻辑字节（约72.82GiB）；逐文件实际SHA/最终路径/uid/inode/nlink已核对，全部硬链接均在这四目录。含54个新有效端点、3个旧失败阶段端点、1个旧校准诊断完整状态。3093份非权重记录继续保留，其中封存生成/Memory的162项只stat，另2931项按字节SHA核对；54份旧H/S模型元数据列为保留例外，H27仍为FOREVER依赖。删除前重新核对157路径及58份内容SHA/身份、完成退出状态和保留对象，服务器20:02:45—20:02:49逐文件unlink完成；[实际回执](../reports/seller_alias_continual/20261101/comparison_result/cleanup_receipt.json)确认157路径全不存在，3093非权重材料按原2931项SHA/162项元数据口径未变，54旧H/S模型元数据未变。已删除78184901449唯一逻辑字节（72.8154GiB）；共享磁盘空闲由200940343296增至279125569536字节，净变化78185226240字节，不能把共享磁盘净变化当成本任务精确物理释放量。Linux原件位于reports/maintenance/20261101/comparison_weights/{inventory,receipt}.json；预训练及其他活动模型不在四个删除根目录。SHA无法恢复已删除模型，原失败和有效正负结果保留。

首次盘点错误地尝试散列封存owners，OS权限拒绝、未读取该文件；未改权限或绕过。见[错误与收窄处置](../reports/seller_alias_continual/20261101/comparison_result/custody_attempt.zh.txt)。第二次仅对授权权重/非封存结果散列，对正式正文/关系/owners及Memory只保留元数据，盘点完成。原始运行、有效正负结果和失败账不变。根可变文档只Windows维护；本轮只本地提交，不推送，不重置大清理周期。
## 2026-11-01 进度只读核对与本地记录交付

按本次明确确认，仅通过SSH读取既有作业元数据、实际代码身份及train-calibration标量状态，在Windows保存[progress_20261101_122209.json](../reports/documentation/20261101/formal_chain/progress_20261101_122209.json)：72658字节，SHA256 `a9ddaaff67c2e2df9a122f26917a65b497676ddce87c7e1cd99b7fbd352fde80`，返回原字节与Linux输出摘要一致。文件记录服务器UTC+8观测时点、输入元数据身份、进程归属/创建时刻/命令/cwd、实际来源核对、运行/等待账、校准状态及工期算术；不含模型、正式正文、分数矩阵、关系标签或新确认结果。该证据只保存在Windows，不宣称已同步Linux。

比较控制器47科学+8编排、H/S44科学+57编排、FOREVER/Vendor共用65来源均与各自正式门一致；修复CAB的47科学来源也一致。旧comparison_finish编排清单中的一次性launcher已按[首步原件](../reports/documentation/20261031/comparison_finish/formal/first_update.json)自删，实查缺少该文件属于已发生的正常自删，不重建、不记成科学来源损坏。旧comparison_finish预算快照已被接管，费用与活动reservation只按comparison_continue现账解释，不能再次相加。

工具自动审批拒绝了最初采集命令末尾在Linux新建进度JSON的动作，理由是本次确认仅允许远端只读；该混合读写命令未执行。随后采用无远端写入的采集并将输出保存在本地，未绕过拒绝或修改活动作业。两份本轮本地一次性采集/接收助手在证据保管后删除，交互SSH与接收连接均退出；没有新服务器脚本、监听、训练或清理操作，不重置大清理周期。根交接/计划/讨论/存储只更新Windows并本地提交，本次不推送。

## 2026-11-01 三槽FOREVER与完整顺序等待链

[formal_chain](../reports/documentation/20261101/formal_chain/report.zh.txt)在Linux独立目录继承原冻结63来源，仅替换queue/forever_formal并新增wait/test，共65项；实际comparison冻结字节保持。六个链接仅BGE及原五份获准train输入，未复制或下载正式文本、关系标签、owners或模型。正式门新建为forever/formal_gate.json与vendorlink/formal_gate.json，旧候选/repair gate及旧批次原件保留；Vendor精确引用最终FOREVER门，不在线覆盖原CAB/H-S来源或作业。

两轮完整审查输入ZIP、原文/audit、请求/提交/菜单观察、主审及修前来源保管在review；首轮CHAIN-R01 OPEN与修后CLOSED分别保留。四组CPU原始receipt/console/evidence及NVML只读记录均回传核对；CPU累计FOREVER141/600、Vendor132/600，native1441/3600及失败账保持。准备181/900和145/3900为明确保守结转，不将网页等待计为计算。

9份部署/启动/观测/最终门/清理小证据经31639字节ZIP回传，SHA256 865388ac11a17d05059ee79f964445457350b16dde112aeed420c0e4e336e1fb；逐项身份见[接收记录](../reports/documentation/20261101/formal_chain/receive.json)。服务器11:53:38启动唯一新链3485029，11:53:52实查仅等待H/S，无新训练子进程。active_formal_wait.py只在首个真实FOREVER更新后自删，workspace源文件、持久state及首步/进程回执保留，不能据此再建第二份监听。

本轮纯中转包、incoming重复副本和新workspace生成pycache在核对必要载荷后清理，完整审查输入包和全部必要原件继续保管。机械清理的manifest落盘/格式核对中断及后续正确处置见[Linux清理记录](../reports/documentation/20261101/formal_chain/transport_cleanup_linux.json)；未重启科研计算或删除科学资产。当前交接每线原位更新，旧上游状态移历史；根可变文档仅Windows维护，独立实施/审查说明同步Linux。本轮本地提交，不扩大旧特定push许可，不重置大清理周期。

## 2026-11-01 FOREVER单worker显存实测与测试权重清理

用户明确确认单worker、最大形状手写输入的限定显存试跑，并要求“测试产生的权重文件要删掉”；首轮测量包装单位错误后，又单独批准一次237秒内补测。两次实际来源/启动/测量、失败原件、单位修复CPU证据、费用与用户决定保存在[forever_memory](../reports/documentation/20261101/forever_memory/report.zh.txt)。Windows scripts/tests保留修后入口和CPU检查；原未提交的失败源码另保存在该目录source下，修后字节可按本次Git版本和回执身份恢复，不复制完整科学workspace。

在确认测试父/子进程均退出后，先保存本次state文件路径/大小/SHA，再于服务器10:50:29完成删除：完整模型/优化器3913191053字节、锚点1304330258字节、手写Memory765307字节及latest元数据99779字节，共4文件5218386397字节（约4.86GiB）。首轮落盘前失败，没有产生权重，只清理空state目录。两个state目录均不存在，本任务.pt/.bin/.safetensors残留为0；[删除前身份](../reports/documentation/20261101/forever_memory/measurement_repair/cleanup_before.json)及[删除后回执](../reports/documentation/20261101/forever_memory/measurement_repair/cleanup.json)保留。原BGE、已有训练资产及CAB/H-S作业未动，不重置项目大清理周期。

31份小证据通过31495字节ZIP回传，SHA256为0c4017ec1d74905c2cc2387923302c6ecdfa8f057785292f8d0e8aa4606b542f，逐项一致见[接收](../reports/documentation/20261101/forever_memory/receive.json)；没有测试权重或正式数据回Windows。两端中转ZIP在身份核对后删除，必要原始证据留存。累计native1441/3600秒（含新63秒失败及104秒成功）、CPU134/600、准备61/900；旧冻结gate不覆盖，后续正式门需承接增量。交接仅替换相应当前项，原状态移入历史；本地Git交付，不扩展上次特定批次push许可。

## 2026-10-31 成功槽接续纠正与原运行接管

[comparison_continue](../reports/documentation/20261031/comparison_continue/report.zh.txt)两端保管纠正范围、47科学＋8编排来源、正式门、原预算/进程身份快照、接管就绪/激活、切换和11:54:20现场原件。只更换两个派发父控制器和唯一H/S等待进程，三条训练PID/创建时间/CPU核/原启动时刻/截止时间保持；原科学workspace、旧completion及失败不覆盖。新workspace仅补既有BGE链接供原确认入口使用，GCR仍从原workspace启动。原comparison_finish单条队列是执行偏差，原记录按历史保留，不追改成正确执行。

CPU五次运行的全部输出/错误/收费、必要旧源码及一次未执行入口错误保留：本次47秒、原106累计153/1200。最终四项及未变共享两项支持当前行为，旧三次错误不隐藏；cpu01/02/03保留对应源码，cpu04复用cpu03版本。小证据ZIP共67874字节、SHA256 83d66bc1af2c80a82d931057ef91a5a8f272ca7213c56c27b57d2f1a831e1864，28份原件按[逐成员清单](../reports/documentation/20261031/comparison_continue/transfer_manifest.json)核对回传；未回传模型/Memory/分数/标签。

本轮input/fix/identity/close/evidence五个中转ZIP的42个成员已逐项与保留文件按SHA核对，两端五包在确认精确路径和身份后删除。所有必要源码版本、原始证据及活动载荷保留，不重置大清理周期。执行报告6372字节、SHA256 5fbce65bd70371a440a199b6e9c66c3b395a163efc52d6c6a28bf2a4c8c6ef60已同步独立目录；根可变交接/设计/计划/讨论/存储仅Windows维护，本轮本地Git交付，不推送。当前交接只替换L²R/H-S两项，旧11:07状态移入历史并明确错误归属。

## 2026-10-31 修复CAB_s6单路启动与H/S重新绑定

[comparison_finish](../reports/documentation/20261031/comparison_finish/report.zh.txt)在两端保管本次明确授权、47科学＋7编排来源身份、4项手写CPU证据、正式门、部署、启动、11:07:57现场及H/S切换原件。Linux独立workspace从已冻结比较／H-S来源复制，只链接原五份train输入与BGE；新CAB从原起点完整补跑，未载入旧诊断模型作训练起点。旧comparison_resume及defaults_repair科学源码/gate/失败/费用均未覆盖；新H/S继续使用原defaults_repair科学workspace，只重新绑定新完整前序。

11项小证据经27832字节ZIP回传，两端SHA256 `87733034950a77c36b83835ae8fc8c0e530312d9137ca93f57a32b3abffd7f8f`及[逐成员清单](../reports/documentation/20261031/comparison_finish/transfer_manifest.json)全部一致。没有模型、Memory、分数或标签回Windows。新launcher在实际首步后按约自删，保留Windows scripts源码、CPU来源身份和first_update；旧H/S等待进程在身份/训练0/无children核对后结束，保留旧记录及全部费用。两端本轮input/evidence中转包核验后删除；必要源码及原始小证据保留，不动活动载荷，不重置大清理周期。准备阶段遇旧launcher已自删的复制错误及修正见report，未误执行旧代码。

当前交接两行原位替换、被替代的10:56—10:59状态移入历史；讨论/设计/存储入口在Windows维护，不同步覆盖Linux根可变文档。仅同步本轮独立目录的范围/核验与执行说明及必要来源，小证据按原件回传；本轮本地Git交付，不推送。

## 2026-10-31 CAB_s6诊断重放现场

通过SSH只读核对已授权comparison_resume作业及defaults_repair监听，读取既有train-calibration拟合JSON和运行元数据，未加载模型/Memory/分数数组或正文/标签。新增[现场原件](../reports/documentation/20261030/comparison_resume/progress_20261031_100640.json)两端保管，32896字节／SHA256 f1408c2c451eeb66614ed9b935127a71f167406db2da8c6ea17b9382320d51da，SCP回传核对一致；其stage_points采集路径错误及不使用范围已在[诊断说明](../reports/documentation/20261030/comparison_resume/report.zh.txt)明确。校准前3898758217字节完整状态和原分数继续留Linux供后续诊断，未下载、加载或删除。原失败、预算、冻结入口及运行状态文件不改；新说明与交接仅Windows维护，本轮仅本地Git，不新增推送或正式执行。SSH/SCP已退出，无临时文件或新监听，不重置大清理周期。

后续用户明确批准一次原失败输入的CPU修复核验。[calibration_repair](../reports/documentation/20261031/calibration_repair/report.zh.txt)保管独立数值几何诊断、一次train CSV解析的访问记录、单次ftol=0拟合及标量梯度核对、9项手写检查和49来源隔离部署；额外6秒CPU在此单列，后续恢复须与原98秒合并为104秒，不在线覆盖活动账。诊断加载了Linux既有分数，未回传矩阵/标签/模型；7份新增小证据经10714字节ZIP回传核对，5个部署增量/测试依赖与本机SHA一致。修复说明随后同步到Linux独立目录，根交接/讨论/存储/设计入口只Windows更新；原冻结workspace/gate/失败均未覆盖。

修复说明6797字节已回读SHA一致：2cf5740ddfb95d2b97ffbd91fa3b51a8479da5607d1dbbf9a238772f70c46c61。两端本轮input.zip/evidence.zip已在源码及7项原件保管、身份核对后逐文件删除，必要来源和结果保留；SSH/SCP全部退出。未改活动来源或创建新训练／监听，不重置大清理周期。

## 2026-10-30 VendorLink与FOREVER正式准备、修复与外审证据

开训前处理量留证增量：VendorLink cpu10四份原始小证据已SCP回传并逐项核对SHA，三阶段处理量、恢复继承及删除latest后保留行为均有手写证据。四份受影响源码版本追加到原source_archive.zip，原有成员保留；当前归档430947字节／SHA256 b6ef7ced3ba493545ed8cadbe763de7b3db6fde70589bf45a34df953549bf516，下文410142字节为初始归档身份。本次两gate继续关闭，仅重绑共用63来源并结转VendorLink CPU130秒，FOREVER133秒不变；实际部署见cpu10/deployment.json。实现说明与源码归档同步到Linux formal目录，根可变文档只在Windows维护。中转材料按本次用途清理，不触及旧回执中另有处置记录的远端包，不重置大清理周期。

两项正式准备分别保存在[VendorLink formal](../reports/documentation/20261030/vendorlink/formal/)与[FOREVER formal](../reports/documentation/20261030/forever/formal/)。Linux共用隔离workspace位于vendorlink/formal/workspace，63份实际源码/手写测试/必要配置已核对；comparison_run/evaluate及其测试采用原资格冻结字节，不覆盖活动实验来源。FOREVER只链接五份已批准train输入与BGE模型目录，不链接整个旧dev/test/owners目录；H27模型仍在原保管位置。两个具体gate均关闭，无正式训练或新确认。

CPU全部尝试、错误、实际收费及获准canonical依赖元数据已回Windows。小证据ZIP先核对两端SHA，再逐成员核对解包文件；84个必要源码版本按SHA/原相对路径归并为source_archive.zip（410142字节，SHA256 5218347ca63bb8049575e48c0c465ae5c4f764dd8db6a1fada34eb285ea015de），所有CPU/扫描receipt.sources均可恢复。旧失败证据未删除。Windows自建中转ZIP及展开副本在必要载荷保管/核对后清理，见[清理记录](../reports/documentation/20261030/vendorlink/formal/transport_cleanup.json)。Linux同批10个中转ZIP的删除被自动审批以缺少明确远端删除许可拒绝，实际保留，不绕过、不影响科研工作；未重置大清理周期。

11份正式方案/实现说明/许可/关闭gate/源码归档/外审记录已SFTP同步到两项独立formal目录；[部署回执](../reports/documentation/20261030/vendorlink/formal/deployment.json)记服务器19:46:14、63份来源一致、无本批活动进程、六个精确链接。该次准备实际向上取整1秒，记FOREVER preparation，启用gate时须与原CPU125/GPU1274一并承接；VendorLink preparation已用25秒、CPU103秒。当前待审gate保留其提交时预算身份，启用前只结转实际费用，不清零。

具体外审包926333字节/125载荷、SHA256 8dee9a952919a8215a1e83058e541db6a933f867548e91655a46ba951acbe731获用户明确批准后，19:36单次提交既有ChatGPT GPT-6 Pro会话。保留待审原包、完整清单、请求、具体许可、提交和页面回执；无正式商品文本/账号名/关系标签/owners/权重/密码。完整报告report.zh.txt、audit_notes.json与主审处置已回收保管，原包用于复核本轮实际输入及固定清单。原审查要求两项定向修复；修后经主审闭合，未冒称再次网页送审。cpu08/04及repair保存新增源码与原始证据，CPU账分别110/133秒；hold_observation保存两gate关闭/各63来源一致/无本批进程的只读现场。用户要求等待明确开始，不创建自动队列。根可变文档本轮只在Windows维护；本地Git交付不等于push或Linux根文档更新。

第二轮与第三轮完整原文、audit及逐项主审分别保存在formal/review/recheck与closure；原提交身份、旧报告与修前来源保留，最新处置只看交接。cpu09新增一行VendorLink生产修复及手写反例，原始receipt/console/evidence与修后源码已保管；两个中转input.zip（15486字节）和evidence.zip（8837字节）逐成员与保留文件核对一致后在Windows删除，Linux中转不在本次删除范围。保留受审源码包用于已批准正式方案在用户开始指令后的实际来源核对，不复制模型或标签。旧部署记录按其时点解释，本轮收尾材料的实际同步另见closure/deployment.json；根文档仍只在Windows维护。未重置大清理周期。修复收尾时仅本地提交；用户随后明确要求全部本地修改提交远端，本次据此交付现有科研分支上的全部未推送提交及最新文档，不把此前仅本地范围继续当作阻断。Git交付不等于再次同步Linux根文档。

## 2026-10-30 H/S失败隔离监听切换

[defaults_repair](../reports/documentation/20261030/defaults_repair/)保存用户范围、独立主审、CPU01/02、来源、正式门及switch/launch/observation。Linux新workspace从旧57份编排来源复制并覆盖H/S控制/测试及已核验共享控制文件，另保留共享测试，共58份资格来源；44科学来源不变，沿用原7个模型/数据链接（含已获准旧dev无标签items），没有解析正式数据。旧comparison_resume/defaults监听在无训练/子进程时结束，原gate/workspace/预算/记录保留；新监听唯一，不改活动L²R。

CPU首轮11通过和一个测试名加载错误（收费6秒）保留，随后仅补漏项通过（3秒），累计68/120秒；全部旧等待/失败/原生费用继续承接。小证据SCP回Windows并按清单逐成员核对，报告/范围/执行说明两端保管。中转input/evidence ZIP核验后两端删除；活动数据/模型/Memory不动，不重置大清理周期。根可变文档仅Windows维护，本地范围提交，不推送。

## 2026-10-30 L²R恢复及S/H交接重新绑定
20:14—20:15只读查询的[进度原件](../reports/documentation/20261030/comparison_resume/progress_20261030_201431.json)（32164字节，SHA256 f5feccdbf0fc0e958db7e33ba9d95ff7a0112d7f401dcd43a02b0eb4da672426）与[工期及等待状态](../reports/documentation/20261030/comparison_resume/eta_20261030_201431.json)（6331字节，SHA256 5f02068eb86eeae1a34317b3e01eca92cfcd52704b6cc22b8ea08216988ec1e3）已SFTP回Windows并核对两端大小/SHA，原件保留；动态计数只在交接顶部维护。本次仅精简可变记录，删除已被取代的状态文字及重复旧工期，不删除冻结证据或活动载荷。根文档未同步覆盖Linux活动workspace。


[comparison_resume](../reports/documentation/20261030/comparison_resume/)为独立恢复目录；Linux workspace50份代码/测试/合同政策核验，沿用原五份训练输入及BGE链接，无dev/test/owners/private_custody链接。14成功及三条旧失败目录用硬链接保管、纳入原存储额度；原formal/gate/budget/failure未覆盖。诊断失败时保留校准前完整状态，成功阶段端点双恢复通过后退役该临时完整状态；不清理活动权重或Memory。

CPU15项原件、来源、新正式门/启动/15:19运行快照和S/H重新绑定小证据通过SCP回传并按transfer_manifest逐项大小/SHA核对。启动后本轮comparison launcher已按首个真实更新自删，源码保留Windows scripts及身份清单。S/H首次导入失败的gate/launch/console保存在defaults的*.import_failure.*，失败记录中的当时原路径按该改名恢复查找；无formal/数据访问，保守5秒计费后修正原PYTHONPATH环境，唯一新监听继续等待。

本轮新增CPU44秒，比较批CPU累计98/1200；旧训练291.508706h和所有失败费继承。新恢复说明/实际执行记录在两端保管，根可变文档仅Windows更新。中转输入/回传ZIP核验后两端删除；旧必要原件、来源与活动运行保留，不重置大清理周期。仅作范围清楚的本地Git提交，不推送，保留其他任务已有修改。

## 2026-10-30 FOREVER实现外审与两项单路GPU资格收尾

完整FOREVER实现审查原文、audit_notes、完成网页及主审保存在[review目录](../reports/documentation/20261030/forever/implementation/review/)；输入包身份400960字节/SHA 7db965e4…6013及91成员/90载荷核对保留，专用页已关闭。CPU05—10、gpu01/02失败和gpu03单路PASS原始console/调用/更新/资源/receipt及各次必要旧来源保存在[资格目录](../reports/documentation/20261030/forever/implementation/qualification/)，最终小证据包170531字节/SHA 4e0c0d5bf1c5538eb840061a8e4d750a07560da5f52464b1e9842acde3dac7e2经SCP回传。Windows不保存BGE或GPU检查点；Linux成功资格状态约5GiB暂留用于本次恢复证据，未作为正式训练起点。原冻结CPU04/送审来源不改，当前独立workspace仅绑定本轮改动与原依赖，不混入并发L²R修复。

[VendorLink单路证据](../reports/documentation/20261030/vendorlink/native/received/)保留两次G1账及最终单路结案/G2未通过事实，包24628字节/SHA 49999a1b…bb8两端核对。原固定导出/标签/模型/Memory留Linux，没有回Windows；旧29日资格及失败原件不覆盖。

用户截图指定的Linux家目录五份中转文件已逐一核对大小/SHA和正式来源保管后删除，合计116780字节，见FOREVER qualification/final/transfer_cleanup.json；当前运行与正式工作目录未动。其他本轮用完的传输ZIP、网页部分输出和重复副本在必要原文/证据保管后清理，不重置大清理日期。仅同步本轮隔离目录的主审/范围补充，不覆盖活动冻结文件；根交接/设计/讨论/本记录只Windows维护。本轮仅范围明确的本地commit，不推送，用户汇报稿改动、另一任务的中断记录和历史删除不纳入。

## 2026-10-30 L²R校准中断与控制器修复

[本轮排查/修复证据](../reports/documentation/20261030/comparison_repair/)在Windows及同相对路径Linux保管，Linux修复来源仅在该目录workspace，未覆盖10月26日正式冻结workspace/gate/budget/failure。49份依赖按sources.json绑定；不相关的本地five_run变更未带入，改用原冻结来源。CPU原始console/receipt和13:47只读现场通过SCP回传，实际大小/SHA核对；CPU新增9秒、累计54/1200秒，原训练失败账保留。没有正式数据/标签/模型/Memory回传，没有诊断重放或正式恢复。临时输入ZIP核验后两端删除，原失败及必要源码证据保留；本轮不重置大清理周期。

Windows代码/测试及可变交接/讨论/本记录更新；Linux只同步本轮独立修复目录及说明，未同步根可变文档。按范围作本地Git提交，不推送，其他任务已有未提交修改及历史删除保持。

## 2026-10-30 FOREVER限定实现与CPU证据

新代码及[实施范围/许可/报告](../reports/documentation/20261030/forever/implementation/)在Windows维护，Linux隔离部署于`/home/yongpeng/cross-lingual/reports/documentation/20261030/forever/implementation/workspace`。最终44项实际源码/必要政策/测试核对一致，只有既有BGE目录链接，没有正式数据链接。前期打包100份无关政策副本已从新workspace及暂存中移除；原仓库及旧活动冻结文件未动。

CPU各次原始console/receipt/evidence、旧版四文件/permit及实际部署/资源回执保存在[资格目录](../reports/documentation/20261030/forever/implementation/qualification/)，通过SCP回传并核对传输包两端实际大小/SHA。资格只用手写输入，临时小模型/Memory在测试结束即由临时目录释放；没有GPU载荷或正式数据/标签/模型回传。用户另行明确批准400960字节新完整实施包后，已于13:05在原FOREVER会话单次提交；唯一授权/提交/完整请求/页面观察见implementation/review，原方案包许可未被泛化。在审新ZIP仍保留，已核对的CPU传输ZIP和打包暂存已清理，原始资格证据及实际旧来源保留。

本轮只同步新FOREVER隔离来源/证据，不更新或覆盖服务器已有活动实验的文档和冻结文件；根交接/设计/讨论/存储文档只Windows更新。本地Git交付排除用户已有汇报稿修改、历史表删除和其他未跟踪材料，不自动扩大先前具体批次push许可。临时传输副本在核对后清理，不重置15天大清理周期。

## 2026-10-30 FOREVER静态方案外审材料

[本轮方案、预算与审查目录](../reports/documentation/20261030/forever/)仅保存在Windows。三份官方源码是固定提交bbeb549a21e59d9490f20a2b02099ee766a67f48的静态阅读原件，没有执行；方案包159582字节、26成员，身份及25载荷大小/SHA核对见[清单](../reports/documentation/20261030/forever/review/input_manifest.json)和[包记录](../reports/documentation/20261030/forever/review/package.json)。用户明确批准该完整包后上传到本轮chatgpt.com专用会话，提交/授权/完整请求及页面观察保留。没有正式文本/标签载荷、模型、Memory或凭据进入包中。

10月30日完整回收后保存report.zh.txt（47114字节，SHA256 b5a00a7be8e145bca216875056b3311da6ddb9af893fcff0d5d8a7e37c4b575c）、audit_notes.json、完整网页观察、[回收身份](../reports/documentation/20261030/forever/review/recovery.json)、主审及revision.zh.txt。上传包25个载荷与现存来源全部同大小/SHA，已由原提交a728ddd2及原项目Git保存；本轮关闭方案处置后删除无独立后续用途的重复ZIP和首次不完整DOM，保留完整外审原文与必要来源。未复制出另一套可执行workspace。

本轮补入同事汇报稿并作范围清楚的本地Git交付，保留用户已有修改。未同步Linux、推送远端、覆盖活动冻结文件或重置大清理周期；其他历史删除及未跟踪文件不纳入。本节中的96GiB及恢复文件清理规则是未来FOREVER提议，不是对现有实验文件的清理许可。

## 2026-10-30 VendorLink GPU校验进度查询

仅通过SSH读取既有监听、资格账、进程、GPU/RAM/disk及七执行来源身份，未读取正式文本、标签、权重或效果指标。新增[现场原件](../reports/documentation/20261029/vendorlink/native/waiting/observation_20261030.json)在Linux/Windows两端保管，4750字节，SHA256 `ca0a2c29a272efe674d652b0003ef4454fdd996c2db8e45559dd9e11dbc41575`；SCP回传后与Linux实际大小/SHA一致。旧观察、配置、许可、启动及失败账不覆盖；无新监听或科研计算，SSH/SCP会话已关闭。根交接与本记录仅Windows更新，本轮只本地Git，未同步Linux可变文档或推送；不重置大清理周期。

同日等待窗口排查另保留[原始运行证据集合](../reports/documentation/20261029/vendorlink/native/waiting/wait_audit_20261030.json)，15333字节、SHA256 `59a4cf38fd192e04c87d3d7067811136a98bd1db4b19d6b18ff204130f67d3e4`，SCP回传并核对Linux/Windows一致。含原补位准入、派发、当前监听/账、来源身份及日志存在性；未包含正式载荷或他人实验文件。旧原件未覆盖，SSH/SCP已关闭；本轮仅本地提交，未改运行或推送。

## 2026-10-29 晚间接手核对与进度补齐

复用已有一次性只读运行观测器，在Linux既有py310采集作业元数据和工期；不调用科研实现、载模、Memory载荷、正式正文或评价标签。两份新原件经SCP回传并与Linux实际大小/SHA256核对：[新对照及原版等待现场](../reports/documentation/20261026/continual_comparison/qualification/status_20261029_evening.json)224416字节／`82248ab47333a6d43c082842739d6b2074e818907411c304320cab40cadb955e`；[VendorLink等待现场](../reports/documentation/20261029/vendorlink/native/waiting/observation_evening.json)7356字节／`0d6a545a952852d9d494902d6642ffa3c37a334b90a854ae277bb55d1917102a`。保留旧观测，不覆盖原授权、配置、门、启动或失败记录。临时Linux采集脚本已删除；Windows中转文件移至各自已有qualification／waiting目录，无新增监听或状态系统。

第一次采集包装因exec共用命名空间中的变量被观测器覆盖，在写出前报AttributeError；未形成文件或触及训练。隔离执行命名空间后重新只读采集，以上原件来自成功的第二次观察，不把第一次写成成功。此次不是科研执行／预算失败，原各实验账不改。交接、计划、讨论、存储四份可变文档只更新Windows；此次两份现场证据两端保管不等于总文档已同步Linux，也不等于Git已推送。原有删除及未跟踪材料保留，不重置大清理周期。

## 2026-10-29 VendorLink导出收尾及10GiB等待

本轮固定导出已在Linux `native/qualification/export` 和 `private_custody` 形成18载荷，共8454714字节；以[机械复用核对](../reports/documentation/20261029/vendorlink/native/waiting/export_reuse.json)为准，Windows未保存正式文本或qrels，仅保存聚合计数/哈希。原导出末尾CLI报错、aux15秒及原trial配置保留；原控制账另存waiting/export_failure_ledger.json，当前账承接失败后复用同一导出，未重分数据。原始15秒账同时在trial/execution按失败来源保管，不以恢复覆盖失败事实。

新来源仅部署Linux `native/waiting/workspace`，旧repair/trial/fix及其他活动实验不覆盖。CPU11/12/13小证据、10GiB配置/门、范围补充、启动及观察回执两端保管；[回传清单](../reports/documentation/20261029/vendorlink/native/waiting/receive_manifest.json)记录当前7份小文件，机械核验一致。CPU11旧源码包和CPU12修复部署包含各次独有源码，继续作为必要来源证据保留；waiting传输ZIP在当前源码逐项核对及CPU13完成后删除，Linux活动副本首次真实更新后按约自删，复现源码保留。根文档本轮只在Windows更新；没有借此刷新其他批次状态、覆盖活动冻结文件或推送未获准新材料。

## 2026-10-29 原生实施外审回收与修复保管

本轮46153字节返回ZIP（SHA7123f89d…）的12载荷逐项核对；[完整原文和附件](../reports/documentation/20261029/vendorlink/native/review/external/report.zh.txt)、最终网页及[独立主审](../reports/documentation/20261029/vendorlink/native/review/primary_disposition.zh.txt)保留。已结案136445字节输入ZIP与临时返回ZIP无额外用途，验收后删除；原输入身份、原配置/门/CPU07/08和Git 5c74f229源码保留，不追改送审历史。

修复部署Linux独立native/repair/workspace，当前九源与CPU10一致；CPU09原九源以42164字节cpu09_sources.zip保留，供cpu09原子发布证据复用，非无用传输包。CPU09/10、新配置/门、资源观察等15个小文件已SFTP核对，见[清单](../reports/documentation/20261029/vendorlink/native/repair/transfer.json)；主审另同步repair目录并核对SHA。原native/workspace、活动实验冻结文件不动；真实数据、模型、Memory没有回传。新门仅解除实施审查前提，没有自动启动作业，qualification仍不存在。

根可变文档只Windows更新，本轮本地Git提交，不扩大先前推送范围；临时浏览器下载页/会话页和日志已关闭/清理，无新自动化或常驻监听，不重置大清理周期。

## 2026-10-29 数据外审回收与限定原生工作区

三市场数据外审返回51161字节ZIP（SHA07b356d6…）已核对全部12载荷；[完整原文与附件](../reports/documentation/20261029/vendorlink/three_market/review/external/report.zh.txt)、网页完成原件和主审保留Windows，重复返回ZIP已删除。原106444字节输入暂供后续实施审查来源衔接，不再作为待回收状态。

新增Linux独立`reports/documentation/20261029/vendorlink/native/workspace`，九源码与cpu08实物对应；手写CPU07/08及资源元数据八个文件SFTP回传后逐项大小/SHA匹配，见[清单](../reports/documentation/20261029/vendorlink/native/transfer.json)。cpu07_sources.zip保留原失败/局部通过的旧源码，不当作无用传输包删除；39336字节后续deploy.zip在当前源码确认后已删除两端。config/permit/implementation三个小文件已同步Linux并核对SHA，permit实施门false、qualification目录不存在；没有真实视图/qrels或GPU产物。新136445字节实施送审包按用户具体授权上传，原件保留待回收。根可变文档此次仍只Windows更新，不覆盖活动实验冻结文件；不重置大清理周期。

## 2026-10-29 三市场规范来源缓存与独立数据工作区

用户明确采用AlphaBay/Dream/Agora后，Linux新增`/home/yongpeng/cross-lingual/data/vendorlink/three_market_20261029/canonical.jsonl`，127537165字节、SHA256 fc80eaf41e712691bd15761a8854a3f094f93ce1153799dabb84a7b9753f4855，包含236457条规范公开商品记录；它服务后续一次数据构建，不是训练器可访问的历史Memory。原两个归档不变，缓存正文未回传Windows、不进入Git/外审。

实际代码部署在`reports/documentation/20261029/vendorlink/three_market/workspace`，与活动实验隔离；cpu06手写输出、联合inventory及来源配置/方案等小证据保管两端，八份回传实物与[Linux清单](../reports/documentation/20261029/vendorlink/three_market/transfer.json)一致。26872字节临时deploy.zip在六源码/产物验证并保留workspace后已删除两端副本，SHA476b6fdc…；旧源码桥及待审包继续服务当前复核。正式视图、qrels和新模型/权重均未生成；既有预训练不变。根可变文档只Windows更新，未覆盖活动冻结来源或推送本轮新材料。详见[方案与保管边界](../reports/documentation/20261029/vendorlink/three_market/plan.zh.txt)，不重置大清理周期。

## 2026-10-29 Agora原CSV取得与Linux保管

按用户“找agora”及此前可用数据只留Linux要求，已取得VendorLink官方指向的Kaggle公开CSV ZIP，最终路径`/home/yongpeng/cross-lingual/data/vendorlink/agora_20261029/kaggle.zip`，8071841字节、SHA256 e982c503097893f5aa07a3122ef2b0a8f96c81458966d09600d9bbf0128b26d8，内含单份32484217字节Agora.csv，直接流式解析未完整解包。Linux直连超时后由Windows现有代理中转，SFTP及两端身份核对后Windows副本/临时目录已删除；Linux1407422字节未完成片段也已删除，下载失败日志保留，见[检查和清理](../reports/documentation/20261029/vendorlink/agora_inventory/report.zh.txt)。

既有11650639字节Agora XLSX与9月归档SHA一致，继续作为不可变历史输入保留Linux，Windows无副本；与新CSV行数不同，不把它当同字节或已证等价转换。小型资格汇总/来源代码/清理记录两端保管，日志原件留Linux；根可变文档只Windows更新，未覆盖活动冻结文件、未推送本轮新材料，不重置大清理周期。

## 2026-10-29 Gwern候选下载、资格与按需清理

用户明确要求下载检查，并指示“确认了的搬运到Linux上，windows上不留副本。不行的就直接删掉”。[本轮检查](../reports/documentation/20261029/vendorlink/gwern_inventory/report.zh.txt)后仅保留Linux `/home/yongpeng/cross-lingual/data/vendorlink/gwern_20261029/grams-20150714-20160417.tar.xz`，67788336字节，SHA256 3163bc149144b8460216ee81e7ded5594b653ad0b89bd49a88f9eff544913c97。未完整解包，AlphaBay/Dream正文只在Linux解析。两个源的大小/SHA256跨端一致，SHA1与发布方一致。

SR1 2013包因仅9个账号满足20广告门槛，不足当前提议的三划分各4账号，已[删除](../reports/documentation/20261029/vendorlink/gwern_inventory/rejected_archive.json)11151256字节，未留解包副本。Windows两包共78939592字节在Linux身份验证后已[删除](../reports/documentation/20261029/vendorlink/gwern_inventory/cleanup_windows.json)；Linux初次直连失败的605903字节也已清理。无新的模型/权重/训练，未动活动冻结来源，不重置大清理日期。

小型报告/来源身份/核验脚本两端保管。原资格JSON/log中异常市场字段含错位商品文本，自动审批拒绝覆盖/删除Linux原件后，改为保留Linux原始审计证据，另导出qualification_summary.json；跨端核对原件SHA及汇总计数一致后，只删除Windows原始证据副本，见[汇总导出](../reports/documentation/20261029/vendorlink/gwern_inventory/summary_export.json)及[Windows证据副本清理](../reports/documentation/20261029/vendorlink/gwern_inventory/cleanup_evidence_copies.json)。Windows无本轮数据归档/正文副本，只有代码、去除文本片段的汇总和元数据。根级交接/设计/计划/讨论/存储文档仅Windows更新；本轮只本地Git，不借先前Harness推送范围扩大外发。

## 2026-10-29 VendorLink数据盘点与英文模型保管

用户明确批准盘点及下载正式所需模型，并要求Windows后备下载完成搬运后删除副本。官方google-bert/bert-base-cased固定revision cd5ef92a9fb2f889e972770a36d4ed042daf221e六文件共436414632字节，最终保管在Linux `/home/yongpeng/cross-lingual/models/step29/bert_base_cased`；[最终回执](../reports/documentation/20261029/vendorlink/data_qualification/model_ready.json)保存逐文件身份。主权重435755784字节，官方SHA256为1d8bdcee6021e2c25f0325e84889b61c2eb26b843eef5659c247af138d64f050；预训练pooler键存在，未载模/推理/训练。

Linux直连小文件成功、权重过慢的尝试及中止保留model_receipt.json与日志；Windows经现有本机代理下载119.479901秒后SFTP，Linux重新核对全部官方身份。已删除Linux2097152字节未完成权重和Windows435755784字节临时权重，见[搬运清理](../reports/documentation/20261029/vendorlink/data_qualification/transfer_cleanup.json)；无Windows模型副本、无全局代理变更、不重置大清理周期。六文件只含一份PyTorch兼容safetensors，没有额外TF/Flax/bin。

实际数据资格原件/限定下一步提议保管在两端同路径reports/documentation/20261029/vendorlink/data_qualification；Windows只保留小型汇总、脚本和文档，没有新拷贝正式商品文本或身份材料。现有market_item.xlsx与9月迁移SHA一致；AlphaBay/Dream/原始Silk Road商品表未找到，未下载新市场数据，不从身份SQL/HTML补造商品表。本轮仅编辑相关可变文档并同步本轮新证据，不覆盖任何活动冻结来源；当前根文档仍由Windows本次版本为准。

## 2026-10-29 Harness与当前文档同步交付

用户明确要求推送本地提交（含Harness优化），并同步服务器相关文档，禁止覆盖活动实验冻结文件。原11个本地提交已普通快进推送至现有科研分支，实际远端从4ca9d6b8774434d6db5591fb7c7f93dd41130adc到0e9a1614b5a17c5217038dcb8bd5e8a70cef6ced，经独立ls-remote核对；本轮同步收尾记录另随当次提交交付，不改变此前各轮授权的历史范围。

服务器同步只更新/home/yongpeng/cross-lingual下AGENTS.md，以及docs中的RESEARCH_DISCIPLINE.zh.md、AI_RESEARCH_HANDOFF.zh.md、CURRENT_EXPERIMENT_DESIGN.md、RESEARCH_PLAN.md、RESEARCH_DISCUSSION.zh.md、STORAGE.zh.md。七个原文件均为本用户普通单链接文件、无路径跳转，原字节全部能在已有Git历史定位，未发现服务器独有修改；经SFTP暂存、核对原文件未变及新文件身份后逐文件原子替换。最终七份大小/SHA及保护核对见[同步记录](../reports/documentation/20261029/harness_delivery.json)。活动新对照47项来源、原版等待44项来源及两份正式门文件同步前后保持同字节；没有向独立workspace、运行目录、模型、Memory或标签文件写入。未在Linux执行Git或刷新训练进度。

新同步记录另存Linux同路径；新回收VendorLink外审正文/附件仍按上一轮Windows保管范围，未借文档同步扩传其他科研材料。原服务器文档由既有Git恢复，临时传输副本核对后清理，不另存重复备份、不重置大清理周期。下方“总文档只Windows”保留其原记录时点。

## 2026-10-29 VendorLink定向复核结案与进度快照

修复复核已完整回收并经[独立主审](../reports/documentation/20261029/vendorlink/implementation/review/repair/primary_disposition.zh.txt)关闭F01—F04。Windows的review/repair/external保管16,939字节原文、检查附件与输出清单，另存完整页面及身份核对。10载荷再次逐项核对后，删除33,237字节返回ZIP和同字节浏览器副本，合计66,474字节；[清理回执](../reports/documentation/20261029/vendorlink/implementation/review/repair/transport_cleanup.json)保留原大小/SHA及实际删除结果。51,586字节修复输入、48,396字节首轮输入和cpu_transfer.zip分别作为当前/首轮源码及cpu01来源桥的明确复核依赖继续保留；不新建备份。专用会话和下载页已关闭，无本轮自动化。下方“在审/保留网页”是提交时点历史。

本轮为记录维护，只读核对Linux已授权作业元数据；[13:50服务器快照](../reports/documentation/20261026/continual_comparison/qualification/status_20261029_harness.json)194,687字节、SHA256 203352cdd92b603ef91f24baeb1508b512adc285a5ddb03832ba9891dcd26561，SCP回传与服务器同字节。只用既有py310读取来源/进程/更新计数/访问账/时间/资源，不载模、不读正式文本/标签或效果；单次临时采集器已删除，SSH退出，活动源码/作业未改。已有12:01快照及其原回执身份另行核对保留。新外审、主审及可变总文档本轮只保管Windows；不冒称已同步Linux，也不重置大清理周期。

## 2026-10-29 VendorLink首轮回收与CPU增量保管

首轮完整外审原文、10载荷、网页公布身份、返回清单核对与[主审处置](../reports/documentation/20261029/vendorlink/implementation/review/primary_disposition.zh.txt)已保存在Windows。35,412字节返回ZIP及同字节浏览器下载副本在载荷验证后删除；原48,396字节输入包继续作为冻结审查源码的复现依赖。cpu_transfer.zip内五份旧源与cpu01逐项同字节，作为F03来源桥保留；核对后的临时解包副本删除。清理明细见[transport_cleanup](../reports/documentation/20261029/vendorlink/implementation/review/transport_cleanup.json)，不重置大清理周期。

Linux独立repair_workspace及cpu05保留本轮修复源码与手写CPU原件，回传的六份证据已核对SHA，Windows另存来源核对。21,462字节CPU传输ZIP及2,117字节证据tar在两端核对后删除；当前源码精确副本也包含在待审51,586字节修复包中，该包因在审及来源复现需要保留。累计CPU26/300秒、无预训练/正式数据/模型载荷，SSH已退出；活动三路和既有等待不改。新主审与总文档当前仅Windows，不冒称已同步Linux。首轮已处置，但同一专用会话正在进行用户明确批准的F01—F04定向复核，因此保留网页待回收，无新增自动化。

## 2026-10-26 五臂中的三臂权重清理

用户要求删除五臂模型权重，获知当前新对照仍需加载 H/S 后明确选择：‘保留 H/S 54份供后续确认，其余三臂权重删除（推荐）’。11:29:59（Asia/Shanghai）已删除 Linux 的 SEQ_ref、ER0.1_ref、LOGIT0.1_ref 各27份阶段权重，共81份独立载荷、105823904280字节（98.556191GiB）。原 formal_20261023、continual_recovery/formal、continual_retry/formal 之间存在硬链接，实际删除171个文件路径；按inode去重计空间，未将路径字节重复相加。81份实际SHA均与最终完成记录匹配，所有选中inode的链接均在清理范围且无打开句柄。逐路径与保留对象见[盘点原件](../reports/maintenance/20261026/five_weights/inventory.json)，实际删除与复核见[回执](../reports/maintenance/20261026/five_weights/receipt.json)，两端大小／SHA一致。

H/S54端点共70436051280字节继续供新确认使用；其余7713个保留文件的inode／大小／mtime／链接数前后不变，含全部非权重证据及未选中的H/S、旧失败载荷。H/S此次按完成记录核对大小并验证元数据未变，所录SHA为原预期身份，不冒称本次重算54份散列。原指标、分数、Memory、日志、代码与审查结论保留；被删权重不能靠SHA恢复，重训仍需授权。Windows对应三目录无权重副本。盘点初次误尝试打开受保护owners文件被权限拒绝，未获取内容、未删除；修正为保留文件只查元数据。首次删除命令因终端折行抄错清单SHA在变更前停止，核实SHA后才执行；均如实留于回执。

删除后当前新对照控制器1198791及三worker1198941／1198946／1198951仍在，campaign无failure.json；未停止、重启或修改冻结队列。删除后共享磁盘可用330027536384字节，仅作现场观察，不把共享空间变化全部归因本次。新清理说明仅Windows／本地Git，Linux保管盘点与删除回执；本次局部清理不重置大清理周期。

## 2026-10-26 五方法结果外审回收与局部传输清理

原会话完整结果外审已回收并独立处置，33743字节原文、网页DOM、11项附件及逐载荷清单保存在[review/external](../reports/seller_alias_continual/20261026/continual_result/review/external/review_report.zh.txt)，主审和具体保管事实见[回收记录](../reports/seller_alias_continual/20261026/continual_result/review/recovery.json)。返回ZIP为277771字节、SHA256 beafdd3d2e248b28f4e4df862f266a1a2feabda865f4ff742344ffd66c03f386，11载荷全部验证后删除review传输ZIP和浏览器同字节下载副本，共555542字节；无递归删除。原9017557字节输入ZIP是返回独立复算程序所需的精确冻结1000载荷目录来源，包含已勘误前审计源码，继续保留该实际复现依赖。原报告与audit01/02不覆盖，必要勘误另存。专用页已关闭，本轮无自动化、SSH或Linux文件变动，新回收材料仅Windows/本地Git；H/S54端点仍供另行批准新对照使用。本次局部传输清理不重置大清理周期。

## 2026-10-24 两种新持续学习对照的方案外审与权重依赖
该次复审收尾：修订完整8376字符正文／146282字符DOM、独立处置及真实review_ready／放行原件两端保管，专用网页已关闭。新33514字节传输ZIP在确认13载荷逐项仍同字节、清单与审查原文保留后删除，见refill_revision/transport_cleanup.json；没有为删除而新建Linux整包备份。较早native／合并包尚未在该轮完成全载荷去重保管核对，不在该次局部删除中处理。等待610563放行后的资源等待、原24成功和两失败保管，以及该轮总文档／必要审查记录SFTP同步，均为当时事实；waiter02/released_status.json只对应该轮观测。后续载模OOM、失败账与失败载荷的接续保管，以[独立重试范围](../reports/documentation/20261024/continual_retry/scope.zh.txt)及[主审](../reports/documentation/20261024/continual_retry/primary.zh.txt)为准；最新运行状态统一从[交接顶部](AI_RESEARCH_HANDOFF.zh.md#当前工作)进入，不用旧放行回执推定仍在训练，也不据旧同步事实声称后续材料已全部同步。

19:42修订保管：原合并审查完整正文／DOM和独立处置已保存continual_recovery/review/external及primary_disposition.zh.txt。旧等待583238的源码／授权／身份／状态及费用承接记录在qualification/waiter01，修订代码、CPU02原日志和回执、替代等待610563回执在对应目录，两端同字节。旧原45冻结workspace和权重／Memory未动；没有正式恢复目录或review_ready。新33514字节／14成员增量包已获具体授权并成功提交原网页，会话在审，原合并包及新包继续保留；不提前删除仍有复审依赖的材料。来源／原件与当前状态分开，旧19:00段落按历史时点解释。本次不重置保管周期。

19:00恢复与合并外审保管更新：原故障8份运行原件已经回收核验，记录见original45_failure；原24成功产物和两中断载荷仍在原Linux目录，没有删除或载入。独立continual_recovery目录保存恢复代码、实际授权、CPU01日志／回执及等待状态；PID583238已运行等待，正式campaign尚未创建。CPU证据回传SHA一致，授权补明确后台启动后保留旧字节authorization.wait_request.json，不追改原CPU来源。最新合并包continual_recovery_native.zip为595997字节／164成员、SHA256 6b6dc3827a8a3759a43c2772d5cf2b126ea978c244c4dbd3c32b8abc85972fbb，已获具体许可并于18:59成功上传外审；当前在审保留，不提前清理。原505696字节native包未直接上传，载荷原字节纳入新包，保持两包身份和历史记录。审查请求／回执及最新可变总文档目前主要保管Windows；部署代码／CPU／等待状态两端有对应证据。本轮不重置大清理周期。

原生资格保管更新（2026-10-24）：Windows已核对回收native01—06及CPU07/08原始证据；native06回传15项、45运行来源结束后全部大小／SHA一致。每轮不同源码已在source子目录保管；native05旧L²R及原测试源码保留，原OOM不改写。native02/03失败临时全状态、native04及native06经验证的临时推理状态均已留证删除；native01—06目前无.pt，本次没有活动资格进程。手写Memory／分数留作资格证据，不是正式商品数据。原45控制器退出且出现failure.json，故障内容随后获用户明确批准读取，8份运行故障／进度原件已SFTP回收并核对大小／SHA，保管于qualification/original45_failure。H_pos中断stage2完整状态3904681285字节及推理状态／Memory保留；SEQ无检查点。全部24条成功状态和旧失败载荷未清理，未载模或恢复。现GPU资源不够且恢复补充未批准，没有恢复监听；新故障原件仅本地保管，未加入待授权外审包。主审区分原生数学／显存证据和并行兼容性遗漏，具体见[原生主审](../reports/documentation/20261024/continual_comparison/implementation/qualification/native_primary.zh.txt)。本轮不重置大清理周期。

15:41定向实现审查已完整回收并独立结案，repair目录保管[原文、DOM／TeX和主审](../reports/documentation/20261024/continual_comparison/implementation/review/repair/recovery.json)；11卡片都是输入引用，无新返回ZIP，专用页已关闭。当前38来源与CPU06匹配；cpu_checkpoint.zip中的CPU05过渡run/test原字节也与回执匹配，必要CPU快照和全部失败／通过证据保留。两份已结束上传ZIP已[逐项确认载荷同字节保管后删除](../reports/documentation/20261024/continual_comparison/implementation/review/repair/transport_cleanup.json)，合计368457字节：原62载荷中57在当前文件、5在核实的29092249提交，增量16载荷全在当前文件，输入清单／身份和本地Git均保留。代码／CPU证据仍两端保管，本轮新审查／可变文档只Windows；未SSH、删除权重／Memory或改活动45条，不重置大清理周期。下方在审保留文字是原提交时点历史。

14:23首版实现已送审：新[具体包及许可](../reports/documentation/20261024/continual_comparison/implementation/review/authorization.json)为306117字节／63成员，处于在审保留期，未当作旧包重复上传。当前源码、首轮及增量源码快照、CPU01—04原失败／通过console及receipt保存在Windows implementation和Linux独立implementation/workspace、qualification；8份回传证据大小/SHA核验一致，未同步正式数据／模型／Memory。源码快照用于重放已发生失败及增量核验，暂有明确用途；不是新增大载荷备份。当前新源码及CPU证据两端保管，外审/可变总文档本轮仅Windows；原45条workspace及H/S权重依赖不改，没有新的训练权重清理。局部实施／送审不重置项目大清理周期。

13:43完整回收最终补正外审，review/closure保管[原文及独立处置](../reports/documentation/20261024/continual_comparison/review/closure/recovery.json)，没有新返回附件／ZIP；请求及最终补正的原始大小／SHA保持。网页已实际关闭，原“待审”段落为提交时点历史。此轮仅本地保管，无Linux模型／Memory操作，不重置大清理周期。

13:39按用户最新要求再次提交最终补正正文，新增[定向请求／回执](../reports/documentation/20261024/continual_comparison/review/closure/submission.json)及实际页面证据保存在review/closure，原补正7459字节／SHA256 2845359d15bae8a462e79dbbb35089b225d49e06648d3b4b65d20acc1a148576未变。原会话再次打开并保留待审；无新ZIP、附件上传或下载，原公开来源及前两轮证据继续复用。下面关闭事实为前轮实际收尾，未追改；本轮不操作Linux、模型或Memory。

修订外审于13:29起完整回收，保留[第二次回复DOM／公式](../reports/documentation/20261024/continual_comparison/review/revision_external_review.json)、[完整转录](../reports/documentation/20261024/continual_comparison/review/revision_report.zh.txt)、[独立处置及最终补正](../reports/documentation/20261024/continual_comparison/review/revision_disposition.zh.txt)和[回收记录](../reports/documentation/20261024/continual_comparison/review/revision_recovery.json)。复审仅引用同名校准脚本，预览关键片段与原本地代码一致；未当作新补丁或下载重复源码，未声称完整附件字节身份已核验。原方案、首次补充及修订提交回执冻结，实际大小／SHA核对留回收记录。无新ZIP、模型／Memory操作或Linux副本；专用页关闭以实际回执为准。新材料只本地Git；局部外审收尾不重置大清理周期。

13:16用户明确要求修订版也外审，已用[修订正文](../reports/documentation/20261024/continual_comparison/review/revision_request.zh.txt)向原会话单次发送定向复核，复用网页已有原附件，未重新打包／上传。正文、来源SHA及[新提交回执](../reports/documentation/20261024/continual_comparison/review/revision_submission.json)留Windows；修订在审，原会话已重新打开保留。此前输入ZIP清理和首轮关闭仍为实际发生历史，不重建冗余包；公开原文、原16载荷及完整首轮审查均保管。

新[设计](../reports/documentation/20261024/continual_comparison/proposal.zh.txt)拟复用formal_20261023中S_tuned/H_pos_tuned各9条轨迹、每条3阶段，共54推理端点及其匹配校准/来源。该批状态在本次对比方案待决期间具有后续依赖，不能只因原批确认结束便按“无用途”清理；尚未形成的状态不记为已存在。当前未SSH查数量、载模、复制或删除权重，原活动状态与已批准生命周期保持；旧Optuna权重已删除，不作为本次可用依赖。

方案、背景请求、公开L²R原文及[外审原回执](../reports/documentation/20261024/continual_comparison/review/submission.json)保存在Windows reports/documentation/20261024/continual_comparison。送审包原为1214368字节、17成员，SHA256 11d3582f12674c1df16b8b6a999dfabf5d8081c3598cdd48331b3b760c8c1332；16载荷含方案/请求、三份公开论文、前次方法审查、十份脚本，不含正式文本/标签、Memory/模型或凭据。首次自动审批拒绝及用户具体许可、11:59单次提交历史不改。

13:04起完整回收后，保留[网页原文及公式](../reports/documentation/20261024/continual_comparison/review/external_review.json)、[可读原文](../reports/documentation/20261024/continual_comparison/review/report.zh.txt)、[独立主审及设计补充](../reports/documentation/20261024/continual_comparison/review/primary_disposition.zh.txt)和[收尾记录](../reports/documentation/20261024/continual_comparison/review/recovery.json)。16个原载荷再次与清单大小/SHA一致，文本来源已有本地Git冻结，公开PDF继续保管；无新增返回证据包。验收后清理本轮冗余输入传输ZIP及临时页面提取，具体事实以收尾记录为准，不新增备份或Linux副本。新材料仅本地Git，未改冻结workspace、既有删除或其他未提交产物；无新监听或自动化。上述是局部外审收尾，不重置项目大清理周期。

## 2026-10-23 五方法正式启动与实现外审闭环

16:26:59已按原授权启动Linux独立`reports/documentation/20261023/continual_implementation/formal_20261023`，冻结workspace/61来源和正式门两端身份核对，首批三worker实际更新见[启动观察](../reports/documentation/20261023/continual_implementation/formal_observation.json)。完整外审9文件原件、主审处置、cpu08增量证据及当前账已同步Linux；135正式阶段推理权重按后续确认依赖留Linux，尚未形成的权重不冒称已有。full仅真实恢复核实后按既有生命周期退休；不清理活动状态或Memory。当前权重/正式小证据均受224GiB全量和2GiB小证据上限约束。

本次72705字节/9成员外审返回包与网页SHA256一致，8载荷逐项核验；原文和必要附件保留review/external，完成处置后删除返回传输ZIP及浏览器重复下载。原836755字节输入ZIP仍是所保留mechanical_checks.py的具体复核输入和审查前来源，继续保留；没有增加重复备份。完整身份及清理事实见[主审核对](../reports/documentation/20261023/continual_implementation/review/primary_verification.json)，本次只本地Git不推送。下面“待回收/未正式训练”为形成时历史。

## 2026-10-23 五方法实现、限定资格与待回收实现外审




用户已批准五方法/零HPO/45轨迹/三路/490累计小时，已在Linux独立`reports/documentation/20261023/continual_implementation/workspace`部署最小实现；61份当前来源1056846字节已逐项与Windows核验一致，见[current_sources](../reports/documentation/20261023/continual_implementation/current_sources.json)。模型只链接既有BGE归档，数据目录仅建立既有数据链接，资格均使用手写群，尚未正式训练/生成确认数据/打开正式标签。旧workspace及模型保管不改。

CPU01—07与native01/02失败/完成原件、账本和必要资源样本在两端保管；GPU临时完整/推理状态在实际恢复验证及留存大小/SHA后退休。当前核验源快照source_cpu01_native01.zip、qualification_fix.zip、implementation_update.zip保留其具体重放用途；不是无用传输重复副本。独立处置、预算投影和控制费保守补记见[资格主审](../reports/documentation/20261023/continual_implementation/qualification/primary.zh.txt)。新正式权重尚不存在，计划135阶段推理状态须保留供确认，存储cap224GiB/free288GiB仍是执行上限，不能写成当前占用。

实现输入包continual_implementation_review.zip为836755字节、158成员，SHA256 6a792263593188bbf89217f928de67626588b11bb5c329c81151e2f63aa9ff63；157载荷1790191字节及ZIP逐项身份核验一致。含源码、冻结方案、手写资格和来源/运行记录，不含正式文本/标签/owners/Memory或模型载荷/凭据。自动审批首次拒绝后，用户明确批准整个具体包上传到chatgpt.com；15:49已单次提交原会话，当前待完整回收，见[具体授权](../reports/documentation/20261023/continual_implementation/review/authorization.json)与[提交回执](../reports/documentation/20261023/continual_implementation/review/submission.json)。GPU问题修复明确免审，不把此次实现审查扩大为GPU复审。输入包因在审和身份复核保留，不能提前删；未创建重复审查或重新上传。新材料仅本地Git，不推送。


## 2026-10-23 五方法修订方案送审材料

新方案、预算、请求、清单、普通字段/算术核对、明确授权和实际页面/提交回执位于Windows [continual_baselines](../reports/documentation/20261023/continual_baselines)。输入ZIP `review/continual_baseline_plan_review.zip`为42232字节、8成员（7载荷+清单），SHA256 `0c95fd669fc41cc7423c36038fb2c6d356a607d0676ad69cf8f070aeb63a8300`，逐项大小/SHA一致；载荷仅本次方案/预算/请求、前次主审处置、selected.json及ER/LOGIT历史结果文档，不含正式数据/标签/Memory/模型/凭据。原六臂输入ZIP字节未改。

自动审批首次拒绝及用户随后对整个新包的明确批准均保留；原包13:18:51提交后的外审已完整回收并独立处置。返回包42613字节/8成员，SHA256 `0f11166ff8259ae7ebd64eb2631521199ae8d4029113f93b5359560aed582daa`，7载荷大小/SHA一致；八原件保留在[external](../reports/documentation/20261023/continual_baselines/review/external)，完整网页、独立核验、主审和完成回执在同级review。验收后删除返回传输ZIP及浏览器重复下载，唯一输入ZIP仍为所保留复核脚本的具体输入而保留；冻结proposal/budget不改。专用页和本轮跟进回收完成后关闭/删除。新材料只作本地Git交付，不推送、不声称已同步Linux，也不改冻结workspace或旧权重保管事实。当前490h/224GiB是未批准的未来预算，不是已发生占用。旧六臂/三臂材料按下面历史记录解释，当前入口见交接顶部。

## 2026-10-23 三域主线静态方案待审材料

12:35后的完整回收：返回continual_mainline_external_review_20261023.zip为55479字节/9成员，SHA256 fc01291f38f46a88d3daf7f76e31ff595d4f387a9bd1d7ea447d2800cf6b29ae。原文、发现、阅读范围、机械复核脚本/输出和交付清单共9原件已核对后保留于[review/external](../reports/documentation/20261023/continual_mainline/review/external)，网页正文、回收核验与独立处置/普通算术记录在同级review。返回传输ZIP验收后无额外用途，按纪律清理该下载副本；原输入ZIP仍为外审复核脚本的必要输入而保留。专用页已关闭、回收跟进已删除，未向Linux新增备份或同步。旧六臂方案原件保持，当前推荐27路径的具体范围与预算另见主审处置，仍待用户确认。

随后用户明确批准该具体ZIP外发，并要求重点审查过度设计及必要实验。2026-10-23 12:04:45（Asia/Shanghai）原包身份不变、经本地Playwright单次提交成功；[授权](../reports/documentation/20261023/continual_mainline/review/authorization.json)、实际请求、提交页面及[回执](../reports/documentation/20261023/continual_mainline/review/submission.json)新增在同目录，回执保留先前拒绝历史。当前在审、待完整回收和独立处置；包内待确认字样是形成时状态，不重打包覆写。实施／预算仍未批准。

新具体方案、预算及外审请求保存在Windows `reports/documentation/20261023/continual_mainline/`。已准备[方案审查ZIP](../reports/documentation/20261023/continual_mainline/review/continual_mainline_plan_review.zip)：233582字节、35成员（34载荷及manifest），SHA256 `17d1bfd3214d6b9f164274527925ef2d14774967edd0ae2d22947e620097f487`；34载荷608117字节的大小/SHA逐项一致，见[准备回执](../reports/documentation/20261023/continual_mainline/review/preparation.json)。仅含具体方案、预算、请求、相关已开放报告/合同/源码/资格字段和权重删除回执，没有正式文本、标签、owners、private_custody、模型载荷或凭据。当前未上传、未连接Linux、未新增科研执行，具体包先供用户审阅确认；旧结果包批准不扩大到本包。新方案及元数据作本地Git交付，ZIP留Windows供待审上传，不推送、不覆盖冻结历史来源。此新增待审包有当前用途，不属于应清理的已验收无用副本；其他Optuna保管决定不变。

## 项目大清理周期

最近一次两端项目大清理完成日期：**2026-11-06**。下一次到期日期：**2026-11-21**，到期须先说明理由及具体范围、取得用户许可，再执行；局部缓存清理不重置该周期。本轮在用户确认完整范围及两组权重处置后完成，保留项已明确，无待决删除项；上次2026-10-22完成事实保留为历史。

独立提醒任务“科研项目大清理到期提醒”（Codex任务ID：`automation-2`）已于2026-10-08创建并启用，附着本对话，每天北京时间09:00仅读取本节与清理规则判断是否到期。同一到期日只首次提醒；未到期及无新决定时保持安静。任务不连接Linux，不执行盘点、删除、Git操作或实验。实际大清理完成后更新上面两项日期，任务据此判断新周期；提醒运行或局部清理均不修改最近完成日期。

2026-10-08大清理按用户要求未另建删除文件清单或逐项回执。Windows冗余历史大载荷仅在Linux原件逐项大小／SHA一致后移除，恢复仍以Linux项目内原相对路径为准；Linux独有科研来源、正式结果和明确保留模型继续保管。Git采用普通删除提交，历史提交保留。旧清单仅证明记录时点的保管事实，不能替代当前实物核对；本次保管位置的变化见下节。

## 2026-10-23 Optuna训练权重清理

用户明确要求同步最新进展并删除Linux遗留Optuna训练权重，覆盖此前仅针对这些权重的暂存安排。2026-10-23 11:15:09（Asia/Shanghai）已删除19个普通文件，共24783347439字节（23.08GiB）；11:15:12完成删除后复核。两端同路径保管[逐文件删除回执](../reports/maintenance/20261023/optuna_weights/receipt.json)，每项保留原路径、大小、SHA、属主与文件身份及删除结果，目标残留0。

| 删除范围（相对Linux项目根） | 文件数 | 字节数 |
|---|---:|---:|
| `reports/documentation/20261021/optuna_reboot/workspace/reports/job/replication_*/run/models/`内九复验A/B端点 | 18 | 23478977354 |
| `reports/documentation/20261019/optuna_recovery/resume_workspace/reports/job/search_08_S_s0/run/models/A_S.pt`，重启前中断遗留 | 1 | 1304370085 |

删除前核对完成状态、当前54份来源、访问账train46/valid37/heldout0/owners0及失败记录；没有相关训练进程，未见可访问描述符打开目标。全部目标属本账号、普通文件且单硬链接，最终路径位于确认的Optuna目录；系统SSH/PAM/FUSE/SFTP辅助进程部分描述符不可读的限制已记录。两次机械预检在任何删除前停止，修正路径排序核对及识别当前SFTP辅助进程后才执行，不计为新的科研尝试。

核对覆盖10月18日implementation、10月19日budget/recovery及10月21日reboot四个Optuna目录：13702个非权重普通文件（434350451字节）和33个符号链接前后身份完全一致。147个Memory载荷（50587385字节）全部保留，其中当前完整campaign的90个为30883495字节；原配置／源码快照、SQLite、完成／失败／费用与访问账、预测／指标／诊断、日志和审查材料保留。未加载或下载权重／Memory，未删除预训练或其他实验模型。权重已无可恢复载荷，哈希仅证明删除前身份；今后训练或载模需求不能据本回执自动重训，也不能把旧A→B权重当作可续C的现存起点。

本轮同时核对最新外审结案记录，更新并SFTP同步五份可变总文档：交接、讨论、存储、计划和设计入口。同步前实际比较两端差异，Linux仍为旧进度，未发现远端独有新决定；同步后核对字节数／SHA一致。冻结workspace不覆盖；审查返回原件及主审仍按下节Windows保管位置查阅，不声称全部结果目录已同步Linux。删除回执与文档作范围明确的本地提交，既有未提交成果不纳入，未推送新材料。本次局部权重清理不改变最近大清理2026-10-22和下次到期2026-11-06。

## 2026-10-23 Optuna完整结果回传与独立分析

原执行workspace为`/home/yongpeng/cross-lingual/reports/documentation/20261021/optuna_reboot/workspace`，正式job及其历史来源保持。09:58:42正常完成，退出及最终stdout费用已回收；本轮成果统一在Windows `reports/seller_alias_continual/20261023/optuna_result`，Linux同相对目录保留分析和回传清单，正式载荷仍以原workspace为原件。完整7639小证据215787086字节及10分析文件3057672字节，经SCP回传全部大小/SHA一致，见[接收核验](../reports/seller_alias_continual/20261023/optuna_result/receive_verification.json)。来源快照2399项和45轨迹/90端点/540捕获核对一致；36,090字段既有分析及179,478标量增量核对完成，详见[报告](../reports/seller_alias_continual/20261023/optuna_result/report.zh.md)。首次SCP认证前连接关闭后重传成功，无新增科研失败。

新独立分析源码在主目录scripts两端保管，不改冻结workspace科学来源；单线程Linux py310、无GPU/正式标签/载模，完整外层0.919040秒保守计1秒，分析A累计12.173054/600秒，总收费325695.192351/979200秒（按10月23日外审独立主审的末位勘误，原账／报告字节不改）。原父账不覆写，增量见analysis_wallclock.json。18份复验权重23478977354字节及90份Memory载荷30883495字节仍留Linux，未加载、下载或删除；旧中断目录亦保留。本轮仅清理新生成且验收后的两个运输ZIP，非用户待处理的历史Optuna文件清理，不重置11月6日大清理周期。本轮完整结果包另按具体许可外发，新成果只本地Git，不扩大原七提交推送许可；同事汇报草稿继续本地忽略。

随后用户明确批准`review/optuna_result_review.zip`上传至chatgpt.com，10:40:04经本地Playwright单次提交；具体包103538522字节、7679成员，SHA256及会话身份见[提交回执](../reports/seller_alias_continual/20261023/optuna_result/review/submission.json)。输入ZIP、request、input_manifest、准备／许可记录及已发送页面证据留在Windows同一review目录，提交时为待回收状态；本轮已回收结案，见下文。已发送包内报告的“尚未送审”等文字保留形成时原貌，当前状态从[交接](AI_RESEARCH_HANDOFF.zh.md)进入，不重打包覆盖已发送原件。随后11:01:55实查最终外审完成，完整回收1256622字节返回ZIP和external/55文件（11321058字节），SHA256 7a0b57de2c24768b6ef9c34318d4b07bb691a269c713947a114fbd1c1e07f9c0，54内部载荷核对一致。完整报告、14项发现、阅读范围、复算代码及证据均保管；[独立主审及勘误](../reports/seller_alias_continual/20261023/optuna_result/review/primary_disposition.zh.txt)完成，专用页已关闭，当前状态由submission.json承接。输入／返回ZIP按既有Optuna保留决定及本次用户查阅需要继续保管，本轮不作实验文件清理；仅处理本次下载新增的浏览器重复临时副本。不把已删除运输ZIP与保留的审查ZIP混淆。

此前文档补齐只维护Windows四份现有总文档，当时没有同步Linux项目根docs、重新连接Linux或新增删除。上文权重及旧中断目录的保管事实是完成时点记录；后续用户明确授权的19权重删除及五份总文档同步见本页上一节，Memory继续保留。10月22日“明天处理Optuna”本身不是删除许可，实际清理依据10月23日新的明确要求；大清理周期保持。

## 2026-10-22 两端项目大清理提前完成

用户先要求盘点并确认后再删，随后明确“Optuna实验的文件都不要动啊，我们明天会处理”，并批准将原定次日的大清理提前到今天，按已呈现的92文件清单执行并更新记录。执行范围严格限于该清单；所有Optuna实验目录、旧／恢复工作区、模型、Memory、SQLite、源码／核验／外审材料和相关浏览器文件均排除。未扩大到其他模型、数据、封存材料、Git历史或其他用户文件。

| 已完成范围 | 文件数 | 逻辑字节数 | 保留依据 |
|---|---:|---:|---|
| Windows已完整解包的历史外审返回ZIP | 19 | 56021348 | 删除前重新核对全部2676个普通成员与解包文件的大小／SHA；必要正文和证据保留 |
| Windows解包目录中重复携带的输入ZIP | 9 | 34083059 | 与正式输入包同字节；正式Windows包也列入下一组时，以已核对的Linux正式包保管 |
| Windows已有Linux原件的历史归档 | 49 | 114620812 | 当次核对最终Linux保留位置的大小／SHA一致；不新建备份或补传包 |
| Linux结果／外审汇集目录中的重复源码归档 | 15 | 2126850 | 保留实际执行目录的源码快照、核验与失败记录；删除前后均核对原件 |

Windows于17:09:27（Asia/Shanghai）精确删除77文件、204725219字节；Linux于17:10:47删除15文件、2126850字节。两端合计92文件、206852069逻辑字节（197.27MiB），目标残留0。Linux15项实际分配2162688字节（2.06MiB），删除前确认属主、普通文件／单硬链接、最终路径和本账号无打开描述符，删除后15项保留原件SHA仍相同。Windows卷可用空间在操作窗口增加204873728字节，Linux增加2162688字节；Windows卷的同时活动可能影响差值，不把它当成逐文件分配大小的精确证明。

保管位置：上述Windows归档通常恢复自Linux项目内同相对路径。涉及两端都移除汇集副本的9项源码归档，须使用实际执行目录原件：`20261014/query_replay_result/qualification/cpu/attemptNN/source.tar.gz`对应`20261013/query_replay_execution/20261013_151500/workspace/reports/qualification/cpu/attemptNN/source.tar.gz`（NN=01—03）；`20261013/crossfit_retention_result/qualification/evidence/{cpu,gpu}/attemptNN/source.tar.gz`对应`20261012/crossfit_retention_execution/20261012_140200/workspace/reports/qualification/{cpu,gpu}/attemptNN/source.tar.gz`（CPU01—04、GPU01—02）。这些路径均位于`/home/yongpeng/cross-lingual/reports/seller_alias_continual/`下；Linux外审解包中的另外6份同内容源码副本也按相同映射保管。未同时删除最终原件，不把SHA当作可恢复载荷。

同步前核对三份可变总文档的实际差异：Linux的存储／讨论文档缺少Windows已有后续记录，交接仍为10月21日11:29旧观测，未发现远端独有的新决定。仅更新Linux项目根`docs/STORAGE.zh.md`、`docs/AI_RESEARCH_HANDOFF.zh.md`、`docs/RESEARCH_DISCUSSION.zh.md`，不覆盖独立workspace内冻结文档或改动Optuna实验文件；本次不重新验收或开放Optuna结果。两端记录按SFTP同步后核对同字节；清理删除和文档作正常本地Git提交，历史／LFS及原有未提交的`reports/documentation/20261022/continual_progress.zh.md`保持，未推送新材料。既有提醒任务按本节新日期判断下一周期，无需另建任务。

## 2026-10-22 Nature Skills本机安装

用户明确要求安装`nature-writing`、`nature-figure`、`nature-shared`并及时更新现有记录。12:02（Asia/Shanghai）验收：使用内置`skill-installer`从[Yuan1z0825/nature-skills](https://github.com/Yuan1z0825/nature-skills)的固定提交[`c6ae86c742705fd968b13c79db1a8518a84089d4`](https://github.com/Yuan1z0825/nature-skills/tree/c6ae86c742705fd968b13c79db1a8518a84089d4)下载并完整安装至本机用户目录`C:/Users/35734/.codex/skills/`。安装前这三个目录均不存在；安装器退出0，228个文件的路径／字节数／Git blob身份与该提交的公开tree逐项一致，两个主技能的9个`always_load`引用均可解析，包括共享依赖。

| 目录 | 上游版本 | 文件数 | 字节数 | 用途 |
|---|---|---:|---:|---|
| `nature-writing/` | manifest 1.5.0 | 77 | 211998 | 论文论证组织、章节起草与首次投稿材料 |
| `nature-figure/` | manifest 2.8.0 | 130 | 35144935 | 科研图、方法图、排版导出及图件检查 |
| `nature-shared/` | 以上固定提交；入口无独立版本号 | 21 | 135606 | 按主技能需要加载的共享参考，不作为独立用户工作流 |

三个`SKILL.md`的SHA256依次为`ce7c4d20329b96b03b0f0c0bb744411b1b4deaa7a7b69b90757705a83c195b12`、`5aefe708a2531c5ddaa8ee74d5155d747e96c0f21a33143d38316b7c5c88ffc4`、`3d914e73f023cab4f247d82f906a948307de8a867fd6c33df999badf67c5d5f2`。完整技能目录保存在仓库外，不把上游源码／资产复制进科研Git；恢复时按上述提交重新安装同三个`skills/`路径。安装器运输缓存及本轮临时源清单验收后清理，不新增长期并行清单。

12:02的安装验收只覆盖文件与引用，当时未运行写作／绘图脚本、未安装额外Python依赖、未连接Linux；该历史边界不追改。随后用户要求“能用吗？你要保证这些skill能用啊”，本轮会话已实际识别两个主技能并加载共享参考，按既有Linux py310完成下列最小功能验收。

- 写作：按`manuscript / algorithmic / experiments / zh-to-en / generic`路径，将已开放且已有独立主审的H_pos结果写成[英文结果段与中文说明](../reports/maintenance/20261022/nature_skills/writing_sample.txt)，附术语表和主张—证据对应；逐项核对原记录中的MAP差值及条件区间，保留新域未确立改善、固定种子／开发集和主次判据边界。没有重算指标或形成新研究方案。
- 依赖：Linux既有py310缺少PDF碰撞检查所需的PyMuPDF。原镜像dry-run因SSL连接失败，无包变更；改由Windows下载PyPI官方`PyMuPDF==1.26.5`的Linux wheel，校验SHA256后经SCP离线`--no-index --no-deps`安装，满足上游`>=1.24.0,<2.0.0`。完整前后包清单留Linux，[差异原件](../reports/maintenance/20261022/nature_skills/package_changes.json)确认仅新增此包；Matplotlib 3.10.7、NumPy 2.2.6、Pillow 12.0.0均未改变。wheel来源及哈希见[固定依赖记录](../reports/maintenance/20261022/nature_skills/wheel.json)。通过上游配置脚本保存本机`C:/Users/35734/.config/nature-skills/nature-figure.json`的`backend=python`，未改技能源码。
- 绘图：在独立`/home/yongpeng/cross-lingual/reports/maintenance/20261022/nature_skills`中，用[手写示例](../reports/maintenance/20261022/nature_skills/smoke_figure.py)实际生成183×80mm双面板PDF／SVG／600dpi PNG及TIFF。首轮源码检查因直接声明DejaVu Sans未匹配上游字体族检查而停止；保留`attempt01.*`，按约定显式配置sans-serif、DejaVu Sans字体并补齐TIFF后复验。最终严格源码检查21通过／0警告／0失败，实际面板对齐通过且无豁免，PDF最小字号6pt、无低于5pt文字，实际PDF碰撞检查0失败／0警告。验收时已目视检查从PDF渲染的预览（图件随后按用户要求删除）：文字、刻度、线条和两面板布局完整，图内明确标识仅为工具检查。执行原件见[execution.json](../reports/maintenance/20261022/nature_skills/execution.json)及相邻审计输出；最终检查与渲染约3.85秒，连首轮失败及PDF预览合计约4.11秒，单CPU、CUDA不可见、子进程峰值RSS约117.2MiB。

26份Linux返回载荷已SCP回传并按[接收清单](../reports/maintenance/20261022/nature_skills/receive_manifest.json)逐项核对；写作例稿、初始运输来源身份及验收原件保存在同目录。Linux保留审计脚本、原件和前后包清单；本机保留完整已安装技能。两端运输ZIP／wheel和本轮临时目录在核对后清理，不改变大清理日期。上游四个审计脚本从上述固定提交原样使用；恢复时按原技能提交及固定wheel版本安装，在Linux既有py310按execution.json所列命令执行，审计脚本须位于工作目录或Python搜索路径。

2026-10-22用户随后要求“删掉这些测试图”。已核对归属、最终路径及原接收身份，删除Windows和Linux该测试目录内各6个图件：figure.pdf、figure.svg、figure.png、figure.tiff、figure.pdf-preview.png及figure.alignment.svg，每端658116字节。两端均核对已不存在；保留技能、依赖、示例源码和文字／JSON验收原件。receive_manifest.json仍是删除前的历史接收事实，不作为当前图件存量清单；本次使用普通Git删除提交，未改写历史，不重置大清理周期。

当前已验证的是项目会用到的证据约束写作与Python科研图导出／审计路径；未将R、外部图像API或所有复杂图型写成已测试。后续具体论文和图件仍按实际证据核验。项目的明确用户决定、现行合同、科研纪律和GPT 6 Pro外审要求继续适用；本次没有正式数据读取、GPU、载模、训练或活动Optuna调度变更，不新增实验／数据／外发权限，也不修改其预算账。

## 2026-10-21 Optuna服务器重启恢复

当前workspace为`/home/yongpeng/cross-lingual/reports/documentation/20261021/optuna_reboot/workspace`，启动和首更新在并列execution。旧optuna_recovery/resume_workspace的4,322文件/1,442,097,470字节按[重启恢复绑定](../reports/documentation/20261021/optuna_reboot/recovery_bound.json)冻结，包括25次完成结果、SQLite残留RUNNING和中断S第9次的A推理权重/Memory；没有载入、覆盖或删除旧状态。新workspace直接复制25次结果用于原45轨迹收尾，未重训重评；新中断尝试访问记录独立留账。新生成的活动模型和Memory继续仅留Linux。

本次用户再次明确免审，见[原话及范围](../reports/documentation/20261021/optuna_reboot/waiver.json)；没有外发新包。CPU attempt13/14分别保留原件及54来源ZIP，均六项通过，外层各15秒，累计188/300；原18原生/876秒复用。初稿C补计1秒被主审核对为遗漏未结控制区间，正式前修正为保守4秒，attempt13原来源保持。详细费用及工程边界见[独立主审](../reports/documentation/20261021/optuna_reboot/primary.zh.txt)。启动/正式门/实际访问和预算小原件已SCP回传，具体身份在execution/receive_manifest.json；所有正式文本标签原表、权重和Memory载荷均未回传。本轮只作本地Git交付，旧公开推送范围不泛化；局部运输件清理不重置10月23日周期。

## 2026-10-19 Optuna资源恢复与9GiB准入修订

当前运行在`/home/yongpeng/cross-lingual/reports/documentation/20261019/optuna_recovery/resume_workspace`，启动/观测在并列resume_execution。原optuna_budget/workspace的三完成结果和资源FAIL保持；原optuna_recovery/workspace及execution保留18:52零派发来源排序失败，未覆盖其门和源码。当前53来源依Linux attempt12原顺序，资格累计158/300；原18调用/876秒复用。旧三个run结果复制到新job用于完整45轨迹收尾，未重训重评；原6权重已退休，未恢复权重或重新加载旧Memory。

三个源码快照分别保留attempt10的qualification/sources.zip（290995字节，08b16c3570b295ae3337c8ca410b7aa1609e0cd71aa890aa232b2b94c87ab0f0）、attempt11的qualification/current_sources.zip（291150字节，b848bca74e45e806ec5370c98c21beda45794683e38ccc819a3fec44aea2898d）、当前qualification/resume/sources.zip（291577字节，62d813a7f0fc5258514e5b70cbc2385bb18bfbc71bd2fc0fcfe979a4375cb076；Linux保管名qualification/resume_sources.zip）。CPU原件、原517文件恢复绑定及零派发失败费用在资格目录；当前门及主审在qualification/resume。attempt10的10载荷、attempt11的9载荷、attempt12连失败的20载荷及恢复首更新15载荷均SCP回传逐项核验，清单分别为receive_manifest、attempt11_manifest、resume_receive_manifest和resume_execution/receive_manifest。科研原件保留，运输ZIP验收后删除。

用户本次明确免审；原1128650字节/125成员新审查包从未上传，已取消并删除无后续用途的ZIP、打包清单和请求，身份与取消状态仍见review/package.json，免审原话见review/waiver.json。首次包含审查归档的closeout同步被自动审批拒绝，未上传；取消冗余归档后，仅按用户新的明确继续授权部署必要执行文件。原272小时72成员审查包仍为native来源复用的实际依赖，继续保留。Windows首次门部署运输件作为失败门快照保留在execution/deployment_snapshot.zip；Linux原失败workspace含相同原件。未公开push，未外发新审查材料。
## 2026-10-19 272小时外审、预算接线与正式启动

[外审回收](../reports/documentation/20261019/optuna_budget/review/recovery.json)保管26返回原件及完整报告，25内部载荷/71输入身份核验；必要失败、阅读范围及主审完整保留，专用页关闭。原获准72成员输入ZIP继续保留，现为具体原生48来源复用的实际门依赖；不得作为冗余上传包删除。外审返回运输ZIP可在解包和同步核验后清理。

当前51来源部署在`/home/yongpeng/cross-lingual/reports/documentation/20261019/optuna_budget/workspace`，保留qualification/budget_sources.zip作为attempt09与正式来源快照；旧clock_sources及native原件不改。七份新增CPU证据9148字节按[清单](../reports/documentation/20261019/optuna_budget/qualification/budget_receive_manifest.json)逐项回传；14份启动/进程/访问/首更新/部署证据28026字节按[执行清单](../reports/documentation/20261019/optuna_budget/execution/receive_manifest.json)核验。门文件置于该workspace内同项目相对路径，原结果/原生/审查依赖不改写。

固定52份S/H已开放基线、8份直接来源及9份固定摘要/锚点文件已核对；首次机械部署因Linux主目录缺S0摘要而中止，未开始训练/CPU/数据访问，随后由已核Windows归档SCP补齐，保留deployment_failure与实际deployment原件。只建立7个合同允许的train/development及公开身份文件链接和既有BGE链接，不链接heldout/owners/private或旧Memory。启动后新权重和Memory仅留Linux，搜索权重按每run成功完整核对后留证清理，复验18端点保留；活动作业载荷不清理。

当前启动状态与PID只看[交接顶部](AI_RESEARCH_HANDOFF.zh.md#当前工作)。运输formal_deployment、anchor_sources及两份receive ZIP在核验后按精确路径清理，来源快照及门依赖保留；本轮只本地Git，不改变10月23日大清理周期。

18:15—18:17状态检查发现正式作业已于16:38因第4个派发的GPU资源预检失败停止。必要停止证据保管于[resource_stop](../reports/documentation/20261019/optuna_budget/execution/resource_stop/manifest.json)：28载荷218477字节（加清单共29成员），含原日志、退出/失败/收费、只读核对的停止SQLite、三个完成/访问/盲门/退休回执及现场观测；从Linux原件SCP回传逐项大小/SHA一致。运输ZIP为39874字节、SHA256 beb473fc5ba9a88412f56b190efb445b1dc1b6355bfaaeaaf222bf0dda83d1e2，验收后两端删除。原job、全部结果/分数/Memory、绑定来源和失败账保留；本轮不载模、不重算、不新建监听、不重启，不删除科研载荷。三个成功搜索的6端点权重已由原控制器按既定生命周期删除，非本轮人工操作；恢复权限与最新阻断只见交接顶部。

## 2026-10-19 Optuna共享起点修复与CPU证据

只同步48份当前科学来源至新`/home/yongpeng/cross-lingual/reports/documentation/20261019/optuna_budget/workspace`；未覆盖旧repair/workspace、未部署模型或正式数据。来源快照qualification/clock_sources.zip为264850字节、SHA256 c210be20811a9ea321b66a05a0611d8797dd9dbd82eba12bf0008e35a0f92cd5，作为attempt08复现源码保留；两端逐文件身份一致。七份CPU原件8611字节按[接收清单](../reports/documentation/20261019/optuna_budget/qualification/clock_receive_manifest.json)SCP回传逐项核验，外层8秒、累计116/300，见[处置](../reports/documentation/20261019/optuna_budget/clock_disposition.zh.txt)。仅三项定向手写CPU、0 optimizer/GPU/正式调用；原生及当前外审输入包不改，新证据尚未外发。新材料仅本地Git，大清理周期不变。

## 2026-10-19 Optuna预算修订静态材料

新提案、保存字段算术和来源身份、独立主审及定向外审请求保存在reports/documentation/20261019/optuna_budget。当前只Windows文件管理/算术，未同步Linux或改生产源码，不声称已实施。具体静态包[optuna_budget_review.zip](../reports/documentation/20261019/optuna_budget/review/optuna_budget_review.zip)377084字节/72成员，SHA256 58c4e04cc7f397fda0909331f6e6eabd9b8e63090cfa6b3a1efcbdd7b786d1b5；71载荷1087615字节及清单逐项验收。包含当前48绑定来源、原生/CPU必要小证据和历史预算/实现审查背景，不含正式文本/标签原表、权重、Memory或凭据。用户2026-10-19明确批准该具体包及272小时修订、条件自动三臂36+9；10:03:43已单次上传并提交[GPT 6 Pro静态外审](../reports/documentation/20261019/optuna_budget/review/submission.json)，实际请求/可见模型/发送页面与最新authorization在同目录保管，未更改冻结ZIP或包内历史待确认字节。外审进行中，为当前审查/后续处置保留输入包及必要证据；未同步Linux或实施新预算，未创建应用监听。新材料仅本地Git，不改变大清理周期。

2026-10-19 10:08:37（UTC+08:00）按用户磁盘询问只读核对实际部署路径：repair/workspace解析仍在`/home/yongpeng/cross-lingual/reports/documentation/20261018/optuna_implementation/repair/workspace`，文件系统`/dev/nvme0n1p2`、ext4、挂载`/`。`df -B1`总3936290357248、已用2986698088448、当前用户可用749563080704字节（749.56GB／698.09GiB），inode可用231526220；挂载选项无quota，quota命令未安装，未独立取得用户配额回执。repair目录当前分配5029888字节；相关Optuna/active_wait进程列表为空。实际search/checkpoint/resource三份来源SHA与Windows一致；此次未启动训练、载模、读取正式数据或清理文件。

按当前[原生保存尺寸](../reports/documentation/20261018/optuna_implementation/repair/native_completed/qualification/native/result.json)及[上次正式端点实物](../reports/maintenance/20261018/positive_ranking_weights/receipt.json)估计：单推理端点约1.3044GB／1.2148GiB，含Adam完整临时状态约3.9047GB／3.6365GiB。45run×2=90端点累计约117.40GB／109.33GiB；其中搜索72份在各run成功退出、完整结果及独立反馈后逐run留证删除，复验18份约23.48GB／21.87GiB保留。完整临时状态每端点恢复核验后即删除，实际写入是直接torch.save而非多副本文件替换；因此最高约18份推理状态+1份临时完整状态=27.38GB，再加全部2GiB非权重预算约29.53GB／27.50GiB，元数据有小幅变化，不能当精确未来实测或新硬限。当前64GiB全输出限额充足，全部用满后按本次可用量仍余680843603968字节／634.09GiB。启动前必须重查当时可用空间；正式阶段资源门按campaign全输出需求核对，不能直接复用原生监听10GiB条件。本轮不改冻结提案、生产policy或外审包，也不增加候选/预算或新清理动作。

## 2026-10-19 Optuna原生完成证据回传

原生三臂完成、正式时间门失败后，28份必要载荷264917字节按[清单](../reports/documentation/20261018/optuna_implementation/repair/native_completed/native_receive_manifest.json)SCP回传并逐项核大小/SHA；运输ZIP188386字节，SHA256 1211a27666bb6b2b0bbd5708dbe9e1d35ac8d0a4c15a6bb2fc7e5d6cf88d1b0e。Windows统一保管在repair/native_completed，清单相对该目录；Linux原件在repair，清单相对该目录。此映射保留原始qualification/native_wait_status.json启动快照，完成快照另存，不覆盖旧dispatch_manifest绑定原件。48科研来源、2调度源码和4门依赖均与原门一致。

原生结果、手写诊断矩阵、临时状态身份/退休原件、控制台、外层计时及完成核对齐全；不含正式数据/标签、模型或Memory载荷。现场三臂临时权重无残留、active_wait.py已自删、PID963024退出，复现源码继续保留，不重建监听。两端临时native_receive.zip在验收后精确清理；必要清单和原件保留，新材料仅本地Git，不改变大清理周期。

## 2026-10-19 Optuna修复、CPU与Linux一次性监听

修复使用独立`/home/yongpeng/cross-lingual/reports/documentation/20261018/optuna_implementation/repair/workspace`；qualification与workspace并列。48份科学来源及2份调度源码逐项两端一致；52份历史已开放结果和8份直接来源核验，既有BGE只建立获准原生所需链接，未同步正式文本/标签或旧Memory。CPU和部署13个载荷311018字节按[接收清单](../reports/documentation/20261018/optuna_implementation/repair/receive_manifest.json)核验；attempt07_sources.zip是必要源码冻结快照，继续保管，非冗余运输包。

当前CPU/native门、批准及修订主审在repair目录两端保管；[实际监听回执](../reports/documentation/20261018/optuna_implementation/repair/dispatch.json)和四载荷[清单](../reports/documentation/20261018/optuna_implementation/repair/dispatch_manifest.json)已回传核验。活动脚本为Linux qualification/active_wait.py，PID963024；Windows scripts/step28_optuna_wait.py保留复现源码。活动脚本在确认首次实际更新后自删，原native_dispatch及日志保留，不重复部署。误建应用内跟进已删除，具体状态只看真实Linux进程及status/result。

临时deployment/native_deployment/dispatch及两份接收ZIP在验收后按精确路径清理；原实现外审输入ZIP因必要来源重放仍保留，不改冻结审查原件和失败账。可变主目录docs在Windows维护，不声称全仓同步；科研来源和必要证据已在独立Linux目录保管。新材料仅本地Git，不重置10月23日大清理周期。

## 2026-10-19 Optuna实现外审完整回收及主审

[回收核验](../reports/documentation/20261018/optuna_implementation/review/recovery.json)：返回ZIP297884字节/58成员、SHA256 f7632fd38f714f0634d013e19b890152a8f1a5d9604268081383927c400bd216；58外部成员、56内部载荷及原140输入回执一致，解包1061502字节。完整外审原文/findings/阅读范围/机械证据/六次失败/辅助原意见均保留，外层身份/成员清单和完成网页齐全。主审及待确认修复范围保存在同review目录。68必要文件1196148字节按[sync](../reports/documentation/20261018/optuna_implementation/review/sync.json)同步Linux同相对目录并逐项核对；不执行外审程序或科研测试，不覆盖隔离workspace。

CPU计数由主审补正为39生产+6参考=45实际更新，82秒预算不变，旧报告/gate原件保持。当前46源码在Windows/Linux一致，00:16:54无本用户Optuna任务。返回整包和浏览器下载/日志重复件在完整验收后清理；原2011821字节输入包仍用于待修复证据重放，仅本地保留。外审专用页已确认无生成/待发送并关闭，新材料仅本地Git，不改变10月23日大清理周期。

## 2026-10-18 Optuna实现与手写CPU

新隔离目录为`/home/yongpeng/cross-lingual/reports/documentation/20261018/optuna_implementation/workspace`；仅46必要源码/合同/政策/批准材料，无正式数据/模型链接。6次CPU原件（含attempt02失败与修复）及各次源码tar快照共49文件1422971字节已SFTP回传，47载荷SHA逐项一致，当前46来源与Linux一致，见[回传记录](../reports/documentation/20261018/optuna_implementation/receive.json)。[CPU汇总门](../reports/documentation/20261018/optuna_implementation/qualification/cpu_gate.json)累计82/300秒、7项适用检查；临时手写模型/Memory已由测试目录清理，既有正式权重/Memory不接触。23:07:05现场无相关作业。

Windows一次性deployment目录在核对当前Linux来源后已按解析绝对路径精确删除；必要源码留主目录，各次冻结快照及成功/失败证据保留。新实现外审包和清单见[package.json](../reports/documentation/20261018/optuna_implementation/review/package.json)，用户已批准上述具体2011821字节/141成员包，23:19:35单次上传并提交[GPT 6 Pro实现外审](../reports/documentation/20261018/optuna_implementation/review/submission.json)；活动输入包字节不变，模型选择/实际请求/页面/回执在同目录保管，专用页等待完整回收。新材料仅本地Git，局部副本清理不重置10月23日大清理周期。

## 2026-10-18 Optuna方案外审回收及主审

已完整回收260847字节/22成员外审返回包；22外部成员、20内部载荷、内嵌原包及25输入载荷身份一致，见[回收核验/保管映射](../reports/documentation/20261018/optuna_plan/review/recovery.json)。保留21必要外审原件220049字节、外部身份/成员清单、最终网页、[主审](../reports/documentation/20261018/optuna_plan/review/primary_disposition.zh.txt)及[待确认实施稿](../reports/documentation/20261018/optuna_plan/implementation_scope.zh.txt)。唯一原186342字节输入ZIP继续保留；冗余返回ZIP、浏览器重复包/清单、内嵌重复输入及下载临时记录共10文件按精确路径核对后清理，实际字节数见recovery。外审机械脚本如需重放，原输入路径映射已记录，不修改外审原件。专用页已确认无生成/待发内容并关闭。

F05计费规格已补清，F06资源可行性未建立；本地预算补充和新实施稿尚未送外审。无新科学执行或Linux同步，材料仅Windows/本地Git；原proposal/ZIP/提交时点记录及已有依赖安装事实不追改。本次局部冗余清理不重置项目大清理周期。

## 2026-10-18 Optuna静态方案待审材料

后续用户指出时间资格与集中评价计费两项缺口，已另存[预算补充](../reports/documentation/20261018/optuna_plan/budget_amendment.zh.txt)，仅Windows保管/本地Git，尚未外发；原26成员输入包及原稿不改。当前9000秒/114小时可行性未建立，集中评价归已有3300秒共享段，详见补充。未新增Linux执行/同步/数据读取或依赖变更。

三臂[具体方案及独立主审](../reports/documentation/20261018/optuna_plan/proposal.zh.txt)、请求和25载荷清单保存在Windows；静态ZIP186342字节/26成员，SHA256 60a248b5599f96e3a944a530fa6818e31d8bfc90133e74a0d7634fd0b9fd81cd，逐成员身份已核对，见[包记录](../reports/documentation/20261018/optuna_plan/review/package.json)。用户已批准具体包，21:18:18单次上传GPT 6 Pro，见[提交回执及页面](../reports/documentation/20261018/optuna_plan/review/submission.json)；活动外审输入保留，不复制权重、Memory、正式文本/标签或旧完整外审返回包。用户另行授权缺失时自行安装Optuna；既有Linux py310现为Optuna 4.6.0，原137包版本不变。安装前后清单、依赖解析/安装报告与回执共11文件159833字节在Linux的reports/maintenance/20261018/optuna_install及Windows同相对路径保管，已[SFTP回传逐文件身份集合核验](../reports/maintenance/20261018/optuna_install/receive_verification.json)。无科研执行，新材料仅本地Git，大清理周期不变。

## 2026-10-18 S_pos/H_pos实现、CPU与实现外审

用户随后明确：“给我把此次训练的权重删掉”。20:46:16—20:46:17已精确删除本次A_S_pos/B_S_pos/A_H_pos/B_H_pos四权重共5217586212字节；先核对四项原端点SHA、最终路径/属主/无运行依赖并回传删除前清单，再删除。models现为空，257个非权重作业文件逐项大小/SHA不变，四Memory共1381626字节保留；见[删除前身份](../reports/maintenance/20261018/positive_ranking_weights/predelete.json)及[实际回执](../reports/maintenance/20261018/positive_ranking_weights/receipt.json)。检查中两次系统会话/proc权限失败发生在删除前，根因与定向处置保留，不属科研失败。冻结审查/20:35观测保持当时事实；新许可覆盖本次四权重，不扩展其他资产。删除记录两端保管，局部清理不重置项目周期。

结果外审收尾：6340418字节/93成员返回ZIP完整回收，92内部载荷及原输入388载荷身份相符；保留92个非重复外审原件5306388字节、最终页面、主审与[回收记录](../reports/seller_alias_continual/20261018/positive_ranking_result/review/recovery.json)。必要审查材料和当前state经SFTP同步原20261018_133500/workspace/reports/result_sync/review及既有state路径，逐项回读见[同步记录](../reports/seller_alias_continual/20261018/positive_ranking_result/review/sync.json)；主目录可变docs仅Windows维护，不声称全仓同步。20:35:34[现场](../reports/seller_alias_continual/20261018/positive_ranking_result/review/closeout.json)核对40来源/完成/访问不变，无相关作业，四权重5217586212字节及四Memory1381626字节保留，未载入/下载/删除。

完整回收及主审处置后，专用页已关闭；精确删除返回ZIP、浏览器重复ZIP、内嵌重复输入和临时控制台日志，合计18191777字节，身份/不存在复核及唯一输入路径映射见recovery。唯一5481875字节获准输入ZIP仍供外审源码精确重放，仅本地一份；冻结原报告/审查原件/失败保留。本次局部清理不重置10月23日大清理周期，新材料仅本地Git。

正式完成后回传：255份必要原件8685752字节及7份Linux独立分析572387字节已逐项核对，见[完整回传](../reports/seller_alias_continual/20261018/positive_ranking_result/receive_verification.json)和[分析账](../reports/seller_alias_continual/20261018/positive_ranking_result/analysis_wallclock.json)。19:32现场四权重5217586212字节、四Memory1381626字节仍为yongpeng普通文件，work为空，原三进程退出；未下载/载入/删除。详细报告及接收记录留Windows并同步原workspace/reports/result_sync，分析源码同步原workspace/scripts；当前可变state两端按旧SHA核对后更新，见[同步核验](../reports/seller_alias_continual/20261018/positive_ranking_result/sync.json)。已验收的4163123字节传输ZIP两端用后删除；此为局部冗余清理，不重置10月23日周期。用户批准5481875字节/389成员、SHA256 4eeeb5df0e93b6c32bf352f42b656b945a8d4e4f71078b73fa4df0e015ae63f7的结果包后，19:43经本地Playwright单次提交GPT 6 Pro，见[提交原件](../reports/seller_alias_continual/20261018/positive_ranking_result/review/submission.json)；唯一输入包及活动网页保留至完整回收/主审，准备时点报告与package原件不追改。新材料仅本地Git。

原生通过与正式启动的新增保管：原生11份必要小证据及8份启动/门禁/执行原件共19文件1000572字节，已按[来源清单](../reports/documentation/20261016/positive_ranking_implementation/qualification/receive_native.json)SFTP回传并核对大小/SHA。原生两臂临时work已由原入口清理，身份原件保留；正式14:14:43启动，14:15:03确认首个实际更新后Linux监听源码自删，Windows复现源保留。14:55原正式进程继续运行，正式模型/Memory/盲分数为活动资产，未下载或删除；本次不新增未完成训练的进度快照文件，只更新唯一当前入口。新材料仅本地Git，局部临时助手清理不重置大清理周期。

最新cuBLAS复审已完整回收：66必要原件970337字节、外部身份、完整正文及独立主审保留；唯一输入ZIP仍供精确审查重放。新执行workspace为`/home/yongpeng/cross-lingual/reports/documentation/20261018/positive_ranking_execution/20261018_133500/workspace`，163必要源码/资格/外审/许可/等待脚本1982089字节[SFTP回读一致](../reports/documentation/20261016/positive_ranking_implementation/qualification/native_deployment.json)。40来源与CPU05一致，12预训练文件仅链接；52开放旧对照SHA核对，7正式输入仅查存在。正式59逐文件链接只会在原生真实通过后建立，不暴露旧Memory/test/owners目录。唯一监听788758已在新目录等待资源；活动源码必须留到首个正式更新确认后自删，Windows保留复现源。原生临时权重按原合同成功后释放，正式新权重不沿用H_cons删除许可。已验收返回ZIP、下载网络临时记录及重复页面共11文件855168字节按精确路径清理，66审查原件和必要页面/身份保留；当前许可/部署/队列状态6文件98717字节[同步回读一致](../reports/documentation/20261016/positive_ranking_implementation/qualification/execution_sync.json)，主目录可变state先精确比对旧SHA再同步，新state两端相同。13:45:24[收尾现场](../reports/documentation/20261016/positive_ranking_implementation/qualification/closeout_observation.json)仍为原监听等待、GPU空闲10862MiB，无原生/正式输出。局部清理不重置10月23日周期。

cuBLAS修复后再审：用户明确要求继续外审后，新[审查包](../reports/documentation/20261016/positive_ranking_implementation/review/workspace/package.json)459440字节/111成员已单次上传原GPT-6 Pro会话并于12:27:57确认接收。该活动输入包、请求、111成员身份、模型选择器及接收凭据保存在Windows的review/workspace；当前state/导航更新也仅本地Git，未声称已把本轮新提交状态同步Linux。原CPU05及前轮完整外审/主审的Linux保管继续以已有sync_workspace.json为准。本轮无新增科研执行、没有重建监听；保留活动会话等待审查。准备页面与自动生成的重复快照/控制台日志核对无独立用途后清理，必要接收原件保留；局部临时清理不重置大清理周期。

修复版外审回收与处置：原会话62成员返回包189487字节、62外部清单/61内部载荷及94输入身份全部核对，保留必要62原件574338字节、完整正文和独立主审，见[回收记录](../reports/documentation/20261016/positive_ranking_implementation/review/repair/recovery.json)。接受唯一新增R03后，仅补cuBLAS专用释放和一个资源测试中的状态/反例；独立attempt05 workspace部署41文件701596字节，CPU1项通过，外层计3秒、累计52/300，新增真实optimizer0。5份新原件7689字节SFTP回传；78份必要外审/主审/资格/状态661752字节同步Linux逐项回读一致，见[同步回执](../reports/documentation/20261016/positive_ranking_implementation/qualification/sync_workspace.json)。主目录state由实际旧SHA精确匹配后更新；12:16:12现场无相关作业，旧来源/失败和权重/Memory边界保持。专用审查页确认完整最终答复、stop0且无待发内容后关闭；14份无用返回ZIP、页面/取回临时文件和本轮日志共598133字节已精确删除，必要原件/唯一输入包保留。新科学代码留独立Linux workspace，不覆盖旧运行。当前导航文档仅本地Git维护；本次局部清理不重置10月23日大清理日期。

11:26:26重新送审：按用户明确指令，将修复后的唯一394296字节/94成员包及必要背景提交原GPT-6 Pro会话，见[提交回执](../reports/documentation/20261016/positive_ranking_implementation/review/repair/submission.json)。原生成已停止且停止状态已确认；部分审查原文和旧输入包保持原字节，不冒称完整回收。新包、成员清单、原/新四文件差异及提交/模型选择器证据保留Windows，供当前审查与随后回收验收；新提交元数据和当前状态更新尚未同步Linux，本轮未连接Linux或进行科研计算。已保留必要页面证据后，只清理本轮产生的8份冗余页面快照；活动审查页及其控制台保留，不动其他浏览器活动。局部清理不重置10月23日大清理周期。

11:14:56增量收尾：用户指出A阶段空缓存评分与第二臂自身CUDA上下文预检后，最小修复在独立`cpu/20261018_111500/workspace`完成四项CPU核验，外层19秒、累计49/300秒；旧失败和原五项成功原件不覆盖。8份新小原件9497字节已SFTP回传核验，见[接收](../reports/documentation/20261016/positive_ranking_implementation/qualification/receive04.json)。6份必要增量资格/部署/接收/局部主审/页面观测及当前状态共45089字节、另2701字节[同步记录](../reports/documentation/20261016/positive_ranking_implementation/qualification/sync_repair.json)已回读一致；主目录可变state在原/新SHA精确匹配后更新并再次SFTP核对。现场无相关作业，SSH关闭，4份一次性助手/清单与1份冗余页面快照已删除；当前审查页及控制台保持。外审尚无完整报告/附件交付，原333034字节包继续保留原字节，四文件修复没有外发。无GPU、正式数据或模型/Memory操作，不改变大清理日期。

三轮CPU分别位于Linux的`reports/documentation/20261016/positive_ranking_implementation/cpu/20261018_095200/workspace`、`20261018_095700/workspace`、`20261018_100100/workspace`，各自部署41份必要文件并SFTP回读一致，失败来源未覆盖。17份必要CPU成功/失败原件共39033字节已回传Windows，逐项身份及实际Linux恢复路径见[接收映射](../reports/documentation/20261016/positive_ranking_implementation/qualification/receive_verification.json)。两次失败各计3秒、成功计24秒，累计30/300秒；不能用成功覆盖失败账。10:16:51现场无属主相关核验/训练进程，attempt03输出目录存在。

15份必要授权、CPU验收/接收映射/状态和实现审查元数据共61933字节，以及3957字节[同步记录](../reports/documentation/20261016/positive_ranking_implementation/qualification/sync.json)，均按项目相对路径SFTP同步Linux主目录并逐字节回读一致。CPU运行原件保留各独立工作区；当前可变主目录docs只在Windows维护，不把本次小证据同步写成全仓同步。没有正式输入链接、BGE/旧Memory载入、GPU或正式训练，H_cons此前权重删除事实不改。

用户批准的333034字节/75成员[实现包](../reports/documentation/20261016/positive_ranking_implementation/review/package.json)仅上传chatgpt.com一次；首次会话发送403未被接受，正常页面恢复后从资料库复用同一上传文件，10:11:34建立一个有效[审查任务](https://chatgpt.com/c/6ad42ab0-aee0-83eb-b53a-81e1302fb008)。包仍有当前审查/回收身份核对用途，唯一Windows原件及必要请求、选择器和接受证据保留，未向Linux补传ZIP。保留活动审查专用页与当前控制台日志；SSH已关闭，4份用后的一次性助手/传输清单与12份冗余页面快照已精确删除，共16文件184786字节，未清理科研原件或模型/Memory。局部收尾不重置10月23日大清理周期，新材料仅本地Git。

## 2026-10-16 S_pos/H_pos静态外审回收

388411字节/47成员返回ZIP、内部46载荷和外部47项TSV均逐项核验；原22输入成员与两补充原件保持。完整报告、读取范围、14发现、手写核验源码/输入/结果/预执行身份/日志/执行账等22必要返回成员保管在reports/documentation/20261016/s_improvement/review/external。外审没有连接项目服务器或执行项目科研，本地主审也未重跑其算例。

验收后删除返回ZIP、浏览器副本、内嵌25重复输入/补充及冗余页面观测；保留原输入ZIP作为冻结源及原字节重放依据，两个补充恢复指向H_cons原审查文件，见[回收记录](../reports/documentation/20261016/s_improvement/review/recovery.json)。不为清理向Linux补传输入ZIP。必要返回、主审、拟实现范围及state按[同步记录](../reports/documentation/20261016/s_improvement/review/sync.json)保管，可变主目录docs仍仅Windows维护。20:14:26现场H_cons仍完成，models空、两Memory在、访问账未变且无相关属主进程；见[现场核对](../reports/documentation/20261016/s_improvement/review/closeout.json)。局部整理不重置10月23日大清理周期，新材料仅本地Git。

## 2026-10-16 H_cons权重删除与新方案送审

用户明确要求先删除此次试验权重，再上传S_pos/H_pos方案静态外审。19:31:39现场确认原正式作业已完成、访问账train1/valid1/heldout0/owners0、无相关属主进程和两权重打开句柄；最终路径为原20261016_124900/workspace/reports/job/run/models下A_H_cons.pt、B_H_cons.pt，两者均为yongpeng的普通单链接文件。逐文件路径/大小/SHA及158份非权重文件身份先存[删除前记录](../reports/maintenance/20261016/h_cons_weights/predelete.json)，并在删除前SFTP回传逐字节核对。

19:31:41精确删除上述两文件，各1304398010字节、合计2608796020字节（约2.43GiB）；models为空，158份非权重作业文件大小/SHA不变，含两Memory共691141字节。既有临时完整状态路径原已不存在；权重未下载/备份/载入，Memory未反序列化。实际[删除回执](../reports/maintenance/20261016/h_cons_weights/receipt.json)两端保管，冻结报告/旧回执保留其历史保管事实，不追改；哈希不能恢复权重，重训仍须授权。

之后于19:36:28单次上传原152565字节/22成员S_pos/H_pos输入ZIP，并附刚关闭H_cons外审正文和主审处置，见[送审记录](../reports/documentation/20261016/s_improvement/review/submission.json)。新审查进行中，原输入及必要页面/请求证据保留，未为删除向Linux补传输入ZIP。当前state和必要新送审元数据按[同步记录](../reports/maintenance/20261016/h_cons_weights/sync.json)SFTP逐字节回读核验；可变主目录docs仍仅Windows维护，不把证据同步写成全仓同步。一次性助手和冗余浏览器观测用后清理；局部清理不重置10月23日大清理周期，新材料仅本地Git。

## 2026-10-16 H_cons结果外审回收与处置

3536092字节/96成员返回包已完整验收，95内部载荷及原275输入身份一致。完整正文、必要统计/源码/命令/失败/补丁及回收清单保留；[独立主审](../reports/seller_alias_continual/20261016/h_consistency_result/review/primary_disposition.zh.txt)已完成，原有效未达目标结论不变。95个非重复返回成员共1352398字节在review/external；审查材料/处置及state按[sync.json](../reports/seller_alias_continual/20261016/h_consistency_result/review/sync.json)SFTP同步原20261016_124900/workspace/reports/result_sync/review，各文件读回字节一致。主目录可变docs未同步，不把本轮证据同步写成全仓同步。

验收后删除返回ZIP、浏览器下载副本、内嵌重复输入与冗余页面快照，精确身份/字节及canonical恢复路径见[recovery.json](../reports/seller_alias_continual/20261016/h_consistency_result/review/recovery.json)。保留唯一原输入ZIP供外审脚本精确重放，不为删除补传历史包。原专用页已关闭。19:11现场仅核对模型/Memory元数据，确认两模型2608796020字节、两Memory691141字节仍留原目录；未下载/载入/删除。局部清理不重置10月23日大清理周期；仅本地Git，无公开push。

## 2026-10-16 H_cons完整结果与独立分析保管

正式运行已完成。158份必要保存分数/指标/查询/计数/诊断/更新日志/44来源及外层账共4700169字节SFTP回传至reports/seller_alias_continual/20261016/h_consistency_result，逐项大小/SHA一致，见[inventory](../reports/seller_alias_continual/20261016/h_consistency_result/inventory.json)与[回传核对](../reports/seller_alias_continual/20261016/h_consistency_result/receive_verification.json)。7份Linux独立分析共386764字节及清单已回传核验；分析源码、报告及回传记录保管在原20261016_124900/workspace/reports/result_sync，当前state两端一致。

18:14:05现场原三PID均退出、无failure，正式completion/访问账未变。两推理权重各1304398010字节、合计2608796020字节；两Memory为342189/348952字节、合计691141字节，继续留Linux原job/run/models与memory，未载入、下载或删除，见[收尾](../reports/seller_alias_continual/20261016/h_consistency_result/closeout.json)。旧H删除许可不扩大到本轮。2176647字节h_cons_result_transfer.zip已在双方必要成员验收后删除；不是正式科研证据删除。具体结果外审输入包3187528字节/276成员随后获准并完成审查；唯一原ZIP仍用于外审verify_input_identity.py的成员/CRC/原字节复核，保留Windows一份，未向Linux补传输入包。局部清理不重置10月23日大清理周期。

## 2026-10-16 H_cons原生重核完成与正式启动

原生attempt03完整通过后，按冻结生命周期保留四次写入的路径/大小/SHA/状态身份，并清理common.pt、paired.pt、H_inference.pt三个最终临时路径，work目录已不存在；原生输出峰值约8.488GiB。见[临时状态身份](../reports/documentation/20261016/h_consistency_implementation/qualification/gpu/attempt03/result/temporary_checkpoint_identity.json)及[实际收尾](../reports/documentation/20261016/h_consistency_implementation/qualification/objective_closeout.json)。10份必要小原件630147字节逐项SFTP核验保留。原BGE未改，先前失败不追改。

本次5个已用部署/回传ZIP共258966字节，两端按相同身份精确删除，见[传输清理](../reports/documentation/20261016/h_consistency_implementation/qualification/objective_transport_cleanup.json)。没有为清理补传历史归档。CPU原件/冻结来源及外审必要材料保留。正式作业15:30:59开始，活动权重/自身Memory及正式输入7文件精确链接继续保留，原旧作业权重删除决定不扩展到新活动模型。本次局部清理不重置10月23日大清理日期。

## 2026-10-16 H_cons合并外审与原生失败收尾

合并外审完整回收及独立主审处置后，删除返回ZIP、浏览器重复下载及内嵌重复输入，共3文件1757356字节；93个非重复载荷、审查正文/主审/清单和唯一用于冻结源码复核的review_input.zip继续保留。见[回收记录](../reports/documentation/20261016/h_consistency_implementation/review/recovery.json)，没有以补传Linux作为删除前提。

获准唯一原生恢复attempt02在3次更新后失败，累计GPU90秒，无后续恢复许可或正式依赖。按合同第8节，回收五份小失败原件并确认作业退出后，14:42:54删除唯一手写common.pt完整状态3904681029字节（约3.64GiB）；路径/大小/SHA、五原件删除前后不变与空work见[实际收尾](../reports/documentation/20261016/h_consistency_implementation/qualification/recovery_closeout.json)。原BGE、CPU/失败/审查证据保留，未创建备份或搬走大载荷；SHA不能恢复已删模型。局部清理不改变项目大清理周期。

## 2026-10-16 H_cons静态审查收尾

完整回收返回ZIP289811字节/26成员，解压472481字节；大小/SHA与页面交付一致，25内部载荷和原输入16成员/15载荷均核对。必要审查原文、数学源码/手写输入/梯度数组/执行日志等25个独立成员344227字节保存在reports/documentation/20261016/h_consistency/review/external；[独立主审](../reports/documentation/20261016/h_consistency/review/primary_disposition.zh.txt)已完成，支持准备受限合同，未授予新执行权限。

按既有清理规则，验收与处置后精确删除返回ZIP及浏览器下载副本、内嵌重复原输入、7份预览和2份本轮已用日志，共12文件814195字节，路径/大小/SHA及实际不存在核验见[回收记录](../reports/documentation/20261016/h_consistency/review/recovery.json)。保留唯一原review_input.zip 128254字节，因为外审evidence/reverify_input.py直接复核其ZIP成员/CRC和原字节；该脚本的external/input/review_input.zip相对路径若将来需要，在独立临时重放副本按canonical原字节恢复即可，不改原脚本、不永久保留重复包。页面正文、选择器/原发送证据、主审与清单保留，专用页已关闭。

本轮仅从网页回收至Windows并作静态主审，未连接Linux、未补传任何包、未修改权重/Memory或旧预算。无公开push许可，仅本地Git；本次局部清理不重置10月23日两端大清理周期。其他用户浏览器活动相关日志不在本次清理范围。

## 2026-10-16 H结果外审回收、处置与静态新包

H结果[外审](https://chatgpt.com/c/6ad18187-f8c0-83eb-8fdc-73440d50847b)已完整回收，359571字节返回ZIP的48成员2476525字节均验收，内部47载荷及原219输入身份一致。[独立主审](../reports/seller_alias_continual/20261016/g_half_result/review/primary_disposition.zh.txt)无开放科学阻断，说明文字缺陷以附注关闭；正式结果/原验收不改。55份必要审查原件、主审及收尾文件共2526229字节SFTP同步至原20261015_235900/workspace/reports/result_sync/review，逐项大小/SHA相同；另同步[sync.json](../reports/seller_alias_continual/20261016/g_half_result/review/sync.json)9872字节并核验。没有补传历史包、原输入ZIP或任何模型/Memory。

原专用页关闭，48必要成员保留于本地review/external和Linux对应位置。验收与处置完成后，删除冗余返回ZIP及旧网页/预览共11文件710467字节，详见[回收及清理记录](../reports/seller_alias_continual/20261016/g_half_result/review/recovery.json)。唯一2798254字节原输入ZIP仍用于外审reconcile_summary_v2.py的--archive精确成员/CRC身份复核，保留Windows一份，有具体用途；不以复制Linux作为删除前提。10:28现场models为空，两Memory690219字节SHA不变，H权重已删事实及预算保持。SSH关闭后删除本轮6份本地助手/传输清单38558字节，远端一次性_verify.py 7852字节亦已删除；不将这些新临时文件计入历史释放量。

新的[H_cons静态包](../reports/documentation/20261016/h_consistency/review/package.json)128254字节/16成员已获具体许可并单次发至chatgpt.com；原方案和15载荷身份冻结。连接错误后沿同会话恢复正在处理，专用页保持，详见[当前状态](../reports/documentation/20261016/h_consistency/review/submission.json)。本包尚有当前审查用途，继续本地保管；无新实现/数据/CPU/GPU/训练，不上传到Linux，不公开push。本次局部收尾不重置10月23日大清理周期。

## 2026-10-16 Linux冗余归档与ER0.1权重清理

用户要求“看看服务器上的项目里有什么无用文件给我删掉，腾出点磁盘空间”，随后针对旧保留例外询问“ER0.1权重能删掉吗？”。按本轮清理范围核对实际依赖：ER0.1于10月3日已完成训练、评价和结果外审／主审，后续对比使用保存的evaluation／collected等证据；现行合同无加载这六份末端权重的安排，无活动作业或精确目标打开描述符。原“保留例外”状态已据本次明确对象与当前依赖重新判断，不按结果正负决定删除。三个共享首域完整状态仍另行保留。

10:09:37在Linux项目内精确删除63文件、8,038,703,295字节（约7.49GiB）：ER0.1的ABC／BCA／CAB各stage2／stage3共六份推理权重7,838,864,652字节（约7.30GiB）；22份同SHA重复ZIP 57,304,763字节；35份已用完的传输／部署／外审返回ZIP 142,533,880字节。后两类共57份199,838,643字节（190.58MiB）。删除前已核对最终路径、属主、单链接、大小／SHA及全部保留依赖，将[删除前身份与逐成员保留映射](../reports/maintenance/20261016/linux_cleanup/predelete.json)SFTP回传核验后再删。精确lsof未返回打开目标，其他用户受限FUSE挂载警告留原记录，不声称全系统可见。

[实际删除回执](../reports/maintenance/20261016/linux_cleanup/receipt.json)确认63路径均不存在、ER models为空；273份ER非权重文件大小／SHA不变，含三份Memory载荷608,088字节及其预算记录；2,031个归档保留对象大小／SHA不变，15份大模型／共享状态的元数据不变，盘点内其余66,208文件元数据无变化。预训练、三个共享首域完整状态、两份排序例外、所有必要结果／冻结来源／审查原文、Memory及正式／封存输入均保留。未载模、未解析Memory或新标签、未重算科研指标，也未补传历史归档或新建权重备份。

本次删除中，文件系统可用空间由27,296,722,944增至35,335,581,696字节，差8,038,858,752字节，与逻辑文件大小分开。随后删除6份本轮临时盘点／助手19,926,535字节（这些是本轮新生成材料，不另算历史空间收益），现场free为35,355,496,448字节，约32.93GiB；共享磁盘后续仍可能变化。必要两份记录在Linux／Windows逐项SHA相符。已删整包的必要内容按predelete映射定位保留文件，22份精确副本可从对应canonical ZIP恢复原字节；35份返回／部署ZIP只保留必要成员，不承诺重建原ZIP压缩字节。未通过完整成员保管核对的归档及当前G_half审查输入仍保留。

旧冻结报告、上传包、历史清单中的“ER六权重保留”是此前时点事实，不追改；当前保管以本次回执为准。SHA不能恢复六份已删权重，将来若需要重新载模，须另定重建范围与授权。SSH已关闭，6份本地一次性助手／传输清单11,444字节亦已删除；仅保留两份必要记录。当前交接已改为删除状态。本轮为用户指定Linux清理，不重置10月23日两端大清理周期；仅本地Git，无公开push许可。

## 2026-10-16 G_half完整结果与外审保管

唯一正式20261015_235900/workspace于10月16日01:40:56完整结束。147份小结果/日志/33冻结来源共4102971字节、8份Linux独立分析证据375924字节已SFTP回传逐项大小/SHA相符，详见[接收记录](../reports/seller_alias_continual/20261016/g_half_result/receive_verification.json)及[原始逐文件身份](../reports/seller_alias_continual/20261016/g_half_result/inventory.json)。本地完整结果目录为reports/seller_alias_continual/20261016/g_half_result；Linux原件仍在该workspace/reports/job，新增独立分析和报告在其reports/result_sync。不回传正式文本/标签、权重或Memory。

[09:42现场](../reports/seller_alias_continual/20261016/g_half_result/closeout.json)确认无原三PID/相关训练及failure；两份推理权重A_G_half.pt/B_G_half.pt各1304396666字节，共2608793332字节（约2.43GiB），两份Memory分别341728/348491字节，共690219字节，仍留Linux。此为现场文件大小核对，未载模/反序列化；预期SHA及恢复证据保留原manifest。这是删除前观测，随后用户明确“推理权重给我删掉，没用的话”。本次核对训练/两恢复/保存分析完成，外审不依赖载模，无已批准后续模型加载或恢复需要；先回传[删除前身份](../reports/maintenance/20261016/g_half_weights/predelete.json)并核对SHA，再于09:53:18—09:53:19删除这两份权重2608793332字节，models现为空。147份其余文件大小/SHA全部不变，含两Memory690219字节；未下载或备份权重，详见[实际删除回执](../reports/maintenance/20261016/g_half_weights/receipt.json)。文件系统free由24709599232增至27318407168字节（约25.44GiB），逻辑释放与文件系统观测分开。两份回执在Linux/Windows均保留，逐项SHA一致；SSH关闭后另删除3份本次权重清理助手/清单1488字节。最初通用进程fd盘点遇到旧sd-pam受保护目录后停止且未删除，改为精确路径lsof核对，受限的其他用户FUSE挂载警告原样保留，不冒称全系统可见。冻结原报告/送审包保管状态不追改。

用户已单独批准2798254字节/219成员结果包（SHA256 c1831a58ba41c7ff46c77b782aad9789730d234e9bf50faa836094cd23cc3880），09:45通过本地Playwright单次发送，结果外审正在进行，见[发送](../reports/seller_alias_continual/20261016/g_half_result/review/submission.json)。唯一输入ZIP、成员清单、请求、授权、可见模型与发送页面保留Windows用于本次审查验收；未向Linux补传输入ZIP或历史归档。仅本次报告/接收回执及外审元数据7文件91144字节同步至result_sync，逐项SHA一致，见[同步回执](../reports/seller_alias_continual/20261016/g_half_result/sync.json)。

SSH已关闭；10份已用一次性传输/打包助手及清单102134字节、6份冗余浏览器快照68558字节已实际删除，共16文件170692字节。必要Linux分析脚本/成功原件/全部小结果、当前专用审查页及活动浏览器日志保留；外审验收/处置后再按实际用途处理包，不提前删除必要输入。局部清理不重置10月23日大清理周期，新材料仅本地Git，无公开push许可。

## 2026-10-15 G_half实施与CPU增量保管

独立Linux workspace为reports/documentation/20261015/g_half_implementation/20261015_224000/workspace。必要CPU失败/成功、两轮源码身份、预算、43份旧结果身份核对等27文件111662字节已SFTP回传逐项一致；[原始回传](../reports/documentation/20261015/g_half_implementation/qualification/receive_verification.json)及[增量验收](../reports/documentation/20261015/g_half_implementation/qualification/cpu_receipt.json)保留首轮错误和累计17/180秒。两轮快照中64份与当前33项来源相同的重复文件已在Linux逐项核验后删除，释放1064841字节，仅保留两份不同的attempt01旧源码及原始失败/成功证据，见[清理记录](../reports/documentation/20261015/g_half_implementation/qualification/duplicate_cleanup.json)。清理前91文件的inventory保留原观测事实，不当作当前存量。

具体322949字节/78成员合并包获准后于23:02单次上传网页，提交时仍在审查（现已完成，见下段），原输入包当前保留用于回收及后续身份验收。只有本轮CPU验收、请求和发送证据7文件25577字节同步至独立workspace，逐项SHA一致，见[同步](../reports/documentation/20261015/g_half_implementation/review/sync.json)；未向Linux补传输入ZIP或历史归档。Windows/Linux当前33来源均仍与CPU第二轮一致，23:02现场无G_half进程。已关闭SSH连接，删除8份一次性传输助手/清单及7份冗余页面快照，共15文件120016字节；保留必要发送页面，活动审查专用页及当前浏览器日志留待审查验收收尾。无模型/Memory清理，不重置10月23日大清理周期。

G_half合并外审已完成验收/独立主审：49份必要返回原件及外部清单/交付回执、页面、主审保留。删除已无独立用途的361104字节返回ZIP、重复下载回执/清单和本轮临时页面/日志，共9份836379字节，详见[回收记录](../reports/documentation/20261015/g_half_implementation/review/recovery.json)。唯一原输入322949字节ZIP仍为外审audit_identity.py精确身份复核输入，留Windows，不向Linux补传。原生新workspace只同步本轮89份必要部署载荷及一份部署清单，逐项核对；旧CPU原件留原独立目录，不复制64份已删重复快照。

原生6次资格完成后12份原件136105字节已SFTP回传逐项一致，含实际诊断/恢复身份、原始计时与日志；完整墙钟采用320/600秒。两完整工作状态及一推理状态已按批准生命周期在Linux释放，work目录实查不存在。正式workspace仅部署40必要载荷及一部署清单；43旧结果与7正式输入用逐文件链接，无旧Memory目录暴露，无历史归档补传。正式活动模型/Memory留Linux，最终两推理权重预计共2608757722字节约2.43GiB，不沿用旧六权重删除许可。启动/部署/首观测及访问账7份35673字节另回传保管，00:03:45当前观测1450字节亦回传核对。收尾关闭SSH，删除10份已用一次性传输助手/清单共54461字节；正式训练由已启动独立进程继续，不受SSH关闭影响。

## 2026-10-15 G_half静态外审保管

本次返回ZIP227026字节/23成员完整回收并逐项验收，独立主审及具体合同准备已完成。必要审查原文、结构化发现、手写核查源码/输入/输出、真实失败/成功日志、阅读范围和清单保留在reports/documentation/20261015/g_refinement/review/external。验收后删除返回ZIP、返回的173263字节重复输入及本轮浏览器临时项，共11文件590958字节；逐项身份及保留用途见[回收记录](../reports/documentation/20261015/g_refinement/review/recovery.json)。

仅保留一份原review_input.zip（173263字节，SHA256 d36f32f225023aeecfe184eb2220c402e4c29d857bf2bcdee05d66ecb5b938cb），因外审check_static.py明确按该压缩原字节/SHA读取并复核；重放可用--input-zip指向这一原件，默认运行助手的input路径可按同字节恢复，无需永久保留重复副本。未向Linux补传历史归档，未删除旧科研证据/模型；专用页已关闭。局部清理不重置10月23日大清理周期。

## 2026-10-15 完整S/G/F结果保管

新作业20261015_131600/workspace于18:40:15 exit0。347份结果/日志/冻结来源共11,094,881字节及7份Linux独立分析程序/表/图/原始输出共491,617字节已SFTP回传，逐文件大小/SHA核验；见[回传记录](../reports/seller_alias_continual/20261015/balanced_pooling_result/receive_verification.json)。原始证据继续留Linux实际workspace，分析和报告另存reports/seller_alias_continual/20261015/balanced_pooling_result；不复制正式文本/标签、模型或Memory到Windows。

用户随后明确：“此次实验的训练权重给我删除”。本次六份新权重A_S/B_S/A_G/B_G/A_F/B_F已于20:24:27—20:24:28删除，共7,826,218,590字节（约7.29GiB），实际models目录为空。删除前核对无活动作业/打开的权重、六端点恢复已完成、当前保存结果分析/外审无载模依赖，逐文件最终路径/属主/单链接/大小/SHA与原记录一致；[删除前身份](../reports/maintenance/20261015/balanced_result_weights/predelete.json)先经SFTP回传核验。349份非权重文件大小/SHA全部不变，包括六Memory共2,071,353字节、完整结果/日志/来源/访问与计费账；临时完整work为空。文件系统可用空间由19,553,755,136增至27,380,006,912字节（约25.50GiB），见[实际删除回执](../reports/maintenance/20261015/balanced_result_weights/receipt.json)，两份记录均在Linux/Windows同相对目录核验保存。未下载或备份权重，不涉及原预训练或其他作业。原[观测](../reports/seller_alias_continual/20261015/balanced_pooling_result/observation.json)、报告和送审包的权重留存状态保留为删除前事实；SHA不能恢复权重，未来重训仍须授权。两端各一个5,766,787字节结果传输ZIP和234,948字节分析传输ZIP已在成功回传核验后删除；必要原件和清单保留，见[收尾](../reports/seller_alias_continual/20261015/balanced_pooling_result/closeout.json)。未补传历史归档，局部清理不重置周期。

结果外审及独立主审现已完成。完整回收的39份外部原件（1,574,378字节）、页面正文、交付身份/清单、主审、18行损失遗忘补正及当前状态共49份/1,632,314字节已通过SFTP同步并逐项大小/SHA一致，当前状态只写运行目录外current_state.json；见[同步记录](../reports/seller_alias_continual/20261015/balanced_pooling_result/review/sync.json)。[20:57现场核对](../reports/seller_alias_continual/20261015/balanced_pooling_result/review/closeout.json)确认完成/访问/31来源不变、无相关活动进程、models为空、六Memory身份相同。外审包内旧保管状态不追改。

按网页外审包清理规则，345,535字节返回ZIP在39成员验收并保留后删除，避免重复副本；6,123,032字节原输入ZIP仍由外审原始核算程序直接按大小/SHA绑定为复算输入，保留唯一Windows原件及具体用途，未上传Linux或补传历史归档。详细身份及处置见[回收记录](../reports/seller_alias_continual/20261015/balanced_pooling_result/review/recovery.json)。专用网页已关闭；不删除必要失败证据，不新建权重备份或科研运行，本次局部清理不重置周期。

## 2026-10-15 失败S/G/F旧权重清理

用户明确要求：“上次训练失败产生的训练权重文件你删了吗？没删的话就删掉并做好记录”。对象仅为Linux旧作业20261014_143200/workspace/reports/job/run/models下A_S.pt、B_S.pt、A_G.pt、B_G.pt、A_F.pt五份权重；此前仍保留。已批准的新完整重训从固定预训练独立起步，不加载旧失败模型／Memory；旧作业无可精确恢复至B/F第247步的完整Adam状态，无当前或已批准后续载模依赖。

删除前现场确认属主yongpeng、最终路径、单链接、五份大小／SHA与原points记录一致、旧作业无活动进程及打开的目标文件；[删除前身份](../reports/maintenance/20261015/balanced_pooling_weights/predelete.json)先经SFTP回传并核验同字节。北京时间13:47:12—13:47:13逐份删除，每份1,304,369,701字节，合计6,521,848,505字节（约6.07GiB逻辑大小）；五路径均不存在，models目录为空。实测文件系统可用空间由25,453,264,896增至31,975,141,376字节（约29.78GiB），变化6,521,876,480字节；与逻辑大小的差异源于块分配等，分别记录，见[删除回执](../reports/maintenance/20261015/balanced_pooling_weights/receipt.json)。两份记录在Linux同相对目录及Windows留存，回传大小／SHA一致。

旧job其余207份文件大小／SHA全部不变，包括失败、来源、配置、盲分数、日志及五份Memory（共1,722,720字节）；未解析Memory或盲结果。FAIL_NO_RETRY、train1/valid0及旧正式18,428.570150秒账不清零；原冻结报告和清单的“权重留存”保留为历史时点事实。当前20261015_131600作业三进程身份／启动时刻／cwd／命令均未变，原预训练及其他作业权重保留，没有权重载荷回传或备份。SHA不能恢复已删权重；局部清理不重置大清理周期，不扩展公开push。

## 2026-10-15 预算增量与训练启动临时包

用户明确免除本次27小时预算/资格继承的补充外审并要求删除包。未上传的299199字节输入ZIP、请求、输入清单和包元数据共4文件315940字节已删除，具体决定见[免审原话](../reports/documentation/20261015/balanced_pooling_repair/runtime/review_waiver.json)。两轮CPU及正式部署的本地上传ZIP副本共3文件779321字节也已删除；删除前再次核对Linux实际大小/SHA，原件分别保留在`/home/yongpeng/cross-lingual/reports/seller_alias_continual/20261015/balanced_pooling_repair/20261015_130100/source.zip`、`20261015_130500/source.zip`及同日期`balanced_pooling_execution/20261015_131600/source.zip`。CPU 4+6秒的失败/成功账、必要原始输出、当前源码/差异、批准/主审和正式准入/进程证据保留。没有为清理补传历史档案，不清理活动训练产物或其他权重；合计7文件1095261字节，仅本地冗余/未使用材料，局部清理不重置清理周期。

## 2026-10-14 本地快照清理

用户明确要求清理 `reports/seller_alias_continual` 无用快照，并指出“不要补传”“没用就删掉”。本次未向Linux补传任何历史归档，也未删除或改动Linux项目文件。原拟补传15份归档的调用被自动审批拒绝，未执行；用户随后明确否决补传，今后不从本次清理要求推导补传授权。

目标目录实际删除526个文件、460,782,500字节（439.44MiB），由1,043,163,605降至582,381,105字节（555.40MiB），减少44.17%。其中49份ZIP为已现场核对Linux原件大小／SHA一致的本地副本，477份为与完整原包成员逐字节SHA一致的解包数组、测试载荷及压缩源码副本。报告、主审处置、原始日志、可读统计表、正式结果与正式冻结来源保留；没有解析正式文本／标签、Memory或模型。

已删ZIP均可从Linux项目 `/home/yongpeng/cross-lingual/` 下原相对路径恢复，唯一异路径恢复项是 `20261001/bge_continual_result/review/bge_continual_result_review.zip`，对应同结果目录的 `review/external/input/bge_continual_result_review.zip`。已删解包载荷在该轮完整外审ZIP内保持原字节：原包在Linux已有同字节归档，或属于仍留本地的14份完整ZIP。冻结历史清单不改写为当前在盘清单，恢复时查包内原成员；未重算任何科研指标。

另清理S/G/F CPU attempt03本地5份冗余文件、181,941字节：源码ZIP（Linux已有同字节原件）、与attempt04完全一致的actual_paths／all_shapes／single_graph_vjp三个输出及空launcher。保留实际失败日志、结果／来源身份、运行脚本和包装耗时；失败原因是测试mutation lambda旧签名，不是科学负结果。91／300秒是累计预算用量，含失败轮向上取整的8秒，账不清零；失败原件不等于每份重复载荷永远保留。

合计移除531份本地文件、460,964,441字节（439.61MiB逻辑文件大小）。这是工作目录释放量，不包含Git历史或LFS对象库；采用普通本地删除提交，不改写历史、不扩大公开push。局部清理不重置上方两端大清理周期。

## 2026-10-14 C_q四臂：结果已回传，五权重已留证删除

固定作业20261013_151500/workspace于10月13日22:55:47正常完成。306份结果／日志／资格小证据共7,760,771字节、从实际workspace另补的29份冻结来源505,635字节及8份Linux保存结果分析472,248字节均经SFTP回传并逐项大小／SHA核对；两端结果归档为reports/seller_alias_continual/20261014/query_replay_result，见[回传核对](../reports/seller_alias_continual/20261014/query_replay_result/receive_verification.json)。原job/run/sources为空，未将首次传输虚称包含源码；补源身份单列于source_manifest.json。完整训练状态已由正式入口按恢复与分支依赖释放，五份推理权重的后续分析／外审均无载模依赖。

首次删除被工具自动审批以缺少本次具体授权拒绝，实际未删；用户随后明确回复“批准删除这五份权重（推荐）”，对应[授权](../reports/maintenance/20261014/query_weights/authorization.json)。[删除前身份](../reports/maintenance/20261014/query_weights/predelete.json)先回传核验，再次确认五文件最终路径、属主、大小／SHA、无活动作业及打开描述符、247份非权重SHA一致后，于10:35:52—10:35:53只删除A_shared、B_S、B_Q、B_C_q、B_C_b，共6,521,871,122字节（约6.07GiB逻辑大小）。五路径均不存在，models为空，247份非权重SHA前后不变，[删除回执](../reports/maintenance/20261014/query_weights/receipt.json)已回传核验。

五份Memory共1,734,069字节仍仅留Linux原workspace/reports/job/run/memory，未反序列化、同步载荷或上传；预训练及其他作业文件不在删除范围。原报告／清单／获准上传包保存形成时字节，“仍保留／待许可”是删除前事实，当前保管以本次回执为准。SHA不能恢复已删除权重，未来重建不在现许可内；局部清理不重置大清理周期。

## 2026-10-13 C_cf三臂：结果已回传，四权重已留证删除

本次固定A→B于10月12日19:53:39正常完成；280份结果／日志／冻结来源／资格证据5,765,044字节和10份Linux分析397,368字节已回传且实际大小／SHA匹配。两端归档为reports/seller_alias_continual/20261013/crossfit_retention_result，见[回传清单](../reports/seller_alias_continual/20261013/crossfit_retention_result/inventory.json)和[分析清单](../reports/seller_alias_continual/20261013/crossfit_retention_result/analysis_manifest.json)。临时完整训练状态已由正式入口在恢复核对及分支依赖解除后释放，身份保留在points。用户随后明确“记得及时删除此次训练产生的模型权重文件”。本次核对四端点完整／推理恢复、评价和保存结果独立复核均已完成；当前网页外审实际使用保存矩阵／分数与源码，现行合同无后续载模用途。现场无该workspace活动进程、打开的模型文件或待恢复完整状态。逐份核对最终路径、uid、大小／SHA与points一致，先将[删除前记录](../reports/maintenance/20261013/crossfit_weights/predelete.json)回传Windows并核验同字节，再于10月13日09:57:18—09:57:19删除A_shared、B_S、B_C_cf、B_C_perm四份推理权重，共5,217,548,401字节（约4.86GiB逻辑文件大小）。四路径均不存在、models目录为空，225份非权重文件SHA前后不变，见[删除回执](../reports/maintenance/20261013/crossfit_weights/receipt.json)。6Memory共1,646,210字节及F/O审计评分继续仅留Linux原20261012_140200/workspace/reports/job/run；未反序列化、回传载荷或上传。预训练及其他作业的保留例外不在本次删除范围。原报告、清单和上传包保留形成时字节，其“权重留存”表述属于删除前事实；SHA不能恢复已删模型，后续重建需另定范围。此次清理由实际依赖决定，局部清理不重置大清理周期。

## 2026-10-11 三臂归因：结果已回传，27权重已留证删除

LOGIT0.1／R0／S的20261009_200600/workspace于10月11日00:24:33正常完成。1,234份小结果、日志、冻结来源及资格证据共38,049,242字节已SFTP回传且逐文件大小／SHA一致；两端归档为reports/seller_alias_continual/20261011/record_attribution_result。Linux保存结果分析与图表共11份亦已回传核对；见[完整分析](../reports/seller_alias_continual/20261011/record_attribution_result/report.zh.txt)、[原始同步清单](../reports/seller_alias_continual/20261011/record_attribution_result/sync_manifest.json)、[分析清单](../reports/seller_alias_continual/20261011/record_attribution_result/analysis_manifest.json)。

用户本次明确要求“确认外审已开始后，核查linux服务器上此次的训练权重是否有用，若后续不再用到，按之前的规定删除”。恢复现有网页会话后已确认外审实际开始，并从网页下载附件逐字节核对为原授权包；没有重复提交。训练、27阶段恢复核对和162项诊断均已完成，当前分析／外审依赖保存分数、统计及冻结来源，现有合同与后续设计没有使用本次阶段权重的安排，现场没有该workspace活动进程、打开的模型文件或待恢复完整检查点。此判断依据实际依赖，不依据结果正负。

先逐份核对最终路径、uid、大小／SHA及阶段身份，并把[删除前记录](../reports/maintenance/20261011/record_attribution_weights/predelete.json)回传Windows核验一致；随后于10月11日10:41:20—10:41:24删除本次27份阶段推理权重，共35,237,126,610字节（约32.82GiB逻辑文件大小）。27路径均不存在，原models目录为空；1,206份非权重证据SHA和23项保留例外元数据不变，见[删除回执](../reports/maintenance/20261011/record_attribution_weights/receipt.json)。BGE预训练、旧三个共享首域完整状态、ER0.1六权重及排序两例外继续保留。

本次27份Memory共8,389,549字节继续保管于原reports/job/run/memory，历史36份Memory也保持原保管；未反序列化、未回传载荷、未上传网页或Git。原[实物清单](../reports/seller_alias_continual/20261011/record_attribution_result/inventory.json)保留删除前身份，不能据其认定27权重仍存在；SHA不能恢复已删模型，未来重建须另定范围。有效结果、冻结来源、诊断和完整指标均保留，局部权重清理不重置大清理周期。

## 2026-10-09 C_plus负结果：两端候选活动实现已撤下

用户明确“撤下候选及五臂入口，保留通用诊断（推荐）”。Windows／Linux项目主目录各删除四个五臂专用入口、核验、shell及测试文件，共享记录重放模块恢复原C/S字节；原C/S、LOGIT和通用诊断保留。冻结合同／policy／建议、两份历史workspace、结果source、原结果和原始审查作为科研记录保留，不再作为活动候选入口。详见[关闭记录](../reports/maintenance/20261009/replay_improvement_closure/closure.zh.txt)及同目录两端回执。Linux1,626份非权重证据SHA及23项模型例外元数据不变，36份Memory仍在；局部代码撤下不重置大清理周期。

Memory是六群有界历史重放状态，含允许的训练文本／标签、教师目标、抽样及附属元数据；36份是27个记录表及9个LOGIT阶段快照，共11,479,595字节（约10.95MiB），不是36个模型或可并入训练的全历史库。原job/run/memory仅Linux保管，未因本次代码清理删除，未上传网页或Git。[结果主审](../reports/seller_alias_continual/20261009/replay_improvement_result/review/primary_disposition.zh.txt)已接收本配置负结果并登记LOGIT历史复现缺陷R1；数值根因未隔离，不以收尾宣称修复。

## 2026-10-09 五臂改进：36权重已留证删除，Memory保留

20261008_151407/workspace正式作业于10月9日13:38:39正常结束。1,591份小结果／冻结来源／日志及资格证据共51,293,424字节已SFTP回传并逐项大小／SHA一致；分析和具体结果外审包亦保留，见[分析](../reports/seller_alias_continual/20261009/replay_improvement_result/report.zh.txt)及[回传清单](../reports/seller_alias_continual/20261009/replay_improvement_result/sync_manifest.json)。这些冻结原件记录的是删除前状态，excluded_payloads不再代表36权重仍存在。

用户随后明确：“模型训练权重确认无用后先保留相关记录再删掉”。本次核对全部训练、恢复和诊断已完成，保存结果分析及已授权外审无载模／恢复依赖，LOGIT历史复现问题当前依靠冻结来源、保存矩阵与更新记录核对；没有获准的后续训练／推理依赖。先逐份验证实际路径、归属、大小、SHA及状态摘要与原manifest一致，保存并回传[删除前记录](../reports/maintenance/20261009/replay_weights/predelete.json)，再于14:48:49—14:48:54删除原job/run/models内36份阶段推理权重，合计46,976,558,793字节（约43.75GiB逻辑文件大小）。实际36路径均不存在，目录剩余0；1,626份非权重证据SHA及23项保留例外元数据不变，见[删除回执](../reports/maintenance/20261009/replay_weights/receipt.json)。共享磁盘空闲变化另存，不全部归因于本次清理。

36份Memory共11,479,595字节继续保留原job/run/memory；只核对字节身份，未反序列化或解析标签，不进入网页/Git。BGE预训练、旧三个共享首域完整状态、ER0.1六权重及排序两份例外保留。删除依据是已结束且无当前依赖及本次明确许可，不是结果正负；权重删除时外审尚未关闭，随后结果处置见上节，LOGIT复现问题及原性能结论不因删除消失。SHA与记录不能恢复已删模型，未来载模须另行确定重建范围。局部权重清理不重置上方项目大清理日期。

## 2026-10-08 记录作用表C/S：15权重已删除

用户明确“训练权重可以删掉吧，如果可以保留对应记录后删掉，删完后提交外审”。20261007_225409训练与全恢复已完成，保存结果独立核验通过，后续结果外审无加载/恢复依赖；09:43:06—09:43:08已逐份删除本次15个阶段推理权重，逻辑释放19,565,561,724字节（约18.22GiB），实际路径/uid/大小/SHA及不存在验证见[删除回执](../reports/maintenance/20261008/record_weights/receipt.json)。224份非权重SHA及23项保留例外元数据前后不变；239份小结果/来源等导出载荷已回传一致，15份完整Memory仅留Linux且不入网页/Git。BGE预训练、三个旧共同首域完整状态、ER0.1六权重和两份排序例外保留。

清理依据是已结束且无当前模型依赖，不是效果正负；C对LOGIT0.1的正向结果及C/S未通过整体原判据均保留。原合同/清单的保管句和权重身份作为历史事实不改，当前本次15份按最新指令已不存在；SHA不能恢复已删权重，未来新训练仍须另行确定范围。结果外审与主审已于10月8日关闭，接收有效固定配置开发结果；见[主审处置](../reports/seller_alias_continual/20261008/record_replay_result/review/report.zh.txt)，主分析与全部指标见[完整结果](../reports/seller_alias_continual/20261008/record_replay_result/report.zh.txt)。

## 2026-10-06 19:53 修改版关闭，九权重已删除

用户最新明确授权删除修改版代码及此次权重。20261006_135937九份.pt已全部删除，实际逻辑释放11758415337字节；Linux34个活动／部署文件清理，Windows10个活动文件删除、共用核验脚本恢复原版。315项科研证据SHA不变，23项保留例外元数据不变。九份Memory和冻结源码／审查包仅保留为记录；无后续加载／恢复依赖。详见[关闭记录](../reports/maintenance/20261006/relation_revision_closure/closure.zh.md)和[删除回执](../reports/maintenance/20261006/relation_revision_closure/receipt.json)。下方本日较早“九权重保留”是清理前历史事实，现已被本记录取代。

本页维护具体保管位置、删除与恢复事实；常设归档／清理规则统一见[科研纪律](RESEARCH_DISCIPLINE.zh.md#9-文件归档与清理)。当前权重按依赖和恢复需要保留，结果正负不直接决定删除；两份排序例外和活动状态保持。下方各日期记录只证明当时状态，恢复前须结合后续删除记录核对实际文件，不按旧清单假定仍有原件。

## 2026-10-06 修订关系目标记忆：结果回传、九权重保留

本次20261006_135937已完成。168份小结果／日志／冻结来源已回传并逐项核对，31正式来源一致；九份本次阶段推理权重共11758415337字节，Linux现时SHA与端点记录一致，仍留独立workspace/reports/job/run/models。Memory九份留同作业run/memory，身份与大小见[清单](../reports/seller_alias_continual/20261006/relation_revision_result/inventory.json)，未载入／下载正文。结果外审与主审尚待完成，旧作业九权重删除许可不移用；不因负结果直接删除。见[结果分析](SELLER_ALIAS_RELATION_REVISION_RESULT.zh.md)。

## 2026-10-06 关系目标记忆：结果审查关闭，九份权重已清理

用户明确授权收尾后清理及后台条件执行。本轮外审／主审已关闭，无后续加载或恢复依赖；10月6日10:39:18完成Linux独立运行20261005_150520的九份阶段推理权重删除，共11,758,382,199字节（约10.95GiB，逻辑文件字节）。删除前逐文件归属、最终路径、大小／SHA均与原清单一致，删除后九路径均不存在；165份源码／配置、完整指标／分数／计数、Memory原始证据、日志及运行回执SHA不变，23项保留例外元数据不变。见[逐文件清理回执](../reports/maintenance/20261006/relation_weights/receipt.json)及[外审／主审处置](../reports/seller_alias_continual/20261006/relation_result/review/report.zh.md)。

预训练、三个共同首域完整状态、ER0.1六权重及两份排序例外继续保留。清理依据是收尾后无依赖，不是固定点的负结果；完整结果、原结论、冻结源码和外审原件保留。权重SHA不能恢复已删载荷，重训仍须另行授权。原inventory保留删除前身份，不按旧保管记录假定权重仍存在。

## 当前结果保管（2026-10-04）

10月4日11:13:39，用户在明确本次两实验12份末端推理权重的范围、依赖和不可直接恢复后回复“那就删”。Linux已实际删除LOGIT0.1六份与risk六份，共15,677,742,102字节（约14.60GiB）。删除前逐份最终路径、归属、大小和SHA一致，删除后12路径均不存在，两个job内511份非权重文件SHA不变。三个共享首域完整状态、两份排序例外、ER0.1六份权重及BGE预训练文件的元数据保持。当前比较和结果外审使用保存分数／矩阵，不依赖这些末端权重；未新增载模、标签解析或训练。保存哈希不能恢复已删模型，旧保管清单仅保留历史事实。见[逐文件清理回执](../reports/maintenance/20261004/logit_risk_weights/receipt.json)。下方10月3日保管及既往清理均为对应时点记录。

19:04:45补充：用户明确要求删除Linux LOGIT0.25六份末端推理权重，实际删除7,838,884,668字节。删除前逐项大小/SHA匹配，删除后179份相关分数、指标及证据SHA不变。当前LOGIT0.1与风险候选恢复共同首域完整状态，比较使用保存结果，不依赖这六份末端权重。原保管清单保留当时事实，但该六份载荷现已不存在，不能按旧清单恢复。详见[删除回执](../reports/maintenance/20261003/logit_quarter_weights/receipt.json)。ER0.1六权重、三个共享首域完整状态和排序两例外继续保留。

ER0.1及匹配LOGIT0.25的正式小结果分别270份8,721,549字节、198份7,785,058字节已回传Windows并逐文件核对。两组各6份新推理权重保留Linux原job/run，合计15,677,749,320字节，10月3日实际流式SHA与端点记录一致；本次未载模或删除这些权重，供结果审查和本轮收尾使用。逐文件位置、大小/SHA见[ER0.1保管](../reports/seller_alias_continual/20261003/er_low_result/payload_custody.json)及[LOGIT保管](../reports/seller_alias_continual/20261003/logit_weight_result/payload_custody.json)。这不改变下方49份旧权重的删除事实，未来保留继续按实际依赖判断。

2026-10-02权重保管更新：用户要求清理近期用不到的权重，并明确“保留s0候选和对照2份，删除其余16份（推荐）”。14:57:31已完成49份64,017,044,472字节清理：排序16、首轮持续学习21推理副本、ER0.5/0.25共12。保留三个首域完整model/Adam/RNG状态及排序s0 C/A第6轮两份；活动0.1目录、全部预训练档案/缓存及793份非权重证据保持，后者SHA复核一致。空间释放后可用约673.15GB。旧18份全保留要求在这16份上被本次明确答复覆盖；历史合同/决定字节和正负结论不改，不能把被删权重都称负结果。见[清理与依赖记录](../reports/maintenance/20261002/unused_weights/report.zh.md)。

后续λ=0.1及匹配LOGIT恢复原三个完整首状态，比较对照读取保存分数/指标，不再加载被删旧推理文件。已删除载荷不能按旧归档清单直接恢复；全部必要小结果和原结论仍可追溯。本节覆盖下方历史段落中当时的全18份保留及21推理文件保留状态。

## 2026-09-30 已完成清理

用户要求检查并清理Windows与Linux两端的过时／无用文件。本轮按实际用途、保留要求和字节核对分类，未按文件年龄批量删除科研材料。Windows实际删除58份原有文件、112,594,084字节（约107.38MiB）：30份历史输入嵌入数组的冗余副本、24份浏览器下载／回执／日志／页面快照、4个内容已全部保留的旧中转包。30份数组共106,032,896字节，在删除前已逐份核对当前Linux原件大小／SHA；原路径继续保留在`/home/yongpeng/cross-lingual/`，需要时按原相对路径SFTP回取并复核散列，不重建字节。必要清单见[删除计划及恢复位置](../reports/maintenance/20260930/project_cleanup/deletion_plan.json)、[Linux实物核验](../reports/maintenance/20260930/project_cleanup/linux_verification.json)、[Windows删除回执](../reports/maintenance/20260930/project_cleanup/windows_cleanup.json)。

Linux实际删除1个已解包且源码原件完整的中转包、18,784字节（约18.34KiB）。另外3个中转包在Linux仍包含当地独有的历史源码／状态字节，保留，未覆盖同名但不同的后续记录。[Linux删除回执](../reports/maintenance/20260930/project_cleanup/linux_cleanup.json)。上述是删除文件的逻辑字节数，不将系统／共享磁盘可用空间变化全部归因于清理；本轮临时扫描清单及辅助脚本已完成收尾，不计入原有垃圾释放量；完整完成状态见[completion.json](../reports/maintenance/20260930/project_cleanup/completion.json)，它取代较早summary中的待办字段。

原生预训练模型、数据档案、有效正负结果、原始外审证据与独有旧源码继续保留。`private_custody/`未进入；私有／封存资产未迁移或删除，未解析正式标签、加载模型、改Git／LFS或触碰其他用户文件。历史嵌入仅按原始字节核对，没有载入数组开展研究。

**9月26日“Linux训练权重为0”只是当时状态。** 当前9月27日排序实验的18份推理权重共23,516,221,872字节按9月28日用户明确例外继续保留，另有持续学习已完成运行目录中的必要共享／对照状态，均不得当临时文件删除。清理后18份冻结科学来源SHA保持一致；18份保留权重大小／mtime／inode未变，Windows其余10,921文件、Linux非活动49,842文件的元数据检查无意外差异。这不是对所有载荷重新计算SHA。见[Windows核查](../reports/maintenance/20260930/project_cleanup/windows_postcheck.json)、[Linux核查与训练观测](../reports/maintenance/20260930/project_cleanup/linux_postcheck.json)。

## 9月26日训练权重清理（历史）

用户明确要求先清理本轮负结果权重，再删除Linux本项目全部训练所得权重，覆盖此前原D和历史有效实验权重保留安排。两项已完成：先删除10份13,067,720,890字节，再删除剩余105份111,917,493,162字节；累计115份124,985,214,052字节（约116.40GiB逻辑文件大小）。每份实际路径、大小与SHA在删除前记录；相关676份非权重文件元数据不变，当前轮95份证据删除后重新核对SHA不变。Linux训练所得权重剩余0；13份预训练载荷15,102,857,017字节及其配置、数据、分数、归一化记录、日志和指标保留。没有处理其他用户进程、封存资产或Git LFS。

Linux原D也已删除；当前只能将D作为已存结果与配置基线，不能再按旧归档回执加载其权重。旧迁移清单中这些训练权重的恢复路径已失效；保留当时清单为历史，不宣称本次删除有字节恢复能力。未来重建须新授权。Windows没有本轮五次新训练权重副本，删除0字节；后续扩大请求只针对Linux，42个其他历史Windows小检查点未处理。

清单／回执：[本轮实物核验](../reports/seller_alias_continual/20260926/pooling_result/weight_cleanup_inventory.json)、[本轮删除](../reports/seller_alias_continual/20260926/pooling_result/weight_cleanup.json)、[全量清单](../reports/maintenance/20260926/trained_weights/inventory.json)、[全量删除](../reports/maintenance/20260926/trained_weights/cleanup.json)。共享文件系统前后可用空间见实际回执；上述字节数指所删文件大小，不将所有共享空间变化归因于本次动作。

### 9月24日存储状态（以下保留为历史）

| 内容 | 位置与状态 |
|---|---|
| 项目代码、合同、文档、小型结果／审查证据 | Windows保留，按已授权范围同步Linux |
| 中文BGE预训练模型 | 仅Linux：/home/yongpeng/cross-lingual/models/step7/embeddings/bge_large_zh；12文件1,302,803,319字节 |
| 中文BGE训练权重 | Linux：reports/seller_alias_continual/20260924/chinese_execution/20260924_152642/job；训练已完成，8权重10,439,069,376字节保留，不当临时文件清理 |
| 启动证明 | Windows／Linux均保留：上述运行phase下startup_status、startup_snapshot及清单；9份52,364字节匹配，清单另计2,175字节 |
| 历史大型载荷与恢复方法 | 下方原迁移记录、实际Linux回执和各实验合同；不按同名推定字节相同 |

训练和valid评价均已完成。80份小型运行／评价文件3,776,103字节逐项回传匹配，另10份独立汇总证据160,770字节核对一致；清单另计。权重仍只留Linux。见[结果回传](../reports/seller_alias_continual/20260924/chinese_result/sync.json)和[结果报告](SELLER_ALIAS_CHINESE_RESULT.zh.md)。中文预训练模型Windows副本此前已实际核对后删除。

用户随后要求清理Git LFS：本地.git/lfs/objects原本0文件、0字节；dry-run和实际prune均成功，删除0、释放0字节。其余11个锁能力缓存共264字节保留。299 retained为工具引用保留统计，不能当成磁盘存在299个载荷。HEAD与index未变，普通Git pack约130.81MiB不是LFS，本次未改历史。见[清理回执](../reports/maintenance/20260924/lfs_cleanup/receipt.json)。

用户随后明确“Git LFS不清理了”，并要求删除工单草稿。已删除草稿，未联系Support；GitHub远端仓库／LFS没有执行删除、历史改写或其他修改，远端释放0字节。后续LFS清理已取消，见[撤销与删除记录](../reports/maintenance/20260924/lfs_cleanup/remote_status.json)。

## 既有归档与清理记录


## 2026年9月24日中文编码器归档

按用户明确要求，`BAAI/bge-large-zh-v1.5`先下载Windows，再以SFTP按原路径复制Linux。固定提交`79e7739b6ab944e86d6171e44d24c997fc1e0116`的12个文件共1,302,803,319字节；Windows先以官方LFS SHA／Git blob核对来源，Linux上传完成后重新读取全部实际文件计算大小和SHA-256，与Windows逐项相同，内容摘要`26d94141907d840eb6890ce9fa690bc3e32229f16704e7a794ae2218fd9685cd`。

归档位置为`/home/yongpeng/cross-lingual/models/step7/embeddings/bge_large_zh`。完整权重仅留Linux；Windows删除12个原文件及空模型目录前再次核对当地SHA和实际Linux回执，删除后目录确实不存在。释放逻辑文件字节1,302,803,319（约1.21 GiB），不冒称磁盘可用空间实测增量。来源记录、转存回执和外审所需的小型配置快照保留，不含重复权重。见[下载清单](../reports/seller_alias_continual/20260924/chinese_model/download.json)、[Linux实际核验](../reports/seller_alias_continual/20260924/chinese_model/linux_verification.json)、[Windows删除](../reports/seller_alias_continual/20260924/chinese_model/cleanup.json)。原中断传输及精确前缀核对记录保留；恢复传输未改模型字节。

## 2026年9月24日结果回传

基础三模型及两点敏感性结果已回传172份小型训练／评价证据，共6,180,044字节，逐项实际大小/SHA一致，见[回传回执](../reports/seller_alias_continual/20260924/result_sync/20260924_112014/sync.json)。12个推理模型共24,121,623,112字节继续位于Linux原base_execution／sensitivity_execution目录，实际哈希在评价完整门重核；没有把权重同步回Windows。Linux正式入口已删除自己的临时完整checkpoint，work为空，保留全部有效结果及历史缓存。

经用户明确恢复授权，把缺失的原valid二值CSV（2,290,729字节）按冻结SHA补齐Linux供本阶段评价；同步本身只核对字节。没有移动test／owners／Audit封存资产。基础实际valid一次、敏感性在明确补充授权后累计两次，次数详见结果报告，不由存储保管权限推导新增评价权限。

已删除两端各3个用完的传输ZIP，每端4,039,407字节；172份必要证据保留。见[Windows清理](../reports/seller_alias_continual/20260924/result_sync/20260924_112014/windows_cleanup.json)、[Linux清理](../reports/seller_alias_continual/20260924/result_sync/20260924_112014/linux_cleanup.json)。这些是文件逻辑大小，不是共享磁盘实测增量。

实际结果外审与本地主审结束后，Windows另外删除上传ZIP及5个浏览器日志／快照，共6文件4,939,700字节，并移除空浏览器临时目录。删除前逐一确认299份提交材料可通过原路径或不可变快照恢复，25份冻结来源和4份恢复来源大小／SHA保持；完整回复文本／HTML、主审记录、核验命令／结果、请求、授权／拒绝和HTTP403历史均保留。见[外审清理](../reports/seller_alias_continual/20260924/result_review/cleanup.json)。本轮没有删除有效科研结果、缓存或Linux模型；外审后资料通过SFTP按原路径同步，回执见[同步记录](../reports/seller_alias_continual/20260924/result_review/sync.json)。

2026-09-19再次按用户要求盘点Windows／Linux项目临时文件及科研产物，两端均未发现新增可确认丢弃的文件，本次删除0文件、0字节。Windows浏览器目录为空、未发现项目命名的临时文件；Linux六个已结束实验work为空，/tmp与/var/tmp未发现属于当前用户且以本项目命名的临时项。剩余压缩包属于外审原件或历史源码／状态快照；模型目录.cache中的小型下载来源元数据、有效负结果和唯一归档检查点保留。未加载模型、解析正式标签、操作封存资产或修改活动实验。此盘点不是对全部历史实验的重新科学评价；详见[本次汇总](../reports/maintenance/20260919_161100/summary.json)。下方为既有实际删除与归档记录。

## 2026年9月18日两端临时副本清理

按用户明确要求检查训练并清理无用文件，Windows实际删除16文件、3389085字节（约3.23 MiB），Linux删除5文件、42069字节（约41.1 KiB）。这是所删文件的逻辑大小，未将共享磁盘可用空间变化当作本次释放量。

Windows清理了9个浏览器日志／快照／重复下载、2份已完成外审的上传包，以及5份已解包的结果传输包。Linux清理同类5份结果传输包；47个成员均逐一核对了保留文件的实际大小和SHA-256。上传请求、清单、完整审查回复、主执行者核验及解包结果继续保留。含独有旧源码或历史状态的包不因扩展名为ZIP而删除，历史审批拒绝记录仍保留原义。

盘点未确认可删除的正式科研结果：Linux大型模型是历史有效实验的保留检查点，包含有效负结果和比较参照；六个已结束实验的work目录已为空。Windows剩余大型私有／封存资产仍遵守原保留范围。本次没有加载模型、解析实验标签、重新评判所有历史实验或修改活动队列。基础模型9份与敏感性21份批准来源均核验一致，去重并加监听、放行记录后的27文件在清理前后未变。

逐项删除路径、原SHA及恢复位置见[Windows临时清理](../reports/maintenance/20260918_150800/windows_cleanup.json)、[Windows传输包清理](../reports/maintenance/20260918_150800/windows_archives_cleanup.json)、[Linux解包成员及保留路径](../reports/maintenance/20260918_150800/linux_plan.json)、[Linux清理回执](../reports/maintenance/20260918_150800/linux_cleanup.json)和[汇总](../reports/maintenance/20260918_150800/summary.json)。下文9月14日记录为既有大载荷归档历史。

2026-09-14 用户要求尽量把 Windows 大型数据集、模型和 checkpoint 迁到 Linux；远端已有且核验一致的直接清理 Windows 副本，并明确列为长期科研纪律。此项是文件归档与副本清理，不改变科学结论、实验合同或标签权限，按现行纪律免外审。

Windows 根目录：`C:/Users/35734/Downloads/2018-10-05-grams-finalexports`。
Linux 归档根目录：`/home/yongpeng/cross-lingual`。两端保持清单中的相对路径。

## 当前清单与执行状态

> 以下为2026-09-14迁移的历史原文，后续措辞“当前保留位置”只指该时点。所述12份旧文本权重后来已按9月26日许可删除，不能据本段从Linux恢复；参见本页“9月26日训练权重清理（历史）”及[清理入口说明](LINUX_CLEANUP.zh.md#当前清理与保留入口)。现在的资产及执行状态见本页顶部和当前交接。

状态：**归档、两端核验、Windows科研载荷清理及临时文件收尾均已完成。最后两个Windows临时文件由用户手动删除，执行者已核对均不存在。**

最初盘点只遍历元数据；执行阶段按原始字节复制和计算散列，没有解析标签或载入模型。原项目文件逻辑大小约40.18 GiB，以下41,308文件、36,663,351,030字节（34.15 GiB）已在Linux完整归档，并从Windows删除。删除阶段观察到磁盘可用空间增加36,751,941,632字节（34.23 GiB）；逻辑大小与文件系统可用空间变化分别报告。

| 类别 | 文件数 | 字节数 |
|---|---:|---:|
| `models/` 的完整预训练模型目录 | 71 | 11,134,845,748 |
| 旧文本实验 `20260907_200309` 的推理 checkpoint | 12 | 22,633,965,684 |
| 原始来源目录与根数据文件 | 41,209 | 1,957,413,921 |
| 大型公开派生数据 | 16 | 937,125,677 |

精确路径与分组保存在[原始元数据清单](../reports/storage_migration/20260914/inventory.json)。其中 `hashes_verified=false`、`deleted_files=0` 是盘点时的历史状态，原记录保持不变；当前完成状态以[两端核验](../reports/storage_migration/20260914/verification.json)、[删除回执](../reports/storage_migration/20260914/deletion.json)和[删除后核对](../reports/storage_migration/20260914/post_verification.json)为准。

远端已有90文件、12,148,939,825字节与本地原件完全一致，直接复用；缺失41,218文件、24,514,411,205字节已补齐。41,206个原始小文件通过299,423,216字节的临时压缩包传输，解包后逐文件核对；12个checkpoint直接复制。新补齐的逻辑字节量不等于压缩后的网络传输量。最终远端41,308文件全部大小及SHA-256匹配，回传核验回执SHA-256为`1601b5fe3c26fdaffb26e5da54f81bd1467b9e2703b0fa03850adf787a831c86`。本地原件的12个checkpoint散列也与原point.json一致。

Windows原生PowerShell于14:41:28—14:50:49执行逐文件删除，实际561.487秒；每项删除前再次计算本地SHA-256。41,308个路径均已不存在，6,683个未选中文件在最终文档更新前核对大小／修改时间均无变化。没有模型反序列化、标签解析或训练；冻结清单、原结论和实验目录标识保持。当前本地代码若依赖已归档路径，先按本记录恢复，不能把本地缺少归档载荷误判为实验失效。

14:52的Linux检查：整盘约3.6 TiB、已用2.5 TiB（71%）、可用1015 GiB；项目实际占用约88 GiB，其中reports约72 GiB、models约14 GiB。GPU在14:18的独立只读复查仍为内核580.173.02、系统驱动库580.178.04，NVML不匹配、PyTorch Error804，未启动训练。

原始来源范围为 `html-rips/`、`2017-12-09-grams-christopher/`，以及 `3z669jwe.sql`、`tijkc3xx.sql`、`market_item.xlsx`、`2017-12-05-philipjames11-darknetmarketplacedataagora20142015.xlsx`、`products_data.csv`。只调整原字节的存储位置，不修改、合并或再生成原始输入。公开派生项仅含清单内16文件；没有将“派生数据”解释为整个 reports 目录。

`private_custody/`、私有审计／oracle、身份资产、封存真值与 qrels 不迁移或删除；Git、代码、文档、原始清单、审查与结果证据，以及当前 Windows 评价所需的小型输入保留。其他未明确属于公开输入的历史大文件暂留，不能只按体积扩大封存访问范围。预训练模型的选择依据本次用户明确的大文件归档要求，不沿用早期无关清理任务的 models 排除范围。

## 迁移与删除方法

本阶段先只读核对资源、项目归属和远端路径；初始远端可用空间1,114,211,950,592字节。使用单连接SFTP串行复制、单进程nice15／idle I/O散列核验。原计划按35 GiB预留，实际新补齐载荷为上述24,514,411,205字节；临时包仅299,423,216字节，未使用GPU或修改共享环境。以下保留当时做法；常设要求统一见科研纪律，不把本次资源值或串行传输细节当作所有未来任务的固定预算／程序。

对每个选中文件核对本地原件和远端实际文件大小及 SHA-256；远端不存在的先按原路径复制，存在但不同的停止该项而不覆盖。核验完成后，删除前再确认本地仍为相同文件、最终绝对路径在 Windows 项目根内，采用原生 PowerShell 精确路径删除。保留逐项复制／复用、核验、删除清单及实际释放空间；中断时保留未核验原件，不能把计划数量写成已完成。

新科研输出直接留 Linux 时，只返回 Windows 审查必需的结果、分数、日志和小型记录；如需本地模型或大型输入，按清单从 Linux 恢复到相同路径并核对原散列，遵守另行适用的实验权限，用后清理多余副本。迁移本身不允许重训或重建文件来冒充原件。

## 与旧归档的衔接

9月10日的[Linux清理](LINUX_CLEANUP.zh.md)删除了旧文本实验的12个Linux模型副本，当时Windows为保留位置。该记录是正确的历史事实。9月14日已按原point.json及两端SHA-256恢复到Linux，并删除Windows副本；**当前恢复方向改为Linux→Windows，Linux为这批原模型的保留位置**。旧删除回执及冻结清单保持不变，不重新训练替代原模型。

## 临时文件收尾

Linux上的临时上传包已由执行者删除，见[远端清理回执](../reports/storage_migration/20260914/remote_cleanup.json)。Windows的`reports/storage_migration/20260914/source_upload.tar.gz`（原299,423,216字节）和`cleanup.ps1`（原4,603字节），合计299,427,819字节，已由用户手动删除；执行者随后检查两个精确路径均不存在，见[清理完成核验](../reports/storage_migration/20260914/cleanup_completion.json)。本次临时文件清理已闭环，不再列为待办。

此前两次自动删除请求及用户再次明确要求后的第三次请求，均在执行前被自动审批以`blocked by policy`拒绝；[原拒绝记录](../reports/storage_migration/20260914/cleanup.json)和[再次尝试记录](../reports/storage_migration/20260914/cleanup_retry.json)保留为历史。不得把用户手动删除记为执行者命令成功，也不根据原文件大小虚构此次手动删除的实测磁盘增量；确认时间仅代表核验时点。此前34.15 GiB科研载荷删除及其磁盘观察值不变。

## 2026-10-07 函数目标记忆固定点回传

20261006_231800正式完成；本次九份阶段推理权重按用户后续“训练权重一起删掉”于10:04:38删除，逻辑释放11758264524字节（约10.950737GiB）；逐项最终路径/uid/大小/SHA、删除时间和不存在验证见[实际回执](../reports/maintenance/20261007/function_weights/receipt.json)。原inventory保留为删除前历史清单。171份非权重原件已回传逐项一致（Memory保管副本不进入网页/Git）；完整指标/分数/计数/日志与冻结源保留。结果外审仍进行中，但只用保存矩阵/证据，无权重加载或恢复依赖；本次明确删除许可已执行。171项非权重SHA和23项保留例外元数据前后不变，共同首域完整状态、预训练、ER0.1和排序例外继续保留。
