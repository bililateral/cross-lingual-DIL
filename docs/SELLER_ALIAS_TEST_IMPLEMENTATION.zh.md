# 固定s0独立test：实现与核验记录

日期：2026-09-28。当前状态：**明确恢复后的正式test已于18:32:14 CST正常完成；原合同8/9，用户已明确接受实际提升作为持续学习研究基础。** 实现外审和本地结果核对均通过，本次结果网页外审仍待完成。首次缺件失败保留第7节，有效执行与接受决定见[完整test结果](SELLER_ALIAS_TEST_RESULT.zh.md)。下方准备、等待及首次失败为历史。

## 1. 核查的问题及实现对应

用户要求候选排序提高并保护基础识别。当前阶段用独立同源合成留出检验已固定C校准系统相对A校准系统，并保留原始A概率保护；不拟合、搜索或选择模型。校准保留C已有排名，不产生C自身新的排名收益。

| 要求 | 实际实现与证据 | 尚未验证的范围 |
|---|---|---|
| 固定s0 A/C E6及已有映射 | 运行器`historical_models`逐项核对原清单、参数摘要、映射、原阈值及参考配置；`inference_one`核对实际权重并严格恢复 | 本阶段没有重新读取两个正式权重；正式前检才验证当前远端字节 |
| 120群完整heldout，标题／描述输入 | `public_inputs`仅白名单文件；核对三域各40群、排序、28账号及商品；ID只供对应；复用未变的分通道表示和对称头 | 手工夹具已执行；正式文本未读取，原生成隔离依据沿用既有记录，不重读owners |
| 两模型盲推理完成后监督一次 | 四分数完整保存／重载、映射重放和群顺序一致后，`parse_once`独占创建访问记录，解析45,360对并逐对对齐 | 手工两模型缺失、分数损坏、映射／群错配均实际拒绝；正式链尚未运行 |
| 一次监督供四角色共用 | 同一truth用于A_raw/A_cal/C_raw/C_cal；无训练、校准拟合或阈值选择；旧自动判定直接比较原logit和原阈值 | 文件SHA读字节与CSV解析分开记账；一次解析不等于底层只读一次 |
| 四矩阵先采集后统计 | `collect`先保存四份120×22矩阵、计数与清单，`finalize`再统计；注入统计失败后用保存结果恢复，禁止重新推理／解析 | 采集自身失败可能没有四份完整矩阵，不因此自动授权重读 |
| 九项同时通过，零退化容差 | 独立逐项失败及浮点边界测试；坏校准A仍被原始A保护拒绝 | 观察值保护不等于总体非劣效，也不保护每个阈值／域／群 |
| 整群配对区间 | 固定s0，三域各40群，5000次PCG64(20260928)；独立频数实现和显式线性分位数核对 | 条件于固定模型、映射、生成机制；不覆盖重训或校准重估 |
| 沿用真实推理语义 | 未改变BGE的`eval`／`inference_mode`、bf16、microbatch4、表示和状态恢复；新包装器用真实微型Torch检查点恢复并前向，输出2.5及参数摘要一致 | 微型张量核验不是原生BGE重测；复用此前已核查的真实BGE与正式保存恢复证据 |

22指标、AP与梯形PR-AUC区分、稳定并列、1／2相关候选、饱和概率仍按已冻结定义。合同中的“尚未连接／编写”是准备时叙述，34份来源冻结后不回写；实际后续状态以本文及运行记录为准。

## 2. 实际Linux核验

已有py310，Python3.10.19、NumPy2.2.6、CPU0，三个BLAS线程变量均1，无GPU；未安装或修改环境。先检查共享资源及进程归属，只读，不干预其他任务。

| 运行 | 实际结果 |
|---|---|
| `cpu/20260928_155700` | 15项通过、0失败／错误／跳过；GNU3.23秒，退出0。目录名不是实际开始时刻，原started／finished保留 |
| 覆盖补充 | 发现漏群用例只有额外身份字段分支；保存旧测试并增加实际删群分支，仅测试文件变化，不改实验语义 |
| `cpu/20260928_155900` | 15:58:26—15:58:30 CST；15项通过、0失败／错误／跳过；unittest2.809秒，核验器2.841085秒，GNU3.26秒，退出0；最大RSS672,860KiB |

最终8份9,655字节小型证据实际回传SHA一致，34份科学来源匹配；26份继承来源未改。1小时／256MiB是CPU核验运行与证据上限，256MiB不是RSS上限。手工生成文件和微型权重在Linux临时目录测试结束自动清理；手工`heldout_parse=1`不代表正式test访问。

实际命令和日志见[启动记录](../reports/seller_alias_continual/20260928/test_implementation/coverage/launch.json)、[最终audit](../reports/seller_alias_continual/20260928/test_implementation/cpu/20260928_155900/evidence/audit.json)、[手工证据](../reports/seller_alias_continual/20260928/test_implementation/cpu/20260928_155900/evidence/handmade.json)、[回传](../reports/seller_alias_continual/20260928/test_implementation/return_sync.json)。已执行Bash语法和两入口`--help`，未跑关闭历史实验的全量回归。

## 3. 主审及历史外审对应

本人已从用户验收问题逐项追踪新方法、运行器、核验器、用例、Bash、合同及政策，并核对直接继承的模型输入、推理、恢复、校准变换、指标与bootstrap。既有校准实现和结果实际外审原文／报告、排序结果和原生恢复证据的适用范围已核对；不把旧审查当新test入口批准。未确认当前必须修改的科学阻断或复现缺陷。补充的漏群覆盖保留原因及前后字节。

完整正式120群推理、正式监督对齐、九项性能、正式峰值资源和实际test结果审查均未完成。不会由手工通过预言正式效果。来源与回传主审记录见[primary_check.json](../reports/seller_alias_continual/20260928/test_implementation/primary_check.json)。

## 4. 下一步及时间

提交合同、源码、实际CPU证据和直接历史依据至网页GPT6 Pro；取得完整实际回复后逐项主审，必要缺陷做最小修复和相关验证。闭环后新鲜检查资源，按已授权固定合同执行一次正式test，**预计10—30分钟，强制上限1小时／4GiB，单GPU单CPU，0训练更新**。尚未启动，无实际剩余时间或北京时间完成区间；外审等待不计入推理预计。完成后回传、独立核对保存结果并做结果外审；九项和有效性均通过才进入持续学习方案设计。

等待外审且无其他必要工作时主动暂停。120群以后不再称持续学习最终未开发留出，后续另立合同；18份模型继续仅留Linux，不因本步自动删除。

## 5. 实际提交记录

自动审批首次因旧授权未覆盖本次新包而拒绝上传，原记录保留。用户随后明确“允许上传本次test实现包（推荐）”；同一116来源、601,670字节包于2026-09-28 16:15:33 CST发送，16:16:01复查Pro页面显示“ChatGPT 正在回应”、空输入框和停止按钮。尚无审查结论。完整[提交回执](../reports/seller_alias_continual/20260928/test_implementation/review/submission.json)、[清单](../reports/seller_alias_continual/20260928/test_implementation/review/source_inventory.json)及批准／拒绝历史均保留。ZIP中的文档是提交准备快照，后续状态更新不回写不可变包。必要同步与临时副本清理后主动暂停。

## 6. 实际外审闭环及后台监听

用户通知外审结束后，已取得实际回复`23534074-8487-4574-960c-022fb55522c0`和390成员／25,727,000字节附件，SHA为`88e2765fa5a43f31a2e68f1ac14abfe733c9e4135c7d0f07192d4ecc5d01f927`。本人完整审读回复、F1—F10、独立v2源码、v1/v2差异、捕获与来源核验程序、原始成功／失败日志及阈值诊断。388份外部清单、389条SHA记录、116提交来源和34当前科学来源匹配。网页15个提交用例及最终10个独立用例通过；首次独立9过1败由参考自身float32阈值比较导致，明确转换float64后通过，项目代码及所有容限未改。外部还真实执行微型Torch恢复及完整继承前向，不冒称原生BGE。无确认必须修复项，不重跑未变项目CPU或修改九判据。[完整主审处置](../reports/seller_alias_continual/20260928/test_implementation/review/disposition.json)。

16:49及16:50资源检查发现GPU0由zhangdele的PID3135644使用；其Python路径`/home/zhangdele/miniforge3/envs/fmwf/bin/python`中的fmwf是虚拟环境名称。未干预该任务。用户明确要求写监听脚本，GPU空闲后自动运行已审test，无须执行者一直监听。[用户原话及具体调度](../reports/seller_alias_continual/20260928/test_execution/listener_authorization.json)。

已部署[后台脚本](../scripts/run_step28_alias_test_wait_linux_20260928.sh)，只调度原已审正式入口，不改变方法、数据、统计或资源预算，不重复送科学外审。Linux Bash语法通过；43份493,812字节来源／回执实际大小和SHA匹配。每60秒检查GPU0没有计算进程、利用率0、显存至少24GiB、主存至少16GiB、磁盘至少4GiB。满足后只调用正式入口一次；任何失败不自动重启或重读标签。等待不读正式文本／标签或加载模型。

监听于16:57:40 CST启动，PID3137352，用户yongpeng、CPU0；16:58:19确认存活，父PID1、独立会话ID3137352、stderr为空。最新资源记录仍有PID3135644、GPU利用率84%，状态`WAITING_FOR_FREE_GPU_AND_RESOURCES`；正式输出目录和heldout尝试记录尚不存在。监听已脱离SSH，关闭连接不会终止它。活动脚本及状态必须保留，正式运行结束后再清理不必要的监听文件。

输出根目录：`reports/seller_alias_continual/20260928/test_execution/20260928_165355`；监听在`listener/`，正式运行将写入`execution/`，实际模型评价输出在`execution/job/`。目录165355是准备编号，不能冒称实际开始时间。新鲜资源准入、34来源和实际CPU／外审回执在正式入口再次核对。正式test预计10—30分钟，上限1小时／4GiB；排队时长无法估计，尚无确定完成时刻。[部署](../reports/seller_alias_continual/20260928/test_execution/deployment.json)、[初始状态](../reports/seller_alias_continual/20260928/test_execution/20260928_165355/initial_observation.json)。

## 7. 首次调度失败：部署缺件，尚无test评价

2026-09-28用户要求“看看test是否已结束”。18:15:14 CST只读检查确认：监听在17:23:43满足空闲准入后调用一次已审入口，17:23:45记录退出1，GNU实耗1.48秒，最大RSS770,284KiB。原监听PID已退出，当前没有后台重试。

错误位于`public_inputs`调用`allowed_input`核对`heldout/items.jsonl`大小时，文件不存在。进一步查到`heldout/supervision/pairs.csv`也未同步。两份原件在Windows，分别4,840,862和4,581,409字节，大小及SHA与政策/生成清单完全一致；此前归档清单将其保留在Windows，43项部署没有包含它们。这是执行者遗漏正式输入同步的复现缺陷，责任在本地部署准备；不是模型的负训练结果或指标验收失败。

`failure.json`明确记录`heldout_parse_attempts=0`；调用顺序和只含failure.json的输出相互印证，尚未解析正文、恢复原生模型、做盲推理或进入标签门。正式入口曾初始化CUDA并检查资源，不能说完全没有触及GPU。没有22指标、区间或T1—T9判定。运行及监听18份20,056字节均已同步，原始退出、日志、事件和失败JSON不覆盖。

仅补齐两份原始字节，不修改34份科学来源、模型、映射、指标、阈值或数据；核对是文件传输和SHA，不是正文/标签解析。两份原件合计9,422,271字节已补齐Linux；18:23新鲜核对五份输入及34份来源大小/SHA全部通过，两权重存在、大小和归属正确（仅元数据，未加载），失败输出仍只有failure.json、heldout访问记录不存在。[复制回执](../reports/seller_alias_continual/20260928/test_execution/monitoring/input_repair_sync.json)、[远端复核](../reports/seller_alias_continual/20260928/test_execution/monitoring/repair_verification.json)。按冻结合同第6节不自动重启，在新目录恢复原范围前取得明确恢复指令。不存在重复CPU测试或重新科学外审的理由：此次修复仅为遗漏的机械复制。真实原生BGE推理和独立评价仍待正式执行，不能用外审手工核验替代。

原估总10—30分钟、强制上限1小时／4GiB；此次失败入口耗时1.48秒，已退出，剩余0。恢复后的执行预计仍10—30分钟，等待确认及资源的时间不计入，尚无确定完成时刻。所有18份权重保留，持续学习设计继续等待完整test及结果核对。

证据：[状态](../reports/seller_alias_continual/20260928/test_execution/monitoring/status.json)、[失败分析](../reports/seller_alias_continual/20260928/test_execution/monitoring/failure_analysis.json)、[回传](../reports/seller_alias_continual/20260928/test_execution/monitoring/return_sync.json)、[本地原件](../reports/seller_alias_continual/20260928/test_execution/monitoring/local_inputs.json)、[Linux缺件实查](../reports/seller_alias_continual/20260928/test_execution/monitoring/remote_inputs.json)。

## 8. 明确恢复与有效执行

用户明确“允许恢复原定test（推荐）”后，18:29新鲜核对41项来源/回执/输入身份及空闲GPU，在新目录20260928_182700于18:30:09启动原入口，18:32:14退出0。只补齐冻结字节，34科学来源未改，旧CPU/外审实证继续适用。heldout实际解析一次，两个原生模型完整120群推理及四角色指标采集均完成。原九条件8过1败，用户随后明确接受观察到的提升，保留事前判定与事后推进决定。完整指标、资源、独立核验和限制见[结果报告](SELLER_ALIAS_TEST_RESULT.zh.md)。
