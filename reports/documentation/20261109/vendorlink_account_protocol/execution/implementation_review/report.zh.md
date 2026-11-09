# VendorLink 账号监督与完整 H：定向科研实现审查报告

日期：2026-11-09（UTC+8）  
审查输入：`implementation_review_input.zip`  
审查性质：冻结科学协议下的实际实现、提供证据、运行有效性与启动条件审查。

## 一、启动结论

**结论：存在必修阻断。当前这个精确包不宜直接启动正式 D/P/M/C。**

我确认了五项具体运行控制缺口：正常原子保存可能误触发目录扫描失败；启动异常后同一轮仍可能派发下一槽；部分共享失败没有进入共享停止路径；正式入口缺少已批准的 64 GiB 启动磁盘门；辅助预算耗尽只停止新派发，不能约束继续发生的监控费用。R01、R02、R04 已用本轮无模型、无真实数据、无真实子进程的受控逻辑检查确认；共享失败分类和辅助硬帽问题依据实际可达代码路径认定，没有冒称现场复现。

**没有发现足以否定本轮完整 H 训练目标、三包当前流、Memory/出生教师、旧 V 分类头、Q2/G4 全池评价或主要科学数值门的确定错误。** 现有 CPU/GPU 证据应按未受影响范围复用；本报告不要求重做数据资格、增加 GPU 补测、增加候选/seed/学习率、扩预算或恢复旧任务。

| 编号 | 结论类别 | 必须修正的行为 | 最小处置 |
|---|---|---|---|
| R01 | 启动前必修 | 正常 `.partial` 原子替换或 `latest` 退役可使另一个 worker 的目录扫描抛 `FileNotFoundError` | 复用已有抗文件消失竞态的 `directory_bytes()` |
| R02 | 启动前必修 | 首槽启动异常后，本轮派发循环仍可启动第二槽 | 本批失败策略下立即停止后续派发；保留已派健康任务排空 |
| R03 | 启动前必修 | worker 已发现的共享磁盘/部分来源失败可能被当成普通单任务失败 | 将明确的共享失败类别传给父控；不要把全部运行异常升级为共享失败 |
| R04 | 启动前必修 | 新正式入口首槽只执行约 34 GiB 空闲门，未执行批准的 64 GiB 启动门 | 将原有 64 GiB 要求接入正式启动/相应派发路径 |
| R05 | 启动前必修 | auxiliary 硬帽只事后发现超额；活动分析承诺也未进入持续判断 | 联合核算未结父活动、活动辅助承诺和清理余量，余额不足时有界停止 |
| E01 | 来源证据闭合事项 | 后续历史评分/开发没有额外使用旧 fit 文本的断言，缺少包内可独立核对的具体来源身份 | 主执行者用已有非敏感来源绑定/核对回执补明；不重读封存材料或重跑资格 |

当前 `gate.pending.json` 的 `review_closed=false`、`review_disposition=null` 是在审期间应有的关闭状态，**不是本报告拿来代替实现审查的缺陷**。用户对本轮实施和按门运行的授权已明确；修正及主执行者独立处置闭合后，更新实际源码/回执绑定和许可即可承接已有授权，不需要重复索取同范围训练许可。

## 二、模型身份、授权与审查边界

我能自述的模型身份是 **GPT-6 Astra Pro**。我不能独立验证后台实际路由、服务部署身份或界面选择器与后台实例的一一对应关系，未声称已经核验这些事项。

本次以当前用户请求及 `execution/scope.zh.md` 第 3 行记录的新实施授权为现行范围。`protocol.zh.md` 中早期“尚未批准实施”的文字按形成时点理解，不覆盖后续明确授权。本次审查保留既审科学选择，不重新设计研究目标。

我实际完成的是附件校验、文本/源码静态追踪、已保存聚合及费用的简单算术、源码身份/差分核对、Python 语法解析，以及为定位具体控制缺陷而做的极小逻辑检查。没有读取包外项目私有路径，没有下载数据或模型，没有载入 BERT，没有运行包内 CPU 资格套件、GPU 探针、真实训练、真实指标计算或重采样。没有修改输入包或其中源码。

提供的 Linux 回执与控制台是本轮审查的证据，不能冒称为我连接项目 Linux 后重新取得的原生运行结果。

## 三、核包与来源绑定结果

### 3.1 附件身份

| 项目 | 实际核验结果 |
|---|---|
| 文件 | `implementation_review_input.zip` |
| 字节数 | **113173**，与请求一致 |
| ZIP 成员数 | **41**，与请求一致 |
| SHA256 | **62159832796833361c5d76bb8b633a27035ef9a4025bcab29f4975698f3d40ae**，与请求一致 |
| ZIP 完整性 | 全成员 CRC 检查通过 |
| 解压路径检查 | 未发现重复成员、绝对路径、`..` 路径穿越或符号链接成员 |

本包没有单独覆盖全部 41 成员的顶层逐成员 MANIFEST。`execution/qualification/manifest.json` 是**数据资格及外部数据文件身份清单**，不是 ZIP 全成员清单。整体 ZIP 的用户给定字节数/SHA256 已匹配；不把缺少另一份顶层清单自行升级成重打包要求。

### 3.2 实际逐项核对

- `execution/deployment.json` 的 **15 个**源码/测试文件绑定，均与包内相应文件的实际大小、SHA256 一致。
- `execution/gate.pending.json` 的 **12 个运行源码绑定**，以及 scope、protocol、qualification manifest、CPU03 receipt、GPU02 shape 共 **5 个文档/证据绑定**，全部匹配；合计 **17 个**。
- 冻结协议本体 SHA256 为 `4589fd6b71e33fdbede84ced7b7c96131d527676a8b748316b2bf73bfcd7bf64`，与 scope、gate 所引一致。
- 对包内 **17 个 `.py` 文件**进行了标准库 AST 语法解析，均通过。这只说明可解析，不作为实现正确性或执行成功证据。
- 数据 manifest 引用的 12 份角色文件及 1 份 provenance 未随包提供；本轮没有访问这些路径，也没有声称核过它们的实际内容或 SHA256。模型绑定同样只作为提供的身份记录阅读，没有载入或重新哈希包外权重。

CPU02 的 15 项来源中，12 项与当前源码相同，差异是 control、run、测试文件；CPU03 和 GPU02 各有 **14/15** 与当前相同，唯一不同是 control。实现说明称最后补了控制器来源完整性检查；本包没有提供那份 13045 字节旧 control 本体，不能仅靠旧哈希独立证明它与当前版本的全部差分。当前控制器已实际静态追读，关闭入口拒绝回执也绑定当前 13614 字节版本。**不把 CPU03/GPU02 说成已经执行了最后这段控制器修改。**

## 四、启动前必修项

### R01：正常并发保存可使目录检查误杀正式轨迹

**来源与实际路径**

`scripts/step34_vendorlink_account_data.py` 第 145 行，`Check.__call__()` 使用：

```python
sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())
```

同一实验根目录内，`step34_vendorlink_account_run.py` 第 73 行及 `step29_vendorlink_run.py` 第 57 行会将 `.partial` 原子替换为正式文件；新 control 第 180 行还会删除已经成功退役的 `latest.pt`。这些都是批准执行中正常发生的操作。

若 worker A 的 `is_file()` 已返回真，而 worker B 随即完成 rename/unlink，A 的下一次 `stat()` 就会抛出未捕获的 `FileNotFoundError`。本协议不自动重试，因此一次正常保存交错就可能使有效正式轨迹失败并关闭下游。

**本轮实际核查**

我从包内 AST 提取原 `Check` 定义，用一个临时文件受控安排“`is_file()` 后原子重命名、随后执行原 `stat()`”。结果确为 `FileNotFoundError`，而目标文件实际存在。没有启动模型或并行训练；这是该错误分支的确定性逻辑复现，不是声称项目 Linux 已发生过此故障。

**必要修正及最小证据**

直接复用已在运行依赖中的 `step30_evolution_io.py` 第 126 行 `directory_bytes()`，其第 131 行已经针对枚举后文件消失捕获 `FileNotFoundError`；或做等价的小修。只忽略单个枚举对象消失，其他 I/O 错误仍应失败，磁盘总量门不放松。

修复后用无模型夹具确认该交错不误杀、仍存在的文件正常计量、真正超限仍拒绝即可。CPU03 的训练测试顺序执行，调度测试子进程只 sleep，不能代替这条竞态验证；无需增加双路训练或 GPU 资格。

### R02：启动失败后同一轮仍可能派发下一槽

**来源与实际路径**

新 `step34_vendorlink_account_control.py` 第 170 行选择 `halt_dispatch_on_failure=True`。但复用 Runner 中，`step30_evolution_control.py` 第 213 行只在派发循环**之前**检查已有 failures；第 264 行的启动异常处理记录 failures/retired 后，未立即阻止同一 `for slot` 循环继续。第 216 行的跳过条件也不检查刚刚产生的 failures。

因此，首槽在 command、Popen 或建立进程身份时失败后，下一槽仍可能被新派发；下一轮才停止。这不符合本批“失败后不再派新作业”的明确策略。

**本轮实际核查**

我执行从包内提取的原 `Runner.run_many()`，将 Popen 替换为始终抛 `OSError` 的模拟对象。实际出现两次启动尝试：`first_launch_failure`、`second_must_not_launch`。没有真实进程被创建。

**必要修正及最小证据**

仅当本批 `halt_dispatch_on_failure=True` 时，在启动异常分支立即设置停止派发状态或退出派发循环；已经运行的健康任务仍按原规则排空。旧调用默认 false 的行为保持。

最小证据是“首个启动调用抛异常，第二个待派任务没有启动调用”，并保持已经运行健康任务的排空语义。CPU03 覆盖的是**已派子进程运行后退出失败**，不覆盖本项；不必重新运行四臂小模型或 GPU 探针。

### R03：共享失败的类别没有完整传递到父控

**来源与实际路径**

`protocol.zh.md` 第 161 行规定共享磁盘/数据有效性问题停止整批相关任务。当前有两处接线不完整：

1. 新 data 第 147 行的共享磁盘失败抛普通 `RuntimeError`；新 control 第 242 行仅对 `ValueError` 写 `shared_failure.json`。父 Runner 第 210 行依赖该标记或自身检测到的 Shared disk limit，才停止相关活动任务。
2. 新 control 第 234 行先调用 `permit()`，第 240 行才进入 worker 异常标记区。若 worker 启动时来源/manifest 绑定核对失败，异常发生在这个标记区之外，父控同样可能只看到普通退出失败。

对磁盘分支，**持续超限通常仍会被父控对另一个活动 worker 的检查发现**，不能说它必然永久漏停。具体漏接时序是：worker 已检测到共享违反并退出，但父控下一次采样前，原子替换或成功 latest 退役使总量回落；父控不见共享标记，只按普通失败让健康任务继续排空。已发生的共享失败因而没有按冻结规则处理。

**必要修正及最小证据**

为明确的共享磁盘/来源有效性失败建立专门异常类别或等价的显式标记，并让受控 worker 的前置来源核验也能向父控报告该类别。只在核定的本批目标目录写标记。

**不要将所有 `RuntimeError` 都升级为共享失败。** 单 worker 的 OOM、RSS、生命周期超限仍应按既定局部失败语义处置。最小夹具可模拟“worker 已报告共享失败，而父控随后看到资源读数恢复”，验证仍停止本批相关活动任务、关闭下游；不必真实填满磁盘或访问真实数据。本项本轮为静态可达路径认定，未实际制造磁盘超限或来源破坏。

### R04：64 GiB 正式启动磁盘门未接入

**来源与实际路径**

`execution/scope.zh.md` 第 9 行明确开始需 64 GiB 空闲磁盘；`protocol.zh.md` 第 196 行也给出该准入安排。新 control 的 `campaign()` 第 163 行起没有执行这一门。新 policy 的 `LIMITS` 仅有 32 GiB `free_disk_reserve`、2 GiB `save_reserve_bytes`。

实际 Runner 的 `resource_observation()` 第 31、36 行使用：

```text
required_free = 32 GiB + (owned_workers + 1) × 2 GiB
```

首槽因此只需 34 GiB，第二槽为 36 GiB。旧 campaign 的磁盘起始检查不会因复用 Runner 自动执行；数据资格 CLI 第 287 行的 64 GiB 检查也不等于稍后正式训练准入已经检查。

**本轮实际核查**

在原 `resource_observation()` 的磁盘/RAM逻辑中，设置空闲 40 GiB、无活动 worker、其余资源充足，返回 `ready=True`。该检查未测 GPU 准入，只确认磁盘谓词低于批准门。

**必要修正及最小证据**

在真实正式启动、以及主执行者按冻结协议落实的相应派发路径接入 64 GiB 要求，保留已有 32 GiB 产物帽和运行余量监控。至少 scope 明确要求的正式启动门必须成立。

用低于 64 GiB、等于 64 GiB 的手写资源边界确认派发行为即可。现有只读观测显示当时磁盘很充足，故本项**不是声称现场磁盘不足**，而是启动条件没有被新实际入口执行。

### R05：辅助预算硬帽只能事后报错，不能控制继续发生的费用

**来源与实际路径**

冻结范围是 training 50h、GPU 6h、auxiliary 4h，总计 60h，活动监控属于 auxiliary。`step30_evolution_control.py` 第 281 行在辅助预算接近用尽时只设置 `budget_stop`。然而第 185 行只要 `active` 非空就继续循环，健康长任务可继续运行，父控监控费用也继续累积；第 308 行才统一计入账本。

`step30_evolution_io.py` 的 `Ledger.charge()` 第 181 行先累计费用，第 185 行才发现超过桶帽后抛异常。这会**事后发现超额**，不能阻止超额发生。派发分析任务时虽在第 229 行附近预约了辅助作业生命周期，但第 281 行的持续判断没有纳入这些尚未释放的 auxiliary 承诺，父监控还可能占用已经承诺给分析任务的余额。

此外，新 control 在 `run_many()` 外部的准备、批次冻结、少量 completion/gate 哈希、选型、判定及写盘没有明确计入 auxiliary。它们主要是小型父活动；大端点和 latest 哈希位于 completed 回调内，已经由 Runner 的 tick 计时，**不应夸大为大权重哈希普遍漏记，也不能重复收费**。

**必要修正及最小证据**

持续判断应共同考虑：

```text
已结 auxiliary 费用
+ 当期未结父控活动费用
+ 活动 auxiliary 作业承诺
+ 有界清理余量
```

当剩余额度不足以继续监控并安全收尾时，按共享预算失败结束本批相关活动、保留失败及费用、关闭下游；正常单任务失败的健康排空语义不能覆盖共享预算已经不足的情况。应给父活动与清理预留可执行空间，不要机械相加后让刚获准启动的贴边分析作业立即被取消。

补记 `run_many()` 外的小段父活动即可，不能用一个 `ledger.active` 包住整个 pipeline/等待/子作业而造成重复收费。只加事后 `charge` 也不能替代上述实际停止路径。

一个虚拟时钟/账本夹具即可区分：存在长活动任务时，辅助余额走到收尾阈值，是否及时结束并使最终账本不越帽；另覆盖活动辅助承诺未被父监控重复占用。不需要真实等待数小时或追加 GPU 运行。本项本轮依据静态控制流认定，未声称现有 317 秒准备费用已经超限。

## 五、科学实现接线：本轮可接受的部分

### 5.1 数据与文本角色

新 data 的 `prepare()` 在旧 fit 原键身份重建、旧 split、原四条选择及正文一致性核对之后，才形成归一类别；没有按归一名反向引入旧 dev/test 对象。归一冲突整项排除。`components()` 针对完整账号候选做图，`prepare()` 明确要求全部 U 在图中；前缀候选过滤后以精确整数 Jaccard 比较决定边，连通分量整体参与污染排除。`allocate()` 固定至少 20 分量、其中至少 8 个未污染，按既定 seed 分为 12/4/2/2。

相关直接来源：`step34_vendorlink_account_data.py` 的 `components`、`allocate`、`prepare`；`step29_vendorlink_data.py` 的 `eligible_ads`、`split_for`。跨账号近重复/模板相关性没有被此账号内图穷尽，保留协议边界。

| 市场 | 包内资格账号数 | 是否达到 128 下限 |
|---|---:|---|
| AlphaBay | 546 | 是 |
| Dream | 157 | 是 |
| Agora | 875 | 是 |
| 合计 | 1578 | — |

由保存聚合可核算：总角色记录 31560；fit 18936，gallery 6312，dev 与 confirm 各 3156。两种不足计数可重叠，不能相加当剔除总数。以上是包内聚合算术，不是重新核验真实账号资格。历史用途 U 的独立来源边界另见 E01。

### 5.2 完整 H、训练流与 Memory

`AccountSettings` 与实际 `h_objective()` 已接入：

```text
alpha = 0.07049361507739538
beta = 0.767438161798796
lambda_rank = 1.3051794547562026

H_SEQ  = L_now
H_ER   = L_now + alpha × L_mem
H_full = L_now + alpha × L_mem + beta × D_birth
```

beta 没有再乘 alpha。旧 Settings 的序列化字段未因新设置扩展而改变，旧调用 fallback 仍为 0.1/0.5。H 的双通道、记录对绝对差/乘积 head、0.5 均值加 0.5 log-mean-exp、自然/重组监督均值、BCE、正例 rank 和 hard-top-5 均保留。

`groups_for()` 每 epoch 使用每账号全部 12 条、分三包，每包独立重排账号，同一 seed 下三 H 的当前流不含方法名。`execute()` 使用 epoch0 三包全部实际群作 Memory 候选；reservoir 与历史抽样不扰动当前流。`Memory.retain()` 只给新增 survivor 生成本臂教师，旧教师不刷新；序列化包含记录、教师及采样状态并检查 1 MiB。H optimizer 跨域延续；V 每域换头并重建 optimizer，旧头与类映射复制到 CPU 端点。

直接来源：`step34_vendorlink_account_policy.py` 的 `AccountSettings/settings`；`step34_vendorlink_account_run.py` 的 `groups_for/execute`；`step29_vendorlink_methods.py` 的 `h_loss/h_objective/update_h/Memory`；`step29_vendorlink_run.py` 的 `set_rates`。

以下仅按资格账号数及实际分群规则作整数算术，可供正式完成核对：

| 市场 | H 每包群数 | H：40 epoch 更新 | V：40 epoch 更新 |
|---|---:|---:|---:|
| AlphaBay | 68 | 8160 | 8200 |
| Dream | 20 | 2400 | 2360 |
| Agora | 109 | 13080 | 13160 |
| 完整轨迹 | — | **23640** | **23720** |

ER/Full 后两域每当前更新回放一群，每轨迹应有 15480 次历史群调用。D 单域 H 每任务 4080 次更新，V 为 4100 次。新 control 的 `validate_training()` 会检查当前曝光、更新数、端点完整性及身份。这些计数不是已完成训练量，也不保证 H8h/V4h 单作业帽和训练50h桶足以完成所有轨迹。

### 5.3 Q2/G4、双 V 读取器及指标

实际评价保留全市场候选，不删掉同账号正候选。H 对 Q2×G4 的八条记录对执行原聚合；Q 与 G 使用不同 role UID。V_SET 用归一记录向量的集合均值实现含自对角的规定集合公式，非正分母拒绝，完整矩阵检查有限性。

V_CLS 只替换相应市场读取头，保留当前端点 encoder；两条 query 各自 softmax 后再平均完整类别概率，另保存逐广告结果。它没有加载旧整模型来消除遗忘。

`metrics()` 按完整 K 类算 macro-F1，零预测类为0，给出 Top1/微F1、MRR/MAP、Recall 与 NDCG；候选 UID 排序及稳定排序实现固定并列规则。九格包括 R01、R12、R23；Q/O/F/N/Z、GB/GC 使用本协议定义，F 保留负值。confirm 分支重新形成全部矩阵，没有将 dev 初学格与 confirm 最终格混算。

直接来源：`step34_vendorlink_account_evaluate.py` 的 `load_role/encode/rectangular/classification/metrics/score/analyze`；policy 的 `CELLS/summarize`。

### 5.4 D/P/M/C 与主张层级

| 层级 | 实际接线判断 |
|---|---|
| D | 四个固定单域末端；H 自身 F1，V 两读取器较高值；差不超过0.001取编号小者 |
| P | 三域绝对/逐seed/相对学习条件、B/C增量和明显遗忘同时检查；V 使用 **max(mean)** |
| M 系统 | Full−SEQ 的 O/F 双门、初学保护、新域保护、对 V_CLS/V_SET 分别的外部优势门均接入 |
| Full−ER | 另检查组件 O/F、ER参照保护；没有擅加第三seed反向1pp门 |
| C | 只有 M 系统通过才开；重复完整确认判据；组件结论与 M 的组件门取 AND，不能追认开发失败 |
| 应用 | Full/V_SET 三seed、市场内/跨市场分别 top10、H/V并集与出现次数；明确无真实马甲准确率 |

直接来源：policy 的 `select/decide`；control 的 `pipeline/run_phase/worker`；evaluate 的 `application`。P停止不额外训练Full；M完整有效但系统失败时，可以按已经获准的分支输出无真值候选，不能打开C或挽回定量失败。

原始逐seed矩阵已能支持后续均值、标准差、范围、配对差和逐域代价展示；当前 `analyze()` 未自动生成完整文字报告。正式结果分析时补全这些冻结展示要求即可，不应因此增加启动测试或临时统计成功门。

## 六、已有 CPU/GPU 证据与费用

### 6.1 证据适用范围

| 证据 | 所能支持的事实 | 不能推出的事实 |
|---|---|---|
| 数据 CPU | 保存回执报告两项通过；测试涵盖精确图与逐对参照、整分量污染、固定角色/容量 | 未独立重算真实数据资格 |
| CPU01 | 夹具资源等待后中断；失败与保守240秒收费保留 | 不是正式任务已启动，也不是240秒真实有效计算 |
| CPU02 | 四项未受影响检查通过；保留日志键冲突与过短生命周期夹具失败 | 不能把整次六项写成全通过 |
| CPU03 | 只补两条受影响路径，实际小BERT四方法三域×两epoch，走 execute→score→analyze；调度故障夹具通过 | 未覆盖本报告新识别的启动异常、目录原子保存竞态或辅助硬帽 |
| GPU01 | 最大形状首次更新 OOM，0次完整更新；原失败源码与控制台保留 | 不是正式科研轨迹失败或效果负结果 |
| GPU02 | 修正后 H1、44广告/946记录对，两次 current+history 完整更新通过，保持项非零 | 不等于真实40epoch全程峰值、长时吞吐或效果验证 |
| 关闭入口检查 | 当前真实 CLI 拒绝未闭合许可，回执声明未创建campaign | 不是允许开启后的全控制器路径已跑通 |

CPU02 的 checkpoint 对照使用四层 tiny BERT、dropout 0.1、两次真实更新，参数容差为 atol=1e-7、rtol=1e-6；另有 alpha/beta 独立加权的梯度参照。CPU03 核实旧头冻结、改变当前 encoder 会改变旧预测，以及三 H 当前流/第一阶段 encoder 一致。

须准确区分：CPU03 实际调用的是 `score(..., 'dev', ...)`。它虽然生成 confirm 夹具文件，**没有执行完整 C 控制器/confirm 全链**；本轮对 C 同代码角色分支及门的结论属于静态追踪。

GPU01 与当前 run 源码的实际差分是 H 非重入梯度检查点及外层日志键改名；GPU 脚本另补 `method='H_full'` 以触发新 model 分支。没有改变科学模型、损失、分群、microbatch 或预算。GPU02 保存的整PID峰值为 **4223664128 字节（约3.93 GiB）**，allocator峰值 **3235251200 字节**，RSS峰值 **2062802944 字节**。这些数值只属于该手写短探针。不能因此增加worker或承诺实际长轨迹峰值。

### 6.2 费用独立加总

| 项目 | 分桶 | 计费秒数 |
|---|---|---:|
| 数据资格 | auxiliary | 42 |
| 数据 CPU | auxiliary | 1 |
| CPU01 | auxiliary | 240 |
| CPU02 | auxiliary | 4 |
| CPU03 | auxiliary | 10 |
| 关闭入口检查 | auxiliary | 2 |
| GPU01 完整生命周期保守计费 | GPU | 7 |
| GPU02 完整生命周期 | GPU | 11 |
| 合计 | auxiliary 299 + GPU 18 + training 0 | **317** |

与 `preparation_budget.json`、`gate.pending.json` 一致。GPU01的7秒包含原探针4秒，GPU02的11秒包含探针本体计费，不重复相加。CPU子账257秒包含于auxiliary299秒，其中含关闭入口检查2秒；不再加一次。

正式 Ledger 再加一次30秒尾部费后，初始账为 auxiliary329秒、GPU18秒、training0秒，总计 **347秒**。R05针对未来实际硬帽执行，**现有准备账本没有发现算术超支或失败费用被删除**。

两槽上限、每槽2核、整PID14GiB、allocator13GiB、RSS32GiB与活动GPU/RAM承诺扣除均有实际接线。H完整作业8h、V4h、D的H2h/V1h、评分1h及整作业预约也接入。梯度重计算可能改变吞吐，仍以实际预算和停止门约束，不能把旧工期估算当保证。

## 七、E01：历史用途 U 的来源证据边界

`protocol.zh.md` 第35行要求 U 覆盖训练、评分及开发选择，并给出每种用途的来源身份。`execution/qualification/usage_sources.json` 的六个实际绑定只覆盖三份旧fit及三份qualification样本；其第37、42、47行对旧正式训练、ABC及两份fullpool的来源是文字说明，未带这些旧gate/输入清单的具体身份及相关条目。

新 data 的 `prepare()` 确实验证了旧fit与资格样本子集关系；第181行起只是保存 `config['usage_evidence']`。数据 CLI 允许核验 `evidence_sources`，但实际资格config及这些额外来源绑定没有随包提供。当前gate也没有直接绑定 usage_sources 或相应用途核对回执。

因此，可以独立追到“2052个旧fit账号、四条记录构成U，资格样本没有超出U”的实现；“后续评分/选择没有额外使用旧fit文本”目前仍是主执行者提供的覆盖摘要。**这不是已发现泄漏，也不是要求重新读取原文。**

最小闭合方式：在最终主审处置中引用已有非敏感来源的 path/bytes/SHA256、相关输入条目/入口定位或既有核对回执，逐项说明旧正式训练、资格、ABC和两份fullpool为何没有扩充U，并使该处置与最终许可绑定。无需索取真实账号、正文、逐条角色、旧qrels、owners、Memory或private_custody，也不要求再次执行数据资格。该项与五项已确定代码必修分开记录，不能在没有具体来源时把外审结论写成“已独立验证全部历史用途隔离”。

## 八、最小关闭路径与可以复用的工作

1. 仅修 R01—R05 所述实际控制路径；不改变科学协议、数据角色、候选、方法、seed、阈值、学习率或预算。
2. 对这些具体分支补最小无模型逻辑证据；已有未受影响的 CPU02、CPU03、GPU02 继续复用。若修复不触及模型/梯度/最大形状，不新增 GPU 资格。
3. 主执行者独立核对修正，补明 E01 的已有来源依据，更新实际部署、源码绑定、费用及主审处置；保持当前许可关闭直至闭合。修复核验费用仍计入原60h相应桶。
4. 闭合后按已有授权和即时资源条件启动唯一控制器，核实真实首个更新；D/P/M/C按原门推进。任何科学停止结果正常保存，不以换seed/扩候选/改确认文本绕过。发生正式失败仍按既定禁止自动重试规则处理。

本报告不代替主执行者最终处置，也没有替用户启动任何作业。**修正目标是让已批准实验按原合同运行，而不是增加新的研究前提。**

## 九、实际阅读范围与未验证事项

### 9.1 阅读范围

完整阅读：

- `protocol.zh.md`；`execution/review_request.zh.md`、`implementation.zh.md`、`scope.zh.md`。
- 六份新源码：`scripts/step34_vendorlink_account_{data,policy,run,evaluate,control,gpu}.py` 所对应的全部文件。
- `scripts/step29_vendorlink_data.py`、`scripts/step29_vendorlink_methods.py`。
- `tests/test_step34_account_data.py`、`tests/test_step34_account_increment.py`。
- `execution/qualification/manifest.json`、`usage_sources.json`；`qualification_cost.json`、`preparation_budget.json`、`gate.pending.json`、`deployment.json`、`cpu_data.json`、`entry_closed.json`、`postcheck_observation.json`。
- CPU01/02/03各自的receipt与console；GPU01的shape、lifecycle_cost、console及两份旧源文件；GPU02的receipt与shape。旧/新GPU相关源码差分已核对。

定向追读、按本轮复用范围审查：

- `scripts/step29_vendorlink_run.py` 的保存/恢复、RNG、学习率及被复用训练状态逻辑。
- `scripts/step30_evolution_io.py` 的来源/目录计量、Budget与Ledger。
- `scripts/step30_evolution_control.py` 的完整资源观测、Runner、退出清理、失败/预约/计费路径，并对照旧campaign的启动门。
- `scripts/step28_formal_queue.py` 的实际NVML读取和相关资源计量。
- `scripts/step30_evolution_policy.py` 的导入依赖、默认预算/资源常量与本轮覆盖关系；未重审旧Evolution科学决策。

没有把旧调度文件中与本轮无关的历史pipeline、数据构建或研究判据重新作为审查对象。

### 9.2 本轮实际执行的小逻辑核查

从原文件 AST 提取相应函数/类定义，以模拟OS读数、模拟Popen及临时文件安排受控事件；没有执行真实模型或项目数据入口：

| 核查 | 实际结果 |
|---|---|
| 40 GiB空闲、首槽磁盘谓词 | required_free=34 GiB，`ready=True` |
| `is_file`后原子rename再`stat` | `FileNotFoundError`；目标文件存在 |
| 首槽Popen模拟抛错 | 第二槽仍发生Popen尝试；真实子进程数0 |

自建启动夹具初次遗漏模拟 `STDOUT` 常量，得到夹具层 `AttributeError`；补齐后才取得上表所述Popen/OSError结果。未将夹具错误当作项目缺陷，未触及项目运行账本。R03/R05没有进行现场故障注入或长时间模拟运行，结论明确来自静态控制流。

### 9.3 未验证事项

- 未独立重算546/157/875资格账号及真实12/4/2/2角色；未读取包外角色/provenance、历史使用逐条索引或模型文件。
- 未独立核验E01所列旧用途覆盖的全部原始来源。
- 未连接项目Linux重新运行CPU/GPU资格或确认其当前实时资源状态；postcheck是保存时点观测。
- 未验证实际40epoch长轨迹可在单作业/分桶帽内完成，也未验证长期峰值、实际Memory文本总量或全池评分耗时。
- 未验证H_SEQ会达到0.70、出现明显且可重复的遗忘，未验证Full保持/新域质量或胜过两种V读取器。这些只能由之后合法完成并按门开放的结果回答。
- 未验证真实不同观察账号由同一自然人控制；本协议和应用输出不支持真实马甲precision/recall/F1、真马甲数量或真实同控遗忘率。

**最终处置建议：本精确版本暂缓正式启动；完成五项局部控制修正、对应最小逻辑闭合及主审来源处置后，继续沿已批准范围推进，不扩大研究。**
