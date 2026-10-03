# 执行、原生证据与来源链独立核对记录

## 裁定范围

本项审查未发现足以推翻当前两项正式结果的执行/来源矛盾，也未确认需要修改项目科学来源或重新训练的 SCIENTIFIC_BLOCKER / REPRODUCIBILITY_DEFECT。判断依据包括实际代码的执行顺序、完整保存元数据及数组的逐项核对、原始 stdout/stderr、前轮原审查原文；不是以 PASS、SHA 数量或测试数为科学证明。结果指标、22 指标端点及 23 项由其他独立参考负责，本记录不替代它们。

独立参考最终为 `provenance_reference_v2.py`，实际执行输出 `results_v2/provenance_result.json`、`results_v2/checks.json`、`results_v2/read_coverage.json`。1697 条内部字段/文件检查均成立；这是核对数量，不能当成 1697 次独立实验。第一次参考有 13 条误断并退出 1，见末节；原件完整保留。

## 路径约定

所有下列输入路径相对于收到包的 `project/`：

- B：`reports/seller_alias_continual/20260930/bge_continual_execution/20260930_143700/job`
- W：`reports/seller_alias_continual/20261001/er_weight_execution/20261001_144452/job`
- E：`reports/seller_alias_continual/20261002/er_low_execution/20261002_124018`
- L：`reports/seller_alias_continual/20261002/logit_weight_execution/20261002_153200`
- ER 新 job：`E/job`；LOGIT 新 job：`L/workspace/reports/job`
- ER/LOGIT 原生 CPU：`E/cpu`、`L/cpu`
- ER/LOGIT 回传与原保存结果审计：`reports/seller_alias_continual/20261003/{er_low,logit_weight}_result`

## 全文阅读与程序覆盖

已首先完整阅读包根 `REQUEST.zh.md`、`project/AGENTS.md` 与 `docs/RESEARCH_DISCIPLINE.zh.md`，完整阅读 ER_LOW / LOGIT_WEIGHT 两份当前合同。AGENTS 引用的当前交接文件未随该特定结果包提供；本轮以用户当前明确授权和 REQUEST 为范围，不因此添加新启动/访问审批。

以下八份原审/主审确实按全文读完，不只是读取结论：

| 原件 | 全文行数 | 用于本轮的实质承接 |
| --- | ---: | --- |
| `reports/documentation/20261002/er_low/review/external/REVIEW.zh.md` | 298 | 完整历史目标 ×0.1，low 全链参数，原共享状态，微型 288 步、正式尚未运行的时点边界，独立参考 REF-01~04 的失败与修订 |
| 同目录 `review/report.zh.md` | 36 | 主审处置、后来原生 CPU 22 例/4 次实更新与正式启动，早期 JSON 字段顺序误报 |
| `reports/documentation/20261002/logit_weight/review/external/REVIEW.zh.md` | 513 | 34 源/7 文件差异，独立 0.5 MSE，缓存旧目标不刷新，手写原生/微型边界，R1/R2 独立参考错误 |
| 同目录 `review/report.zh.md` | 47 | 原生 CPU 28 例/4 次更新与 MSE 探针；用户后来允许资源足够即启动及资源监听，不再强制等 ER0.1 结束 |
| `reports/seller_alias_continual/20261001/bge_continual_result/review/external/REVIEW.zh.md` | 466 | 原首轮 shared/ER 的恢复/随机供给/计数/原生记录，旧 F1–F12 承接与解释边界 |
| 同目录 `review/report.zh.md` | 86 | 初轮有效负结果与 LOGIT 对 ER 的局部正结果，缺失 observation 链接已有限定处置 |
| `reports/seller_alias_continual/20261002/er_weight_result/review/external/REVIEW.zh.md` | 519 | 原 ER0.25 与原 ER1 的完整日程/共享状态核对，原 23 项与开发选择，旧数值审计口径、失败修订 |
| 同目录 `review/report.zh.md` | 24 | 旧四组比较、原执行者实际核对原件、已关闭错误与当前参考身份 |

`read_coverage.json` 记录 603 个实际打开文件的大小/SHA/处理方式。其区分 full_json_parse、all_update_array_values、file_identity、全文人工审读，不声称 603 个文件均逐行人工审过科学语义。两份新 train.log、两份原生 CPU stdout/stderr/GNU 资源流均全文读过；两个新作业 manifest/startup/progress/points/updates、四个原生分支结果、保存审计原 stdout/stderr/execution 及回传清单均完整解析。未取得也未声称读取旧审查嵌套的大型证据 ZIP、正式文本/标签/缓存正文/权重。

## 31/34 来源的实际身份

ER0.1 执行根是 `project/`，LOGIT 执行根是 `L/workspace/`。分别由每份 job 的 `execution.json.source_files` 开始，对照实际源文件逐份 bytes/SHA；再分别与 `run/manifest.json`、`run/startup.json`、`run/progress.json`、对应原生 `cpu/audit.json` 的完整 source_files 比较。ER **31/31**、LOGIT **34/34** 均一致，两份 policy SHA 分别为：

- ER0.1：`201c7a766965df127803be89a9f3b7b1a5246053d95b2bcafa4dcc08cfa08c3d`
- LOGIT0.25：`2aaf41724055ed3dff93ad65a8be2cc4e83c8f35d4515dace34bcdc44b081deb`

两套执行来源各对 B 原首轮 18 份来源做实际身份核对，**18/18 原字节保持**。从 ER31 到 LOGIT34，新增合同/policy/decision 三项；变动恰为两 Bash、四 ER Python 模块与 `test_step28_er_weight_contracts.py` 共七项，其余来源不变。不能将 LOGIT 根的七个新字节当作 ER 作业实际执行过的字节。

本次结果包没有独立同名 `freeze.json` 文件；本轮可直接核对的是实际执行冻结 source_files 与其真实31/34载荷、policy和原生证据的绑定。完整原实现外审文本记录当时 freeze 的核对（LOGIT 原文附录 B 还列全34源）。不能写成“本次实际打开了两份 freeze.json”，但没有必要为缺少这个重复容器增加新门。两份 `execution.authorization` 引用的原授权 JSON 正文亦未随包提供，现有内容为 path/bytes/SHA 与 user decision/原主审承接；不得假装重新认证了其原文。

## 起点、配对和正式更新证据

对每种顺序的两份新 `restored_starts`，逐项核对 B 中该顺序 shared point 的完整 checkpoint bytes/SHA/state_SHA、model_state_SHA、首映射 a/b、Adam=288。三原共享 checkpoint 每份 **3,910,983,797 字节**。两新任务各三起点同时与 W 的 half/quarter restored_starts 的完整状态、模型摘要、Adam 和首映射相等。ER 使用 B 原 `*_er_after1` 缓存记录，LOGIT 使用 B 原 `*_logit_after1`；原首缓存 file bytes/SHA 与 members 精确对应。

六起点均记录 `first_scores_replayed_exactly=true`。这属于来源绑定的正式运行记录，符合实际代码先恢复完整状态再回放60群的路径；本网页没有实际加载这些大状态，不能把该布尔改写为网页做过原生张量与60群推理。

两新作业的六份 update NPY 分别为 288×14、288×16。独立逐行重算并核对：

| 核对量 | ER0.1 | LOGIT0.25 |
| --- | ---: | ---: |
| 从 NPY 行数相加的新更新 | 1728 | 1728 |
| 从 current_ids+history_ids 长度相加的群梯度呈现 | 3456 | 3456 |
| 新 point / 盲分数数组 | 6 / 24 | 6 / 24 |
| 当前完整目标分解与历史完整目标分解的最大绝对差 | 3.427267074584961e−7 | 3.2782554626464844e−7 |
| 历史加权项 `weight × history_total` 最大差 | 0 | 0 |
| 记录总目标对 `current_total + weight×history_total (+0.5 MSE)` 最大差 | 0 | 0 |
| 独立 LR 公式最大差 | 1.6940658945086007e−21 | 同左 |
| gradient_norm>1.0 的保存步数 | 1728 | 1728 |

分解使用不同浮点归约顺序，低于独立选定绝对 1e−6 核对容差；未修改原验收的零退化条件。0.5 MSE 系数列所有1728步都是0.5，没有再乘0.25。每个当前48群恰各出现6次，历史群只有6个且共抽288次；Adam 末步为576/864，观察步1/29/30/288的 head 参数均改变，encoder仅阶段288步不变，对应 LR=0。四观察步的有限非零合并梯度记录与该解释一致。每项有72个每24步日志事件，时间单调、逻辑计数正确；除已知接口改名 FutureWarning 外无生产异常文本，末状态与 completion 一致。

### 旧 ER0.25 更新日志的覆盖缺口

本包 **未包含 W/run/updates/ABC_quarter_stage2.json 等旧 quarter 的逐步 JSON/NPY**。W manifest.training 只有相应 path/bytes/SHA，没有展开288个群序列。本程序直接逐行核对的是两新作业与 **B 原ER1六份日志** 的 current_ids、history_ids、current_dropout_stream、stage/domain/Adam/update数；全部相等。不能说本次直接逐行读了旧quarter日志。

桥接依据具体而有限：两新 train_stage 都调用 old_training(reference, order, stage) 与同一 B 原ER日志逐项配对（ER run.py:240–248；LOGIT:246–254）；旧ER权重结果原外审 §3.3/§4（原文约108–138行）与主审已实际核过旧half/quarter和原ER的完整配对日志；本次 W 的五份 policy 钉住记录与真实文件全部 bytes/SHA一致，新两任务与W的共享起点记录逐字段一致。此证据链可承接已审旧比较，不能提升为本次直接重做缺席旧日志的审查。该明确范围缺口本身不否定当前结果，不要求索要标签/权重、重训或重开历史。

## 恢复、记忆与预算的时点区别

十二新point均有 `full_model_adam_and_rng_restore_verified=true`，full_checkpoint/inference模型记录和四评分文件绑定齐全，首映射固定。实际 checkpoint 代码依次保存完整state、恢复model/Adam、恢复RNG、完整12校准+60valid分数回放、保存推理state并恢复，最后再次恢复RNG（ER run.py:109–152；LOGIT:110–155）。网页只认证保存记录和源码因果顺序，未重做原权重恢复。

每点 `learner_auxiliary` 用独立规范 JSON 重新计算其字节数，ER为21290–21404字节，LOGIT为21318–21430字节。六个 `_budget.json` 表示每点checkpoint后、stage2 retain前的完整记忆计费摘要：ER 198747–217165字节，LOGIT 242229–260873字节，均小于1MiB。缓存正文被排除，因此完整缓存大小仍是原程序记录，而辅助JSON大小可以网页独立实测；两者不能混称。

train_stage 的 memory_after_training 比 `_budget.json` 早一次 auxiliary 更新，两个整个摘要不必相等。十二份的 members、seen、draw_*、references、reference_origins、with_logits完全相同；全记忆字节差恰等于auxiliary字节差，后者与该point保存的auxiliary独立字节数相等。这种SHA变化是元数据更新，不是旧logit目标刷新。

## 完整盲门、一次解析、退出和资源

每个 job 的 `before_valid.json:1` 都绑定该次完整 `run/manifest.json` 的实际 bytes/SHA，账目为 train1/valid0/heldout0/owners0；`access.json:1` 和 `completion.json:1` 同为 train1/valid1/heldout0/owners0。执行入口源码顺序为 train→blind_gate→before_valid→parse_once(development)→collect→finalize（ER:453–469；LOGIT:505–521），parse_once在loader之前记尝试（BGE run.py:76–87）。证据支持正常程序执行次数；不是操作系统全文件访问监控，也没有在本网页重跑要求权重/缓存正文的完整native blind_gate。

| 项目 | ER0.1 | LOGIT0.25 |
| --- | --- | --- |
| 正式开始，+08 | 2026-10-02 12:55:35 | 2026-10-02 16:31:29 |
| 正式结束，+08 | 2026-10-02 18:16:57 | 2026-10-02 21:53:13 |
| 秒级时间戳差 | 19282秒 | 19304秒 |
| GNU wall time | 5:21:21 | 5:21:44 |
| 程序 completion elapsed | 19279.312225695秒 | 19302.0049627251秒 |
| 退出码 / GNU exit | 0 / 0 | 0 / 0 |
| 原程序观察产物峰值 | 10,446,866,117字节 | 10,447,072,442字节 |
| GNU最大RSS | 7,021,972 KiB | 7,082,784 KiB |
| Python / Torch / CUDA | 3.10.19 / 2.9.1+cu130 / 13.0 | 同左 |
| GPU名称 / CPU affinity | RTX5090 / [47] | RTX5090 / [46] |

不同计时边界造成秒级差异，未发现实质矛盾。两项均低于43200秒/16GiB **新产物** 上限，16GiB不是GPU/RSS上限。产物峰值是定时/预留检查观察峰值，不能冒称连续采样精确峰值。十二阶段合计纯训练秒数18045.5562 /18053.0914，各约5小时。额外恢复、评分、校准与评价解释总时长差别，不将记录的GPU异步host时间解释为独立核计算时间。

LOGIT等ER0.1结束后才启动并非本次实际时序：两项有重叠。完整LOGIT主审原文“当前资源顺序”与“15:48:31观测”段（约29–47行）保存用户后续明确允许资源足够就开始及监听请求，因此不因旧冻结合同的“等low释放后”字样误判违约。但具体后来监听触发、GPU资源就绪观测/自删脚本不在本包，网页不认证这些现场动作；正式命令、执行来源与完成记录已存在。

## 原生 CPU与网页复算严格分层

原生CPU raw stdout/stderr、GNU日志、起止/exit及quarter/tenth或quarter/logit JSON都已实际读、逐份hash核对。原生记录为Python3.10.19、NumPy2.2.6、Torch2.9.1+cu130，CPU47/46，GPU false、正式inputs/labels false。

- ER：12:46:28–12:52:32(+08)，GNU364.25秒，22个不同unittest条目均OK，quarter和tenth各一次真实warm和一次真实加权更新，总4次原生BGE手写更新。
- LOGIT：15:37:17–15:44:20(+08)，GNU422.82秒，28条均OK，quarter和logit各2次实更新，总4次；另有不step的MSE梯度参考。
- 四个分支均真实warm Adam1，然后声明把计数夹具1→288，再一次update到289。不能称原生已经做288步连续运行。两项中的当前初态、warm后模型、Adam摘要、全部378当前及历史logit逐值配对；每分支29个encoder/head参数探针记录变化且梯度正有限。
- 原生梯度分量 **仅提供norm/SHA，未提供向量**。ER原程序记录逐元素最大差为encoder2.546585e−11/head1.047738e−9；LOGIT原程序记录`g_LOGIT = g_ER + g_0.5MSE`最大差2.546585e−11/7.566996e−10。这些是原生程序执行结果，网页无法从norm/SHA重新逐元素算出。独立范数比仅作诊断，分别2.50000358787728/2.499998207283172，不拿它证明完整向量比例或参数步长比例。
- LOGIT原生手写保存logit与origin_eval_reference均378维，可以独立复算MSE：0.0001965899888976513，与其独立记录一致。另一个代理从手写关系独立核BCE/Rank/Hard及导数；均不是正式标签重算。

网页当前参考由 `run_record.py` 绑定一个允许CPU、BLAS线程1运行，只读上述小结果，不导入项目模块，不加载Torch/BGE，不连接服务器、不安装依赖。真实版本、argv、cwd、起止时间/退出和原stdout/stderr在 `runs/execution_provenance_{v1,diagnosis,v2}/`。不得将其毫秒级核对记为项目原生再次执行。

## 回传映射和权重证据

ER的return_inventory有270项、8,721,549字节；LOGIT有198项、7,785,058字节；所有实际Linux path均存在且bytes/SHA吻合，各returned_path唯一，逐项满足从该job根映射到Windows显示 `job/...`。不能把结果报告中 `*_result/job/...` 显示链接与包内Linux路径差异误报为漏包。两份 audit_return_inventory各五文件均匹配；audit_stdout.log与audit/audit.json逐字节相同，audit_stderr均0字节，audit_execution退出0，审计源/辅助源hash也一致。

两份 payload_custody 各6个原生推理权重记录，与十二point.model的path/bytes/SHA精确匹配；ER总7,838,864,652字节，LOGIT总7,838,884,668字节。只有记录，不是本网页重新流式hash了这些大文件，更不是载模验证。三缓存排除清单的大小亦与manifest记忆保存摘要一致。return_sync中的一次粘贴wrapped digest重复字符误报，包内仅给当时处置说明；原始终端两次hash流未提供，不能声称本次独立重验了那个粘贴错误。

项目10月3日保存结果审计原生退出记录为 ER 02:23:57.866168–02:23:58.532662 UTC、LOGIT 02:23:58.533720–02:23:58.999446 UTC，CPU47。其413532/368052数值比较与7.105427357601002e−15最大差只对其计数器跟踪量成立；本参考实际loss分解更大约3.4e−7，两个口径必须分开。该原生数值审计本身不是独立于项目实现的新科学证据，本轮另写公式/索引才是独立复算。

## 必要分类与限制

- **SCIENTIFIC_BLOCKER：当前已提供证据未确认。** 正式来源、目标记录、共享起点、同供给桥接、访问顺序、完成量/预算没有实质矛盾。负结果不是程序无效。
- **REPRODUCIBILITY_DEFECT：当前项目未确认需修项。** 对两不同时点memory摘要全相等和norm比例的误断属于本审查参考，已修并留原件；不能转嫁给项目。
- **未验证范围：** 原生大状态tensor/RNG实际恢复、正式cache正文和logit向量、正式AP/MAP等真值重算、原生梯度向量逐元素再算、旧quarter逐步日志本次直读、独立freeze/authorization原件正文、现场监听资源触发/进程状态、完整旧网页证据ZIP，均不在本次直接证据范围。
- **OUT_OF_SCOPE_OVERDESIGN：** 要求以上所有私有大载荷/新训练/新标签/服务器操作、重开无变化原审、重造同名freeze或新签名/系统审计门、将23项扩大到新门，均非本次必要修正。

## 本参考真实失败与修订

`execution_provenance_v1`退出1，stdout列13个失败：12个memory摘要不同时点相等误断；1个原生float32归约norm ratio自设阈值误断。整个输入未改。`execution_provenance_diagnosis`实际退出0，逐份输出字段差及字节差；`provenance_reference_v2.py`仅修上述比较语义，并撤去无法由缺席向量认证的阈值。v2实际退出0、1697项成立。初稿/修订稿/原始两个版本输出/诊断/精确diff均保留，详见 `FAILURES_AND_REVISIONS.zh.md`。早期交互式查看未逐条生成独立计时回执，不伪造；全部实质程序执行均由统一记录器保存。
