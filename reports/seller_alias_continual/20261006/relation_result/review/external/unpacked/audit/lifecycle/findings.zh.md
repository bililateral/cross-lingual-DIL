# 本轮结果：正式完成、数据资格与状态生命周期独立复核

## 审查范围与结论

对象是 `relation_memory_result_review/reports/seller_alias_continual/20261006/relation_result/` 的冻结源码、门控、执行、训练端点、逐步轨迹、记忆摘要、盲分数、集合身份与完成账。先完整阅读了该目录 `review/context.zh.md`、`review/request.zh.md`。这次审查不访问项目 Linux/GPU，不加载正式文本、逐对标签、模型或记忆容器，不导入项目模块，不启动训练。原 ZIP 总身份由主审核验，本子审另外核验所用可见文件本体。

**在附件可见证据范围内，未发现数据供给、调度、记忆生命周期或完成资格方面的新阻断问题。可以把这批材料作为已完成的固定候选结果审查，不能再称“只有原生一步、正式效果尚未训练”。这不等于效果、机制归因或创新性通过。**

独立脚本 `independent_visible_audit.py` 只使用 Python 标准库和 NumPy：用 AST 重建 23 个文件的科学依赖闭包并哈希实际源码；核对 162 个可见文件本体；根据固定种子派生规则独立重建 9 条当前群完整调度、6 条历史抽样完整调度和 Algorithm R 成员演化；逐行核对更新、学习率、有限损失/梯度与总目标；核对 37 组盲分数及 18 个校准变换；核对 28 套指标/计数文件绑定。结果无不一致。这些是三个到达顺序各自的派生随机流，不是三个独立初始化种子。

保存的完整输出是 `independent_visible_audit.json`，控制台输出是 `independent_visible_audit.stdout.json`。检查条目数量不是方法正确性的统计保证。

## 逐项意见

### L1. 23 个科学来源与正式执行链绑定一致

- **位置：** `source/scripts/step28_relation_memory_run.py:31–54,461–475,543–550`；`gate.json/source_files`；`job/execution.json/{gate,sources}`；`job/run/manifest.json/source_files`；`job/evaluation/{collected,evaluation}.json/source_files`。
- **证据：** 独立 AST 闭包恰为 23 文件；实际每个文件的字节数/SHA 与 gate、execution、manifest、collected、evaluation、CPU 集成回执中的名单一致。`execution.gate` 也与本包 `gate.json` 本体一致。四项前置回执/审查文件本体哈希匹配；原生回执绑定的核心 `step28_relation_memory.py` SHA 与本次冻结核心相同。
- **影响：** 可以把已有受限数学/原生核验与本次正式入口对应起来。新根目录 `step28_relation_memory_result_verify.py` 是结果分析脚本，不应混入训练时 23 文件；父层被复用的旧合同常量不等于启动了旧路线。
- **最小修正：** 无。引用训练实现时使用本轮 `source/`，不要用之后编辑过的工作目录替代。
- **性质：** 已核实的证据闭环；非新增科学错误。

### L2. 2592 次更新、4320 群呈现及三个自身首域可独立核对

- **位置：** `source/scripts/step28_relation_memory_run.py:251–311,314–350`；`source/scripts/step28_relation_memory.py:69–85,346–399`；`source/scripts/step28_bge_continual.py:50–100`；`job/run/updates/*.json`、`points/*.json`。
- **证据：** 每个顺序三个阶段，每阶段 48 个当前 fit UID 恰各出现 6 次；逐步顺序精确等于 `seed_for(schedule_seed,order,stage,"current")` 的独立重算。每条 288 行，阶段 Adam 端点分别 288/576/864，逐行步号、学习率及 current+history 总和一致；三顺序合计 2592。首阶段 864 次只有当前群，后两阶段共 1728 次各含当前及历史一群，因此是 864+2×1728=4320 群梯度呈现。三份首阶段盲分数均与初始化不同，也彼此不同。
- **证据边界：** 逐行日志本身不是重新执行梯度证明；冻结核确实每次 `optimizer.step()` 后检查 Adam 步数，并沿正式入口产生这些端点。原生梯度路径核验的适用核心 SHA 相同。
- **影响：** 不存在把原先共享首域的 288 步或旧模型状态当本候选首域的可见迹象。入口每个顺序重新从核验的 BGE 预训练构造新关系头、新 Adam 和空 Memory；初始模型 digest 一致核对，随后各自产生首阶段。旧 LOGIT0.1 只在统计时读保存矩阵/计数。
- **最小修正：** 无。保持“单 s0、三个到达顺序”，不能把三个顺序升级为三种子重复。
- **性质：** 完成量与比较身份已核实；范围限定。

### L3. valid 一次开放有控制流与前后账支持，但不是 OS 文件审计

- **位置：** `source/scripts/step28_relation_memory_run.py:530–556`；`source/scripts/step28_bge_continual_run.py:76–87`；`job/before_valid.json`、`access.json`、`completion.json`；`job.wrapper.txt`、`job.console.txt`。
- **证据：** `before_valid` 绑定实际完成 manifest，9 点齐全，账为 train=1、valid=0、heldout=0、owners=0。随后实际 `access` 与 `completion.access` 均为 train=1、valid=1、heldout=0、owners=0。`parse_once` 在打开标签 CSV 前记访问尝试，拒绝重复；正式入口顺序明确为全部训练→所有盲分数/恢复/缓存身份门→before_valid→一次 development 标签解析→28 套收集→完成。108 条控制台进度按 ABC/BCA/CAB、阶段 1/2/3、每 24 步严格递增，耗时递增；包装正常退出 0，开始/结束与同步记录一致。
- **影响：** 支持此次固定流程没有在训练阶段用 valid 标签选点；valid 文本用于预定盲评分和当前域 calibration 标签拟合，是合同允许的不同路径。它仍是已经开发过的 valid，不因此成为新独立留出集。
- **最小修正：** 不需要为本次添加系统审计平台或重读标签。表述为“冻结控制流及前后回执支持一次开放”，勿声称网页拿到了 valid CSV 真实打开时间或 OS 系统调用日志；本包没有这类证据。
- **性质：** 证据支持与可见边界；不是已证实访问违规。

### L4. 数据供给与淘汰历史隔离按既定离线调度约定成立

- **位置：** `source/scripts/step28_continual_expression_run.py:58–128`；`source/scripts/step28_chinese_base.py:52–79`；`source/scripts/step28_bge_continual.py:65–100`；`source/scripts/step28_relation_memory_run.py:257–269,284–306`；PILOT 文档第 13、40 行。
- **证据：** 入口使用 `public_inputs` 只载 train/development 正文和公开群元信息，不走旧 `Archive`；只允许 train/development 的完整二元 supervision。可见 partition 中 144 fit、36 calibration、60 valid 的全部 240 UID 不相交，每域分别 48/12/20；从 fit+cal 的 60 UID 独立按固定哈希拆分可再现各域 48 fit。`Supply.current` 只按当前到达域、下一合法阶段发 48/12 群；学习器 update 只接当前群和存活 Memory。
- **影响：** 离线调度器保有待供给档案，与学习器可用历史预算是两层合同；不能把存在调度器本身误判为 180 群免费回放。阶段固化只读取仍存活旧六群和当前 48 群；旧端点归档在 blind gate 被哈希，不被重新反序列化为训练供给。冻结执行没有重新加载淘汰群的调用路径。
- **最小修正：** 无；保持该供给边界。原始群正文、标签和卖家身份本体未附，网页不能独立重新验证每群 28 账号/每条文本，这一层来自冻结解析器的校验和运行回执，不要称为网页重验原数据。
- **性质：** 源码/UID 证据支持；本体权限边界，不是要求补开数据。

### L5. 六群与累计统计生命周期吻合，CAB 零个最早域原始群是合法抽样结果

下表为独立重建 Algorithm R 后与点摘要逐一吻合的结果；“字节”为正式回执给出的完整容器大小，未在网页加载容器本体。

| 顺序/阶段 | 累计统计 N=seen | 缓存域构成 | 完整容器 B |
|---|---:|---|---:|
| ABC/1 | 48 | A6 | 495919 |
| ABC/2 | 96 | A4+B2 | 497879 |
| ABC/3 | 96 | A4+B2 | 504583 |
| BCA/1 | 48 | B6 | 502923 |
| BCA/2 | 96 | B2+C4 | 495717 |
| BCA/3 | 96 | B2+C4 | 502427 |
| CAB/1 | 48 | C6 | 484440 |
| CAB/2 | 96 | A6 | 491835 |
| CAB/3 | 96 | A6 | 498554 |

- **位置：** `source/scripts/step28_relation_memory.py:179–191,197–275`；`source/scripts/step28_continual_population_data.py:211–248`；各 `points/*/memory_summary,retention`；全部历史 `updates/*/history_ids`。
- **证据：** 使用保留专属随机流、各阶段按 UID 排序插入独立重建，与所有成员顺序、旧新 count、seen 精确一致；历史抽样用另一个专属随机流，6 条各 288 次，与 trace 完全一致，当前/历史 UID 不重叠。第一、二阶段均有实际固化及 H 最小特征值/容差回执；第三阶段 `retention=null`、N=96 与约定相符。
- **影响：** CAB 阶段 2 后缓存全为 A，虽没有 C 原始群，累计监督统计仍含已完成两域对应 N=96。无域配额 Algorithm R 允许这一构成，不能改称抽样错误，也不能事后增加域均衡。它是否解释局部增益/退化，本轮不能因果归因。第三阶段未固化不代表漏掉训练，更不支持声称无限流每步或第四阶段成本已核验。
- **最小修正：** 无算法修复。结果解读保留该具体构成和终端范围。
- **性质：** 实现吻合与外推边界；非新增 bug。

### L6. 1MiB 是完整有效历史口径，阶段开始/抽样新增随机状态已纳入检查

- **位置：** `source/scripts/step28_relation_memory.py:197–223,277–325`；`source/scripts/step28_relation_memory_run.py:203–215`；执行定义第 27–28、46 行；PILOT 第 40 行。
- **证据：** 序列化头含完整六群/标签/索引、reservoir RNG、抽样 RNG/count/stage、映射、参考对应 key；数字部分含 H/b/c/N、六个 FP32 X。校准 first/stage 参数以及 Torch CPU/CUDA RNG、完成阶段/Adam 步数在 `memory.maps` 内，赋值后先 `to_bytes()`，不在预算外。`begin_stage` 新增 RNG 与每次 `draw` 变化后都执行完整预算检查，失败回滚后抛错。前轮已发现的“端点只计 draw_rng=None、下一阶段新增约 7KB 未检查”漏洞在冻结核心中已修复。
- **影响：** 所有端点回执 484440–504583 B，均低于 1048576 B，最大回执约 48.1% 上限；stage3 比 stage2 多约 6.7KB 与仍保留活动历史抽样状态的生命周期相符。常规模型/一份 Adam 依合同另计；归档的推理点和旧缓存不回喂学习器，归档本身不自动变成有效历史。
- **证据边界：** 各 9 个缓存和模型本体未附。9 个缓存的大小/hash 在端点与 Linux inventory 间一致，9 个模型元数据同样一致；网页没有 fresh hash 这些本体，也没有重算其 H/b/c 或正文总大小。不能把源码计量设计与回执核对表述成网页亲自加载容器复核。
- **最小修正：** 保留上面证据层级，无需补开正式材料或训练。资源代理另评总磁盘/RSS/显存，不将 1MiB 与活动模型大小混淆。
- **性质：** 已修复缺口、当前一致回执及权限限制。

### L7. 真实保存恢复有具体读回路径；端点布尔值本身不是全部证据

- **位置：** `source/scripts/step28_relation_memory_run.py:186–248,300–301`；`source/scripts/step28_continual_population.py:157–210`；`source/scripts/step28_bge_continual_run.py:58–73`；各 `points/*/{full_state,model,full_restore_verified,intermediate_deleted_bytes}`。
- **证据：** 正式 checkpoint 写当前 Memory，实际从文件读回并逐字节一致；full checkpoint 写 model/Adam/metadata，`torch.load` 后实际装入 model 与 optimizer，比较实际载入后全状态 digest；从恢复 Memory 中恢复 CPU/CUDA RNG 并读回核对。之后重打当前 12 校准和全部 60 盲 valid 分数，逐元素精确比较。推理状态也实际保存/装入；用于下一阶段的 Memory 是恢复对象。9 点都记录 `full_restore_verified=true`，完整文件删除字节数等于该 full 文件字节数，保留模型 metadata 与 inventory 一致。
- **影响：** 支持这不是只记 hash 而没有调用恢复的纸面实现。冻结源码中的数值和状态重读路径、源绑定的集成扰动恢复证据、正式端点及后续调度相互支持。但由于本体未附，网页无法独立执行 BGE/Adam 或二进制恢复；也不意味着正式中断训练可任意续跑。
- **最小修正：** 无。谨慎用“服务器正式执行回执与冻结恢复路径一致”，不用“网页亲自恢复九个正式模型”。
- **性质：** 正式恢复证据支持，独立复核范围限定。

### L8. 完成资格来自 completion，未把纯统计 PASS 当正式全程完成

- **位置：** `source/scripts/step28_relation_memory_run.py:420–458,478–527`；`job/completion.json`；`job/evaluation/evaluation.json`；inventory、wrapper。
- **证据：** 保存统计状态为 `STATISTICS_COMPLETE_REQUIRES_VALID_COMPLETION`。实际 `completion.json` 是 `COMPLETE_RELATION_FIXED_POINT`，绑定实际 evaluation 文件的大小/SHA，访问账与正式范围一致，2592 更新，随后资源检查才保留该完成文件。完成中的继续建议与统计六项合取一致；本次为 false。可见 inventory 没有 failure/recovery 文件，包装退出 0；未发现本次执行走了统计恢复的证据。
- **影响：** 前轮 P1 所担心的“纯统计可越过资源失败签发通过”问题已经在当前源绑定入口里区分；这批不能因负效果被贬成未正式完成，也不能只凭 `verification.status=PASS` 宣称效果好。
- **最小修正：** 无新增工程要求。引用有效完成用 completion，引用效果用已保存统计；总预算资格由负责资源的子审复核。
- **性质：** 已核实的资格闭环。

### L9. 可见盲分数与校准身份完整；MAP 仍需依赖已保存指标

- **位置：** `job/run/scores/*.npy`；`points/*/scores,maps,calibration_ids`；`maps/*.json`；`source/scripts/step28_alias_calibration.py:61–94`；`job/evaluation/collected.json`。
- **证据：** 初始化及 9 点 raw 均为 60×378 FP32，各点 calibration 为 12×378 FP32；stage-cal/first-cal 均为 60×378 FP64，有限且实际哈希匹配。18 个变换均严格等于 `a * float64(raw) + b`，全对顺序及精确并列保持；a 正且在固定边界内。各顺序 first map 均为自己的已训练首域参数，后续保持；9 点 current calibration UID 与当前域 12 群精确对应。28 套矩阵/计数文件本体哈希及集合身份匹配，基线两份 collected 的 group/domain/metric 列顺序也一致。
- **影响：** 没有可见的首域映射错用、校准顺序损坏或数组换位证据；从 raw 到 calibration 的运算可以网页独立复算。没有逐对标签，所以不能从这些 raw 分数重新计算 MAP/AP，也不能独立重拟合 calibration 目标；应从提供的已保存群指标复算端点/配对统计（由统计子审完成）。
- **最小修正：** 无需真值或重训。结果报告把“独立复算保存指标统计”与“重新从 raw+truth 评分”分开。
- **性质：** 可见身份/变换已核实；权限边界。

## 分类处置

本分工没有发现需要修代码、补训或重开正式读取来修复的新科学错误/复现阻断。必要的是维持证据层级：附件可见文件可以实际重哈希/重算；未附正式本体与精确访问时刻不能认证。三顺序单 s0、已开发 valid、第三阶段不固化、无原始卖家身份材料等边界均已在当前上下文主动说明，不宜再次当作作者遗漏而加码工程流程。

对最终总体裁决的建议：本分工支持“这是一份完成资格成立的正式固定点结果”，效果筛选与创新评价应由对应证据独立给结论。不能因资源和流程完成而要求继续训练；不能把本次外审当作新参数、配额、数据或预算授权。
