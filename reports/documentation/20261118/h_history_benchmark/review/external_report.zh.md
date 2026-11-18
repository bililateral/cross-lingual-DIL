# 固定 trial3 Full 与五个原定义基线：中文共同确认方案与实现合审

审查日期：2026-11-19（北京时间）  
审查对象：本次附件 `combined.zip`  
模型自述：GPT‑6 Astra Pro。我不能独立核验界面选择器或后台部署身份，因此不把上述自述写成后台身份认证。

## 一、总评与放行意见

**方案的比较对象、六方法分路、共同确认设计和 raw MAP 配对主表可以保留；当前实现应先修复两项具体工程缺陷，再办理正式开训准入。**

本次发现的开训前必修项是：

| 编号 | 判断 | 触发条件 | 实际影响 |
|---|---|---|---|
| HB-R01 | 生成新群前跳过必要端点文件的重新核验 | 训练验收成功后、generation 开始前，某个必须保留的推断端点丢失或损坏 | 仍可能生成并消耗唯一新确认批次；随后恢复失败。**不等于 pairs 已提前开放** |
| HB-R02 | 控制器验收期间的可捕获中断可能漏记任务费用 | `finish()` 已移除活动槽，随后 `stop()`／`validate()` 抛出 `KeyboardInterrupt` 等 `BaseException` | 实际 elapsed 和失败回执可被跳过，外层清理也无法再找到该槽 |

两项都是源码中可直接追踪的路径，不是根据“尚未做全量实验”推测出来的问题。它们分别影响生成准入和失败账目，不构成六种科研方法定义已经错误、结果已经无效或已经发生标签泄露的证据。具体定位与最小修法见第五节。

**积极结论同样明确：**本包没有把六方法换成一个统一的替代模型；H 的两个历史权重进入实际损失；ER／Logit 保留账号级独立分支；L²R／GCR／FOREVER 走各自原生入口；正常确认路径先完成全批盲分数和校准核验，再一次打开 pairs；六方法 raw MAP 的同顺序、同 seed 配对公式成立。【T：`native()`、`train()`；E：`confirm()`、`paired_table()`；N：各原生更新函数】

正式效果幅度、区间、多重比较、预算、并发及资源额度尚未固定。这些是开训前必须另行落定的合同选择，不能因为当前交付只提供配对点估计，就判定其统计代码有错误。本报告不授权开训、生成正式新群或自动等待。

## 二、包身份、实际阅读范围与证据等级

### 2.1 实际核包

我对附件本体及逐成员进行了字节和 SHA-256 核验：

| 项目 | 实际核验结果 |
|---|---|
| ZIP 大小 | 40,584 字节 |
| ZIP SHA-256 | `0a79d2c369dd96c944c043fb062a55327241a5be0d64b12053f76b4b5dec5b30` |
| ZIP 成员数 | 10：9 份载荷加 `package_contents.json` |
| ZIP CRC | 未发现坏成员 |
| 载荷清单 | 9 份载荷的大小和 SHA-256 全部与清单一致，无漏项或额外载荷 |
| `source_manifest.json` | 声明 50 个生产／政策／测试文件；本包仅提供其中 3 个完整文件的字节 |
| 原生节录 | 11 个源文件、28 个函数／类节录；节录标题所列 SHA 与来源清单相符 |

**后两项不能混同：**我核实了来源清单内部的对应关系，不能仅凭节录标题上的 SHA，声称已重新计算其余完整生产文件的哈希。来源清单不是未随包源码的替代品。

### 2.2 本报告的源码引用约定

| 代号 | 附件内文件 |
|---|---|
| Q | `questions.zh.md` |
| S | `scope.zh.md` |
| T | `scripts/step28_history_benchmark.py` |
| E | `scripts/step28_history_benchmark_evaluate.py` |
| U | `tests/test_step28_history_benchmark.py` |
| N | `native_source_excerpts.txt` |
| C | `synthetic_cpu.json` |
| G | `synthetic_gpu.json` |

上述八份文件以及两份清单均已完整阅读。引用 N 时，同时写出原始源文件、函数名及节录标注的原始行号；不是把节录文件的行号冒充原文件行号。

本次执行仅包括解包、文本／JSON 阅读、文件哈希核验和已给数值的简单算术。**没有导入或运行科研项目、执行测试、加载模型、构造正式数据、读取包外私有数据、重新计算科研指标或统计重采样，也没有外部检索。**CPU／CUDA 的“通过”来自所附既有回执，不是本次审查重新运行得到。

## 三、问题一：固定身份、真实更新与原生端点

### 3.1 54 条轨迹的规模和身份正确展开

入口固定六种方法、ABC／BCA／CAB 三个顺序、seed 4／5／6，得到：

- 轨迹：6 × 3 × 3 = **54 条**；
- 阶段端点：54 × 3 = **162 个**；
- 每条当前更新：3 × 48 × 6 = **864 次**；
- 全批当前更新：54 × 864 = **46,656 次**；
- 每条六个已学域格，全批 **324 格**；
- 每格 40 群，全批 **12,960 次群前向**；
- 每格保存三个角色，共 **972 个盲分数数组**。

这些是计划规模，不是已完成的正式运行数量。FOREVER 的额外回放更新、各方法的梯度选择、维护、校准与恢复开销不包含在“46,656 次当前更新”中。【T：L25、`plan()` L49、`cohort()` L55；S：L16、L22】

### 3.2 六方法确实分路

| 论文对象 | 新入口的实际路径 | 可见原生行为 | 本轮判断 |
|---|---|---|---|
| trial3 Full／H | `history.Spec("replication", 3, …, ALPHA, BETA)` → `five_run.train_trajectory(..., history, ...)` | 记录表示、自然／重组两视图监督、历史监督及出生评分 MSE、实际 Adam 更新 | 两个历史权重实接，未改为账号级基线 |
| ER | `five.Spec("ER0.1_ref", …)` → `five_run` | `account_separate_moments`，原 rank，历史监督 0.1，不要求教师 reference | 保留原独立账号级基线 |
| Logit 0.1 | `five.Spec("LOGIT0.1_ref", …)` → `five_run` | 同一账号级表示，历史监督 0.1，原始 logit 保持系数 0.5 | 没有冒充 H 的组件消融 |
| L²R | `comparison.Spec` → `comparison_run.train_trajectory` | `stage.backward` 后真实优化更新，原生 L²R 评分分支 | 原生调用链成立 |
| ER+GCR | 同一 comparison 入口中的 GCR 分支 | `stage.backward`、实际更新、随后 `stage.after_update`；端点维护显式重新启用梯度 | 没有把选择维护变成无梯度占位 |
| FOREVER | `forever.Spec` → `forever_formal.execute` | 时钟事件、当前更新、巩固、端点、保留与切域 | 原生阶段顺序可定位 |

依据：T 的 `native()` L37、`train()` L138；N 中 `step28_continual_five.py::Method.metadata` 原 L145、`Method.update` 原 L178；`step28_continual_comparison.py::Method.update` 原 L129、`Method.score` 原 L112；`step28_forever_run.py::stage` 原 L217。

### 3.3 H 的两个历史权重确实进入损失

入口固定：

- 历史监督：**0.07049361507739538**；
- 出生评分保持：**0.767438161798796**。

`history.Objective.objective()` 对自然与重组视图分别计算 BCE、`rank_pos` 和 hard 项，再平均得到监督项。当前群监督权重为 1；存在历史 reference 时，监督项乘历史监督权重；教师 MSE 另外乘保持系数。其形式为：

```text
当前群：L_current = L_supervised
历史群：L_history = 0.07049361507739538 × L_supervised
                    + 0.767438161798796 × MSE_birth
```

历史目标使用出生参考表经自然视图聚合得到的得分。更新调用真正的 `record.backward_group(..., objective_fn=self.tuned.objective)`，随后裁剪梯度、执行 `optimizer.step()`，并检查 Adam 步数。手写测试还检查这两个乘法贡献及正的教师贡献。【T：L27、L39；N：`step28_history_optuna.py::Objective.objective` 原 L105；`step28_continual_five.py::Method.update` 原 L178；U：L86、L91】

**核验边界：**本包没有提供 `history.Spec` 的全部属性体及完整 `config/fresh`，也没有旧 trial3 的完整原始配置。所以能确认“trial3 复现分支和两系数实接”，不能声称已逐值独立比对全部 `lambda_rank`、`lr_multiplier`、随机根和旧训练参数。正式冻结材料应保留完整 native identity 与实际配置；这可以通过配置与源码补证完成，无须重训。

### 3.4 L²R 的唯一 ftol 例外被保留

`gradient_stopping()` 仅对 **`CAB_s6_L2R_text_fixed`** 返回真，训练入口把该值显式传入原生 comparison runner；原生 checkpoint 再传给校准函数。验收逐阶段要求该轨迹 `ftol=0.0`，其余轨迹为原 `OPTIONS["ftol"]`，并要求拟合 PASS、stage 映射与文件一致、first 映射与第一阶段一致。【T：L60、L145、L176；N：`step28_continual_comparison_run.py::train_trajectory` 原 L240、`checkpoint` 原 L154】

因此，新入口没有把全部 L²R／GCR 都统一改成这一例外。至于“旧 18 条成功轨迹中仅这一条采用该设置”的历史事实，本包没有附全部旧轨迹原件，本次按已声明背景复用，不写成重新核实过的历史结果。

### 3.5 阶段端点没有被统一成不适当的定义

H／ER／Logit 与 L²R／GCR 的原生路径检查连续 Adam 计数，每阶段对应 288、576、864 次当前优化更新，保留当前校准、模型和校准映射的实际保存／恢复检查。

FOREVER 的端点发生在 `consolidate()` 之后，再执行 `retain_and_switch()`。它的 Adam 计数对应 current／replay／fixture 的合计，原生完成门要求 current=864、fixture=0、replay 在原规定的 0 至 228 范围内。不能为了使六者计数形式一致，删除 FOREVER 原有的时钟回放或全参数锚点，也不能把它的总 Adam 计数强制当作 864。【N：`step28_forever_run.py::stage` 原 L217、`endpoint` 原 L181；`step28_forever.py::Learner.update` 原 L274；`step28_forever_formal.py::validate_completion` 原 L93】

原生 L²R／GCR 的选择内部、FOREVER 的完整事件与巩固内部、全参数锚点实现未完整随包。本轮确认新入口调用这些原生路径，不重开所有既有方法的逐式实现审查。

## 四、问题二：当前域权限、共同分区及省去旧开发评分

### 4.1 区分调度器读取训练档案与学习器取得未来域

`comparison_run.training_inputs()` 校验冻结数据来源，只读取 train 元数据、train 文本和 train 标签；检查 180 个训练群、每域 60 群以及账号／记录身份。每域按冻结 partition seed、domain、group UID 的哈希排序，取 48 个 fit 群，其余 12 个用于当前校准。【N：`step28_continual_comparison_run.py::training_inputs` 原 L97】

该函数确实会在调度层索引完整授权训练档案，包括后续域。**这不等于后续域已传给当前学习器。**可见训练循环逐阶段调用 `Supply.current(..., stage)`，更新函数只接收当前群和抽出的幸存 Memory／reference；没有把完整 Supply 或全部未来群作为更新参数。H／ER／Logit 的 `train_stage` 还核对历史成员、目标和抽取次数在阶段内的约束。【N：`step28_continual_five_run.py::train_trajectory` 原 L233、`train_stage` 原 L193；`step28_continual_comparison_run.py::train_trajectory` 原 L240】

可支持的结论是：**在已给出的调用边界上，未见未来域或已淘汰历史被直接交给学习器。**它不是操作系统层的硬隔离证明；完整 `Supply.current`、FOREVER Supply 包装及配置定义未随包，不能仅凭函数名称补出全部内部检查。访问计数字段也不是独立系统调用审计。

### 4.2 共同 fit／校准分区有实际阻断检查

H／ER／Logit 显式传入同一个 `comparison_run.training_inputs`；L²R／GCR 直接使用它。FOREVER 调其原生 `run.training_inputs` 与 Supply 包装。最终 `completed()` 对所有 54 条的完整 `partition` 作相等比较，任何方法或轨迹不同都会在生成之前报错。【T：L149、L201；N：`step28_forever_formal.py::execute` 原 L140】

相同当前样本时序由原 `parent.schedule(current, c, order, stage)` 继承；函数实现和各配置的全部随机根未完整提供，因此不能仅凭“调用了同一个 schedule”断言已逐群独立比对 54 条未来正式时序。正式回执应保留现有 schedule／current UID 证据，冻结 native identity 和分区摘要。

可以把共同分区的检查前移以避免训练后才发现不一致而浪费资源，但现有最终检查已能阻止不同分区进入共同新群。这属于验证时机的改进，不是已发现当前训练使用了不同分区。

### 4.3 去掉旧开发评分，当前校准与保留路径仍在

在新的具名训练输入路径中，`valid=None`。原 checkpoint 仍然：

1. 用当前 12 个校准群评分并拟合映射；
2. 在第一阶段创建 Memory，执行原 `method.retain`；
3. 真正序列化 Memory，并重新解析恢复；
4. 保存带 optimizer 的工作状态并恢复验证；
5. 保存推断状态并恢复验证；
6. 对当前校准分数作逐值相等检查。

被条件跳过的是旧 60 群开发 raw／校准分数；输出 `development_ids=[]`。原 `Method.retain` 对 H／账号级基线仍只在前两个阶段更新保留内容。【N：`step28_continual_five_run.py::checkpoint` 原 L122、`train_trajectory` 原 L233；`step28_continual_five.py::Method.retain` 原 L239】

这支持“省去旧开发评分没有从新入口删除校准、保留和恢复”的结构判断。它不证明正式 BGE 在所有环境下与旧入口逐步逐比特相同；本包没有这样的对照运行。必要的恢复文件也应区分：允许清理的 full 工作检查点，与必须留供盲评的阶段推断端点，是不同对象。

## 五、问题三：全批门、一次盲评、失败与费用

### 5.1 正常路径中的顺序与身份检查成立

正式控制器按 training → generation → confirmation → analysis 执行，后处理串行。全部训练成功以前，不调度生成。之后：

| 关口 | 可见检查 | 结论 |
|---|---|---|
| 54 个成功回执 | ID 集精确齐全，success 为真，training 桶，批次回执与单任务回执相同 | 缺轨迹、失败轨迹不能正常放行 |
| 训练记录 | spec、source_files、864 次当前更新、三个阶段、当前校准 ID、恢复标记与映射 | 身份和端点元数据有具体检查；模型文件重验存在 HB-R01 |
| 共同新群 | 固定根种子与命名空间；ABC 各40群；生成角色核对群、账号、记录、控制分量；owners 封存 | 新群资格有生成路径，尚未实际生成 |
| 推断恢复 | 按六方法恢复，核文件哈希，检查阶段与 stage／first 映射；FOREVER另核 spec／counts | 没有用另一方法的通用评分器替代 |
| 324 个格 | 54条×6格、120个唯一群、真实域、共同群顺序、三个角色、具名文件路径 | 缺格／错域／错群序／缺角色会被拒绝 |
| 分数内容 | 每角色文件哈希，40×378浮点矩阵、有限性；从 raw 依端点映射重算两套校准并逐值比较 | 不能只靠“分数文件存在”通过 |
| pairs 开放 | 全部盲分数检查后写 `all_blind_complete.json`，随后才调 `open_pairs()` | 正常代码顺序满足共同盲评 |

主要依据：T 的 `completed()` L187、`validate_training()` L153；E 的 `validate_cells()` L21、`load_blind()` L41、`blind_trajectory()` L48、`confirm()` L111；N 的 `step28_formal_queue.py::dispatch` 原 L171、`step28_continual_comparison_evaluate.py::generate` 原 L86。

`open_pairs()` 在实际读取前登记一次尝试，检查文件名、字节数，按群严格对齐、要求每群 20 个正例，并核对整文件 SHA。它在这一遍读取完成后保留 truth 供指标计算，不需要每个方法重新开标签。生成器此前构建和验证 pairs 属于生成角色，不应误称为评分者已经利用标签。【E：`open_pairs()` L91；N：`step28_continual_five_evaluate.py::access` 原 L26、`confirmation_public` 原 L131】

这里的“一次开放”是正式入口的逻辑访问保证，不是针对任意外部进程的权限安全证明。

### 5.2 HB-R01：缺端点仍可能先生成新群

**严重程度：开训前必修，生成准入缺陷。**

**源码定位：**

- T：`completed()` L200，固定传 `verify_models=False`；
- T：`validate_training()` L157、L182，只有开关为真才执行 FOREVER 原生完成核验或其他方法的模型文件核验；
- E：`run()` L210、L212，随后进入 `shared.generate()`。

**触发情形：**54 条训练已经通过各自验收，成功回执、completion、partition 和校准文件仍完整；但在生成前，一个必须保留的 `point["model"]` 文件丢失、损坏或转存遗漏。

**实际路径：**`completed()` 仍可通过，正式新群开始生成；到 confirmation 的 `blind_trajectory()` L74，才因文件验证或恢复失败停下。由于后面无法完成全部盲分数，正常路径到不了 `open_pairs()`。

因此，此项违反的是 S L20“162 个阶段端点完整后才生成一次新确认”的承诺，会消耗一次性批次与费用；不能扩大为已经提前开 labels。

**最小修法：**生成前重新核验全部 162 个应保留的推断端点的存在性、字节数及 SHA。现成办法是让 `completed()` 接受核验开关，并让 generation 必须使用 `validate_training(..., verify_models=True)`。不需要载模；不要求保留本来允许清理的 full 工作状态。confirmation 恢复时的原有逐文件检查继续保留。

**最小闭合证据：**完整成功回执和其余元数据不变，只删除或损坏一个推断端点，确认 generation 在写出正式群／登记生成访问之前拒绝。补这一针对性检查即可，不要求新增训练或全套 GPU 核验。

### 5.3 HB-R02：控制器中断时，已移出活动表的任务可能漏账

**严重程度：开训前必修，失败计费及收尾缺陷。**

**源码定位：**N 中 `step28_formal_queue.py::dispatch()` 的内部 `finish()`，原 L203 起：

- 原 L204：先 `item = active.pop(slot)`；
- 原 L207、L211：再执行 `stop(process)` 和 `validate(row, job)`；
- 原 L212：仅捕获 `Exception`；
- 原 L216、L218、L225：之后才计算 elapsed、累加费用、写 dispatch receipt；
- 外层原 L338 捕获 `BaseException`，原 L341 只遍历仍在 `active` 中的槽。

**具体可触发情形：**子进程已经退出，父控制器正在做模型哈希／端点验收，此时收到 SIGINT，`validate()` 抛出 `KeyboardInterrupt`。它不属于 `Exception`，所以只执行关闭日志的 finally，然后跳过后面的计费代码。该槽早已被 pop，外层也无法补记。验收回调抛出 `SystemExit` 同理。

这个路径能够遗漏整条已经运行的任务时间和调度回执。若中断发生在停止子进程期间，同样存在其不在外层排空列表中的窗口；由于完整 `stop()` 未随包，我不进一步断言某种中断一定会遗留活进程。

**最小修法：**

1. 在子进程终止和账目收尾完成前，不提前移除活动项。
2. 对可捕获的 `BaseException` 先记录失败原因，并在收尾段结算实际 elapsed、写失败回执，再传播中断。
3. 确保外层清理再次进入时不会重复计费；如果停止尚未完成，保留该项供外层继续排空。

这只要求局部完善现有控制器的可捕获中断路径，不要求支持断电、SIGKILL 后自动恢复，更不要求新增 OOM 重启平台。

**最小闭合证据：**使用伪造的已退出子进程和一个抛 `KeyboardInterrupt` 的验收回调，核对恰好一次真实时间结算、失败回执、其余自有子进程收尾以及不进入生成。无需 GPU 或科研计算。本次审查未执行该测试。

### 5.4 唯一领取和失败政策可以保留

`campaign.mkdir()` 领取整批，任务 `job.mkdir()` 领取具名任务；worker 检查 PID、命令、cwd 与 deadline，再用 `worker_claim.json.open("x")` 阻止同任务二次入口。后处理串行，因此不能单凭 `prior.access()` 自身没有文件锁，就判定正式入口必然可以并发二次开 pairs。【T：`check_dispatch()` L207、`main()` L255；N：`step28_formal_queue.py::dispatch` 原 L175、L301；`step28_continual_five_evaluate.py::access` 原 L26】

两种 worker 失败政策均可由资源合同选择：

- `stop_batch`：任一失败停止整批；
- `retire_failed_slot_continue_healthy`：失败槽退役，健康槽继续领取；最后只要有一条失败，仍不进入生成和评价。

本包明确无自动重试，也没有同控制器自动 OOM 恢复。**这与一次性批次的定义相容，不是必须新增恢复功能的理由。**读 pairs 后若指标阶段失败，同样保留一次尝试而不自动重开。【T：`read_gate()` L90；N：`step28_formal_queue.py::dispatch` 原 L184、L284、L326】

### 5.5 费用口径及尚待固定的资源合同

普通完成和普通异常路径会把失败任务计入相应桶，采用从启动至停止和父端验收后的真实 elapsed，超过限额不截短费用。完整队列先预留所有训练和后处理时限，再加既有费用；generation、confirmation、analysis 分桶。【N：`step28_formal_queue.py::dispatch` 原 L178、L216、L229】

但该 elapsed 包含启动、导入、CPU 准备、轮询延迟及验收，不是纯 GPU kernel 时间。控制桶又计入扣除显式 sleep 后的控制器工作时间；验收可同时出现在任务生命周期账和控制账中。正式合同应明确这一保守口径，不把累计任务小时当成墙钟工期或纯 GPU 计算小时。HB-R02 修复前，也不能声称全部可捕获失败均已完整留费。

正式资源合同至少需要固定逐方法时限、三项后处理时限、并发、全部分桶、既有费用、整 PID 与 allocator 额度、他任务保留、磁盘和保存空间。还需注意两个具体接口条件：

- 调度器按 `seconds - 60` 留停止时间，而 `read_gate()` 只要求正数。正式任务时限应大于这 60 秒预留，并可加相应入口校验；不能把微型短测的 47 秒直接填为正式任务额度。
- gate 的 sources 会与当前部署源码逐项比对，但授权 JSON 的相等字段列表没有 sources。若正式许可承诺冻结精确生产来源，应把 sources 纳入授权比对，或让授权记录绑定冻结 gate 摘要。当前尚无正式门，此处是待落定的来源绑定合同，不是已经发生来源偷换的结论。【T：L101、L105、L115；N：`step28_formal_queue.py::dispatch` 原 L280】

S L32 披露旧 L²R 九条成功轨迹共 **299.627344 任务小时**，约每条 33.291927 小时。本包没有该历史账本，所以这只能作为方案提供的成本提醒，不能写成本次已独立复核的费用。它也不能替代其余五方法和全批盲评分的预算。

## 六、问题四：六方法 raw MAP 主表与科学解释

### 6.1 主表确实取 raw MAP，并做同顺序、同 seed 配对

`analyze()` 读取每格 `metrics["raw"]["matrix"]`，校验具名路径、文件哈希、22 列定义及 40×22 形状，抽出有限的 MAP 列后对 40 群等权平均。随后按阶段位置组合字段；每种方法再对九条固定轨迹等权平均。五个 H 减基线的比较，逐个匹配相同 order 与 seed，并保留全部九组差值。【E：`analyze()` L178、`field_means()` L154、`paired_table()` L161】

没有把不同确认批次混在一张配对表，也没有把 stage-cal／first-cal 冒作 raw。保存单位是比例，转换为百分点时乘 100。状态 `SIX_METHOD_PAIRED_TABLE` 是运行完成状态，不是科学效果通过。

### 6.2 六格与字段定义正确

令 M(s,p) 表示训练到阶段 s 时，对该顺序第 p 个域的 40 个新群取 raw MAP 均值：

| 字段 | 实际公式 | 可解释的量 |
|---|---|---|
| O | [M(3,1)+M(3,2)]/2 | 两旧域最终能力 |
| A_acq | [M(1,1)+M(2,2)]/2 | 两旧域各自初学完成时的能力 |
| F | [M(1,1)−M(3,1)+M(2,2)−M(3,2)]/2 | 两旧域从取得端点到最终的平均下降 |
| N | [M(2,2)+M(3,3)]/2 | 第二、第三域取得端点的平均能力 |
| Z | M(3,3) | 第三域最终能力 |
| final_mean | [M(3,1)+M(3,2)+M(3,3)]/3 | 三域最终均值 |

以上直接对应原 `field_terms()`，没有发现分母、正负方向或学习位置映射错误。【N：`step28_continual_five_evaluate.py::field_terms` 原 L225；E：L154】

两个恒等式应在解释中保留：

```text
F = A_acq − O
ΔF = ΔA_acq − ΔO
final_mean = (2O + Z)/3
```

因此，ΔF<0 可能部分来自 H 初学更弱；必须同时看 ΔO、ΔA_acq、ΔN 和 ΔZ。三域最终均值的改善也可能由第三域补偿旧域，不能替代旧域结论。

此外，A_acq 是“学习完成后的能力水平”，不是相对于初始模型的净学习增益。本轮六格不包含所有未学域／初始端点，不能从这些字段推导未保存的净获得量。F 可为负，表示最终超过取得端点；代码没有将其截断成零。

### 6.3 手写算术复核正确

测试给定六格：

```text
M(1,1)=0.8；M(2,1)=0.7；M(2,2)=0.6；
M(3,1)=0.5；M(3,2)=0.4；M(3,3)=0.9。
```

复核得到 O=0.45、F=0.25、A_acq=0.70、N=0.75、Z=0.90、final_mean=0.60。只把某一个 H 配对的 O 增加 0.09，九组平均差为 0.01，即 **1 个百分点**，与回执一致。这是手写公式验证，不是正式六方法效果。【U：`test_table_and_single_logical_opening()` L226；C：`paired_report`】

### 6.4 当前没有区间或联合通过标记，并非当前代码缺陷

本次明示交付配对点估计，实际源码也没有生成显著性或效果通过布尔值。若要作确认性论文主张，应在开训前书面固定：

1. 主比较及其家族：H 对五基线分别检验，还是要求某个预定联合结论；
2. O／F 的实际有意义幅度，以及 A_acq／N／Z 的保护条件；
3. 区间方法、单侧或双侧方向、同时性与多重比较处理；
4. 采样和配对单位，以及对三个顺序、三个 seed 的条件解释；
5. 全批失败、缺件及任何后续重做的处置，避免看结果后选择性保留轨迹。

同一 120 群被所有方法、顺序和种子共同使用，九个固定配对不是九套独立新数据；12,960 次群前向更不是 12,960 个独立样本。若以后做重采样，必须保留方法间、阶段间以及同群被重复使用的对应关系。具体区间方案可另定，本轮不替用户擅自新增阈值或改变旧门。【S：L24；E：`paired_table()`】

这批新群能够回答固定试验条件下六方法的共同确认问题。它不能把旧异批比较自动升级为直接配对，也不能单凭同机制合成群结果推出真实市场马甲识别效果。

## 七、CPU／CUDA 手写证据能支持到哪一步

### 7.1 应分清四种不同层级的证据

| 证据项 | 实际做了什么 | 可以支持 | 不能据此声称 |
|---|---|---|---|
| 六方法分派 | mock 三类 runner，观察 `train()` 的目标、spec 和参数 | 新入口选对原生 runner | 已经通过正式 runner 完成一条完整轨迹 |
| 六方法单步更新 | 微型网络调用实际 update，整体 state digest 改变，输出378个有限得分 | 原生单步参数更新和评分接线可运行 | 正式 BGE 已完成864步，或每个组件梯度都独立验收 |
| train-only checkpoint | ER0.1 微型网络，12校准群，valid=None，真实保存与 full／inference 恢复 | 该端点分支保留校准、Memory和恢复 | 六方法全部阶段均动态走完原生 checkpoint |
| 六方法盲恢复 | 每法3个手写微型端点，patch fresh，实际 restore／score／保存与双校准比对 | 六种端点格式和推断评分路由可接通 | 162个正式端点已由训练产生 |
| 324格结构门 | 构造完整元数据字典，再删除格或打乱群序 | 完整性和对齐门的相应正反例 | 已实际算完324格盲分数 |
| 一次开 pairs | 一个手写标签群，第二次调用拒绝 | 串行逻辑访问登记有效 | 完整正式队列、故障恢复或并发压力已测试 |

依据：U 的 `test_six_native_training_routes_and_real_update()` L24、`test_endpoint_with_only_current_calibration()` L104、`test_missing_receipt_and_misaligned_blind_cells()` L130、`test_blind_loader_all_six_native_endpoint_formats()` L160、`test_table_and_single_logical_opening()` L226。

### 7.2 Adam 289 和三种方法的历史分支覆盖

H／ER／Logit 的测试把 Adam step 手填为 288、动量手填为零，然后实际更新一次到 289。**它证明一个人造第二阶段边界后的真实一步，不证明此前发生过288次训练。**这一设置在问题说明和测试注释中都已披露，没有必要将其误判为伪造正式训练。【U：L80；Q：L10】

L²R／GCR 只做第一阶段的一步；FOREVER 只调一次 current update。因此，三者的后阶段历史选择、时钟回放、巩固和锚点行为没有被本次短测动态覆盖。源码调用存在与这一动态覆盖不足应同时报告。

### 7.3 实际动态覆盖是每设备18端点、36格

盲恢复测试在每种设备上，对六方法各构造 ABC／seed 4 的三个端点。端点由手动参数增量、手写映射与计数产生，保存时没有 optimizer；恢复时 fresh 被替换为微型工厂。随后实际调用原生恢复和评分，处理同一套120个手写群。

每种设备的动态规模为：

- 6 × 3 = **18 个微型端点**；
- 6 × 6 = **36 个盲格**；
- 6 × 240 = **1,440 次群前向**。

`complete_cells=324` 属于另一个元数据结构测试，其中分数记录仅具备路径字段；不能把两个测试拼接成“已动态完成324格”。【U：L139、L183、L196；C／G：`native_blind_restore`】

ER 端点测试的 Memory 序列化为 CPU **64,352 字节**、CUDA **64,367 字节**，都小于1 MiB。它证明对应手写实例的往返，不证明所有方法或正式数据下的 Memory 上界。【C／G：`train_only_endpoint`】

### 7.4 CUDA 回执数值相容，但不能外推正式容量

| 项目 | 所附回执 |
|---|---|
| CPU | 6项，通过；6.548753秒 |
| CUDA | 3项，通过；exit_code=0，fault=null |
| CUDA 测试内部时间 | 44.745079秒 |
| CUDA 外层生命周期 | 46.601874秒 |
| 单列47秒 | 与外层生命周期向上取整相容 |
| 整 PID 采样峰值 | 748,683,264字节 = **714 MiB** |
| allocator allocated峰值 | 68,814,848字节，约65.627 MiB |
| allocator reserved峰值 | 69,206,016字节 =66 MiB |
| 短测上限 | 整PID 1.5 GiB、allocator 512 MiB、180秒 |
| 正式预训练／正式数据 | 回执均记0次加载／读取 |

上述是回执及简单单位换算。测试源码确实调用梯度检查点相关设置、真实 optimizer 和评分路径，但其断言主要是整体模型状态改变及评分有效，不应扩大为逐层 encoder／head 梯度全覆盖。

`other_worker_remained_verified=true` 是外层回执的声明；包内没有他任务 PID／内核启动身份的完整原始观测序列，也没有该采样器完整源码。因此本次不能自称独立重新核验了“同卡其他正式任务全程未受影响”。【G；S：L28；U：L47、L65、L77、L97】

本包也没有实际运行完整 `queue.dispatch()` 的证据，未动态覆盖正式预算耗尽、worker OOM、健康槽续领、控制器中断收尾。正式 BGE 并发容量与时长应由适用的原生规模依据和另外批准的资源合同决定，不能由714 MiB／46.6秒按比例外推。

## 八、必要处置、补证边界与最终意见

### 8.1 开训前必须修复的范围

**只对两项已定位问题要求代码修复：**

- HB-R01：generation 前重新核验全部必要推断端点。
- HB-R02：受控中断也必须完成失败计费与收尾，避免活动槽提前移除导致漏账。

修复应附对应的两个定向拒绝／失败用例、更新后的源码摘要及主审处置。本轮不要求重新训练、重新跑全部既有资格、增加确认群或建立通用恢复平台。

### 8.2 必须另行固定，但不应伪装成本次算法缺陷

正式开训仍需落定统计分析约定、资源与全部费用分桶、失败槽政策、精确来源绑定、实际配置及共同分区证据。尤其应把 trial3 的完整 native identity、原排序权重与学习率倍率写进冻结材料，不能用“沿原设置”替代可复核的值。

本包仅提供原生节录。未附的配置／fresh／Supply、L²R与GCR选择内部、FOREVER事件与巩固内部，以及控制器 stop／资源采样／Budget 内部，不在本次完整源码复核范围。可复用既有审查，但应保持“本次独立核到哪一层”的表述准确。

### 8.3 非阻断文档小修

S L28 的 CPU／GPU 链接写成 `cpu.json`、`gpu.json`，附件实际名为 `synthetic_cpu.json`、`synthetic_gpu.json`；Q 开头“只交付实现与合成 CPU 证据”的旧表述也宜同步为 CPU 与 CUDA。两份实际回执存在并已核验，这些编辑问题不否定回执。

### 8.4 最终放行意见

**科研比较结构：可保留。六方法接线和配对点估计：在上述证据层级内认可。当前正式入口：暂不签署开训通过，先闭合 HB-R01 与 HB-R02。**

两项修复闭合后，应结合另行冻结的分析约定、真实规模资源预算和本批明确授权决定是否开训。没有必要因为当前只交付点估计入口，就重写方法、补造效果门或重开全部历史审查；同样不能把手写 CPU／CUDA PASS 写成正式训练、全批盲评或 H 优于五基线已经得到证明。
