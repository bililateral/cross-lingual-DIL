# 当前 H 消融：暂停派发、显式恢复与一次性控制器交接定向外审

日期：2026-11-07，Asia/Shanghai。当前一个审查实例完成，未委派子 agent。本轮只审查新增控制路径及其证据，不重开旧科学实现、预算、效果或 GPU 资格。

## 一、结论

**当前版本尚不建议进入对活动旧父进程的实际交接。存在两项可由具体代码路径推出的局部实现必修；普通暂停、真实退出码读取、原账和失败槽承接的主体设计可保留。**

1. **HPAUSE-01：冻结一致性检查漏掉“待派发任务目录已经创建、子进程尚未启动”的中间状态。** 该状态能够通过现有 dispatch/子进程比对，交接后恢复到同一任务时，新的 `job.mkdir()` 会因目录已存在而使批次失败。
2. **HPAUSE-02：生成/分析的辅助时限在暂停前按余额计算，暂停中的主动检查费用会使该预留过时。** 显式恢复时，可能在辅助桶仍有充足余额、也尚未启动该后处理的情况下，被过时的“完整下一任务预留”拒绝。

两项均不需要新科研实验、BGE 载模、GPU 资格或通用恢复平台。修复并用两个定向 CPU 反例闭合后，可沿现有单次交接设计继续主审；未变证据按适用范围复用。当前 `request.draft.json` 应保持 `approved=false`，不据本报告对旧父发送 SIGSTOP/SIGKILL。

**本报告不要求停止或重训当前 workers，也不撤销原 H 消融正式运行许可。** 包内新的开训授权已经取代本会话上一轮“正式训练关闭”的历史状态；当前关闭的是这次尚未结案的控制器交接门。开放科学方案问题 0，开放定向实现必修 2，新增 GPU/科研实验要求 0；无须另外制造可选扩展。

## 二、附件身份、模型身份与证据边界

### 2.1 附件核对

| 项目 | 独立机械核对 |
|---|---|
| 名称 | h_ablation_pause_review.zip |
| 大小 | 64029 字节 |
| SHA256 | `f463faae97642d63de08298f5ff3d8dea5c95b54fa0505f871fe005d32f8f0a1` |
| 成员 | 25：24 个载荷加 package_manifest.json |
| 内部清单 | 24 项大小与 SHA256 全部一致，无缺项或额外载荷 |
| ZIP 检查 | CRC 通过，无重复成员、不安全路径或符号链接 |
| package_manifest.json SHA256 | `f3bdef63a40781e34ed30b8b7525866428e68466e538826f2c42422b15188771` |

最终四来源的实际字节同时符合 `pause/source_manifest.json`、`pause/attempt03.json`、`pause/request.draft.json` 中的身份。四份 `frozen_context` 源码也逐一匹配原正式 gate 的对应来源。这里的身份核对不等于本审查重新访问了远端全部 50 来源。

### 2.2 模型身份

- 网页选择器：本轮未独立观察或操作。包内 `context/review_primary_disposition.zh.md:7` 记载的是前次提交者观察到 GPT-6／Pro，不能冒称本轮新观察。
- 自述身份：当前运行上下文为 GPT-6 Astra Pro；当前一个实例，未主动换模型或委派其他模型。
- 后台身份：未取得可独立验证的后台模型 ID、路由或服务档位证明。没有将网页记载、自述名称与后台认证混为一谈。

### 2.3 实际阅读与未核范围

先核包身份，随后完整阅读 `pause/review/request.zh.md` 和 `pause/scope.zh.md`。完整阅读三份当前源码、最终测试源码、三轮 attempt JSON/日志、request draft、authorization、budget、observation、source manifest，以及原开训授权和前次主审处置。对原正式 gate 作结构化字段、来源、队列及授权核对；对新旧两份控制器作完整差异检查，读取冻结 runner 的门、验收及调用路径，以及冻结 queue 的 NVML/资源检查直接依赖。逐文件深度列在 `reading_scope.json`。

本实例仅执行附件解压、文本/AST/JSON读取、哈希、差异和机械算术。**没有连接项目 Linux，没有执行项目脚本或测试，没有发送任何进程信号、创建暂停标志、改 request/gate，也没有读取模型、正式数据、预测数组或标签。** CPU 运行事实来自包内主执行者回执，本实例没有将其重新执行一遍。

未核：当前远端父子进程及资源状态、实际交接效果、正式 GPU/NVML 实测、新控制器完整接管后的正式任务；原 50 来源中未附的其余 46 项实际文件；`original_launch` 所指 launch.json 原文；scope 提及但本包没有成员的 `cpu02_sources.zip` 和 Git 941dc8bb 初版源码；前次 CPU04—06 原始回执。后者只按包内主审记录及未变范围复用，不冒称本次已读原件。

包内 18:50:39 观察是 **4 完成＋3 在跑＋56 待派＝63，0 退休，无 batch failure，pause 标志不存在**。旧父身份与 draft 的 PID、start_ticks、属主、命令和 cwd 一致。这只是该时点的回传观察；不能充当稍后交接瞬间的现场证明。

## 三、实际调用路径与已接入行为

### 3.1 普通暂停及显式恢复

新 `dispatch` 在每个待派发槽取下一条前读取 `dispatch_paused()`。暂停不跳过前面的 ledger、资源检查、GPU 快照、退出检测、验收、结费和释放。循环末保存 `PAUSED_DRAINING` 或 `PAUSED_IDLE`；即使 pending、active 都为空，只要仍 paused，就继续等待，不能自行返回下一阶段。

训练及 generate/confirmation/analyze 都经控制器传入相同的暂停回调。`change(..., paused=False)` 只有在初次排空交接已经进入 `PAUSED_CONTINUATION` 后才移除标志；命令返回只是请求，不冒称控制循环已经确认。scope 已明确最后一次检查与实际启动之间的一条在途派发窗口，可按下一份状态确认，不需要改成强制杀掉已启动任务。

证据：`scripts/step28_history_optuna_control.py:143–277`；`scripts/step28_h_ablation_control.py:131–181`；`scripts/step28_h_ablation_pause.py:305–318`；`pause/scope.zh.md:7`。后处理预算的暂停增量缺陷见 HPAUSE-02。

### 3.2 一次性交接

`takeover` 先检查交接 approved、review 和更新源身份，获得一次性锁，并加载独立的新控制器/派发器；科学入口及原 gate 仍从冻结 workspace 导入。它只对旧父发送 SIGSTOP。`freeze` 核旧父身份、停止状态、保存 dispatch 和直接 children 对应，以及每个活动 worker 的命令/父进程/cwd/槽位；初始化前置失败会恢复旧父。

`drain` 使用 `HeldChild` 读取仍由停止的旧父持有的 zombie 的内核退出状态。桥接不派发新任务，继续检查在跑 workers、退出、截止与资源；成功仍调用冻结 `validate_completion`。单任务失败加入原退休槽集合；全排空后形成 continuation，才杀掉旧父，再在暂停状态下进入新版控制器。该路径不是训练 checkpoint 恢复，不重新构造在跑模型或 worker。

证据：`scripts/step28_h_ablation_pause.py:38–79,102–240,243–302`。冻结边界仍有 HPAUSE-01，不能只因流程顺序合理就判整个交接已可执行。

## 四、逐项发现

### HPAUSE-01｜确证实现必修｜冻结检查不能排除半完成的派发准备

**严重度：REPRODUCIBILITY_DEFECT；阻断当前版本的实际一次性交接，不否定原科学实现。**

旧父派发的顺序是：`job.mkdir()` → 打开 console → command 写 spec → `Popen` → 登记 active/process → 移除 pending → 保存 dispatch。`freeze` 只比较前后 dispatch、实际 children 和已保存活动 workers，没有检查 pending 对应目录是否已被旧父创建。

**精确静态反例：**在一个合法的两 worker 在跑、一个健康槽空闲的状态下，旧 dispatch 已保存两条 active 和下一条 pending。下一轮资源满足，旧父为该 pending 创建目录后、Popen 前被 SIGSTOP。此时：

1. 保存的 dispatch 没变；
2. 旧父的 children 仍正好是两条保存的 active；
3. 两条 worker 的身份、命令、父进程和 cwd 都正确；
4. 没有 campaign failure/completion。

因此当前 `freeze` 可以通过，原 workers 随后正常排空，下一条仍留在 pending。旧父被退役后，显式恢复触发新版 `job.mkdir()`，已有目录造成 `FileExistsError`，继而进入 STOP_NO_RETRY。这个结果由源码执行顺序决定；**本审查没有在正式进程或新建本地进程上演练此反例，也不声称 18:50 的三 worker 快照已发生该状态。**

**证据位置：**`frozen_context/step28_history_optuna_control.py:219–238`；`scripts/step28_h_ablation_pause.py:102–139,200–216,280–292`；`scripts/step28_history_optuna_control.py:238–255`。最终三子进程 CPU 测试是在三个 child 及 dispatch 都写完后让假父 sleep，不覆盖该中间窗口：`tests/test_step28_h_ablation_pause.py:205–219`。

**最小修正：**在旧父已 STOPPED、尚未提交 frozen/账本之前，将“队列边界”同时与任务目录状态核对：pending 的作业目录不得已存在，且不能已进入后处理准备。发现半完成准备就走现有 SIGCONT 回滚，保留旧父继续完成自己的动作，并明确本次未成功接管；不自动重试、删除半建目录或放宽新任务目录创建规则。当前用途只需一个小范围边界拒绝检查，不需要事务平台或任意断点恢复。

**最小核验：**受控 CPU 夹具把旧父停在 mkdir 后/Popen 前，断言交接拒绝、旧父恢复、原队列/账本未被桥接提交；也可对等地模拟这些已固定的边界状态。不能仅再次运行“三个子进程已稳定启动”的成功例。

**影响范围：**一次性交接在合法派发边界的可恢复性、同一队列连续性。仅限制论文主张不能修复运行错误；不需要科学实验。

### HPAUSE-02｜确证实现必修｜长暂停会使生成/分析的整桶余额预留失效

**严重度：REPRODUCIBILITY_DEFECT；影响普通显式恢复后的辅助后处理。**

控制器进入后处理前一次性计算 `seconds = caps[bucket] - used[bucket] - 120`，再把固定 row 交给可能长时间暂停的 dispatch。问题发生在暂停于后处理 row 已形成、尚未启动的窗口被接收时，并非每次暂停都会失败。暂停中的主动检查按合同计辅助费。恢复时仍用旧 row.seconds 检查“已用＋预留”，生成和分析恰好与这些检查共用辅助桶。

**机械反例：**辅助 cap=14400 秒，构造 generation row 时 used=1000，于是 row.seconds=13280。暂停中的主动检查累计121秒，恢复时 used=1121，实际余额仍13279秒；但旧预留检查得到 `1121+13280=14401>14400`，因此在尚未启动该任务时就失败。若按实际派发时的余额使用原公式，任务限额应为13159秒，仍满足原120秒余量且远大于60秒。这里的121秒是主动检查累计费用，不是纯 sleep；反例没有扩大或豁免预算。

**证据位置：**`scripts/step28_h_ablation_control.py:16–23,162–181`；`scripts/step28_history_optuna_control.py:204,219–225,237–246,256–263`。测试中的 `Ledger.control()` 是空操作，初始 CPU 后处理等待例用固定10秒任务，不模拟暂停期间辅助余额变化：`tests/test_step28_h_ablation_pause.py:34–45,82–96`。

**最小修正：**只对原本使用“剩余桶余额”的 generation/analyze，把该时限计算移到最终真正派发前，并让预留检查、传给子进程的 deadline 和回执的任务时限使用同一个新值。训练仍固定21600秒；确认的独立桶及全部 cap 不变；余额确实不足仍停止，不跨桶借用或自动加预算。

**最小核验：**一个 CPU 可控账本反例，在 paused 时累加121秒主动费用、桶尚充足，再显式恢复，确认按当前余额预留且可派发；另保留余额真正不足时停止。无需等待真实长时间或启动生成器。

**影响范围：**长暂停后的生成/分析连续性和停止原因准确性。单靠说明“长等待不豁免预算”不能消除这个错误，因为反例中的桶没有耗尽。不是新增科研实验需求。

### HPAUSE-03｜核对通过及解释边界｜暂停确实覆盖派发，最后任务完成后也等待

主体暂停逻辑放在派发点，没有用暂停替代监测循环；训练与三个后处理均实际传入回调。`PAUSED_DRAINING` 表示仍有活动任务，`PAUSED_IDLE` 表示该 dispatch 已无活动任务；后二者都不是整个 campaign 科学成功。完全完成仍暂停的情形不会越过下游门，失败/预算上限也不会因等待而豁免。

**证据：**`scripts/step28_history_optuna_control.py:200–277`；`scripts/step28_h_ablation_control.py:157–181`；最终测试 `:82–129`。CLI `change` 的身份核对、阶段限制及请求回执见 pause 源码 `:305–318`；九项测试使用 Event 模拟暂停信号，没有实际调用该 CLI 的完整 pause/resume 正向路径，后者是本次静态接线判断。

**最小处置：**除 HPAUSE-02 外无额外必修。继续以保存的 dispatch 状态确认生效；不把“已发请求”写成“从这一时刻绝无一条在途启动”。影响范围：用户可见暂停语义。无需追求跨全部指令的瞬时零窗口保证。

### HPAUSE-04｜核对通过及证据边界｜旧父保留、真实退出码、监测与失败规则

只冻结旧父；当前 workers 使用原 PID、进程组、参数、模型和截止时间。保持旧父停止，使其不能提前回收 zombie；`HeldChild.poll()` 校验 start_ticks 和 parent，从 `/proc/<pid>/stat` 提取退出状态，并用 `waitstatus_to_exitcode` 解释。completion 存在不等于 exit=0：真实退出非零优先成为失败，只有退出/释放成功才执行原验收。

桥接直接对旧 worker PID 获取 NVML 使用值并续存旧峰值记录，不能使用“桥接自己的 children”来代替这些旧子进程；代码实际显式枚举 rows。RSS 合计包含桥接、旧父和原 worker 家族，补足普通 `check_running` 只遍历当前父系的限制；共享磁盘门、单条 deadline−60秒、进程10.25GiB门仍执行。个别失败退休槽位，不重跑；共享故障清理本批活动组，清理错误明确留账。

**证据：**`scripts/step28_h_ablation_pause.py:38–79,160–240`；`frozen_context/step28_formal_queue.py:65–98,139–168`；`frozen_context/step28_history_optuna_control.py:47–79`；测试 `:175–254`。非零退出7的真实 CPU child 证据支持退出状态读取；三 child 排空例也保留失败槽1，只验收成功的0、2。

**最小处置：**无独立新增必修。`released=True` 在这条路径表示已无存活 worker 进程组、资源已释放，不表示旧父仍持有的 zombie PID 已从 `/proc` 消失。GPU/RSS超限和异常清理分支本轮主要由源码检查支持；CPU 夹具将 GPU 对象换成空进程表，并没有验证正式 NVML/显卡行为。影响范围：监管连续性和退出证据的正确解释。

### HPAUSE-05｜核对通过及现场前提｜排空后的同一队列、失败槽和费用承接

continuation 只接受 active 为空、sources 一致的 dispatch，逐项检查完成回执 index/id/released/slot，拒绝重复 index；用原行号重建 pending，并要求与保存列表完全一致；失败回执重建的退休集合必须等于原集合。因此已完成轨迹不重派、失败槽不复活；存在失败时，即使健康槽完成余下任务，整批仍不通过下游门。

桥接和新控制器都用 `Ledger.__new__` 承接原 caps、used 和 events，不重调构造器、不再收一次 final_tail。原 worker 费用从其命令 deadline−21600 推得原启动点，到释放时计 training；验证及控制按辅助账。纯轮询/资源等待 sleep 排除，主动检查仍计费。新控制器重新使用原科学入口、原 gate、原63清单，batch 结构核对后不覆盖原文件。

**证据：**`scripts/step28_h_ablation_pause.py:89–99,128–135,188–216`；`scripts/step28_history_optuna_control.py:149–162,176–199,264–277`；`scripts/step28_h_ablation_control.py:119–160`；测试 `:131–164`。

三轮 CPU elapsed 合计 **19.724903613096103秒**。600秒是包含这些 CPU 的保守准备记账值，属于原辅助14400秒，不是新增额度；包内明确尚未写入活动控制器账。本轮观察中四条完成 training 事件求和 **45969.69135101582秒**，与 used.training 一致。600只应在实际接管结算时加入一次；如果现场后续准备超过这一保守数，应更新真实费用，不能截断或清零。

**最小处置：**上述承接机制无独立必修；后处理动态预留按 HPAUSE-02 修。现场继续用冻结后最新账，不拿18:50快照覆盖后续已发生费用。影响范围：队列连续性和预算解释。原启动30秒准备费、既有final_tail和本次600秒不得混成同一项反复收费。

### HPAUSE-06｜核对通过及授权边界｜冻结科学来源与新控制器来源分开保管

新版控制模块从独立路径加载，依赖仍来自固定 workspace；`run.gate()`继续检验原50来源、63身份、预算及原授权。新 controller_update 另存更新来源和审查绑定，进入 dispatch、最终 completion 和 execution_time；不伪称新控制代码早已包括在原50项中。旧父日历段保留 original_launch 引用及结束信息，新段另有开始/结束/含等待日历；累计任务费用不分段清零。

交接 `approved=false` 在 `takeover` 最前面拒绝，早于源码加载、状态目录和进程检查；最终第九项负例覆盖这条路径。原正式 gate 则是 approved=true、review_closed=true，且包含四动作，其有效开训授权不能被前次历史“训练关闭”覆盖。

**证据：**`scripts/step28_h_ablation_pause.py:243–279,290–302`；`scripts/step28_h_ablation_control.py:108–114,194–202`；`scripts/step28_history_optuna_control.py:168–174`；`frozen_context/step28_h_ablation_run.py:22–67`；`context/h_ablation_start_authorization.json`；`context/h_ablation_start_formal_gate.json`；测试 `:166–173`。

**最小处置：**无额外代码必修；HPAUSE-01/02修改后应重新绑定实际更新来源及定向证据，由主执行者独立处置，再决定是否填写交接批准。当前报告交付不自动把 draft 改 true。影响范围：来源可追溯性、历史授权与本次动作边界。

### HPAUSE-07｜现场前提｜不能把静态审查和18:50观察当成功交接

实际接入必须使用同一活动旧父及其当前 PID/start_ticks/属主/命令/cwd、当前保存 dispatch/worker/process/账本对应；旧父停稳后再确认一致边界，包含 HPAUSE-01补齐的未完成派发准备排除。实际状态仍应处于本实现支持的训练队列交接阶段，没有已经开始的后处理或封闭 campaign。若现场已变，不应凭旧快照继续发信号。

桥接要能持续运行到排空并承担监管；全组释放后才退役旧父。初始交接排空尚未完成时，不应直接删除 marker 绕过阶段检查；显式恢复应经既定入口。仅看到 PAUSED_IDLE 也不证明新控制器已完成来源验证、接管和记录；应同时看 state/controller 的交接阶段及活动进程身份。

**证据：**`pause/observation.json`、`pause/request.draft.json`；`scripts/step28_h_ablation_pause.py:102–139,243–318`；`pause/scope.zh.md:13–19,29`。

**最小处置：**这些是后续实际接入前提，不要求本次外审连接 Linux、冻结正式父进程或追加正式演练。影响范围：本次静态结论能否用于稍后的具体进程。现有观察支持“当时尚未暂停”，不支持“此刻已成功交接”。

### HPAUSE-08｜证据边界｜最终 attempt03 九项有用，但不覆盖两项新缺口

| 最终测试 | 可支持的结论 | 明确边界 |
|---|---|---|
| initial_pause_then_resume_cpu_followup | 初始暂停阻止CPU后处理，解除后只派一次 | 假Ledger.control不计费，不覆盖HPAUSE-02 |
| running_job_drains_and_last_job_waits | 在跑任务正常退出，最后一条完成仍等恢复 | 手写短进程，无模型/GPU |
| three_slots_finish_without_taking_pending | 三槽排空时不领取第4/5条，恢复续领 | 暂停在三条均已启动后 |
| continue_preserves_completed_and_retired_slots | 完成不重派、失败槽保留、健康槽续领、原失败仍挡整批成功 | 构造的continuation，验收被替代 |
| continue_rejects_changed_queue | 错pending清单被拒 | 不覆盖半建目录状态 |
| ledger_does_not_reset_or_double_final_tail | 原费用/事件保留，只加一次准备费，不重提尾费 | 是构造账本核对，不是正式总账审计 |
| closed_handover_cannot_touch_processes | approved=false时takeover不接触进程、不建state | 只测关闭负例，不是完整批准接管正例 |
| held_child_exit_status_and_release | 停止假父后仍读取真实child退出7并释放 | 真实Linux CPU进程；不是正式worker |
| actual_freeze_and_drain_three_cpu_children | 实际SIGSTOP假父、三CPU child原地运行至排空、失败槽和费用保留 | GPU空表、gate/验收/资源检查替代；不执行正向takeover→退役旧父→新controller.execute整链，不覆盖mkdir/Popen窗口 |

代码位置为 `tests/test_step28_h_ablation_pause.py:82–254`。attempt03日志九个名称与当前 AST 中九个测试逐一对应，四来源全匹配，状态PASS/returncode0，外层 elapsed=6.982622978975996秒；日志内测试计时5.522秒是另一层口径，并不矛盾。

attempt01/02分别7项、8项PASS，外层4.768837294075638秒和7.973443340044469秒；它们有旧版本身份，不能改写为对最终四来源全部执行的证据。最终九项本身已覆盖其适用行为。scope所说cpu02_sources.zip未入当前25成员包，本审查不声称读过它，也不要求为旧中间版再扩包/重测来替代最终证据。

**最小处置：**保留这九项适用证据，只针对HPAUSE-01/02补能区分错误的边界反例；不自动重做全部旧科学或GPU资格。影响范围：实现资格的可支持程度。CPU玩具确实执行了Linux进程/信号的部分接线，但不是正式进程或GPU验证。

### HPAUSE-09｜过度设计｜本轮不应扩成通用恢复或新实验

不要求把暂停扩展成任意训练checkpoint恢复、跨主机容灾、通用事务队列、应对任意外部管理员改信号的防护系统；不要求改变63条队列、失败不重试、394小时桶、八配置、seed或科学主判据；不要求新增GPU压力/资格、BGE短测或科研实验。

两项必修都来自本次新增暂停/交接对现有操作时序的具体影响；它们不是要求系统对所有可能故障提供自动恢复。修正后的“边界不一致就回滚/拒绝，仍然不自动重试”符合本轮范围。没有另外必须列出的可选扩展项目。

**证据：**`pause/review/request.zh.md:3–14`、`pause/scope.zh.md:3–19,21–29`、`pause/authorization.json`。最小处置是不追加上述范围。影响范围：保持研究与运行边界。

## 五、结案与下一步判断

本轮完整定向审查可以交付，但当前源码不能据此标成“无开放实现必修”。建议主执行者：

1. 补齐冻结时半完成派发准备的拒绝/回滚检查；
2. 将 generation/analyze 的剩余辅助额度限时计算移到真正派发点；
3. 用两个受控CPU边界反例及实际修改后的来源身份闭合，再独立处置；保留已有适用九项证据。

在此之前保持本次交接draft关闭，不改变正在运行的原控制器/worker。修复后仍须满足现场身份、来源、队列、账本、资源与接管阶段条件；本报告不冒称这些已经满足。成功暂停/交接的事实必须来自后续回执和活动进程观察，不能由本报告或CPU PASS代替。

交付支持材料中的两个反例均明确标为静态执行序列/机械算术，没有运行项目实现或正式进程。原科学方案、旧负结果、已关闭的未变审查和原预算仍保持。
