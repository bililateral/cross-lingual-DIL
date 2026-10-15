# F 终端 Q 合同补充与最小实现合并外审

审查日期：2026-10-15。审查对象仅为本次获准上传的合同、源码和保存证据。本次未连接项目服务器，未使用 GPU、BGE、正式文本、标签或旧模型／Memory；未委派其他审查者。执行范围另列于第八节。

## 一、分项裁决

**当前合同与实现：未发现新的 SCIENTIFIC_BLOCKER，也未发现必须先修改生产算法才能进入固定原生资格的实现缺陷。** 实际代码实现的是互斥 O／T、仅实际第 4096 次开放 T、返回同次 Q、保持原输出精度和安全检查；实际包络反向、分块 VJP、统计与盲门、来源继承及剩余预算已经接入，不是占位提案。

**本次增量 CPU 证据：接受其作为当前来源下、限定手写输入和受影响路径的资格证据。** 此结论来自原始失败／成功日志、实际源码、独立参考适用条件和针对性复核，不以“四项 PASS”代替判断。首轮失败有效保留，26 秒没有清除；第二轮没有放宽求解或梯度容差。但“第二轮只改 fixture”和“每轮一次轻量 AdamW 调用”的字面口径需作两处非阻断性勘误，详见第七节。

**后续执行：当前没有新版本 GPU／BGE 原生通过证据，当前不能正式运行。** 合并外审回收及独立主审处置后，可按已经批准的补充推进一次固定三次 F 原生资格；只有其真实结果、当前来源、计时投影和资源门全部满足，才进入已批准的一次完整三臂新运行。这里不要求用户重复批准相同补充，也不把本报告当作新的执行 gate 或本次网页审查对项目 Linux 的操作许可。

授权依据为 `BP/authorization.json:3–13` 与 `amendment:7–13、96–102、138–156`。合同形成时的“待批准”仍是历史事实，已被对应授权在其明确范围内取代。报告正确保留了“尚未执行 GPU”的状态（`BP/report.zh.txt:10–11、25、28`）。

### 证据路径缩写

以下所有行号均对应原文件，不是本报告重新排版后的行号；单行 JSON 用“L1＋字段”定位。

| 缩写 | 原包路径 |
|---|---|
| BP | `reports/documentation/20261015/balanced_pooling_repair/` |
| CPU1／CPU2 | `BP/qualification/cpu/attempt01/`、`BP/qualification/cpu/attempt02/` |
| solver | `scripts/step28_balanced_pooling.py` |
| run | `scripts/step28_balanced_pooling_run.py` |
| verify | `scripts/step28_balanced_pooling_verify.py` |
| tests | `tests/test_step28_balanced_pooling.py` |
| policy／shell | `schema/step28_balanced_pooling_policy.json`／`scripts/run_step28_balanced_pooling_linux_20261015.sh` |
| amendment／original | `docs/SELLER_ALIAS_BALANCED_POOLING_AMENDMENT.zh.md`／`docs/SELLER_ALIAS_BALANCED_POOLING.zh.md` |
| OLD | `reports/documentation/20261014/balanced_pooling_implementation/` |
| FAIL | `reports/seller_alias_continual/20261015/balanced_pooling_failure/` |

## 二、实际包身份与继承范围

收到的文件为 `review_input(20261015-032717).zip`，实际 **497,290 字节、66 个 ZIP 成员**，SHA256 为：

`c1ba62c3b84368a66e3276bee1c66fef8a346d156d70b7eb85e5a915e8cc4f57`

三项均与请求吻合。实际清单位于 `BP/review/manifest.json`，不是包根：65 个载荷的原路径、大小和 SHA256 全部匹配，另计清单自身；没有清单外载荷或缺项。文件工具未提供 ZIP 内正文，故本次在隔离工作区解包读取，并对全部成员逐字节核验；不是从请求摘要推断文件内容。

`BP/source_manifest.json` 的 **30 份来源**与 CPU2 `result/result.json` 的 `source_files` 完全一致。我还不加载模型地按 AST 重建 `run.sources()` 的依赖闭包，结果同为这 30 份。CPU1 内嵌 `source.zip` 的 37 个成员也已核对，其中 30 份来源逐项匹配 CPU1 的原始 result。`BP/qualification/received_files.json` 的 14 个回传成员记录均匹配本包原始输出。

旧 `formal_gate.json` 的 29 来源与旧 native result 的来源完全一致。与当前相比，公共交集中 **23 份不变、5 份改变**；旧 shell 退出当前来源，新 shell 与补充合同进入当前来源。将 `implementation.patch` 对五份当前文件反向应用到隔离副本，得到的五份旧文件大小／SHA 全部精确匹配旧 gate。这既支持补丁与实际文件一致，也使未变部分继承有字节依据。新 shell 和补充合同是单独提供并实际审读的文件，不把五文件差异补丁冒称完整部署清单。

本次核验的边界同样明确：输送用 ZIP 已按保管说明删除，不能据保留的成员回执独立重验历史 SFTP 传输包本体；报告中的服务器实时观察不是本实例亲见。这里确认的是**收到包的身份、保存记录的一致性及其源码解释**，不是远程主机的独立现场认证。上述边界不需要补传正式资产或重建保管系统。

机械输出见返回包 `checks/mechanical_audit.json`、`checks/input_identity.json`、`input_inventory.tsv` 与 `attempt01_archive_inventory.tsv`。

## 三、F 停止条件、返回值与反向接线

### 3.1 O／T 是实际互斥分支，不是更改日志

实际代码先完成全部原数值检查，再计算：

\[
\begin{aligned}
qpass&=(q\_error\le10^{-12})\land(checked\_gap\le5\times10^{-9}),\\
O&=qpass\land(p\_error\le10^{-10}),\\
T&=qpass\land\neg O\land(actual\_iteration=4096).
\end{aligned}
\]

在 `qpass` 为真的条件下，`~ordinary` 正好对应 P 残差超过旧门槛。因此 O 与 T 不重叠；第 4096 次满足 O 的块不会误计 T。判断使用冻结 `SOLVER.maximum_iterations==4096`，不是本次传入的局部迭代上限；`maximum_iterations=1` 不会因为循环结束就得到终端例外。第 1—4095 次仍只有 O，已接受块继续从活动集合移除。（`solver:65–90、120–126`）

原非有限输入／结果、负可行化缺额、Q 非负、显著负 gap 等检查均仍在接受之前；`checked=max(gap,0)+eps`、`eps` 的原系数 256、Q 门和 gap 门未改。实际返回是 `plans=q`、`values=mu+primal`，没有改填 P、常量、旧计划或另开求解器。（`solver:70–76、91–126；amendment:30–41`）

这里需要精确措辞：**中间 P 的验收条件确实改变了；输出误差要求没有放宽。** 本版不能称“所有验收条件完全未变”，也不能仅称日志修复。新版本身份 `log_sinkhorn_terminal_q_v1` 将允许返回集合的变化与旧版区分开。

### 3.2 数学依据成立，但不是任意浮点输入的严格证书

对均匀边际、中心化 W，令中间量 \(P=U\exp(W+a+b)\)，以及

\[
D=\sum P-1-\langle r,a\rangle-\langle c,b\rangle.
\]

对精确可行 Q，直接展开得

\[
D-J_W(Q)=\sum_{ij}\left[Q_{ij}\log\frac{Q_{ij}}{P_{ij}}-Q_{ij}+P_{ij}\right]\ge0.
\]

所以对偶上界本身不要求中间 P 先达到可行性门槛。进一步，在精确算术条件下，

\[
0\le F(W)-J_W(Q)=\mathrm{KL}(Q\Vert P^*)\le D-J_W(Q),
\qquad \|Q-P^*\|_1\le\sqrt{2(D-J_W(Q))}.
\]

精确 gap 为 \(5\times10^{-9}\) 时，右端给出 \(10^{-4}\) 的计划 L1 界，说明窄终端 Q 方向与原 \(2\times10^{-4}\) 资格要求相容。

但程序中 Q 仅近可行，上式会增加边际残差与势的乘积及总质量误差，还存在中心化、指数、对数、求和和 dtype 转换舍入。当前 `eps` 不是这些误差的通用严格界。故我同意有限修订及其针对性资格，不同意把两个打印门值解释为任意输入、任意形状或整段训练轨迹必达原误差界的证明。这一限定已在补充中明示，不是本轮另加通用区间平台的理由。（`amendment:39；FAIL/review/external/unpacked/review.zh.md:138–174`；本段等式为独立代数核对。）

### 3.3 统计保持真实 P，并具有足够的联合可辨识性

一次 `_solve` 内先积累局部候选统计，只有 `done.all()` 后才把成功块数、误差和 T 数并入外部统计。若混合批部分块已内部合格、另一些最终失败，则整次没有成功返回，不提交那些部分成功计数；`solve` 包装层另增加一次失败调用。不能把这一行为扩大成跨多个独立 `solve` 调用的全局事务，但它符合本补充对“整次批求解成功”的定义。（`solver:54–62、128–150`）

T 数跨成功调用求和；全体实际 P 最大残差、仅 O 的 P 最大残差、所有 Q／gap 最大值分开保留。没有 O 时普通残差为 `None`，不伪造为零。T 的最小实际迭代与全体最大实际迭代共同受门检查：有 T 时最小必须为 4096，全体最大不超过且达到 4096，从而排除“部分 T 提前接受但另一个块迭代到 4096”伪装同一摘要。`run.solver_gate` 还检查整数计数、T≤blocks、失败为零，以及有 T 时真实 P 超过普通阈值等一致性；统一盲门实际调用它。（`solver:128–146；run:432–451、537–538`）

### 3.4 Q 确实进入目标与真实更新

`_Envelope.forward` 保存求解器返回的 Q，按输入 dtype 转换；`backward` 返回非平凡 upstream 与所保存 Q 的逐项乘积。这是近似最优值的包络梯度约定，不是有限迭代程序的精确自动微分。（`solver:153–163`）

实际 `account_logits` 按账号记录数分桶构造完整矩形块；current 的 A0/A1、history 的 A0/A1 和 teacher_A0 均经本臂 F，历史项仍为 0.1 倍监督加 0.5 倍保持项。`method.update` 将该目标交给原 `record.update`；后者真实建立代理表、取得其梯度、执行分块 head 反传和 encoder 反传，最后调用 optimizer。不是独立 helper 通过却未接入训练入口。（`solver:176–242；scripts/step28_record_replay.py:159–208`）

## 四、两轮项目 CPU 原始证据

### 4.1 原始失败、第二轮成功与预算

| 原始记录 | CPU1 | CPU2 |
|---|---:|---:|
| wrapper 实际墙钟 | 25.86000084877014 秒 | 44.89070701599121 秒 |
| 本次向上计费 | 26 秒 | 45 秒 |
| 累计计费 | 123 秒 | 168 秒 |
| unittest | 运行 2 项，第二项失败 | 4 项通过，0 失败／错误／跳过 |
| wrapper 退出码 | 1 | 0 |

两轮合计为 **97＋26＋45＝168／300 秒，剩余 132 秒**。失败费用保留。CPU2 原始 result 的内部 elapsed 为 44.461580976843834 秒，不等于 wrapper 墙钟；两者没有被混用，实际费用按外层 45 秒记账。CPU2 峰值 RSS 为 721,809,408 字节、资格工作目录峰值为 8,887,088 字节，低于对应限额。（`CPU1、CPU2/wrapper.json:2–8；CPU1/result/unittest.txt:1–21；CPU2/result/unittest.txt:1–9；CPU2/result/result.json:L1；BP/qualification/state.json:1–66`）

`run.sh` 显式选择项目 py310、CPU0、单线程并禁用 GPU。原始 result 记录 Torch 2.9.1+cu130、确定性设置，以及 `formal_inputs=false`、`formal_labels=false`。带 CUDA 的安装版本不等于本次使用了 GPU；资格入口在 CPU 分支不加载 BGE。保存的 result、日志与代码路径相符，但本实例没有重新登录该环境。（`CPU1、CPU2/run.sh:1–19；verify:300–317`）

### 4.2 固定数值例用的是实际舍入输入，而不是预设零最优值

`rounded_reference` 将实际 K 元素用 `Decimal.from_float` 转为 80 位运算。对于手写对称结构

\[
K=\begin{pmatrix}a&b&c\\b&a&c\\c&c&z\end{pmatrix},
\quad P^*=\begin{pmatrix}u&v&x\\v&u&x\\x&x&s-2x\end{pmatrix},\quad s=1/3,
\]

令 \(\beta=e^{2c-z}/(e^a+e^b)\)。边际和一阶条件给出

\[
x^2=\beta(s-x)(s-2x),\qquad
x=\frac{2\beta s}{3\beta+\sqrt{\beta^2+4\beta}},\quad
u=\frac{s-x}{1+e^{b-a}},\quad v=s-x-u.
\]

这一路不复用生产 Sinkhorn。实际 FP32 输入的参考最优值是 **−8.017996194460732×10⁻⁹**，不能因为舍入前构造的理论值为零就继续以零作为参考。（`tests:575–594、662–685；CPU2/result/terminal_precision.json:L1`）

对保存数组独立复算得到：

| 固定 e=10⁻⁴ 例 | FP64 输入 | 先舍入 FP32 的实际输入 |
|---|---:|---:|
| 内部值对独立参考绝对误差 | 1.837259972273088×10⁻¹⁰ | 1.837275501445285×10⁻¹⁰ |
| Q 对独立 P* 的全块 L1 误差 | 7.667136977687703×10⁻⁷ | 7.667140823922286×10⁻⁷ |
| Q 转换输入 dtype 后 L1 误差 | 7.667136977687703×10⁻⁷ | 7.701291376063508×10⁻⁷ |

这些值均满足冻结阈值，FP32 对外值也满足其原转换界。两轮 `terminal_precision.json` **逐字节相同**；第一轮并不是先在数值资格上失败、第二轮再降低阈值通过。当前记录还包含真正返回 Q 的 VJP、正负 upstream、转置／翻转／零块混合、e=10⁻⁸ 拒绝、局部上限 1 拒绝，以及失败混合批不合并部分成功的断言。（`tests:662–693`；返回包 `checks/reference_arithmetic.json`）

对 e=10⁻⁸ 拒绝，原始小记录证明异常路径被断言捕获，并未逐次保存其全部轨迹或最终全部数值；我不补造未保存的拒绝数表，也不把手写例称为原正式三个失败块重放。

### 4.3 首轮为什么失败：hard-negative 分支换序，而非已证明的求解精度失效

首轮原始 unittest 明确在实际代理梯度与单图参考梯度比较处失败：3,486 元素中 773 个不匹配，最大绝对差 **0.0011897629592567682**；容差仍为 `atol=2e-6, rtol=2e-5`。日志不是 optimizer、预算或文件身份失败。（`CPU1/result/unittest.txt:4–17`）

这里的监督目标用 detached 分数稳定排序选 top‑5 负例。近乎并列时，即使两个评分表达式只差约 10⁻⁷，也可能选择不同的负例集合；这时比较的是不同分段上的梯度，不能用“评分很接近”推出代理梯度应在原容差内一致。（`scripts/step28_alias_ranking.py:106–129；tests:68–81`）

我在隔离环境仅复核了原先这一个固定 integration fixture 的首轮和当前版本。首轮精确重现上述 **773／3486、同一最大差值**。其 current_A0 有 25 个查询、另外三个监督视图各有 28 个查询的 top‑5 集合不同，参考第五与第六负例分数边界为零。当前 fixture 的四个监督视图集合差异均为零，最小参考第五／第六边界约 **0.0033330917358398438**。这直接支持报告给出的近并列换序解释，不仅是相信“加偏移后 PASS”。（返回包 `local_checks/attempt01/`、`local_checks/current/`）

第二轮类型／账号 token 是专门构造的手写模型输入，不是正式身份特征。三色构造保证两个 assignment 视图中每账号都有三类记录；实际生产 `assignment` 仍执行，观察钩子只委托原函数并记录输出／梯度，没有替换求解、计划或优化器。（`tests:603–658、709–715；scripts/step28_record_replay.py:66–99`）

### 4.4 第二轮独立参考究竟证明什么

对实数 K 加任意行势和列势，在固定双边际下只增加常数 \(\langle r,\alpha\rangle+\langle c,\gamma\rangle\)，最优计划不变。因此用独立闭式 P* 对固定偏移族形成单图包络参考是成立的设计，不是换一个同源 Sinkhorn 重算来“自证”。当前测试从轻量模型**实际产生的 FP32 对称基础块**取得参考，而不是只读理论初始化表；我对保存的这个实际块另作 80 位闭式算术核对，参考值 **0.009999982540628076** 与保存 P* 完全复现。（`tests:699–727；CPU2/result/terminal_real_update.json:L1；checks/reference_arithmetic.json`）

不过，这个精确不变性属于实数算术。各实际 FP32 表经模型运算后，并不逐位保持完美行列加性；也不能据单个对称基础块把参考外推成任意非对称矩形块的精确解。我的隔离固定输入检查中，去掉基础块及行列均值加性成分后的最大非加性残差约 **4.239×10⁻⁷**。对精确可行的熵最优计划，最优性条件给出对称 KL 等于残差与计划差的内积；结合 Pinsker 可得 \(\|P_E-P_0\|_1\le\|E\|_\infty\)，从而说明这一固定扰动量级与 2×10⁻⁴ 要求相容。

这里的残差测量、浮点计划表示和上述条件性不等式**不构成区间认证**；返回 JSON 中相应 perturbation bound 字段也只能按这一条件解释。它们用于界定当前固定 fixture 的参考范围，不为所有形状或更新后轨迹背书。teacher 的评分只进入 MSE；不应把 teacher 上人为计算出的 top‑5 差异算作未实际使用的排名损失缺陷。（`tests:728–740；local_checks/fixed_fixture_check.py`）

### 4.5 真实集成与模拟编排必须分开

CPU2 真实轻量集成确实经过五次账号投影，共 **5×378＝1890 个 T 块**，零失败、最小终端迭代 4096。当前监督为 4.501949787139893，历史监督为 4.501948833465576，历史 D0 为 0.014166662469506264，历史加权项为 0.45727822184562683。代理表梯度对独立单图最大差为 **2.0489096641540527×10⁻⁸**；encoder／head 梯度最大差约 1.008×10⁻⁷，更新后参数对独立更新最大差约 8.498×10⁻⁹。（`CPU2/result/terminal_update_proxy.json:L1；CPU2/result/terminal_real_update.json:L1；tests:695–756`）

这些“参数差”是两条实现之间的差，**不是初始化到更新后的位移**，原报告第 22 行对此已经作了正确限定。源码真实执行生产 AdamW，且独立参考也真实执行一次 AdamW；调用口径勘误见 RD‑02。项目原始 JSON 没有单独保存每个模块的初始→更新后位移，不能替它补称保存过该证据。我自己的观察式复核记录了当前 fixture 的 encoder 最大位移约 9.358×10⁻⁶、head 约 0.001，二者非零；这是隔离环境的额外实现核查，不是补写项目 Linux 原始记录。

另一个编排测试模拟了更新和前向。其 `orchestration.json` 明示 `real_updates_in_this_fixture=0`：可以支持 1728 调度、6 端点、36 诊断和拒绝不完整状态的控制流，**不能证明 1728 次真实训练已经执行**。（`tests:486–570；CPU2/result/orchestration.json:L1`）

## 五、新旧资格与固定三次 F 的真实接线

旧原生 result 确认旧版实际八次更新，来源是旧 29 文件；旧资格账计费 364 秒。其 F 求解摘要最大迭代只有 22，不能拿来证明新终端路径、4096 成本或新元数据恢复已经通过。旧 gate 中原 54,322.234253062634 秒投影同样不能原样用于当前新作业准入。（`OLD/qualification/gpu/attempt01/result/result.json:L1；OLD/qualification/state.json；OLD/qualification/native_disposition.zh.txt:1–13`。原生 result 的实际包路径另见逐文件范围表。）

实际 gate 链可分为三层：

**继承层。** `prior_evidence()` 核对七份固定旧证据、旧 gate／native 来源、旧 8 次／364 秒、正式失败费用和旧 train／valid 账；授权和补充身份也受核对。23 份未变依赖才继承，五份变化的实现不借旧 PASS 冒充已重测。（`run:732–761；policy:85–120`）

**当前审查／CPU 层。** `reviewed_gate` 要求当前 30 来源、具体授权与主审处置身份、`NO_OPEN_BLOCKERS`，并绑定当前 terminal‑q CPU 成功结果及累计预算；不是只检查一个布尔 PASS。当前包没有以本次外审尚未完成为前提预造新的正式执行 gate。（`run:764–780；BP/report.zh.txt:28`）

**新原生／正式层。** 新 GPU 入口只允许 `terminal-q`，原八次流程不能再次作为默认许可运行；新结果必须实际三次、累计 11 次、旧费 364 秒、累计不超 900 秒。新来源、F 两类恢复、两阶段作用、诊断后下一更新一致、GPU 终端 VJP 和完整 CPU／GPU 压力摘要均受检查。CPU 门测试里构造的 GPU JSON 只是准入反例 fixture，绝不是当前已经存在的新原生实测。（`verify:318–328；run:783–830；tests:776–807`）

固定原生代码确有如下实际安排：第一步 F 无历史更新分配 Adam；以此完整手写状态构造明确标注的 B 阶段计数 336，做第二次 current＋history 更新；恢复相同 model／Adam／梯度、RNG、模式与 Memory 基线，执行真实 48 fit／6 cache／40 blind 诊断，再做第三次相同更新。它比较完整下一状态的字节、RNG／Memory／模式和相关日志，而不是仅比较一次输出 loss。人工计数 336 不是已执行 336 次。（`verify:29–41、117–159；scripts/step28_record_replay_numerics.py:45–74；scripts/step28_replay_diagnostics.py:31–43、65–129`）

随后还有实际 F 完整／推理状态保存恢复，核对本臂原始分数、pooling 身份和元数据；临时文件按原生命周期释放。GPU 固定硬数表单独完成 terminal 分支和实际 VJP；它不声称最大形状 BGE 手写文本自然产生过相同的硬块。这个证据分工合理，但**本次只审到代码，未执行这些 GPU 操作**。（`verify:173–231`）

## 六、剩余预算、压力乘数与读取边界

### 6.1 旧费只扣一次，五处使用同一有效限额

原正式失败 `budget.elapsed_seconds` 为 18428.570150192827。按当前规定，

\[
\lfloor86400-18428.570150192827\rfloor=\mathbf{67971}\text{ 秒}.
\]

`policy` 固定旧费／总额／有效剩额；`projection` 与正式 gate 比较 67971；派生 `Budget` 按新作业自身时间检查 67971，并另报旧费＋新费的累计值；watchdog 调同一 `Budget`；新 shell 从包装开始记录唯一时间，扣除已耗包装时间和 **5 秒终止宽限** 后计算 timeout。它没有先从有效限额再减一遍旧费，也没有让新目录恢复到 86400 秒。（`policy:35–54；run:197–230、822–886；shell:3、11–23；scripts/step28_record_replay_run.py:60–104`）

CPU 反例把投影构造为 70000 秒，实际 `execute` 在 `public_inputs` 之前拒绝，未创建 job；新 Budget 的旧费用不为零，并能在新限额处拒绝。这验证的是剩余预算接线，不是 70000 秒科研实测。（`tests:766–807；CPU2/result/terminal_gate_budget.json:L1`）

shell 的 `bash -n` 已在隔离环境通过；我没有实际启动项目 shell 或声称验证操作系统在任意调度情况下的硬实时终止。代码层面没有把宽限额外加到 67971 秒之后。CPU 累计 168／300、原生旧 364／900、正式旧费用是三本不同用途的账，不彼此抵扣。

### 6.2 压力测量不是单块成本冒充整表成本

新原生将固定 3×3 硬例扩成 **28 账号、84 记录、378 个跨账号块**，真实调用 `account_logits`。GPU 量为整表 forward 加非平凡 VJP 的 `c_gpu`，CPU 量为整表 forward 的 `c_cpu`；二者各测一次，均要求 378 个真实 T，且摘要同时包含 CPU、GPU 两侧。GPU 还核对保存 Q 的实际全表梯度。（`verify:191–219；run:809–819`）

进入成本表的增量实际为：

\[
t_{A,F}\!\leftarrow t_{A,F}+2c_{gpu},\quad
t_{B,F}\!\leftarrow t_{B,F}+5c_{gpu},\quad
v_F\!\leftarrow v_F+c_{gpu},\quad
h_F\!\leftarrow h_F+4c_{cpu}.
\]

2 是当前 A0/A1；5 是当前两次、历史监督两次和教师一次；1 是自然评分；4 是历史诊断 live／teacher 各 A0/A1。S/G 四项维持旧实测，L、保存恢复耗时和大小取新旧较大值。代入冻结公式后，单就硬例附加项，最终投影增加 **4327.5×c_gpu＋180×c_cpu**，不是只加一次资格耗时。（`verify:220–224；run:218–230、799–819；amendment:104–119`；乘数展开为本次算术核对。）

GPU 量已含 VJP却也加给无反向项、自然测量又已包含普通求解，因此这属于明示的保守附加假设，不是互不重叠的耗时分解。它覆盖固定已知困难构造，不保证任意 2—8 条记录混合分桶、所有块均跑 4096 次的最坏情况。未来新投影超剩额即停止，不能删掉压力成本、诊断、样本或精度来挤入；即使投影合格，运行硬限仍有效。当前没有新 `c_gpu/c_cpu` 实测，因而不能给出“新正式运行一定装得下”的结论。

### 6.3 七文件、跨作业账与统一开放

当前正式输入函数仍限原 manifest、validation、groups、train/items、train/supervision/pairs、development/items、development/supervision/pairs 七文件；字节身份核验不等于语义标签开放。合表 C 行可作必要技术筛选，但在 Group、模型输入及标签数值对齐之前剔除，不打开 manifest 引用的其他正式对象。（`original:127–144；amendment:140–152；run:88–172`）

旧失败保存的 access 为 train=1、valid=0、heldout=0、owners=0，当前 `prior_evidence` 实际核对；新作业本地 train 由 0 到 1，`parse_once` 先登记消费再调用 loader，失败不退还。因此跨两作业 train 最多 2，而不是将本地账伪写成 2。统一盲门要求本地 train=1、valid=0及全部三臂端点、恢复、来源、预算、数值、诊断成立，之后才登记并解析唯一一次 valid。（`FAIL/evidence/reports/job/access.json:L1；run:375–429、454–591、732–761；scripts/step28_bge_continual_run.py:76–87`）

新作业三臂从各自固定起点独立训练 A，再续各自 A→B；不加载旧失败模型／Memory 续 41 步，不混入旧 S/G 端点。完整新编排的成本取舍已经批准，不说明旧 S/G 结果失效。固定三比较和一次结果开放不自动授权据结果调 tau、容差、seed、epoch、候选或再训练。当前附件无正式标签、模型或 Memory 载荷；原失败定位 ID 和数值摘要只按获准故障审查用途处理。（`amendment:9–13、140–156；run:375–429、594–728`）

## 七、逐项发现与最小处置

### SCIENTIFIC_BLOCKER

**无新增项。** 我未发现实际 O/T、Q 返回／VJP、输出阈值、CPU 独立参考或预算／读取链中的具体新矛盾，足以要求重开已经关闭的 S/G、数据、评价或历史研究。这里的“无”不是任意输入正确性证明、性能结论，也不解除尚未执行的原生前提。

### RD‑01｜REPRODUCIBILITY_DEFECT：第二轮差异不应概括成“只有 fixture”

**证据。** 从 CPU1 原始 `source.zip` 对当前来源逐文件比较，变化涉及 `tests`、`run`、`verify` 三文件。除手写 token／embedding／head 的行列加性偏移及对应维度／归一化补偿外，还增加了代理梯度证据输出，调整了预算反例中浮点值的重组方式；`run:813–814` 增加 CPU／GPU 两类压力摘要必须齐全的检查；`verify` 删除重复 `import os`。求解器、policy、shell 和原数值／梯度容差未因这次失败修改。完整差异保存于 `checks/attempt01_to_current.patch`。涉及的简写口径见 `BP/report.zh.txt:16`、`BP/review/request.zh.txt:9` 及本次请求。

**影响。** “修复近并列梯度比较失败的机制只改手写 fixture”成立；“两次尝试之间所有源码只改 fixture”不成立。新增压力摘要检查属于收紧资格完整性，没有发现其放宽门槛或改变 F 数值的效果；当前 CPU 已针对当前来源运行。

**最小处置。** 在主审处置中说明上述三文件差异及各自目的，保留两轮原件；不必改 solver、改容差或为这些已有字节重跑全套资格。此项是报告范围勘误，不是生产数值实现阻断。

### RD‑02｜REPRODUCIBILITY_DEFECT：轻量 optimizer 调用需明确统计对象

**证据。** 生产路径在 `scripts/step28_record_replay.py:206` 调用一次 `optimizer.step()`；独立参考在当前 `tests:739–740` 调用一次 `ob.step()`。两轮 integration 都执行到参考 step 后才作比较。`tests:755` 与 `CPU2/result/terminal_real_update.json` 的 `actual_optimizer_calls=1` 可以解释为“生产被测路径一次”，但 `BP/report.zh.txt:20` 的“两轮各发生一次轻量 CPU AdamW 调用”若理解为整项测试总调用就不准确。隔离观察实测每轮均为两次 step。

**影响。** 每轮是 **1 次生产更新＋1 次独立参考更新**，不是两次生产轨迹更新，也都不是 BGE 原生更新。wrapper 全墙钟已经包含两者，因此不是少记 CPU 费用、也不是偷用原生三次预算；1728 模拟编排仍不能算真实训练。

**最小处置。** 主审明示该字段统计“生产被测 optimizer”；总观察调用另写为 2，并说明其构成。无需改写原始 result、撤销已完成 CPU 或追加运行。此项是记账口径勘误，非执行阻断。

### 合理后续建议 RS‑01｜保持参考与真实位移证据的适用范围

项目 CPU 更新 JSON 保存的是实现间差值，不是基线位移；独立闭式参考限于当前对称块和固定加性族，FP32 小非加性残差按有限扰动解释。我已给出隔离核查及限制，建议后续主审沿用这些准确措辞。计划中的原生 `updated()` 已要求 encoder/head 梯度及基线位移为正，执行既定检查即可；不建议为补同类表述再增加 CPU 调用。（`BP/report.zh.txt:22、25；tests:747–755；verify:117–131`）

### OUT_OF_SCOPE_OVERDESIGN

本轮不要求新求解器、8192 次、warm-start、温度／参数／seed 网格、通用区间或验证平台、所有历史重测、额外原生调用、正式 K／模型／Memory 补传、选择性恢复或新授权仪式。这些不是上述两处勘误的最小处置。旧八次证据保持旧身份；新三次尚未发生就继续记未发生，不能为了“材料齐全”伪造 PASS。

## 八、本实例实际执行的小核查与限制

我只在隔离网页工作区执行了**同一个已提供手写 integration test 的两个既有版本各一次**，没有运行完整测试类、资格入口 main、正式 `execute` 或项目 shell。观察钩子委托原 `account_logits` 和原 AdamW.step，不替换计算；另外做了只读 ZIP／SHA／AST／反向补丁／shell 语法核对，以及保存手写数值的闭式算术复核。

隔离环境为 Python **3.13.5**、Torch **2.10.0+cpu**、NumPy **2.3.5**、Linux 6.18.44，Torch 单 CPU 线程，`CUDA_VISIBLE_DEVICES=''`；它不同于项目 py310／Torch 2.9.1+cu130，**不是项目 Linux 原生资格**。来源由各 result 的 SHA／大小记录绑定；输入由原固定 fixture、保存实际基础 K、确定性构造源码和实际输出分数记录确定。

| 本实例实际执行 | 结果 | 记录耗时 |
|---|---|---:|
| 首轮来源固定 integration | 预期失败，重现原代理梯度差 | 22.796497352 秒 |
| 当前来源固定 integration | 通过；四监督视图无 top‑5 集合分歧 | 22.290516390 秒 |

耗时覆盖脚本记录区间内导入、测试和固定输出分析，两次合计约 45.087 秒；不计入项目 CPU 168／300 账。静态处理和保存数组算术的各自耗时另保存在 `checks/*.json`。首轮失败不隐藏为成功，当前通过不改名为项目 GPU PASS。

返回包包含实际执行源码 `local_checks/fixed_fixture_check.py`、两轮原始 unittest／console／result、保存的手写评分数组，以及必要机械核查源码和输出。条件性扰动界的说明见第四节；JSON 中的 bound 名称不能脱离该条件使用。没有保存或输出任何正式模型、标签或 Memory。

## 九、实际阅读范围与没有声称完成的工作

本次完整审读了本轮 report、authorization、补充合同、973 行 implementation.patch 和六份受影响实际文件：solver 274 行、run 896 行、verify 340 行、policy 122 行、tests 809 行、新 shell 23 行；逐项读取两轮 CPU result／unittest／console／wrapper、数值和更新小证据，以及资格、来源、接收清单。

对直接依赖只追到实际 assignment、分块 VJP、optimizer、hard-negative 排序、诊断中立性、完整状态比较、保存恢复、Budget 和 parse_once 等受影响调用；旧合同读取本次权限／资格／预算条款，旧审查原文读取终端数学、统计、资格、预算及边界相关段落，旧主审处置按原文复用。旧 native 单行大 JSON 的来源、计数、资源、测量、恢复／状态比较摘要和 solver 摘要已读取核对，未重新审查所有旧梯度逐元素效果记录。

**全部 66 成员完成字节核验，不等于全部 66 文件都作了完整语义重审。** 返回包 `reading_scope.json`／`reading_scope.tsv` 为每文件列出实际行范围、结构字段范围、仅身份核验或 AST 分析；未读段落明确不计入“完整阅读”。输入和首轮内嵌归档均有逐文件 SHA／大小清单。

没有执行项目 GPU 或新正式训练，没有读取旧模型／Memory、原正式 K/P/Q 或任何新正式标签；没有取得新原生计时、也没有复算已关闭的全部 S/G／数据／22 指标结果。旧正式三个失败块仅有定位和摘要，本轮手写核验不是其轨迹重放。

## 十、最终处置

**合同与最小实现：可接受，无新增生产算法阻断。增量 CPU 证据：在本文限定的手写参考与实现范围内可接受；两处报告口径由主审作最小勘误，不要求重跑。当前原生资格：仍未执行；当前正式执行条件：尚未满足。**

主审回收本报告、独立处置上述勘误并绑定当前来源后，按既有批准范围继续一次固定三次 F 原生资格。只有真实三次结果、F 保存恢复／诊断后更新一致、终端 GPU VJP、378 对压力成本、新投影≤67971 秒及资源／读取 gate 均满足，才进入已经批准的一次完整三臂运行；不需重复取得相同合同批准。任何一项失败或超剩额即停止、保留失败与费用，不删成本或放宽条件。

本轮不产生 F 优于 S/G 的效果结论，不否定旧 S/G 有效证据，也不承诺下次正式输入必然成功。
