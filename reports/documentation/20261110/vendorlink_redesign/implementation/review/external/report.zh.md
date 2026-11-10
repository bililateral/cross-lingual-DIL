# VendorLink 账号监督 CE 适配：实际实现定向外审报告

**审查日期：2026 年 11 月 10 日**  
**审查对象：** `vendorlink_implementation.zip`；附件声明交付 Git `813b1837`。  
**审查性质：** 已审方案的实际实现外审，重点是新增 CE、阶段与科学门接线、资源调度及必要增量证据，不重新设计方案。  
**模型自述：** 我是 ChatGPT，本会话可报告的模型自述为 **GPT-6 Astra Pro**。用户指定了 GPT-6、Pro 强度；我没有可独立核验的界面选择器记录、后台路由或算力档日志，因此不另行声称已验证后台模型身份或实际资源档位。

## 一、结论与处置

**结论：实现层可接受。当前实现可在本轮外审结论按既有程序登记闭合、并另获正式启动授权后进入 D。** 在此次可访问的源码、保存回执和限定复用证据范围内，未发现需要先闭合的 `SCIENTIFIC_BLOCKER`，也未确认必须先修复的 `REPRODUCIBILITY_DEFECT`。不需要为进入 D 再新增候选、种子、完整轨迹或重复 GPU 资格。

这个判断的依据是：CE 确实进入了原 H 的编码器更新；辅助与原优化器的参数、更新和跨域状态边界正确；D 的 A/B 冷启动与 V_CLS 选择、P/M/C 的完整性和组合门存在于实际调用链；88h 预算及用户指定的 OOM 续领政策也由正式入口消费。旧方案外审 R01 的 `/sqrt(2)`、R02 的辅助 AdamW 规格均已落实。[S1—S6、S8—S10]

**当前仍未获得正式启动许可。** 包内 `gate.json` 的 `approved=false`、`review_closed=false` 与送审状态一致，`permit()` 在这两个条件未满足时拒绝执行。本报告没有修改这些字段，也没有启动 D/P/M/C、监听、角色文本评分或应用输出。外审登记闭合及另获正式启动授权属于现有执行边界，不是本报告新增的科研门。[I/gate.json；S6 `permit()`、`main()`]

**本结论不表示 CE 已追平 VendorLink，也不表示 Full 已证明抗遗忘。** 旧 H_SEQ 的平均遗忘 4.3657pp、A/B 初学不足、原 5pp 与原学习门失败、旧费用 71450 秒，继续按原结论保留。3.5pp 是用户在已见旧结果后选择的新目标，不追认为旧实验的独立确认；本轮没有重算或改写旧结果。[I/review/request.zh.md，第 9—15 行；N/proposal.zh.md]

### 分类结果

| 分类 | 本次结论 | 处置 |
|---|---|---|
| `SCIENTIFIC_BLOCKER` | 未发现 | 不增加新的科学门或实验 |
| `REPRODUCIBILITY_DEFECT` | 未确认有阻止进入 D 的确定缺陷 | 无必修代码项；保留第八节的证据表述边界 |
| 合理后续建议 | 三项非阻断的证据、措辞及结果报告建议 | 最小文字处置或按原方案履行报告义务 |
| 涉及新科学选择的建议 | 本审查不提出新的选择 | 不调整 γ 网格、阈值、种子或市场顺序来争取正结果 |
| `OUT_OF_SCOPE_OVERDESIGN` | 第九节列出 | 不作为进入 D 的条件 |

## 二、附件身份、实际核查范围与证据层级

### 2.1 清单核对

| 项目 | 本轮实际核对结果 |
|---|---|
| ZIP 大小 | **143125 字节**，与用户提供值一致 |
| ZIP SHA256 | **`1ea3f81020f46ee03979701473a85efef520faba8ea38d7faa2e8d02a29c1ab1`**，一致 |
| 成员数 | **40 项：39 份载荷＋MANIFEST.json**；无重复成员名，ZIP CRC 检查通过 |
| MANIFEST | 39 份载荷的路径集合、逐文件大小、SHA256 均与实际附件一致 |
| MANIFEST 自身 | 9265 字节；SHA256 `d498dc1e972173c7d3b140727e020fb517393a33894206ef457b4ba7d88eed4a` |
| 当前运行源码 | 13 份，均与 `gate.sources` 的大小及 SHA256 一致 |
| 部署/回传记录 | `sync.json` 列出的 34 项，本地附件字节均与其记录一致 |
| 当前方案 | 31971 字节；SHA256 `82b21ea771f75294bd8cd91f880b2249dbd3bdc27e11928f653c2b1a3345a515` |
| 旧证据复用的字节基础 | 共享 `step29_vendorlink_methods.py`、`step29_vendorlink_run.py` 与上一轮方案包逐字节相同 |

我按用户指定顺序读取 MANIFEST、实现外审请求、实现报告和当前方案，再检查 13 份运行源码、历史测试/探针恢复源、CPU/GPU 回执、来源清单、gate、费用及收尾记录。正文以附件内实际字节为审查对象；`813b1837` 是附件记录的交付标识，本轮没有查询实际 Git 对象库来独立证明该提交内容，也没有访问项目服务器。

### 2.2 三类证据分别使用

1. **本轮直接做过的核查：** ZIP/文件身份核对，文本和 JSON 阅读，源码静态调用链追踪，保存数字的预算算术与来源链比对。没有导入或运行项目模块、测试、模型、训练、评分或科研指标计算。
2. **包内保存的原生执行证据：** CPU01、CPU02、GPU01 的源码、回执和控制台记录。这些支持执行者已完成的 Linux 核验，不等于我在本轮重新复现了这些运行。
3. **限定复用的旧证据：** 原角色资格、原 H 表示/历史机制及未变的 V 科学路径，只在此次没有改变且适用的范围内复用。不把旧 PASS 直接覆盖新增 CE 或调度行为。

`bindings.json` 中的真实角色 manifest、BERT 文件路径及其 SHA 是可读元数据；相应载荷不在附件内，我未访问其包外路径。`preflight.json`、`cleanup.json` 对远端身份、无 campaign、无活动进程和短测权重已删除的陈述，属于保存的执行者回执，不能写成本审查者对远端当前状态的实测。包内的两个关闭字段可以直接核对；远端无监听、无真实角色读取等运行事实，本轮依据所交材料判断，没有独立的服务器遥测。[I/bindings.json、preflight.json、cleanup.json、gate.json]

本次没有使用网页资料推断项目行为；没有索取或读取账号正文、逐查询分数、封存标签、Memory、权重或凭据。`evidence_sources/` 中的文件按历史证据源处理，不把已经退役的核验脚本当作当前正式入口。

## 三、CE 与完整 H 的实际计算、优化和状态

### 3.1 CE 没有与编码器脱节，也没有替换 H 主评分

原记录表示由共享 `Model.records()` 生成：title、description 各最多 256 token，sum-last4 masked mean，每通道归一化后拼接并除以 √2，得到 1536 维表示。新 `AccountModel.backward_h()` 直接调用该函数，没有另做不同尺度的 CE 表示。[S3 `backward_h()`，第 37—66 行；S8 `Model.records()`，第 190—200 行]

当前项实际等价于：

\[
L_{\mathrm{current}}=L_H(z)+\gamma\,\frac{1}{N}\sum_{r=1}^{N}\operatorname{CE}(Wz_r+b,y_r),\qquad\gamma\in\{0,1\}.
\]

这里的标签取自当前记录所属账号 `v.identity`，按当前市场完整类别映射构造；CE 按实际记录求均值，不按 H 自然/重组视图重复加入，也不是只对当前小组账号进行局部分类。[S3，第 55—63 行]

代码先对原 H 的记录对目标取得梯度，分块反向到同一个 `leaf`，随后把当前 CE 梯度累加到该 `leaf.grad`，最后通过 `z.backward(leaf.grad)` 回到原编码器图。因此，看到 `detach()` 不能据此认定 CE 被截断：这里使用的是明确的手动链式反传。历史样本另走原历史图，累加到同一组模型参数，但跳过整个 CE 分支。[S3 `backward_h()`、`update_account()`]

最终 H 评价仍调用原记录表示和 pair head，按 Q2/G4 的记录对分数做原 mean/log-mean-exp 聚合；辅助分类头不参与主分数。紧凑评分端点保存所需 encoder/pair head，未保存辅助头用于评分，符合其仅为训练辅助项的定位。[S4 `save_endpoint()`、`restore()`；S5 `encode()`、`rectangular()`]

### 3.2 优化器、裁剪、换域和随机流均有实际边界

| 核查点 | 实际实现与影响 | 结论 |
|---|---|---|
| γ=0 对照 | 仍计算 CE 诊断，但不做 CE backward、不 step 辅助 optimizer；辅助梯度为 None | 保持 γ0 对照，未暗中训练辅助头 |
| 辅助 AdamW | lr=3e-4，betas=(0.9,0.999)，eps=1e-8，weight_decay=0.01，amsgrad=False；W/b 在同一组 | R02 已落实，规格明确 |
| 参数归属 | 原 H optimizer 仅 encoder＋pair head，辅助 optimizer 仅辅助头；更新前检查参数对象无重叠/重复 | 不会由两个 optimizer 重复更新 encoder |
| 更新顺序 | 两者清零；累计当前 H＋CE 及历史梯度；参数并集共同裁剪一次；原 optimizer step 一次，γ1 辅助 optimizer step 一次 | 符合已审算法，未把历史拆成额外 encoder 更新 |
| 换市场 | 每域重建辅助头及辅助 optimizer；原 H optimizer 只在尚不存在时创建 | 保留 encoder/pair Adam 状态；辅助状态按域重置 |
| 学习率 | 辅助头沿 H1 pair-head 的阶段调度因子，实际固定 lr_multiplier=1 | 未另引入未披露调度 |
| 初始化随机流 | `fork_rng` 覆盖 `nn.Linear` 构造及显式初始化全过程 | 不只隔离最后一次 normal 初始化，未扰动原 pair head |
| 当前/历史随机流 | 当前更新种子不含 H 方法名；当前、历史反向分别定种子；分组与 Memory 保留各自流 | 没有因新增辅助初始化破坏已定方法配对 |

证据：S3 `start_account_stage()`、`update_account()`、`set_account_rates()`；S4 `make_model()`、`execute()`；S8 `make_optimizer()`；S9 `set_rates()`。原 V 仍按自身新市场分类头的既定方式重建优化器，未被 H 的状态政策覆盖。

共同裁剪包含辅助头参数，因此 γ 干预也包含这个已披露的裁剪耦合。它是当前训练算法的一部分，不要求额外增加“等梯度”对照。γ=1 是否使 CE 过强，应由冻结的 D 能力门和后续科学结果回答，不能把工程反向正确解释成尺度已最优。

当前 H 的 `make_model()` 还启用了 `use_reentrant=False` 的激活检查点，用于反向时重算 Transformer 激活。这是实际显存执行安排，没有增加新的目标项或候选；GPU01显式断言其已启用，并沿该路径完成真实 BERT 更新。不能把该短测扩大为与所有历史执行方式逐位等价的证明。[S4，第 57—61 行；T3，第 54—66 行]

### 3.3 保存恢复支持当前用途，不构成自动重跑授权

`save_account_state()` 保存模型、原/辅助 optimizer、辅助类别、Memory、游标、RNG 和模式；`load_account_state()` 先重建正确形状的辅助头，再 strict 载入、恢复两 optimizer、Memory、模式及 RNG。CPU01 与 GPU01 对下一步更新的一致性均有保存证据。[S3，第 110—138 行；T1，第 140—180 行；T3，第 83—106 行]

此处恢复能力用于核验状态完整性；正式控制器没有启用失败续训队列，已有 campaign 目录会拒绝自动重启，失败任务不重入队。无需为了已有恢复函数另建通用恢复框架。[S6 `campaign()`；S10 `Runner.run_many()`]

### 3.4 GPU 结果支持的结论恰好是“CE 起作用”

保存回执显示，同一手写最大形状下，γ0/γ1 的当前 encoder 梯度范数分别约 **0.017459 / 0.363649**；跟踪 encoder 参数的更新差最大值约 **2.50041485×10⁻⁵**。这与探针实际比较参数更新的代码相符，不只是 loss 非零或辅助头发生变化。[I/gpu01_receipt.json；T3，第 48—82 行]

这些数字证明 CE 在该次真实 BERT 更新中影响编码器；它们不能证明 CE 已提高 H 主评分、已追平 V_CLS 或改善抗遗忘，也不支持据此改变 γ、学习率或冻结阈值。

## 四、D/P/M/C 与组合科学门的实际接线

### 4.1 任务、冷启动、选择器

| 阶段 | 当前代码构造与入口行为 | 判断 |
|---|---|---|
| D | H_base/H_CE/V0/V1 × AlphaBay/Dream；8 个独立冷启动，seed41，每任务单域40 epochs | 与已定开发设计一致 |
| D 的 Dream | task 显式带 `domains=['Dream']`；训练、暴露核对、评分都读取 `task_domains(task)` | 不会误训/误评 AlphaBay |
| V 选择 | 仅以 A/B 的 V_CLS 均值选择；V1−V0>0.001 才选 V1，其余选 V0 | 未遗留按 max(V_CLS,V_SET) 选参的旧行为 |
| H_CE 筛查 | 两域各≥0.70，且各域相对入选 V_CLS 不低于−0.02；保存 CE−base | γ0 对照确实被评价，但不会事后改选 γ0 进入 P/M |
| P | H_CE_SEQ 与入选 V，各 seed51/52/53，6 条 A→B→C 轨迹 | 不继承 D 权重；固定新种子和顺序 |
| M | 同配置 H_ER/H_full × seed51/52/53，6 条 A→B→C 轨迹 | 通过 P 才开放 |
| C | 复用 P＋M 共12条既有轨迹的完整端点集合，只新做确认文本评分 | 不新增训练；M 系统通过才开放 |

证据：S2 `settings()`、`tasks()`、`select()`、`task_domains()`；S4 `make_model()`、`execute()`；S6 `worker()`、`pipeline()`、`campaign.run_phase()`。

D 的 Dream `R11` 是该单域任务的终点，而不是全局 AlphaBay 的终点：`evaluate.score()` 用 `domains[j-1]` 解释该格。CPU01 曾以手写 Dream、tiny BERT 和2 epochs 实际经过训练、完整性核对及 Q2/G4 评分，确认只访问 Dream 的 gallery/dev；不是仅检查一个 domains 字符串。其历史版本范围见第七节。[S5 `score()`；T1，第 182—208 行]

单 seed、逐域2pp的 D 门仍是已接受的保守成本筛查。失败即 `STOP_ADAPTATION`，并不否定所有可能的 CE 适配；通过也不等于已独立证实 CE 的统计贡献。D/P/M 使用 dev 的既定开发性质没有被实现变成独立确认，只有满足条件后 C 才消费 confirm 文本。

### 4.2 P 的所有前提均进入决定函数

`policy.decide()` 先要求每方法恰为按序的 51/52/53 三个 seed，结果有限；缺 seed 不按剩余结果放行。P 的实际 AND 条件为：

- H_SEQ 三域初学 R11/R22/R33：每域均值≥70%，每 seed≥60%。
- 每域 H_SEQ−V_CLS：均值≥−2pp，每个配对 seed≥−5pp；V_CLS 各域均值≥70%。
- H_SEQ 的 GB、GC：各自均值≥5pp，每 seed 非负。
- H_SEQ 的 F：均值≥3.5pp，至少2/3 seed≥3pp，每 seed 非负。

两类前提同时通过才有 `PASS_P`；否则分别按学习不足或无实质遗忘停止，不启动 M。代码用原始浮点值作比较，没有先四舍五入再过门。[S2 `decide()`，第 101—124 行；S6 `pipeline()`，第 102—108 行]

CPU01 的保存反例包括 F=0.0349 不通过及缺 seed 拒绝，直接覆盖本次目标与完整性的重要边界；不能据此声称每一个阈值均做过动态反例测试。[T1，第 216—227 行]

### 4.3 Full 的主要改善与质量保护是组合判定

以下均按冻结的三 seed 配对，表中 pp 为百分点。

| 条件 | 代码中的实际要求 |
|---|---|
| Full 对 SEQ 的旧域最终与遗忘改善 | ΔO 与 F_SEQ−F_Full 各均值≥3pp；各至少2/3 seed正向；各 seed均≥−1pp |
| Full 初学相对 SEQ | A/B 各均值差≥−1pp、逐 seed≥−3pp |
| Full 自身能力及相对 V_CLS 初学 | A/B/C 各均值≥70%、逐 seed≥60%；相对 V_CLS 各均值差≥−2pp、逐 seed≥−5pp |
| Full 新域保护 | B/C 相对 SEQ、V_CLS、V_SET 分别满足均值−2pp/逐 seed−5pp；GB/GC各均值≥5pp且逐 seed非负 |
| Full 对外部 V 的旧域最终优势 | 对 V_CLS、V_SET 的 O 各均值差≥2pp、至少2/3正向、逐 seed≥−1pp |
| Full−ER 组件增量 | 系统门本身通过，O/F各改善均值≥2pp且各至少2/3正向；对 ER 的 A/B初学1pp/3pp、B/C新学2pp/5pp保护同时成立 |

P 的学习/遗忘前提及表中前五项系统条件实际进入 `system = all(checks.values())`，控制是否开放 C；Full−ER组件增量另由 `extra` 判定，并以系统通过为前提。组件失败不会阻止已满足系统门的 C 确认，也不能被系统 PASS 替代。组件门没有擅自加入已定设计之外的“第三 seed 至多退1pp”。[S2 `decide()`，第 125—146 行]

Q=(R11+R22)/2、O=(R31+R32)/2、F=Q−O 的实现正确。因此 O 提升与 F 改善相关，仍不能作为两个独立统计证据。初学保护的作用是防止以先学得更差来获得表面上更少的遗忘；本轮只核查该保护是否落实，没有重新选门。[S2 `summarize()`、`decide()`；N/proposal.zh.md]

### 4.4 C 使用完整确认矩阵，组件结论要求 M∧C

C 的任务集合是原 P/M 任务，因此评分走完整9格矩阵；每格的 query role 统一为 confirm，并从该次矩阵重算 Q/O/F/GB/GC。没有把 dev 的初学格与 confirm 的最终格拼接。[S5 `score()`、`analyze()`；S6 `worker()`、`run_phase()`]

C 再次调用完整 `decide(c, True)`。随后执行 `c_gate['old_score_increment'] &= m_gate['old_score_increment']`，所以 M 组件失败不能由 C 的局部正向追认。系统可通过而组件不通过，应分别报告。[S6 `pipeline()`，第 115—121 行]

C 是同一账号体系、既定 gallery 和冻结轨迹上的条件文本确认，不能表述为未见账号、独立真实同控身份或对所有开发选择无条件独立的验证。这个既定解释边界仍成立。

## 五、数据可见性、完整性与两种 V 读取器

### 5.1 完整性屏障确实位于实际执行路径

正式入口经 `permit()` 核对 Linux/解释器、授权状态、协议、CAPS/LIMITS、失败政策、13份源码及方案/scope/主审/manifest/CPU/GPU绑定。CPU02 已把 CPU 资格判断改为实际回执的 status/tests/errors/failures 字段；不再读取旧格式的 exit_code。[S6 `permit()`；T2，第 286—308 行]

阶段执行顺序为：**冻结任务 → 所有训练成功 → 训练完成清单 → 整批评分成功 → 评分完成清单 → 分析与科学决定**。`validate_training()` 检查 task/identity、状态、0至stages的端点集合、全部当前广告暴露数、更新数及每个端点绑定。worker 和父控成功回调都调用它，核对完成以后才退役 latest。评分入口再次核对整批训练；分析入口要求整批评分任务序列完整。[S6 `validate_training()`、`worker()`、`complete()`、`run_phase()`；S5 `analyze()`]

因此，单个 worker 故障后即使其他任务继续完成，也不产生完整阶段的放行条件。保存部分端点不等于该阶段已完成；缺项不会被静默排除后计算平均值。C 的开放还在 pipeline 和 worker 内分别检查 M 系统门。[S6 `pipeline()`、`worker()`；S10 `Runner.run_many()`]

### 5.2 V_CLS 没有用旧 encoder 回避遗忘

V 每市场分类头保存为 CPU 快照。评价旧市场时只换回该市场的分类头，保留当前端点 encoder；每广告 softmax 后按 query 的2条广告平均概率。V_SET 使用同一条 V 训练轨迹的 encoder 作集合评分；两个读取器都保存、共同评价，没有另训一条有利的 V_SET，也没有结果出来后择优挑读取器。[S4 `execute()`、`save_endpoint()`；S5 `classification()`、`score()`]

未来市场未学头对应的预学习 V_CLS 只具有已披露的诊断性质；已学市场若缺旧头，代码会拒绝。此次共享 V 方法文件和原训练辅助文件与上一轮包字节一致，支持在未变范围复用 V 证据；不必为新增 H CE 重跑 V 完整资格。[S5，第 153—160 行；S8、S9 的身份比对]

### 5.3 应用输出没有越过无真值边界

应用只读取 gallery，用 H_full 和同轨迹 V_SET，固定51/52/53，区分市场内/跨市场，排除同一观察账号，输出 top10、双方候选并集、both/only_H/only_V 及跨 seed 出现次数。完成记录保留 `true_alias_precision=None`、`known_owner_labels=False`。[S5 `application()`]

P 失败时没有 Full，应用关闭；M 技术完成但科学门失败时可生成无真值候选，C 保持关闭；技术缺项直接使执行异常结束，不走应用分支。候选输出不会把定量失败改成科学成功，也不支持马甲同控准确率主张。[S6 `pipeline()`、`campaign()`、`worker()`]

## 六、无 GPU 硬 cap、OOM 续领与 88h 预算

### 6.1 取消的是人为显存终止线，派发预留仍有效

`policy.LIMITS` 中 `allocator_bytes=None`、`process_gpu_bytes=None`；新 `configure_compute()` 不调用 allocator 配额接口。Runner 的整 PID GPU 终止分支只有在 cap 非 None 时生效。新 campaign 显式指定当前 step35 worker 入口和新 LIMITS，因此共享旧文件中仍存在的旧 configure 函数不会成为新 worker 的执行路径。[S2，第 11—16 行；S6 `configure_compute()`、`campaign()`；S10 `command()`、`run_many()`]

| 控制 | 实际行为 |
|---|---|
| worker 数 | 最多2槽；已有 V 时不派发第二个 V |
| H/V 预留 | 各5/17GiB，写入实际 job 与 PID identity |
| 尚未体现的占用承诺 | 按 max(0, reserved−observed_used) 从物理空闲中扣除 |
| 可用组合 | H＋H、H＋V；资源不足可等待准入 |
| 真实占用超过5/17GiB | 不会仅因超过预留而触发固定 GPU cap 终止；仍受物理 OOM 及其他已定边界约束 |
| 其他边界 | RAM、磁盘、生命周期和累计费用边界保留 |

证据：S6 `make_jobs()`；S10 `resource_observation()`、`Runner.run_many()`。5/17GiB 是启动预留，不是本轮给出的长跑峰值保证；GPU01 的3.973GiB采样值也不能反向充当新硬 cap。

### 6.2 OOM 后是“退休失败槽、健康槽续领”，不是只等当前健康任务结束

实际流程为：

1. step35 `main()` 将 CUDA OOM 与普通 RuntimeError 区分，前者写 `CUDA_OOM` 标记。
2. Runner 结合非零退出和该标记识别 OOM，并调用释放检查；释放异常会取消 OOM 豁免，按其他故障处理。
3. 失败槽进入 retired；已消耗的任务位置不回退，不重入队。
4. 当前入口开启 `continue_after_oom=True`；其余未退休健康槽可以继续领取本阶段剩余独立任务。
5. 全部槽退休则不再派发；无自动恢复失败槽、自动重跑失败任务或自动重启 campaign。
6. 最终只要存在失败或缺项，整个批次为 INCOMPLETE，Runner 拒绝下游调用；阶段不会生成完整训练清单并继续效果分析。

证据：S6 `main()`、`campaign()`；S10 `release()`、`Runner.run_many()`，特别是第 158—187、194、224—262、339—347 行。

这里的 worker 是逻辑槽位；正常领取下一任务也会创建新 PID。判断是否违规补位，应看退休槽是否复活、失败任务是否重入，不能只看是否创建了新进程。当前代码没有恢复退休槽。

非 OOM 个体故障停止新派发并按既定边界处理在运行任务；共享有效性/资源故障走共享停止路径。释放未确认或仍有未知 GPU PID 占用，不会被当作已经腾出可用空间而豁免。[S6 `main()`；S10 `release()`、`resource_observation()`、`run_many()`]

CPU01 直接经过真实 Runner，启动一个模拟 OOM 子进程及两个健康 CPU 子进程；保存结果显示失败槽0退休，两个健康任务都由槽1完成，失败费用只记一次，整批 INCOMPLETE，下游拒绝。其20GiB显存值由 FakeMemory 注入，用来检查不触发旧16GiB终止分支，**不是实际 GPU 分配或耗尽实验**。CPU02 则实际进入 main 的异常分类分支，但同样没有制造真实 GPU OOM。[T1，第 242—284 行；I/cpu01_dispatch.json；T2，第 309—319 行]

### 6.3 88h 修订已进入 Ledger 和派发预约

| 桶 | 当前实际配置 |
|---|---:|
| training | 288000秒＝80h |
| gpu | 14400秒＝4h |
| auxiliary | 14400秒＝4h |
| 总计 | **88h** |

本轮没有增加训练任务，仍为8个单域开发任务＋12条三域轨迹。job 局部帽为 D：H3h/V1.5h；P：H_SEQ6h/V4h；M：ER/Full各10h；评分 job1h。Ledger 和 Runner 使用相同的新 CAPS，配置不一致会拒绝。[S2 `CAPS`、`tasks()`；S6 `make_jobs()`、`campaign()`；S10 `Runner.__init__()`]

此前预约上沿按已给投影计算：

\[
6+19+4\times\frac{46}{6}+2\times10
=75\tfrac{2}{3}\mathrm{h}=75\mathrm{h}40\mathrm{m}<80\mathrm{h}.
\]

CPU01 的保存测试对这一条件实际调用 Ledger 预约末两条各10h任务，没有通过缩短局部帽或挪用其他桶绕过问题。[T1，第 228—233 行]

正式派发继续预约下一任务全帽及同桶活动任务的完整承诺，完成或失败后按生命周期实耗向上取整计费。aux 同时覆盖父控活动、活动 aux 作业及清理余量；接近边界会停止相关活动任务。父级 measure 不包住子作业执行或资源等待，避免把整段等待/训练再作为父控活动登记。[S10 `Runner.run_many()`；S11 `Ledger`；S6 `measure()`]

**88h 是预算配置，不是实际工期保证。** 既有约63—78h仍是投影，未因本次短测变成双路长训实测；物理资源等待、任务实耗或费用超出投影仍可能使阶段正常停止。此次88h修订解决的是已指出的预约余量，不承诺覆盖全部任务同时耗满各自最坏局部时限。[I/review/request.zh.md，第 15 行；N/proposal.zh.md]

### 6.4 本轮已发生费用未重复或回溯冲账

| 项目 | 保存的生命周期/费用 |
|---|---:|
| CPU01 | 4.68170255秒，计5秒 |
| CPU02 | 2.18858532秒，计3秒 |
| GPU01 | 17.54767092秒，计18秒 |
| 准备、核对及收尾 | aux保守计120秒，明确不是全部测得科研计算时长 |
| 本轮累计 | training0＋gpu18＋aux128＝**146秒** |

CPU8秒已包含在 aux128秒内。未来 campaign 的 Ledger 另登记一次30秒保守收尾费，初始记账为 training0/gpu18/aux158，共176秒，之后再计真实新增活动；没有再次把CPU8秒相加，也没有挪用旧71450秒预算。CPU控制台 unittest 内部计时与 receipt 全生命周期计时不同，不构成矛盾。[I/cpu01_receipt.json、cpu02_receipt.json、gpu01_receipt.json、budget.json；S11 `Ledger.__init__()`]

## 七、旧证据＋本次增量是否足够

**足以支持当前实现进入 D 前的有限工程判断；不足以支持任何正式科学效果结论。** 必要增量对应本次实际变化，而不是依靠“总计五项通过”这一数字替代覆盖分析。

| 证据 | 已直接支持的内容 | 应保留的范围限制 |
|---|---|---|
| CPU01：独立参照 | 普通 autograd＋手写 logsumexp CE，与实际分块反向的梯度和更新比较；γ0与旧H参数更新逐位等价；γ1改变encoder梯度 | 使用 tiny BERT/手写输入；H目标本身复用旧函数，不是重证全部旧H数学 |
| CPU01：状态 | 辅助初始化隔离、换域主optimizer保留、辅助optimizer重置、保存恢复下一步逐位一致 | 不代表正式任务恢复队列已经启用 |
| CPU01：阶段 | 手写Dream实际训练/评分、V_CLS选择、D停止、部分科学门反例、80h预约 | 是缩小2epoch任务；没有完成真实40epoch D，也没有动态跑完M/C campaign |
| CPU01：调度 | 真实CPU子进程＋真实Runner，OOM退休、同健康槽续领、无旧GPU阈值、缺项阻断 | 显存观察为模拟；没有实际跑混合H＋V GPU任务 |
| CPU02 | 当前permit的回执正反例、真实main异常分类入口 | 一项受影响检查，不伪称重跑全部CPU01 |
| GPU01 | 真实BERT，44记录、两通道256token、946记录对、875类；4次更新，CE实际encoder作用、历史项、双optimizer与Memory恢复后下一步一致 | 单路手写短测；不是完整训练控制器、双路长训或效果验证 |
| 旧不变证据 | 角色资格、原H/历史结构、未变V科学路径 | 只在其原范围复用；不为新增CE/OOM行为代签 |

来源为 T1/T2/T3、对应 receipt/source 清单、当前 S1—S13。GPU 的空 console 不单独构成缺证：探针将结构化结果写入 receipt，包内还提供了对应断言源码；不能只凭文件非空与否判断有无执行证据。

### 7.1 当前版本覆盖必须如实拆开

当前13份运行文件中，**11份与 CPU01_sources 完全一致**；两个变化文件经增量来源链覆盖：

- 当前 `step35_vendorlink_account_run.py`：10442字节，SHA256 `1b99cdb9dbbcaec39b914613498b069e19072f49a5c980732885ecad29e29718`，与 GPU01 记录一致。
- 当前 `step35_vendorlink_account_control.py`：16097字节，SHA256 `68eb1a42deb99dc11680037f4e4898f0ebc1ed21b69a50c3e1593bfa5939d9f7`，与 CPU02 记录一致。
- 三份 evidence_sources 分别与对应历史源记录的大小、SHA256一致；CPU02历史测试源比CPU01新增受影响测试和单项运行选项，未改变原四项测试正文。

**但来源绑定与函数被实跑是不同事实。** CPU01 实际执行 Dream 完整循环时，run 的 SHA 是 `d7bbccb6…f8602a`；GPU01绑定当前 run，却只调用其中的 `make_model()`、`groups_for()`等，并通过新 methods 做更新，没有调用 `run.execute()`。包内没有 CPU01 那份旧 run 的完整正文或精确差分，所以我不能从哈希独立重建其全部变化，也不能写“当前完整训练循环已由GPU01重新验证”。[I/cpu01_sources.json、gpu01_sources.json、cpu02_sources.json；T1、T3]

当前 run 的完整正文可读，实际域、暴露、学习率、更新、状态保存和端点路径均已静态检查，未发现可达的计算或完整性错误。核心 methods 与 CPU01完全一致，且 GPU01直接调用其真实BERT更新。因此上述版本精度限制宜记录，不足以据此要求重复GPU或完整训练。若希望补齐历史变化的展示，最小动作是提供已有Git中的小差分，不是新科研执行。

### 7.2 不把下一步逐位一致扩大成任意状态的全覆盖

GPU01 比较未中断与恢复后下一步完整模型状态及日志，显式给定相同 step_seed。它直接支持该确定更新路径所使用的参数、优化状态、Memory与恢复逻辑一致；不能仅凭这一项声称已独立测试所有潜在随机数消费者。源码本身保存/恢复 RNG，正式更新也按固定种子定位，没有因此发现具体缺陷，不需要补通用随机流平台。[T3，第 83—106 行；S3 `save_account_state()`、`load_account_state()`]

整 PID GPU **3.973GiB**、RSS **3.488GiB**是保存的采样峰值；不是连续观测上界，更不是双路长跑保证。长时 CUDA 占用变化、真实 CUDA OOM、混合 H＋V 并发没有在本次增量核验中实测；各科学门也没有逐条补做所有动态反例。相应实现判断来自静态接线与适用旧证据，这个边界不构成无依据增加资格的理由。[I/gpu01_receipt.json；T3 `check()`]

## 八、具体意见、影响与最小处置

本次没有必修项，因此不存在以笼统风险充当 `SCIENTIFIC_BLOCKER` 或 `REPRODUCIBILITY_DEFECT` 的条目。以下均不改变“授权后可进入D”的结论。

| 编号与分类 | 证据位置及触发表述 | 具体影响 | 最小处置 |
|---|---|---|---|
| A01：非阻断证据精度建议 | I/report.zh.md §2；cpu01/gpu01/cpu02_sources；T3。若后续写成“当前13文件所有入口均原生完整复验” | 会把来源绑定扩大为函数执行覆盖，尤其混淆历史run.execute与当前GPU部分函数 | 在后续证据摘要保留第7.1节的版本/调用范围；可附已有小差分，但不要求新增测试 |
| A02：非阻断措辞建议 | I/review/request.zh.md第23行、I/report.zh.md第13行称“C复用12端点/12条端点”；S6 run_phase、S5 score | 容易被理解为只复用12个最终模型，掩盖完整确认矩阵 | 改写为“复用12条既有轨迹的完整端点集合”；不改代码、任务或确认政策 |
| A03：合理后续报告义务 | S5 analyze保存逐seed矩阵和指标；N/proposal.zh.md已有结果报告要求 | 自动JSON尚不等于最终科学解释，不能只摘PASS、平均O或F | 正式结果存在后，按原约定报告逐域/逐seed、配对差、均值、样本SD及范围；系统、组件、外部比较分别下结论，无需新建通用报告框架 |

这些建议不要求打开确认文本、读真实角色、恢复旧权重或重新计算旧负结果。A01可直接用本报告的范围说明完成；A02是术语精度；A03是未来结果报告的既有工作，不是启动D前新增的完成条件。

## 九、哪些要求属于本轮过度设计

以下不能由本次已发现的问题推出，不应自动追加为外审必修：

- 重跑V完整资格、全仓历史测试、旧数据资格、旧P结果或全部已结案审查。
- 为证明代码存在而启动真实D/P/M/C、提前打开confirm，或索取账号正文、逐查询预测、Memory、权重及凭据。
- 增加γ/学习率网格、γ0完整三域轨迹、更多种子、市场顺序或新基线。这些改变科学选择和预算，不是当前实现缺陷的最小修复。
- 要求双路长训GPU资格、故意耗尽真实GPU、所有方法/seed预演。现有单路最大形状增量、真实CPU调度故障注入和静态接线已有明确分工。
- 恢复人为allocator或整PID固定GiB硬cap；把3.973GiB采样峰值转换成新的固定终止线。
- 构建通用恢复、随机流证明、队列、权限或报告平台；用无法穷举的所有异常场景作为阻止当前D的理由。
- 以“预期效果尚未出现”为由调整种子、顺序或门槛，或者把工程通过写成CE贡献/Full抗遗忘已经成立。

## 十、正式进入 D 的最终判断

**建议本轮实现外审登记为“可接受，待另获正式启动授权进入 D”。无必须先修改的代码项，无必须追加的CPU/GPU资格项。** 既有gate关闭状态应随本轮审查处置及正式授权按原流程更新；本报告本身不是训练启动授权。

进入D后，冻结选择与停止条件仍照常生效：D未过就停止适配；P不能同时满足学习与遗忘前提就不进M；M系统未过就不开C；组件不能由系统PASS或后续局部正向追认；技术缺项不变成科学通过。是否学好、少忘、旧域最终更好并保持新域，必须等待正式阶段结果回答。

## 附录：证据定位索引

本文 `N/` 表示 `reports/documentation/20261110/vendorlink_redesign/`；`I/` 表示其 `implementation/` 子目录。所有源码位置均指本次已核对字节的附件文件。

| 代号 | 文件 | 本次重点核查函数/内容 |
|---|---|---|
| S1 | scripts/step35_vendorlink_account_io.py | Check、SharedDiskLimit、pinned别名 |
| S2 | scripts/step35_vendorlink_account_policy.py | CAPS/LIMITS、settings、tasks、select、task_domains、summarize、decide |
| S3 | scripts/step35_vendorlink_account_methods.py | start_account_stage、backward_h、update_account、set_account_rates、save/load_account_state |
| S4 | scripts/step35_vendorlink_account_run.py | groups_for、make_model、execute、save_endpoint、restore |
| S5 | scripts/step35_vendorlink_account_evaluate.py | load_role、encode、rectangular、classification、metrics、score、analyze、application |
| S6 | scripts/step35_vendorlink_account_control.py | configure_compute、make_jobs、permit、validate_training、pipeline、worker、campaign、main |
| S7 | scripts/step29_vendorlink_data.py | Group、seed_for、group_chunks、assignment；区分旧准备函数与当前复用路径 |
| S8 | scripts/step29_vendorlink_methods.py | 原表示/H目标、optimizer、V更新、Memory |
| S9 | scripts/step29_vendorlink_run.py | rng_state、restore_rng、set_rates；区分旧execute/CLI与当前入口 |
| S10 | scripts/step30_evolution_control.py | resource_observation、release、Runner；区分未被新入口调用的旧configure/worker |
| S11 | scripts/step30_evolution_io.py | verify、record/write、directory_bytes、Ledger；区分旧Budget/permit与当前复用 |
| S12 | scripts/step30_evolution_policy.py | 共享模块依赖及旧默认值；确认新入口显式覆盖CAPS/LIMITS，科学门使用S2 |
| S13 | scripts/step28_formal_queue.py | GpuMemory的NVML观察；旧dispatch/gate不是当前campaign |
| T1 | evidence_sources/cpu01_test.py | Git cc8dd742对应历史4项测试源码 |
| T2 | evidence_sources/cpu02_test.py | Git f7e1f51f对应历史源；实际只运行受影响入口检查 |
| T3 | evidence_sources/gpu01_probe.py | Git1b3caa2c对应唯一GPU增量探针源码 |

此外核对：MANIFEST、I/review/request.zh.md、I/report.zh.md、I/scope.zh.md、N/proposal.zh.md、N/review/primary_disposition.zh.md、原方案外审报告，以及I内bindings、gate、cpu/cpu01/cpu02/gpu01回执与来源清单、控制台、dispatch、preflight、budget、sync、cleanup。上述范围不包含项目私有载荷或当前服务器的独立实时验证。
