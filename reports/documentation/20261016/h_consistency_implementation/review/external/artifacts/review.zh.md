# H_cons 批准合同与最小实现：合并、反对性审查

日期：2026-10-16。对象：本次收到的 H_cons 实施包。项目证据中的 UTC、Asia/Shanghai 时间按原件保留。本审查由当前实例独立完成，未委派、未连接项目服务器。

## 一、结论与停止位置

**核心训练实现及现有 Linux 手写 CPU 增量可以在明确边界内接受；没有发现目标、两学生回传、共享 VJP、独立 A→B 轨迹或统计方向的已证科学阻断。但是，当前包还不能出具“APPROVED_NO_OPEN_BLOCKERS”结论：原生资格缺少批准合同明确要求的两组 encoder/head Adam 变化记录。建议以“有一项待补正的复现性缺陷”退回定向处置，补正并由主执行者独立关闭前，暂停依赖本包的 GPU 准入。**

这不是宣判 Adam 更新错误，也不是推翻已经完成的 CPU 证据。发现的具体问题是：合同第166行要求在调用2/3和4/5分别记录两组件的裁剪前后梯度及实际参数/Adam变化；当前 `component_effects` 只比较并输出梯度、参数，虽已捕获完整 Adam 状态，却没有消费其中的 Adam 矩做这两组记录。第5/6次完整状态一致性检查解决的是恢复/诊断中性，不能替代第2/3、第4/5的开关C比较记录。详见 R01。

本次结论分为四层，不能合并为一个 PASS：

| 层次 | 本次判断 |
|---|---|
| 已批准范围、目标及最小实现 | 批准已存在；核心公式和实际入口一致，未发现已证科学阻断 |
| Linux 手写 CPU 增量 | 现有三轮可按实际方法变化继承；累计41/180秒，不是第三轮重跑全套 |
| 新目标 BGE/GPU 原生资格 | 尚未执行；先补 R01，再由主执行者独立处置，仍须完成唯一六调用及真实资源测量 |
| 唯一正式运行及性能结论 | 尚未执行、尚无新性能结果；本报告不是可执行的 NATIVE/FORMAL 门，也不代替用户的条件授权或主审 |

**最小处置范围仅是原生比较证据的生成/保存及其受影响核验。** 不需要修改 JS、H、D0、权重、六调用计划、训练轨迹、比较方法、精度或预算；不需要重训旧 S/H、不需要第七次 GPU 更新、不需要读取旧 Memory 或正式数据补动机。已有41秒计费、原CPU原件和旧结果均保持。

## 二、输入身份、批准时序与审查可见性

### 2.1 本次实际输入

实收附件为 `review_input(20261016-053330).zip`：

- 424,454字节；SHA256：`b8f344c4eb7f50276bf1967ed6c1579aa47d871828d0bf69015c02534c9a4e73`。
- 92个成员，名称唯一；91项载荷加 `manifest.json`。
- ZIP CRC无坏成员；没有路径越界、重复成员、未列载荷；91项载荷的大小和SHA全部与清单一致。
- 42项当前直接来源逐项核对；三轮各42项来源映射、各轮 `sources.json` 和 `result.source_files` 相互一致。

这些是实收身份及包内自洽核对。当前消息没有另给一个外部预期包SHA，不能把包内一致夸大为额外第三方来源认证。Files 接口未返回ZIP正文，随后读取开发者提供的已挂载原始路径；没有由文件名猜造附件。原ZIP以不改字节方式附在交付包中。执行核对见 `evidence/mechanical_audit.py`、`evidence/mechanical_outputs/mechanical_summary.json`；完整成员身份见 `artifacts/input_identity.json`。

`changes.diff` 的10个文件区段、全部当前侧hunk均与实际源文件一致。三份修改过的共享文件可由差异逆向重构；`g_half.py` 的顶层定义变化确实只有新增 `metadata`。其余7个区段是新文件。没有取得项目Git对象库，所以能验证“差异当前侧与提交文件相符”，不能独立证明旧侧确实来自Git提交 `3cd20ad3`。没有因此要求额外拉取仓库。[`changes.diff`；`evidence/mechanical_outputs/diff_verification.json`]

### 2.2 已批准，不是“静态未授权”仍在生效

`authorization.json` 的状态是 `APPROVED_FULL_CONTRACT_GATED_SEQUENCE`，观察时间为 `2026-10-16T04:35:10.990Z`，绑定合同30,924字节、SHA256 `57e39227b2a789316f3475af7aa97a88c834db883b0b7c9a7a45e9db6feb68bf`，与本次合同字节一致。它批准的是包含外审、独立主审、原生资格、投影和唯一正式运行的条件顺序，不是这些条件已经满足。[`sources/reports/documentation/20261016/h_consistency_implementation/authorization.json`:1–11；合同:201–209]

因此，合同形成时、旧静态外审或主审中“未授权实施”的句子不能否定这项后续批准。`manifest.pending` 保留制作时“specific package upload permission”等表述，也不覆盖本次用户已提交这个具体包的事实。反过来，上传审查不代表 GPU 已通过，不准把旧 H 的原生PASS、六次调用或余额转给 H_cons。[`request.zh.txt`:1–27；`manifest.json` 的 `pending`；旧静态主审与批准文件]

### 2.3 实际阅读与身份边界

正文的路径相对于原输入包；简写如下：C＝`sources/docs/SELLER_ALIAS_H_CONSISTENCY.zh.md`；M＝`sources/scripts/step28_h_consistency.py`；R＝`sources/scripts/step28_h_consistency_run.py`；S＝`sources/scripts/step28_g_half_run.py`；V＝`sources/scripts/step28_g_half_verify.py`；T＝`sources/tests/test_step28_h_consistency.py`。行号指原始UTF-8文本，单行JSON另给字段。

完整阅读批准合同、授权、本请求、新目标/运行器/资格入口/policy/定向测试；沿调用阅读受影响共享单臂运行器、六调用核验器、record目标回传/Memory、诊断和必要保存/解析/统计依赖。历史外审原文采用定向章节阅读，结合对应独立主审，仅继承未变范围；未全量重新审查旧F、旧数据项目、全部旧测试或全部历史审查。逐文件层次及范围见 `artifacts/reading_scope.json`，不能将92项字节核验写成92份全文科学审读。

旧H两份权重已删除、旧Memory仍受原保管边界约束，按 `evidence/H_result_primary.zh.txt`:36–38继承；本次未重新核对服务器删除现场，更没有据此请求载模或读取旧Memory。

本实例自述为 GPT-6 Astra Pro。本次没有独立读取模型选择器或后台路由，不宣称已验证网页“GPT 6 Pro”的实际服务路由，也不改写旧审查身份。本次没有外网文献检索，因为问题是冻结合同/实现的合并审查，不重开创新性搜索。

## 三、目标与梯度：已经接通的是什么

### 3.1 H评分没有被JS替换

`M.identity` 返回旧H评分身份；`M.account_logits` 调用原 `g_half.account_logits`，不是复刻一个近似H。原H对一个记录块K计算

\[
H(K)=\tfrac12\operatorname{mean}(K)+\tfrac12\{\operatorname{logsumexp}(K)-\log |K|\}.
\]

`g_half.block_scores` 将输入K升为FP64，两个池化分量在FP64计算，然后一次转回输入dtype。实际生产K是FP32；JS接收的正是这份FP32 H输出，再升FP64，而不是拿一份未量化的另算H作为JS学生。推理仍只经自然A0的旧H，评分入口没有JS。[M:17–20、42–44、117–124；`sources/scripts/step28_g_half.py`:39–73、109–137；C:11–36]

原监督没有改变成JS监督。每个视图仍有BCE、多正例query rank和固定0.5 hard项，A0/A1监督先平均。hard候选由该视图自己的H分数，在已知负例中稳定选top5；只对离散索引选择detach，所选分数仍live。JS的27列则按规范候选索引排列，不按各视图分数或top5重排。[M:70–93；`step28_alias_ranking.py`:89–129；`step28_chinese_base.py`:165–191]

### 3.2 27候选、28平均和双向边均正确

`query_logits` 要求378个完整账号对分数，用对称索引映射为28×28，再去掉自环形成28×27；同一无向边在两个查询方向分别出现。两个视图使用同一规范候选顺序。`consistency` 在候选轴做 `log_softmax`，混合分布用live `logaddexp(l0,l1)-log(2)`，候选轴求和，仅沿28个query取平均，最后一次转回FP32。没有额外除27、补epsilon、clamp、单向KL或从日志浮点数重建C。[M:47–67]

记某query的两分布为p、r，m=(p+r)/2，群级目标为

\[
C=\frac1{28}\sum_q\frac12\left[\sum_jp_{qj}\log\frac{p_{qj}}{m_{qj}}+\sum_jr_{qj}\log\frac{r_{qj}}{m_{qj}}\right].
\]

对视图0某logit的独立解析导数是

\[
\frac{\partial C}{\partial z^0_{qi}}
=\frac1{56}p_{qi}\left[\log\frac{p_{qi}}{m_{qi}}-\sum_j p_{qj}\log\frac{p_{qj}}{m_{qj}}\right],
\]

视图1对称。每个无向账号边的梯度必须先汇合它在query两端的贡献；每个H块的记录对导数为 `0.5/|K| + 0.5 softmax(vec K)`；最后再把A0/A1两条分配路径加到同一完整记录表。只比较一侧query梯度或某一视图梯度不能证明总K回传正确。[C:38–53；M:47–67；T:30–109]

两个学生以及m都在当前图上。原件 `js_precision.json` 保存实际 `LogaddexpBackward` 的两个 `LogSoftmaxBackward0` 父节点；测试同时检查源码没有学生detach。仅detach m的一阶导数等价性是已关闭事实，不要求用一阶更新去识别它，更不新增Hessian资格。[T:156–198；`cpu/attempt01/result/js_precision.json`]

### 3.3 总目标与五项日志重建一致

当前分支无reference，权重1；历史分支有自己的出生表，监督与C各乘0.1，D0独立乘0.5：

\[
J_A=\overline B_{cur}+C_{cur},\qquad
J_B=\overline B_{cur}+C_{cur}+0.1\overline B_{hist}+0.1C_{hist}+0.5D_{0,H}.
\]

`supervised` 只记录原Bbar，`consistency` 独立记录C；每个分支还记录 `supervised_contribution`、`consistency_contribution`、`teacher_contribution` 和总weighted。D0先将live自然A0和自己的冻结出生K表各经H投影，再平方差均值；没有把0.5放进历史0.1内变成0.05，D1没有进入优化。[M:70–93]

新的 `loss_total` 不信一个总损失字段：验证双身份、阶段/分支结构、qualification flag、current无教师、mse1为空、各分量权重及weighted，再重建五项总量。正式模式的更新入口拒绝 `qualification_without_c` 或 table observer；盲门也拒绝去C日志。这是训练目标与审计重建一起变更，而非只把名字改成H_cons。[M:96–114；R:113–139；S:422–451]

### 3.4 C确实经head及encoder参与同一次更新

实际链路为：Linux shell → 新run → 显式entry的共享单臂流程 → 新 `method.update` → `record.update(objective_fn=...)` → 每个current/history的 `record.backward_group`。后者保留一次encoder表示z的图，取得完整记录对K并在proxy上对**完整J**求 `dJ/dK`；随后每512记录对重算head，对该块使用上述总K梯度做VJP；所有块累积到leaf后，`z.backward(leaf.grad)` 返回encoder。当前和历史先累积，之后一次clip-norm=1、一次AdamW.step。[M:96–114；`step28_record_replay.py`:159–208]

proxy的detach不是丢弃C，也不要求另造一张跨全部记录对的永久大图。关键是其导数实际作为每个head块的上游梯度，并最终传回原encoder图；当前代码满足。新的JS没有增加encoder前向分支，但不能由此推断运行时间或峰值不增加。

原学习率日程仍是阶段内1…288，A/B optimizer时钟连续；阶段末encoder学习率为0，所以“576 optimizer调用”不能改称每一次encoder都发生非零位移。资格选择step2和step49避免把零学习率端点拿来要求不可能的作用。[C:65–69；S:177–206；`step28_bge_continual.py` 的阶段日程；V:148–191]

## 四、身份、独立轨迹、Memory及恢复

### 4.1 显式entry，而非生产全局替换

新run用 `functools.partial(...,entry=sys.modules[__name__])` 指向既有单臂函数；共享函数内部使用entry的method、policy、baseline、loss重建、诊断检查和metadata。资格入口显式传入新entry、手写模块和consistency=True。没有在生产代码中用monkey-patch把旧模块全局method替换；测试内的mock不应被误报为生产替换。[R:160–182；新verify:1–7；S中各entry函数；V:253–356]

H的评分身份与H_cons训练身份是两个不同对象。训练身份包含候选数/轴、排自环、28平均、lambda/温度、两学生及m live、精度路径与各权重。它进入来源、日志、模型、端点、Memory metadata和诊断；完整/推理恢复后均重新bind并拒绝缺失身份。这里保留旧 `pooling.arm=G_half` 是正确的评分身份继承，不是新训练名没有生效。[M:17–39；R:33–72、142–151；S:209–302、368–503]

### 4.2 新臂自己的A、出生表和B连续性

`train` 只初始化一次新H_cons，从共同预训练及初始空Adam身份出发；A训练后生成自己的A端点和六个出生表。B从恢复后的自身A模型/Adam/RNG/模式/Memory继续，不从旧H或S的训练状态启动。[S:305–365]

六组Memory、Algorithm R和1MiB约束复用；A阶段保留的完整记录表及origin hash在B过程中不刷新，新臂自己的表值不要求与旧H相同。分配/抽样/随机命名空间仍为ABC；实际日程、current/history UID、A1摘要和dropout种子在日志中保留。新训练身份不会改变随机键。[`step28_record_replay.py`:234–332；S:177–216、368–503]

每个端点对12 calibration＋40 blind做原始评分、完整恢复后评分、推理恢复后评分；逐元素相同才继续。Memory是独立序列化的自身载荷，完整连续状态还绑定Adam、RNG、模块模式和Memory摘要。`blind_gate` 重建并核对的是新臂自己的Memory与轨迹，旧对照仅用批准保存输出/metadata。未见调用旧完整blind_gate或跟随旧manifest中的Memory路径读取载荷。[S:236–302、368–514]

### 4.3 诊断实际使用H，A1机制诊断没有新增

新capture同时传入自然评分回调和H池化回调，避免“预测已换H，D0/D1仍用mean”的混合。B诊断的48 fit/6 cache/40 blind角色完整，D0/D1以live记录表及出生表独立投影，D1只在原诊断中出现，不是新增正式A1机制研究。[M:127–136；`step28_replay_diagnostics.py`:61–129]

诊断中立域保存并恢复Python/NumPy/Torch/CUDA RNG及所有子模块模式；实际手写测试还比较model/Adam/Memory与下一更新结果。六轮曲线只有在新blind gate后得到valid标签时才做评价。原九项严格判读、whole-group抽样及“不显著不等于无过拟合”的解释继承，没有新增epoch选择。[`step28_replay_diagnostics.py`:18–43、132–208；T:322–368]

## 五、旧S/H双根比较与统计含义

### 5.1 52文件的身份和允许读取范围

批准列表锁定52文件、合计2,940,294字节。S来自 `reports/seller_alias_continual/20261015/balanced_pooling_result/job`，H来自 `reports/seller_alias_continual/20261016/g_half_result/job`，H保存臂名仍是G_half。`baseline_runs` 先核对批准列表SHA、52项和8项未变直接来源，再独立读两根的manifest/collected/partition，核对40UID、22列、三角色和初始身份；S/H的public-input metadata和partition还需彼此相等。[R:75–110；C:75–86]

我逐项核对了批准列表与 `deployment_receipt.baseline_files` 的大小/SHA对应，8项未变源码与本包字节也一致。**我没有连接Linux重哈希那52份原件，包内也不含其全部载荷。** 可以说“Linux回执记录与批准列表一致”，不能说“网页审查独立读过52份实际保存输出”。列表含已开放群定位标识；这些标识不授予旧Memory、模型或正式数据载荷读取权限。

新盲门还将新臂实际288+288日志的UID/分配摘要/随机轨迹与两根旧日志直接配对。起点和抽样路径应一致；训练后参数和出生表值可以不同，后者正是干预的结果，不应错误地强求相等。[S:146–148、497–502]

### 5.2 双根映射确实接入finalize，但项目夹具未充分区分它

`finalize` 为每一个端点维护 `origins[name]`：新端点使用新evaluation目录，S与G_half分别使用各自旧根的evaluation。读取matrix、counts、queries都由这个按点映射的origin决定，不由最后一次循环遗留的old_root决定。[S:549–591]

但当前CPU比较夹具把S和G_half都映射到同一个oldroot，并复制同一批保存数值；编排夹具也用同根。这能够检查三角色、字段、对齐和统计代数，却不能单独检出“两个真实旧根被混读”。不能把它的通过描述为一个有区分力的双物理根读测。[T:435附近、482–508；发现N01]

本审查额外构造了**三个不同根、故意相同相对文件名、S/H不同数值**的保存代数夹具，真实调用当前finalize，只mock允许的baseline元数据供给；得到H_cons−S/B=0.30、H_cons−H/B=0.15，证明本例实际按根区分。此结果是网页隔离读取/统计检查，不是项目Linux资格，更不是模型改善。[`evidence/probe_outputs_v2/probe_summary.json` 及三个distinct_root目录]

### 5.3 判据分离、遗忘方向和N/A均正确

B末主要判据是H_cons−S的raw MAP在A、B各自条件95%区间下界严格大于0；新B直接改进H是独立的H_cons−G_half/B区间判据。两者由逐群直接差计算，绝不相减两个相对S区间。H_cons−H/A没有新增非劣门，但完整保存在主比较中；A末差、旧A原始变化、遗忘差也分别输出。[S:593–612；R:33–43]

高好指标遗忘是A末−B末，Brier/log-loss低好指标遗忘是B末−A末；候选相对对照的遗忘差再相减。`old_changes` 始终是原始B−A，不因指标方向改号。此区分已落实到实际算式，不只在解释文本中改名。[S:593–610]

统计使用唯一PCG64(20260930)的(5000,2,20)整群索引，共用于比较、角色和差的差。40群固定为每域20群；不按28个query独立重采样。22列与query10列保存矩阵相互校验。precision/MCC的N/A按混淆矩阵分母确定；固定组内任何必需操作数缺失，该均值/区间为N/A，不能各臂nanmean后制造不同样本比较。[S:565–591；`step28_balanced_pooling_run.py`:648–680]

隔离夹具进一步验证：故意使S/B旧A中的1个必需群precision/MCC无定义，比较保持20群并返回N/A，不删群；MAP遗忘差−0.1，log-loss遗忘差+0.1。其退化区间和true布尔值完全由人为常数构成，不是效果估计。

即使未来两判据都通过，也仅支持本固定、单seed、反复开发valid下的系统比较。不能由此识别C在B阶段的孤立作用、H与JS的特有协同、已定位的捷径根因或独立论文贡献。新臂A也被C改变，B起点和出生表随之改变；本合同没有S+JS或正式A1机制诊断，不应暗中追加这些比较，也不能借其缺失取消用户已经批准的有限开发问题。[C:30、92–106；旧静态外审定向章节]

## 六、三轮Linux CPU证据：采纳范围和不能采纳的表述

### 6.1 计费与来源继承成立

| 轮次 | unittest内层 | 结果计时elapsed | 外层墙钟 | 本轮收费 | 累计 | 实际范围 |
|---|---:|---:|---:|---:|---:|---|
| attempt01 | 22.351s | 23.74305018s | 24.13945484s | 25s | 25s | 六项；三种计数均0 |
| attempt02 | 0.681s | 2.07406818s | 2.38010406s | 3s | 28s | 只补共享K/冻结原生几何项 |
| attempt03 | 10.882s | 12.24273089s | 12.64404964s | 13s | 41s | 只补新臂盲门/双对照输出收集项 |

三个进程exit0，failures/errors/skipped均0。收费按外层与result内层elapsed较大者逐次向上取整，不能以attempt01内部24秒替代应计25秒。累计41/180秒，原生0次、正式0次。[各轮 `wrapper.json`、`result/result.json`、`result/unittest.txt`；三份CPU shell；`evidence/mechanical_outputs/CPU_provenance_and_charges.json`]

三轮42项来源中，除新测试文件外全部相同。AST核对表明：01→02只改共享表测试方法，增加224记录/254汉字/三阶段角色的固定几何；02→当前只改保存比较和新臂编排方法中的两个证据文件名前缀。其余方法可按原来源继承，不能把最后1项写成当前全套6项重新执行。[`source_mappings.json`；`cpu_receipt.json`；原测试两版本；`evidence/mechanical_outputs/test_AST_changes.json`]

### 6.2 数值证据支持什么

共享表独立参考不调用生产JS生成期望：手工规范候选、稳定概率及解析logit导数，再经独立H块枚举回传。保存数值为：56记录FP32共享K总梯度L1误差 `1.0874424223753439e-8`；224记录三例最大 `1.1430479432532657e-8`，低于合同 `5e-7`；10坐标FP64差分最大误差 `1.3083286827509331e-12`，低于 `2e-9`。同时覆盖双学生缺失、错误候选对齐/额外平均、合法零例及稳定极值。[T:30–114、156–252；`cpu/attempt02/result/shared_K_precision.json`；`cpu/attempt01/result/js_precision.json`]

真实tiny更新的开/关C对照分别观察到：current encoder最大参数差 `7.450580596923828e-9`、head `1.0728836059570312e-5`；paired encoder `1.1920928955078125e-7`、head `3.33935022354126e-5`。独立完整图还逐张量检查分项、裁剪前/后梯度、更新参数及Adam step/一阶矩/二阶矩。保存JSON是检查摘要，不包含每个梯度张量原件；实际逐张量比较依据是对应测试源码及该进程成功执行记录，不是仅相信 `adam_moments_compared=true` 这个字段。[T:254–320；`cpu/attempt01/result/complete_graph_and_update.json`]

这些数字证明该tiny非退化例中新增项确实传到两个组件、与独立单图一致，不证明BGE上也能保留可辨参数差，不证明λ=1有效。attempt02的最大形状是手写文本结构和合成K回传，不是已加载BGE、实测256 token编码或BGE的224记录真实更新。

实际tiny capture走了48/6/40角色、H评分、H的D0/D1、完整/推理恢复与下一步逐元素一致。576/864、2端点/12诊断则来自另一个明确mock forward/update/metrics/calibration的编排夹具：它的实际训练更新数是0。不能把两个测试拼成“已经训练576步”，也不能把人为保存值产生的双true当模型结果。[T:322–537；`cpu/attempt01/result/capture_restore_identity.json`；`cpu/attempt03/result/h_cons_orchestration.json`]

### 6.3 覆盖缺陷已关闭，但原件没有被追改

attempt01的新臂orchestration/comparison使用旧H测试的文件名，之后被旧默认回归覆盖。unittest可以证明相应方法当时通过，不能证明被覆盖的新臂具体JSON还在。当前T的两个 `h_cons_` 前缀和attempt03的实际输出补足此缺口；01原件不改，成本不清零。该项为已关闭复现缺陷R02，不应再要求全套重跑，也不能在报告中把01那两份JSON认作H_cons细节原件。[T:477–480、536；`cpu_receipt.json/reproducibility_correction`；attempt01与attempt03对应JSON]

## 七、六调用原生计划：大体接通，但R01必须先处置

### 7.1 六次实际调用与固定手写状态

| 调用 | 代码实际路线 | 应形成的证据 |
|---|---|---|
| 1 | 共同预训练、空Adam，去C的原H目标，stage1 step1 | 一次真实准备更新后W，非空Adam；不冒称历史训练 |
| 2 | 从W恢复，同current，去C，step2 | current对照 |
| 3 | 从同W恢复，加C，step2 | 与2比较总目标差、两组件梯度/实际参数作用及同K的H/JS解析回传 |
| 4 | W参数/矩不变，只将Adam时钟显式置336构成P；paired去C，stage2 step49 | 配对对照，活跃历史监督与D0 |
| 5 | 同P、同current/history/T/RNG，加C，step49 | 与4比较Ccur＋0.1Chist及两组件作用 |
| 6 | 从P恢复，实际48fit/6cache/40blind capture，再重复5 | 与5比较model/Adam/梯度/RNG/模式/Memory及分项日志逐元素相同 |

[V:124–210；C:153–168]

资格去C通过新method的专用flag执行，但实际目标等价旧H；不是重训旧对照，也不是新增正式臂。我在同一手写K上分别核对FP64/FP32、current/history四种情况，去C与旧H目标及K梯度逐元素一致。[`evidence/probe_outputs_v2/probe_summary.json`]

教师构造确为W eval下历史完整FP32记录表加 `2*sin(.31*i+.71*j)`；i、j取224条记录的上三角索引。它不是正式出生表，也不是旧Memory。P的参数和矩来自一次真实准备更新，只合成时钟336。手写Memory额外抽48次是为了诊断时钟，不能叙述成之前已训练336步或已经正式抽样48次。[V:171–202]

原生最大current/history要求224记录、448个通道序列、每序列256 token；CPU固定的是254汉字构造，真正tokenizer长度仍须原生实测，不合则本次失败，不换文字反复尝试。[V:91–121；C:153–164]

### 7.2 计数、精度和恢复检查不是空接线

`updated` 开始先限制不足6次；实际update返回后先追加updates并写progress，再核对捕获的同一K上H/JS值及独立回传，因此局部精度失败不能把已经完成的更新当作未消耗。只有加C的调用3、5、6捕获table，分别1＋2＋2，共5份表检查；资格成功要求实际6次，失败可以更早停。没有追加第7步、换fixture或自动重试循环。[V:124–147、210、253–356；R:147–151]

两对current/paired分别要求encoder/head有裁剪后梯度差和**真实参数差**，还比较选定关键参数的裁剪前梯度；J差与对应C贡献要吻合。恢复/诊断后下一更新比较包含Adam及所有模型张量；完整和推理保存恢复后H分数逐元素相同。资格文件生命周期最多两个完整fixture加一个推理文件，大文件删除前保留身份。[V:44–72、158–243]

这些是实际将执行的检查代码，不是已有BGE观察值。CPU的微小非零差不能保证BF16 encoder、FP32 head/Adam在原生预算内通过；C作用若被clip/Adam舍入压掉而不满足合同，就应按原计划失败，不能降低标准或寻例。

### 7.3 R01：两组Adam变化记录没有落地

**类别：REPRODUCIBILITY_DEFECT；状态：OPEN；建议阻止当前包作为“无开放问题”的GPU前提。不是SCIENTIFIC_BLOCKER。**

**合同依据：** C:166明确要求2/3和4/5分别记录两组件的裁剪前后梯度、实际参数/Adam变化。

**代码事实：** V:29–31的 `state` 确实捕获 `adam=optimizer.state_dict()`；V:158及178分别保存gcurrent/gpaired。但 `component_effects(model,on)` 没有optimizer参数，V:44–72只读取 `on['model']` 与 `on['grads']`，输出参数/梯度delta，没有访问 `on['adam']`。这两份状态在V:170、192删除，返回报告V:244–250也没有两组Adam矩比较摘要。该helper虽然复用旧实现，也不能因此豁免本次批准合同的明确记录要求；本发现不追改旧H资格结论。R:147–151的新原生验证同样没有这项记录要求。

**触发条件：** 现代码完成一次看似全部通过的六调用资格时，结果仍可能包含 `both_phases_encoder_head_effect=true`、`next_update_bitwise_equal=true`，却没有合同约定的2/3和4/5分组件Adam变化记录。

**科学/复现影响：** 无法从该资格报告审计“开关C比较”的优化器矩变化；CPU的独立全图Adam比较和第5/6的恢复一致性是不同问题，不能移作缺失记录。现有真实参数差已提供下游作用证据，因此不能进一步宣判Adam坏了、C未训练或算法无效。这里限制的是证据完备性及原生准入说明的真实性。

**最小处置：** 利用已经在内存的off状态和当前on optimizer，在删除gcurrent/gpaired之前，按当前模型的命名参数与optimizer组对应，为current/paired各自的encoder/head保存 `step`、`exp_avg`、`exp_avg_sq` 的实际对应/有限性/差异摘要；参数和梯度比较保留。至少记录合同要求的实际变化，可明确采用同一起点开关C后状态差，不必另存巨大的Adam副本。无需新增“每个矩都必须超过某阈值”的门，因为合同硬要求是每组件实际参数差，不应借补记录新造精度标准。

只补受影响的辅助函数/输出和必要定向测试；更新来源及CPU增量回执，41秒原账继续，不能把旧回执直接贴到新源码。主执行者应独立核实并决定如何关闭；不需要全量重审、不需要修改生产训练算法或增加GPU调用。正式门生成前应确认这两组记录实际存在，而非再用一个笼统布尔值代替。

## 八、资源、数据权限与正式入口

### 8.1 三本新账与旧费用

新CPU180秒、新GPU600秒/六调用、唯一正式28800秒分别计量，CPU当前41秒；旧H CPU17、GPU320且6次、正式6050秒保持，不挪余额。旧原生内部计数即使有不同展示值，也不能据此改写主审已采用的外层保守计费。[C:176–182；旧H结果主审:38；本次CPU wrappers]

CPU shell在启动前隐藏CUDA，单CPU/单线程，使用既有py310；外层与内部较大值向上取整且保留失败。正式shell记录包装开始时刻并用统一截止时间约束解释器/执行及TERM宽限；运行器另检查资源、输出、时长并在异常保留失败。没有自动正式恢复或重试。[三份CPU shell；`run_step28_h_consistency_linux_20261016.sh`:1–23；S:656–711；V:253–356]

### 8.2 投影不漏工作量，但数值仍待原生实测

合同及实际 `projection` 一致：

\[
T_{proj}=1.25\{288(t_A+t_B)+1446v+36h+L+2(s_F+r_F+s_I+r_I)+1800\}.
\]

1446v的构成为：A六诊断 `6(48+40)=528`；B六诊断 `6(48+6+40+6)=600`，最后6是残差需另取live全表；两端点三种评分过程 `2×3×52=312`；A出生六表6。另有B六轮×六cache＝36h，h覆盖取得K/T后D0/D1所需H投影。训练288次current＋288次paired计入tA/tB；L、完整/推理保存和恢复分别计入；1800秒是扫描/校准/指标/整群bootstrap/保存分析及小IO的未实测预留，再整体加25%。[C:184–195；S:163–174；正式训练/诊断/端点入口]

输出投影为

\[
W_{proj}=\left\lceil1.1(W_F+2W_I+2\,GiB)\right\rceil\le10\,GiB.
\]

正式生命周期允许一份临时完整状态和两份保留推理状态；输出监控同时计入临时/侧车，保存前还有reserve检查，不只计算最终归档。原生两完整fixture＋一推理状态同样受硬峰值约束。来源/诊断/Memory/原子写入等不能因最终删除而从峰值中扣掉。[S:151–174、236–302；V:81–88、219–243；`step28_record_replay_run.py`:60–106；`step28_continual_population_run.py`:63–76]

代数测试得到4833.75秒是把各测量设为1的算式例，不是H_cons预测。25%和1800秒不是已证明的上界；实际H_cons测量、峰值显存reserved≤28GiB、RSS≤64GiB、输出≤10GiB以及项目现场资源仍须满足。R01补记录的实际开销也仍在原600秒内，不授予额外预算。

### 8.3 七输入、A/B筛选、一次尝试及盲门

公共输入入口只走合同七文件：manifest、validation、group metadata、train文本/标签、development文本/标签。技术扫描会遇到ABC，但在构造模型Group/编码之前筛掉C；标签文件先按许可定位群，再只对A/B做数值对齐。不能把“扫描过一个文件”说成C进入训练，也不能把小文件/元数据视为放开test、owners或private。[C:131–141；`step28_balanced_pooling_run.py`:89–173]

`parse_once` 在调用标签加载器**之前**把对应尝试计数加一并写账，异常也消耗该次尝试。新正式run只解析train一次；`open_valid` 要求train1/valid0/heldout0/owners0，完整新blind gate成功并写 `before_valid` 后，才尝试解析valid。原52文件是已开放对照，不通过它们载旧模型或旧Memory。[`step28_bge_continual_run.py`:76–87；S:305–365、368–514]

对权限的本次判断来自实际代码路径与保存证据，不是已经在项目现场观察到正式数据访问。GPU尚未运行，正式更未运行；没有新valid结果可评价。

## 九、发现分类、最小处置与不采纳的扩展

| 编号 | 分类/状态 | 位置、触发与影响 | 最小处置 |
|---|---|---|---|
| R01 | 复现性缺陷／开放；GPU前补正 | C:166；V:29–31、44–72、158–192、244–250。两组开关C资格未记录分组件Adam变化，不能声称完全兑现合同 | 用已捕获状态补摘要，定向CPU及来源更新；不增GPU步、不改算法/容差 |
| R02 | 复现性缺陷／已关闭 | attempt01同名输出覆盖；T:477–480、536及attempt03证明补存。原01具体新臂JSON不可使用 | 接受03受影响项；原件、41秒和原通过范围保持 |
| N01 | 证据范围／非阻断建议 | T:435、508同根模拟S/H，不能单独检出双根混读；实际S:549–591映射正确，隔离反例已过 | 限定现有CPU表述；后续受影响测试可用不同根/值，不强制扩建或全量重测 |
| P01 | 尚未满足的阶段前提，不是已失败 | 原生6调用、BGE作用、真实时间/峰值/投影尚无数据 | R01和独立主审后按原计划唯一尝试；不得以CPU或旧H PASS代替 |
| L01 | 科学主张边界，不是本轮新缺陷 | 单seed、复用开发valid、A也被干预；双域胜S与新B胜H不同，缺少S+JS不支持特有协同 | 保留限定和旧A/H完整披露；不新增正式臂或非劣门 |
| E01 | 审查者自身错误／已关闭 | 隔离脚本v1错把Group字段写为records，在开始数学核算前AttributeError退出1；不是项目源码问题 | v1/traceback/环境/退出留存；v2仅改字段及独立输出目录，退出0，不计项目账 |

没有把R01提升为已证算法/优化器科学错误；没有把P01“尚未执行”写成失败；也没有把N01的夹具区分力不足夸大为实际双根读取错误。反对性审查应区分这几种不同强度的结论。

明确不采纳：要求重训/恢复旧H权重、读旧Memory或封存载荷；追加S+JS、seed、网格、正式A1机制诊断或第七次GPU调用；要求用一阶差异区分仅detach m；把对称全图的局部offset争论重开；为一个记录缺口创建通用资格平台；把旧审查冻结“未授权”当成当前拒绝全部实施的理由。它们均超出本轮最小必要处置。

## 十、审查侧隔离核算与完整证据

本次审查没有加载任何encoder/head模型，没有调用项目CPU/GPU资格入口，也没有运行optimizer更新。独立数学核算只构造28账号×2记录的手写完整表和标签；统计核算只构造40群保存指标/计数/query数组。生产target、H/JS及finalize等纯数表函数被直接调用，项目正式文本/标签、模型、Memory均未读取。

环境：Python3.13.5、PyTorch2.10.0+cpu、NumPy2.3.5；CUDA隐藏、单线程/CPU0、offline环境。与项目py310/PyTorch2.9.1+cu130不同，不能迁移为项目原生资格。

保留的执行包括：

1. `reviewer_probes.py` v1：3.978567秒、退出1；Group实际字段为items，误写records，尚未开始数学/统计检查。源码、完整stdout/stderr和环境/时间保留。
2. `reviewer_probes_v2.py`：5.532002秒、退出0；只修上述字段和输出目录，未改手写K、算法、精度或判据。FP64 C误差 `6.2450045135e-17`，共享K梯度L1 `9.2109115542e-17`；FP32 C误差 `8.6008965922e-10`、共享K梯度L1 `6.6774025461e-9`。四种去C与旧H目标/梯度逐元素相同；不同根及N/A代数检查如第五节。
3. `mechanical_audit.py`：0.869257秒、退出0；CRC/载荷/42来源/三轮计费/方法级差异/52文件回执/合同绑定/diff当前侧核对和R01的AST事实检查。它是字节与结构核对，不是科研效果重算。

完整手写输入、两dtype的实际/参考H及query/K梯度数组、三个根的完整人工保存值及完整evaluation、脚本和退出记录都在 `evidence/`。输出中出现的true与置信区间属于人工夹具，不属于H_cons正式结果。第一次审查脚本错误不算Linux项目失败，也不能从交付中删除。

## 十一、最终移交意见

主执行者可接受本次对目标、共享VJP、身份/自身恢复、新盲门、保存对照映射和统计方向的有限实现结论，接受三轮Linux CPU的按方法继承及41秒账。**当前仅有R01作为必须独立处置的开放复现性补正；关闭前不把本包写成无开放问题并启动GPU。**

补正不改变科学方案；关闭后仍只进入既定唯一六调用原生资格，不直接跳到正式训练。原生未过、真实资源/投影超限或某项作用未显出，均按现有合同停止，不追加候选、调用、预算或容差。正式结果是否支持双域胜S及新B直接胜H，要等将来按原合同完成的实际保存证据，本报告不给性能承诺，也不追改旧H双域目标false的结论。
