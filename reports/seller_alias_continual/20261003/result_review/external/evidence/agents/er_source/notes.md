# ER λ=0.1：冻结来源、直接科研依赖及独立 CPU 公式核对

## 1. 本分工的结论

在此次获准小结果包和完整源码的范围内，**没有发现足以改变 ER0.1 当前科研结论的实现阻断，也没有发现必须修改本轮生产源码才可交付的未决复现缺陷**。这个判断来自实际输入到评价的代码语义追踪、原生执行小证据，以及新写的独立 NumPy 公式和 ID 流重建；没有把项目自己的 PASS、哈希一致或测试数量当作结论。

主比较必须是 ER0.1−ER0.25，完整的 ER0.1−SEQ 结论另算。相对 quarter 满足开发替换标准，不会自动使相对 SEQ 满足原验收。ER0.1 的有效局部正结果、相对 SEQ 的当前域退化、已有开发选择以及最终确认尚缺这几件事可以同时成立。本分工没有重新计算全套 5000 bootstrap 和五组 23 项；这些交由本次独立统计分工，最终总评须合并其结果。

## 2. 实际阅读与身份边界

先完整阅读了包根 `REQUEST.zh.md`、`project/AGENTS.md` 和 `docs/RESEARCH_DISCIPLINE.zh.md`。实际 ER 来源是下列运行目录 `job/run/manifest.json` 的 `source_files` 所列 31 份，而不是仅搜索 `freeze.json` 或沿用旧外审的文件数量：

`reports/seller_alias_continual/20261002/er_low_execution/20261002_124018/`

这 31 份合同、policy、决定、脚本与测试均已全文阅读；包括 880 行的 `step28_chinese_base.py`、575 行首轮 runner、448 行公共输入 runner、底层群和 pair 定义、损失、排序、校准、评价、保存恢复等。活动调用链内部的项目科研依赖均在这批来源中。`read_coverage.json` 提供每个实际路径、行数、大小、SHA-256、阅读主题及与冻结清单相符结果。旧文件中未被本轮调用的 `Archive`/独立 `main` 也已经读到，并明确区分活动路径；不能把旧函数里存在其他数据入口误判为本轮访问了该入口。

还完整读了 ER0.1 结果说明、前次外审 `REVIEW.zh.md`、主执行方处置、当前原生 `execution.json`、`manifest.json`、训练进度原日志和 CPU stdout/stderr/资源记录。运行 JSON/NPY 的完整字节由独立脚本读取并解析，所有实际读取路径和大小/SHA 记录在 `independent_er_source_result.json/input_files`。对大 JSON 的字段解析不冒称为“所有字段逐字语义审查”，与源码全文阅读分栏记录。

两套冻结树中 SHA 相同的共同依赖，可以由本分工负责一次内容全文审查、LOGIT 分工逐文件证明字节对应；LOGIT 的不同源码、policy、合同、决定及新增来源由其独立全文读，不能用 ER 版本代替。第三方 PyTorch、transformers、SciPy 本体的库源码未在附件，本轮没有审阅其全部实现。

本分工使用当前工具环境及 GPT-6 Astra Pro 上下文；不据此认证另一个网页 GPT 6 Pro 会话。原包总大小、613 成员与给定总 SHA 的身份核对由主审入口负责，不能用本分工的 31 文件匹配替代 ZIP 级完整性。

## 3. 合同与实际数学目标

本轮合同第 7–11 行明确只增加 λ=0.1，复用旧 0/0.25/0.5/1；这是一项看过前轮结果后的开发扩展。第 9 行目标是

\[
L=L_{\rm base}(c)+0.1L_{\rm base}(h),\qquad
L_{\rm base}=L_{\rm BCE}+L_{\rm rank}+0.5L_{\rm hard}.
\]

`scripts/step28_chinese_base.py:165–191` 的 BCE 是 378 个无序边的等权平均。每群 28 个账户、4 个三账户簇和 8 个双账户簇产生 20 正边；每个查询有 1 或 2 个正例、27 个非自身候选。其 rank 项是每查询全部候选 `logsumexp` 减该查询正例分数均值，再对 28 查询平均。

`scripts/step28_alias_ranking.py:89–129` 从每查询**已知负例**中选分数最高的 5 个，排除自身和所有正例，按候选位置处理并列。只把选择索引 detach，选中负分数与所有正分数仍留在梯度图内；每查询的正例×5 个负例 softplus 均值再对查询平均。BCE 对每无序边计一次；排序项中一条边关联两个查询是定义本身。

`scripts/step28_er_weight.py:149–203` 对 current 的完整 `total` 先反向，再对 `coefficient * history total` 反向。因此历史 BCE、rank 和 0.5 hard 全部乘 0.1；不只乘历史 BCE，不除以 1.1，没有 logit MSE。独立源码没有导入项目公式，而以无邻接矩阵的无序边/逐查询遍历重新实现上述目标和解析梯度。

## 4. 输入、可见资格和模型活图

| 检查点 | 实际代码与结论 | 未越过的证据边界 |
|---|---|---|
| 公共输入 | `step28_continual_expression_run.py:58–104` 活动 `public_inputs` 只构造 train/development 的群和 title/description 输入；验证群内 28 账户、账户不跨群/分割重复。 | 正式标题、描述正文未提供也未读；“文本确已屏蔽”的内容真实性只能依赖冻结输入链，不能声称网页重新逐条审定。 |
| 标签资格 | 同文件 `107–128` 只允许 train/development 指定一次 attach，校验列名和群覆盖；`step28_bge_continual_run.py:76–87` 尝试前记账，失败也不自动再读。 | 不打开任何正式标签。保存计数和手写标签只用于允许的独立检查。 |
| pair 顺序 | `step28_continual_population_data.py:105–116` 先排序账户，再按组合列出 378 对，标签按具名左右账户对齐、完整覆盖，不依赖行位置碰巧一致。 | 用户明确禁止的数据保持不可见。 |
| 48/12 分区 | `step28_chinese_base.py:52–79` 按固定种子、实际域和 UID 的哈希顺序切 48 拟合与 12 校准；开发 60 群每域 20。 | 本轮独立从已保存 UID 重建了切分，检查拟合/校准/development 不重叠，没有用正式内容或标签。 |
| 阶段可见域 | `step28_bge_continual.py:65–105` 的 Supply 只给当前拟合和当前校准；`step28_er_weight.py:138–147` 从第二站续训。 | 老 dispatcher 持有全档案的能力与实际调用资格分开；本轮没有新取第一站群来重建缓存。 |
| 特征 | `step28_chinese_base.py:134–162` 只把 title/description 送 encoder；每 item 向量 L2，再按账户对标题、描述分别做均值和总体标准差，再拼接/L2。账户向量 4096 维；ID、域、数量标量不进入模型。 | ID 仍合法用于排序、分区、抽样和并列确定；这与把 ID 当特征不同。 |
| 活图 | `step28_chinese_base.py:110–131,194–220` 原生本地 BGE、CLS 1024、微批 4、最大长度 256、训练路径保留图，checkpoint 保留 RNG。`step28_continual_population.py:14–29` 头用 \(\lvert u-v\rvert\) 与 \(u\odot v\) 拼接，8192→128 ReLU→1。 | 网页没有加载 BGE 或正式权重，也没有模拟成原生 encoder 执行。 |
| 评分 | `step28_chinese_base.py:223–230` 评分走 eval/inference_mode；训练调用恢复 train 状态。 | 保存评分矩阵支持评价复算，不支持重新生成全部网络前向。 |

## 5. 梯度、单次更新与恢复

### 5.1 一个 current/history 对应一个 AdamW 更新

`step28_er_weight.py:149–203` 每步只有一次 zero_grad；current 和加权 history 分别 backward 后，全部参数一起只调用一次 `clip_grad_norm_(..., max_norm=1, error_if_nonfinite=True)`，再一次 `optimizer.step()`。这保留 \(g_c+0.1g_h\) 的合成梯度；不存在先独立裁剪两支再合成，或两次 Adam 更新的等价性偷换。

`step28_continual_population.py:32–40` 的 AdamW 对 encoder/head 分组，encoder weight decay 0.01、head 0，betas=(0.9,0.999)、eps=1e−8、foreach=False。`step28_bge_continual.py:50–53` 每站 encoder 的前 29 步 warmup，随后线性降至第 288 步 0；head 仍以 0.001 更新。因而“288 次优化器调用”与“287 次正 encoder LR”同时成立，最后一步 encoder LR=0 是合同调度，不是更新漏计。

新 1728 行记录中合成梯度范数均大于 1，范围约 5.31–31.22，说明正式记录的总裁剪一直触发。但只凭总范数，不能推出 current/history 梯度冲突、方向抵消程度或 λ 变小收益的机制；生产记录并未给每步两支完整梯度向量。

### 5.2 共享起点的代码和小证据

`step28_er_weight.py:85–135` 检查原始、前轮结果与环境、三首域 shared 记录、公共群 ID、校准首映射和来源。`step28_er_weight_run.py:178–239` 在新监督解析前验证原三个 full checkpoint 和原第一站缓存的身份，恢复 model/Adam/RNG，并精确回放原首站保存的 60×378 分数。起点是原共享第一站，不是 quarter 的最终端点；缓存 `with_logits=false`、seen=48、draw_count=0，首映射原样复用。

真实保存和加载实现不是只写一个“恢复成功”字符串：`step28_continual_population.py:157–210` 计算模型、optimizer 全张量和元数据摘要，`torch.save` 后真实加载，严格 model 状态及 Adam 摘要相等；`step28_bge_continual_run.py:58–73,230–241` 捕获/恢复 CPU 与全部 CUDA RNG。当前运行三个 restored_starts 的 checkpoint/model/cache/Adam288/首映射，与原 shared 和前轮参考记录独立逐项相等。

这是源码加原生记录支持的真实恢复流程。因为正式权重和缓存正文按要求未提供，本轮网页不能再次执行原生恢复，更不能把 metadata 的一致性当作自己已经重跑了该模型恢复。

## 6. 抽样、dropout 流与保存预算的独立重建

`step28_bge_continual.py:109–219` 与 `step28_continual_population_data.py:202–244` 实施 capacity=6 的 Algorithm R；每站结束按排序后的 48 个新拟合群各插入一次。历史抽样从存活的 6 群中均匀、有放回抽取，288 次，而不是预设每群恰好 48 次。序列化大小包括群内容、标签、随机状态、计数、校准附属状态及索引，全部受 1 MiB 限制；ER 没有 logits reference。

`step28_er_weight_run.py:43–96` 将当前日程、历史 UID、current/history dropout seed 流与原 ER 配对；`240–258` 第二站结束后才插入该站群，第三站不再生成新历史库。本轮独立脚本使用新写的 canonical seed、shuffle、Algorithm R 和有放回 draw 逻辑，从保存 UID 元数据重建了三个顺序的第一站库和第二站保留库，以及所有 1728 当前/历史步骤；与 ER0.1 和附件中的原 ER 六段日志逐项全同。每段当前群均为 48 个、每群 6 次，历史每段 6 个 UID，实际频数范围 34–63，并非声称每个都抽到 48 次。

原生阶段记录由当前 Adam 288→576→864，六段各 288 新更新，总 1728 次/3456 群梯度呈现。六份保存更新数组均为 288×14、全部有限；逐行重算 current/history 目标以及 LR 最大目标分解误差为 **3.427267074584961×10⁻⁷**，与保存 float32 损失分量再在 float64 组合的量级一致。总数不证明科学效果，它在这里用于检查具体目标和更新流程没有与合同走样。

| 顺序/阶段 | 已保存历史状态字节 | Adam 步 | 校准 a | 校准 b | 该段训练秒 |
|---|---:|---:|---:|---:|---:|
| ABC/2 | 210077 | 576 | 0.7466920394605547 | 0.9109138059174582 | 3001.206900279969 |
| ABC/3 | 212110 | 864 | 0.6159492336364464 | 0.3210021847909775 | 3034.451065602014 |
| BCA/2 | 217024 | 576 | 0.6971622408383085 | 0.43552903807805526 | 3040.0321237458847 |
| BCA/3 | 209968 | 864 | 0.7265845987862745 | −0.03487049281530061 | 3028.2180122029968 |
| CAB/2 | 198571 | 576 | 0.6962149177865552 | 1.0095779933057942 | 2948.299650341971 |
| CAB/3 | 206145 | 864 | 0.5991312386262947 | 0.24444952326081537 | 2993.3484757731203 |

## 7. 校准、盲门与评价实现

`step28_er_weight_run.py:99–175` 每站校准只使用当时当前域的 12 群，4536 对、240 正；首映射复用。其 checkpoint 路径先保存/真实恢复 model/Adam/RNG，回放校准和 development 分数，再保存最终推理状态及全部数组。校准结束恢复原 RNG，避免评分、保存和拟合干扰后续训练随机流。

`step28_alias_calibration.py:41–49,97–147` 的目标是未加权、未裁剪的 Bernoulli NLL；解析梯度为 `mean((sigmoid(a*z+b)-y)*z)` 与 `mean(sigmoid(a*z+b)-y)`。正仿射参数 a∈[0.001,100]、b∈[−100,100]，初始 (1,0)，L-BFGS-B 最大 200 次迭代/2000 次调用，projected gradient≤1e−6、结束 NLL 不高于初值；没有在 valid 上选择映射。`70–94` 检查正斜率、排序和并列完全不变。这里独立核对了六份映射 metadata 的参数、群资格、计数和收敛字段；没有利用正式校准标签重拟合，也不能独立确认每个 NLL 数字与不可见标签相符。总审另从允许的保存分数检查变换数值。

`step28_er_weight_run.py:271–370,402–474` 要求三条路径全训练完、6 个原生终点、24 份分数数组及数据/更新盲门全部完成，之后才统一解析一次 valid；`373–399` 保存新增 18 指标/计数集及 collection，完成后统计可从保存矩阵重启而无需再读标签。`step28_er_weight_evaluate.py:71–83,108–180` 把 18 新集与原 45 集、前轮 36 集合成 99 个集，核对 60 群 UID/实际域/22列/角色一致，复用 81 集不算新增样本。

评价公式全文核对见 `step28_continual_population_evaluate.py:28–100,225–228`：AP 用台阶式精度召回积分，PR-AUC 是梯形面积，两者不能互换；ROC 处理成组并列；Recall@FPR≤1% 是标签诊断阈值；Brier 基于 sigmoid，logloss 的数值概率裁剪至 1e−15；工作点固定 logit=0。排序每查询 27 候选、1/2 正例，固定候选位置并列，报告 MRR、MAP、Recall/NDCG@1/3/5/10。`step28_chinese_base.py:680–696` 的 pooled 分类先汇总计数再算比例，不能用组均 precision/recall 冒充。

`step28_bge_continual_evaluate.py:91–137` 的汇总按顺序位置定义，但 bootstrap 按**实际域**配对索引。原始排名/曲线与 stage-cal 概率为主，raw/first-cal 留完整诊断。三顺序均值以相同实际域 20 群重采样，PCG64(20260930)，5000 draw，线性分位数；同一 draw 跨方法、阶段、顺序和角色复用，不把三个顺序当三种训练种子。七种汇总为：

\[
\begin{aligned}
O&=(R_{31}+R_{32})/2, &N&=(R_{22}+R_{33})/2, &Z&=R_{33},\\
F_{first}&=R_{11}-R_{31},
&F&=(R_{11}-R_{31}+R_{22}-R_{32})/2,\\
G&=(R_{22}-R_{12}+R_{33}-R_{23})/2,
&final\_all&=(R_{31}+R_{32}+R_{33})/3.
\end{aligned}
\]

Brier/logloss 的 F/G 按较低更好反向。原 23 项判据见同文件 `141–163`：O MAP 要求均值和 CI 下界>0；其他要求的观察值方向保护必须原样保留，不因其 CI 跨零而判观察值无退化。`step28_er_weight_evaluate.py:16–28,31–68` 从 policy 取固定 quarter 主对手，只有相对 quarter 的原 23 项全通过才替换当前 ER；SEQ 单独完整判定，不替换主对手，也不追加 λ 搜索。

## 8. 原生证据和此次网页复算分栏

### 8.1 包内原生证据

原 `execution.json` 记录 Python 3.10.19、Torch 2.9.1+cu130、CUDA 13.0、RTX 5090、CPU affinity [47]，31 份来源与当前全文读的一致。原生 CPU stdout/stderr 及资源记录实际完成两个手写 branch：quarter 和 tenth 各 **2 次真实模型更新**（先一次 warm，手工将 Adam counter 1→288，然后一次目标更新）。这是必要的分量缩放与继承计数示例，不能描述成“CPU 原生跑完了 288 次 warmup”。原 stdout 的两个分支耗时约 173.870/169.223 秒，外部 CPU 墙时 6:04.25、退出码 0；22 个合同测试的通过数只是它们实际运行的记录。

原生记录提供 29 个参数 probe：24 层 query 权重、embedding 和 4 个 head 参数，观察到有梯度及实际参数变化；两分支 current 的 norm/SHA 完全一致，history 梯度范数按 0.25/0.1≈2.5 缩放。完整梯度向量不在本包，因此原审计所记 encoder/head 最大逐元素误差（约 2.55e−11/1.05e−9）只能作为**原生程序记录**引用，不能声称本轮网页重新逐元素验证。

正式 `job/train.log` 从 ABC stage2 直至 CAB stage3，每 24 步一事件，六段共 72 个进度事件，最后明确完成；只有 SentenceTransformer 维度 API 重命名 FutureWarning。日志的 `peak_observed_bytes` 是新产物预算观测字段，不能误标成 GPU 显存峰值。本分工没有重新启动这些命令，也没有连接该服务器。

### 8.2 当前独立 NumPy 核对

环境是 Python **3.12.14**、NumPy **2.3.5**、Linux x86_64/glibc 2.39，CPU affinity [0]；通过记录器固定单线程，无 Torch。全部程序只导入标准库和 NumPy，formal text/label/cache body/weight/server/training/test-owners/project-module-import 均为 0。

从包内**手写 CPU 例**保存的 378 logits，与独立构造的 4×3+8×2 账户簇标签重算得到：

| 项 | 独立 current | 独立 history |
|---|---:|---:|
| BCE | 0.6633192276984121 | 0.6633684505929200 |
| rank | 3.291688478997472 | 3.292110873997935 |
| hard | 0.6929290613008465 | 0.6933351772194255 |
| BCE+rank+0.5 hard | 4.3014722373463075 | 4.302146913200568 |

ER0.1 总目标独立值 **4.731686928666364**，原生记录 **4.7316868782043455**，绝对差 **5.0462018563735e−8**。quarter 独立值 **5.37700896564645**，原生记录 **5.37700891494751**，差 **5.069894015719001e−8**。这直接核对完整目标而非仅接受 PASS。

此外新写一组非正式 378 连续分数，以解析 BCE/rank/hard 导数核对 56 个有限差分位置，最大误差 **4.4362813769738274e−11**；再用一个共享 3 参数的手写 Jacobian 验证 current+0.1 history 合成梯度，3 个参数有限差分最大误差 **5.159286886602388e−11**。例中的正确总值 5.100652159620462，与只缩历史 BCE 的 8.81006349799016、错误除以1.1的4.636956508745874均明显不同。还按一次总裁剪和已有一阶/二阶矩手算一次 AdamW 更新；这是公式参考，不是实际 PyTorch 或 BGE 的替代执行。

## 9. 精确命令、失败与修订

工作目录均为 `/workspace/scratch/ca13b63db3f0/review_work/evidence/agents/er_source`。原始运行、源码快照、精确 argv、cwd、公开环境覆盖值、起止 UTC、退出码和 stdout/stderr 的字节/SHA 在 `evidence/runs/er_source_*`。

成功参考执行的精确入口：

```bash
python -B /workspace/scratch/ca13b63db3f0/review_work/evidence/src/run_record.py --label er_source_independent_v2 --cwd /workspace/scratch/ca13b63db3f0/review_work/evidence/agents/er_source -- python -B /workspace/scratch/ca13b63db3f0/review_work/evidence/agents/er_source/independent_er_source_v2.py
```

v2 于 2026-10-03T03:55:17.340913+00:00 开始、03:55:17.596869 结束，0.255795811 秒，exit=0；源码 SHA `c5e22495240f7b1a73c0af25771b07280c65f8d43f89233e8746fbc50932a804`。完整结果在 `independent_er_source_result.json`；stdout 原文 11410 字节、SHA `de326c1e7302ef7b7f90ac60fc8c5d26101b2ff9cf6834d669118251041aa91f`，stderr 空且保留。

**ER-REF-01（已修复的独立参考错误）**：v1 假设小包还包括原 SEQ 及 half/quarter 的逐步更新 JSON，试读 `.../20260930_143700/job/run/updates/ABC_seq_stage2.json` 时 FileNotFoundError。原件未覆盖，2026-10-03T03:54:16.293842 开始、03:54:16.370455 结束，exit=1；stderr 1747 字节/SHA `44719e7c6e56dfc982cac6f90ee80eaa4a9b9b14158ad2ffec21ee791fafb4b0`。它是审查脚本对附件范围的错误假设，不是训练或结果错误。

随后由 `revise_er_source.py` 留下 `revision_v1_to_v2.diff`、修订 JSON 和新 v2 原件。v2 把逐步直接对照限定为实际提供的原 ER 六段；原 SEQ/half/quarter 仅按提供的 shared/manifest/metrics 记录核对，不虚构其原逐步文件已读。失败 v1 的 SHA 为 `532a6fc1ff0fffb4fae5e2a0d4b74261b1532788c70948dd312a0399cf69f185`。相应 `er_source_independent_v1`、`er_source_revision_v1_to_v2`、`er_source_independent_v2` 运行目录全部原样交付。生成阅读清单的 `finalize_coverage.py` 也是单独记录的整理操作，不计为又一个科学验证测试。

## 10. 四类处置与只需保留的边界

| 类别 | 当前判断及必要处置 |
|---|---|
| SCIENTIFIC_BLOCKER | 本分工未发现已确认的阻断。若将相对 quarter 的开发胜出写成相对 SEQ 原验收通过、三种种子稳健、独立最终确认或创新充分证明，会构成结论越界；现有合同已明确这些边界，交付继续保留即可。 |
| REPRODUCIBILITY_DEFECT | 未发现必须修生产代码的未决问题。ER-REF-01 已在独立参考代码中修复并保留失败原件；不转嫁为项目缺陷。文件身份能证明版本对应，不能独立证明不可见原始内容。 |
| 未验证范围 | 未重新读取文本/标签、核验屏蔽正文、重拟合真实校准、加载权重/缓存正文、重跑原生恢复或正式前反向。没有完整原生梯度向量，未重新逐元素比较。小包没有原 SEQ 和旧 half/quarter 逐步更新 JSON；其原运行链只按提供的历史记录与共同起点核对。历史外审的旧 evidence.zip 不在本包，不能声称已读其源码/原流。 |
| OUT_OF_SCOPE_OVERDESIGN | 本轮无需追加 λ、种子、训练、留出、缓存、伪造方法名或新审查制度来“修复”有效负结果；不得因此索要正式数据或连接服务器。若未来申请泛化或最终确认，那是新的科研设计，不是当前交付必修项。 |

已开发 valid、单 s0、三个顺序及条件 bootstrap 的范围限制要写在报告正文。条件区间不含重新训练、校准拟合和自适应选择的不确定性；观察值下降不会因为区间跨零就消失。对于已有相对 SEQ 的当前/最终保护未通过，应保留为有效负结果与局部权衡，不以仅保存正面 O MAP 或换对手来改写结论。
