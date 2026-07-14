# Step15-v8：表征桥接审计与上下文证据融合实施方案

更新日期：2026-07-14
代码分支：`method/step15-v8-contextual-evidence-fusion`
当前状态：代码、policy、Linux runner 与纯合成 contract tests 已实现；真实数据计算尚未在 Linux 执行。

## 1. 为什么启动 v8

当前同一批 `200` 条中文内部开发测试上存在明确差距：

| 方法 | AP | 科研解释 |
|---|---:|---|
| Step15-v7 clean LR/L2 | 0.463904 | 去标识、严格边界下的当前 clean 基线 |
| Step15-v6 M3 | 0.594897 | 旧表示较强，但含有不能直接恢复的潜在 identifier shortcut 与旧选择过程 |
| Step15-v6 normalized retrieval | 0.597533 | 表明合法检索信号可能仍有价值，但旧语义输入不满足 v7 的去标识要求 |

这个差距不能通过恢复旧特征或继续看测试集调权重来解决。v8 的任务是把差距拆成可检验的原因：

1. 64 维 E5 随机投影是否在当前样本规模下引入了噪声；
2. 单一 E5 是否不足，多编码器去标识共识能否恢复合法语义；
3. v7 删除 retrieval-only 特征时是否同时删掉了合法的内容/结构信号；
4. LR/L2 是否是主要瓶颈，线性 pairwise ranking 是否更适合 AP；
5. v7 Stage B 是否因为 token-wide 风险聚合和固定 `score * 0.1` 误伤 direct positive。

v8 不以当前内部测试高分为目标。方法选择只允许读取 train 的 seller-component-disjoint OOF 预测；代表性 valid 只在方法冻结后检查一次；当前中文 test 永远只是内部诊断；论文结论必须等待 Step20 prospective holdout。

## 2. 不变的数据边界

v8 继承并冻结 v7 的以下边界：

- 英文与中文监督标签文件不改变；
- v7 representative validation assignment 不改变；
- train、valid、internal development test 的 seller component 仍然物理隔离；
- factorized sample weights 的公式和参数完全继承 v7；
- LR/L2 的 `L2=10`、无自动 class weight、最大迭代和收敛容差保持一致；
- 使用相同十个 seed：`20260320` 到 `20260329`；
- threshold 仍由 representative valid 上的既定 balanced-accuracy 规则冻结；
- 当前 `200` 条中文 test 不参与 representation、模型家族、evidence expert 或 threshold 选择。

v8 不覆盖任何 v6/v7 结果。每次运行必须指定独立 `run_id`，输出在：

```text
reports/step15_v8/<run_id>/
```

同一路径已存在时所有阶段均 fail closed。

## 3. Bridge Audit：B0-B3

### 3.1 共同的去标识语义文本

语义输入只由 category、title、description 相关内容构成。以下 profile 字段被明确排除：

- seller raw name；
- normalized alias；
- market/seller ID；
- contact concatenation；
- structured identity snapshot；
- 旧 `profile_text`。

随后复用 v7 的 fixed-point identifier redaction：反复应用 Step3 高精度 identifier 正则和 seller-specific literals，并在每轮后归一化空白，直到文本和上一轮完全一致。最多允许八轮；仍有 PGP、email、Telegram、URL、钱包、电话、QQ、微信、Jabber 等残留时直接失败。

E5 使用已经冻结的 v7 identifier-redacted cache；v8 重新编码 identifier-redacted BGE-M3 和 LaBSE，并对同一 clean text pair 计算 identifier-redacted GTE reranker score。v8 会逐 pair 复算 E5 cosine，并要求与 v7 冻结字段最大差异不超过 `2e-10`，同时要求本次 redacted text corpus hash 与冻结 E5 cache 完全一致。

GTE reranker 不是天然对称函数。seller-pair verification 不应受 left/right 排列影响，因此 v8 对 `(left,right)` 与 `(right,left)` 分别打分后取均值；两个方向都只读取 identifier-redacted text。

### 3.2 四组表示

#### B0：v7 表示在 v8 协议下的控制实验

```text
v7 strict-clean 20d + E5 symmetric pair latent 64d
```

64 维 latent 来自：

```text
[abs(e_left - e_right), e_left * e_right]
```

再经过固定 seed 的 Gaussian Johnson-Lindenstrauss projection。B0 保留 v7 的特征表示，但按照 v8 的 seller-component OOF、折内 corpus reference、折内 imputation/standardization 重新训练。它用于隔离“表示变化”，不是旧 v7 数值结果的逐字节复刻，也不是候选方法。

#### B1：去掉随机 latent

```text
v7 strict-clean 20d
```

20 维中已经有 identifier-redacted E5 cosine。B1 只删除 64 维随机投影，直接检验 v7 下降是否来自高维投影方差或小样本噪声。

#### B2：多编码器去标识共识

```text
v7 nonsemantic 19d
+ redacted E5 cosine
+ redacted BGE-M3 cosine
+ redacted LaBSE cosine
```

B2 不使用任何旧 `profile_text` embedding。三个 encoder 的作用不是投票生成标签，而是让低维 LR 判断跨编码器一致性与分歧是否能恢复合法语义信号。

#### B3：合法 retrieval bridge

B3 在 B2 基础上只加入：

- `candidate_rule_count_non_identifier_v8`；
- fold-train/domain-only normalized lexical similarity；
- fold-train/domain-only normalized structural support；
- identifier-redacted reranker score。

`candidate_rule_count_non_identifier_v8` 不信任历史 aggregate 字段，而是从以下显式 allowlist 重新计算：

```text
profile_lexical_neighbor
shared_title_clone
shared_description_clone
structural_support
```

因此 `shared_contact_exact`、supplemental contact、PGP 和 positive-component closure 都不能进入 B3 count。

### 3.3 永久禁止的输入

包括但不限于：

- `candidate_rule_count_raw`；
- 历史 `candidate_rule_count_non_identifier`；
- shared contact/PGP 字段；
- English uppercase gap；
- 未清洗 `profile_text` embedding；
- 未清洗 reranker；
- 从 valid/test 拟合的 IDF、OOV、percentile 或 normalization statistics。

### 3.4 OOF 预处理边界

对每个 seed，train 按 `domain::seller_component` 分为五折。每一折都重新执行：

1. 只用 fold-train sellers 拟合 title/description DF、IDF、OOV、market-relative percentile reference；
2. 只用 fold-train rows 拟合 B3 lexical/structural domain mean 与 standard deviation；
3. 只用 fold-train rows 拟合 median imputation；
4. 只用 fold-train rows 拟合 LR standardization；
5. 训练模型并预测完全未见 seller component 的 held fold。

这比仅把模型分折更严格，防止 held component 通过无监督 corpus statistics 影响表示。最终对 representative valid 与 internal test 的模型只使用完整 train sellers/rows 拟合 reference。

### 3.5 表示选择

每个 B 组获得十个 seed 的完整 train OOF prediction。主选择指标是：

```text
mean seed macro-domain OOF AP
```

英文 OOF AP 和中文 OOF AP 先分别计算，再取平均，避免英文较大数据量淹没中文目标域。差异在 `0.002` 以内按预注册复杂度顺序选择更简单表示。representative valid 与 internal test 都不能改变 B0-B3 选择结果。

## 4. 线性 pairwise ranking 对照

选出 B 表示后，仅比较两个模型：

1. LR/L2；
2. linear pairwise RankNet。

RankNet 不训练 Transformer。每个 domain 内构造：

```text
x_positive - x_negative -> 1
x_negative - x_positive -> 0
```

每个 positive 最多确定性采样 24 个同域 negative；pair weight 取两个真实 row factorized weight 的较小值。正反 pair 同时加入以消除方向偏差。模型仍是强 L2 的线性 logistic pairwise objective。

模型家族也只按 train OOF macro-domain AP 选择；`0.002` 以内优先 LR/L2。这样只回答“直接优化相对排序是否有帮助”，不扩散为分类器搜索。

## 5. Step16-v8：occurrence context 专项审查队列

规则从中文 Step3 item identity occurrences 构建共享 token seller pairs，但排除已经监督的 pair。输出三个队列：

### 5.1 Risky-only public noise

共享 identifier，但没有双侧 clean seller-facing direct occurrence；token 只在 product-data、victim-like、support-only、公共 URL 或高频 context 中出现。

### 5.2 Mixed-context identifier

两侧均有 seller-facing direct occurrence，但同 token 同时出现在 risky/support context。它正是 v7 token-wide veto 最容易误伤的状态。

### 5.3 Verified direct both sides

同 token 在两侧都有 clean seller-facing direct occurrence，且不属于 train-seller 高频公共 token。

队列文件包含原始 normalized identifier 和两侧 context preview，供 blind review 使用；不包含任何模型分数。脚本不会写 `positive` 或 `negative`，review label/evidence type 字段保持空白。规则命中不能写回 Step5，只有完成盲审、证据判定和 component-safe split adjudication 后才允许更新监督集。

候选 split 资格按共享标识符形成的 seller connected component 整体分配，而不是逐 pair 分配。同一候选 component 只能进入一个 split；若 component 已触及多个既有 split，则整组标为 `blocked_cross_split_seller_overlap`。超高频 token 的卖家截断使用固定 SHA-256 抽样，不取字典序前缀，避免来源命名顺序造成系统偏差。

当前预注册目标是：

| 类型 | Train | Valid |
|---|---:|---:|
| risky-only public noise | 40 | 20 |
| mixed-context identifier | 30 | 15 |
| verified direct both sides | 60 | 20 |
| component-anchor positive | 25 | 15 |

如果候选或复核结果达不到目标，应报告证据稀缺，不能用规则自动补标签。

## 6. Occurrence-level evidence expert

### 6.1 为什么不用旧 hard veto

v7 的旧逻辑只要共享 token 的任一 occurrence 出现 risky/support，就把整条 pair 乘以 `0.1`。因此一个真实 seller-facing Telegram 同时被商品内容引用时，也会被当作 public noise。v8 删除固定 multiplier。

### 6.2 输入

约 15 个低维 occurrence 特征：

- pure direct token 数；
- risky-only token 数；
- support-only token 数；
- mixed-context token 数；
- high-frequency token 数；
- shared token 总数；
- distinct item 与 market 数；
- public URL/domain 标志；
- Telegram、email、financial/phone 类型标志；
- 三个 `ZH x evidence-state` interaction。

clean probability 在 train 上必须来自 seller-component grouped OOF prediction。不能把同一 row 的 in-sample clean probability 交给 expert。

### 6.3 训练

模型是 compact offset logistic LR/L2：

```text
logit(P_final) = logit(P_clean_OOF) + delta(context)
```

英文和中文共享主系数；仅增加三个中文 evidence-state interaction，其 L2 penalty 是主系数的四倍。这样英文的 direct/public evidence 提供一般规律，中文少样本只允许小幅目标域修正。

训练不使用八分类辅助 head、不使用 MLP、不把 evidence type 或 review label 当作输入特征。review label 只作为监督目标。

### 6.4 离散状态与方向约束

| 状态 | 允许动作 |
|---|---|
| `verified_direct_both_sides` | 只允许非负 logit uplift |
| `direct_with_mixed_context` | 强制不改变 clean score |
| `risky_only_shared` | 只允许非正 downgrade |
| `support_only_shared` | 只允许非正 downgrade |
| `high_frequency_public` | 只允许非正 downgrade |
| `ambiguous` | 强制不改变 clean score |
| `no_shared_identifier` | 强制不改变 clean score |

因此即使 expert 系数学错方向，也不能把 risky evidence 提升或把 direct evidence 压低；mixed context 不再被硬 veto。

## 7. 晋级门槛

所有门槛只在 representative valid 上计算：

1. selected clean AP 相对 B0 至少 `+0.03`；
2. public-noise FPR 至少降低 `0.20`；
3. direct/component recall 下降不超过 `0.05`；
4. template-clone FPR 不增加；
5. contextual fusion AP 不低于 clean；
6. grouped component bootstrap 的 clean-B0 与 fusion-clean AP delta 95% CI 下界都不得低于 `-0.03`；
7. model selection 从未读取 internal test。

另有数据充分性门槛：representative valid 至少包含 `20` public-noise、`20` direct positive、`15` component positive。当前已知 valid 大约只有 `6/18/16`，因此第一轮即使模型指标方向较好，也很可能因 public-noise slice 不足而保持 `promotion.eligible=false`。这是设计要求，不是运行错误。

## 8. Step20 与图谱阶段

Step12-v8 通过只表示方法具备进入 prospective 评估的资格，不代表论文验证完成。后续必须：

1. 冻结 selected B、模型家族、expert artifact、threshold 和 hashes；
2. 使用 v8 freeze 后新收集、双盲审、seller-disjoint 的 Step20 holdout；
3. 所有配置冻结后一次性评估；
4. Step20 evaluation lock 成功后，才允许 Step11/17；
5. Step11/17 只能显式 allow-list，禁止 auto selector 和 reports glob。

`scripts/step15_v8_downstream_gate.py` 同时检查 Step12 promotion 与 Step20 one-time lock。任何一个缺失都会以退出码 `3` 阻止下游。

Step20 lock 按 v8 `run_id` 隔离，并必须写入对应 `step15_v8_model_freeze_manifest.json` 的 SHA-256、`evaluation_count=1`、模型与阈值在 holdout 解封前已经冻结，以及 holdout 未用于模型选择。只有同名文件而哈希不匹配时仍然 fail closed；历史运行的 lock 不能放行新模型。

## 9. Linux 同步与执行

从 Windows 同步到 Linux 的 v8 必需源码清单：

```text
schema/step15_v8_contextual_evidence_policy.json
scripts/run_step15_v8_linux_20260714.sh
scripts/step15_v8_common.py
scripts/step15_v8_preflight.py
scripts/step15_build_v8_clean_semantics.py
scripts/step15_run_v8_bridge_audit.py
scripts/step16_build_v8_context_review_queues.py
scripts/step15_train_v8_contextual_evidence.py
scripts/step12_v8_statistical_robustness_audit.py
scripts/step15_v8_downstream_gate.py
scripts/step15_v8_build_sync_manifest.py
tests/test_step15_v8_contextual_evidence_contracts.py
```

文档同步项：

```text
docs/STEP15_V8_CONTEXTUAL_EVIDENCE_FUSION_PLAN_20260714.zh.md
docs/PROJECT_PROGRESS.md
```

Linux 工作区还必须保留 policy 引用的 v6/v7 冻结 schema、既有 Step3/4/5/15-v7 artifacts、Step7 semantic model policy，以及本地 BGE-M3、LaBSE、GTE reranker 模型目录。runner 的第一阶段会只读校验这些文件、冻结 manifest/assignment/input hashes、Step4/feature pair universe 和本地模型目录；任一不匹配都会在 GPU 编码前停止。

完整运行：

```bash
cd /home/yongpeng/cross-lingual

export CUDA_VISIBLE_DEVICES=0
export V8_RUN_ID=bridge_v1_20260714
export STEP12_RESAMPLES=5000

bash scripts/run_step15_v8_linux_20260714.sh
```

GPU 只用于 identifier-redacted BGE/LaBSE encoding 和 GTE reranker。Bridge LR/RankNet、evidence expert 和 Step12 主要使用 CPU。

如果同一 `run_id` 已存在，不得删除后覆盖。应使用新路径：

```bash
export V8_RUN_ID=bridge_v1_20260714_rerun1
bash scripts/run_step15_v8_linux_20260714.sh
```

## 10. Windows 同步回传

运行结束后同步整个目录：

```text
reports/step15_v8/<run_id>/
```

重点文件：

```text
step15_v8_sync_manifest.json
clean_semantics/clean_semantics_manifest.json
bridge_audit/step15_v8_bridge_audit_summary.json
bridge_audit/step15_v8_grouped_oof_fold_manifest.json
context_review/step16_v8_context_review_summary.json
contextual_evidence/step15_v8_contextual_evidence_summary.json
step12/step12_v8_statistical_robustness.json
step12/step12_v8_model_metrics.csv
step12/step15_v8_model_freeze_manifest.json
```

必须保留 predictions、artifacts、OOF files、embedding metadata/matrices 和 review queues，因为 sync manifest 对每个文件记录 SHA-256；只同步 summary 不足以复核结果。

## 11. 结果解释纪律

- B1 胜出：64 维随机 projection 在当前数据规模下主要增加方差；
- B2 胜出：多个 identifier-redacted encoder 的低维共识有价值；
- B3 胜出：v7 删除 retrieval features 过度，合法非 identifier retrieval 可以恢复部分 v6 信号；
- RankNet 胜出：当前主要瓶颈更接近排序目标与 log-loss 不一致；
- contextual expert 只降低 public FPR 且不伤 direct recall：支持 occurrence-level evidence fusion；
- B0-B3 全失败：v6 优势更可能来自不能恢复的 shortcut 或当前监督不足；
- contextual expert 失败：现有 occurrence context/labels 仍不足，不得继续靠 multiplier 调参；
- Step20 失败或无法构建：只能写内部验证、数据集/概念漂移/负结果论文，不能宣称可泛化性能提升。

v8 的科研价值不取决于一定超过 v6，而在于用严格 OOF、去标识语义、合法 retrieval bridge、occurrence context 和 prospective gate，把“有效身份信号”和“泄漏/公共噪声”拆开验证。
