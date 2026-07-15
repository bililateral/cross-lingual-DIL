# Step15-v8：表征桥接审计与上下文证据融合实施方案

更新日期：2026-07-15
代码分支：`method/step15-v8-validation-slice-expansion`
当前状态：正式 Step16-v8 双审、隔离 refreeze 与 readiness 检查已完成，valid `24/23/15`、train `20/30/10` 均达到预注册门槛；32 项 contract tests 通过。identifier-redacted embedding、v7 数值特征、B0–B3、evidence expert 和 Step12-v8 尚未在 Linux 正式运行，因此不能宣称方法已晋级。

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

- canonical 英文与中文监督标签文件不原地改变；Step16-v8 只生成独立、hash-bound 的 readiness overlay；
- 旧 v7 assignment 对已有可用监督的 split 归属保持不变；新增 reviewed rows 只写入新 assignment，component ID 在隔离 refreeze 中重算；
- 固定 200 条 internal-development-test pair UID 集合和哈希完全不变；
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

不可变内部队列保存完整规则诊断，但 reviewer 不读取这些内部队列。只有已经存在 Step4/v7 pair feature、redacted E5 cache 完整且不触及 test/cross-split component 的候选会进入 blind packet，避免浪费审查。系统为这批候选生成两个顺序独立打乱的 `reviewer_a/reviewer_b_blind_packet.template.csv`，只展示原始 normalized identifier、seller UID 和两侧 context preview；`queue_kind`、occurrence state、split 资格、feature-ready 状态及模型分数全部隐藏。规则命中不能写回 Step5，只有两个不同 reviewer 给出高置信一致的 identity/evidence 结论，或由不同于两人的第三 reviewer 完成高置信裁决后，才具备进入监督 overlay 的资格。

候选 split 资格按共享标识符形成的 seller connected component 整体分配，而不是逐 pair 分配。同一候选 component 只能进入一个 split；若 component 已触及多个既有 split，则整组标为 `blocked_cross_split_seller_overlap`。超高频 token 的卖家截断使用固定 SHA-256 抽样，不取字典序前缀，避免来源命名顺序造成系统偏差。

当前预注册目标是：

| 类型 | Train | Valid |
|---|---:|---:|
| risky-only public noise | 40 | 20 |
| mixed-context identifier | 30 | 15 |
| verified direct both sides | 60 | 20 |
| component-anchor positive | 25 | 15 |

如果候选或复核结果达不到目标，应报告证据稀缺，不能用规则自动补标签。审查通过的 pair 还必须已经存在于 Step4 与 v7 pair-feature universe；否则只能列入“需要上游 Step4/feature expansion”，不能直接塞入训练。

### 5.4 独立 refreeze，而不是原地修改 Step5

`scripts/step16_apply_v8_context_reviews.py` 读取不可变队列、单独的 completed review 文件和既有冻结边界。它执行：

1. 校验两个 reviewer 身份不同；
2. 一致结论必须两人都是 `high` confidence；
3. 不一致结论必须由第三人裁决，且裁决者不能是前两人；
4. `diagnostic_test_only` 与跨 split component 候选一律不能进入新监督；
5. 对所有既有与新增监督 pair 重新闭合 seller connected components；
6. 保证一个 component 只属于 train、valid 或 internal development test 中的一个；
7. 校验 200 条 internal development test 的 pair UID 集合逐条不变；
8. 只在独立目录输出 Step5 labels overlay、evidence labels overlay、assignment、manifest 和生成后的 v8 policy。

原始 Step5、旧 evidence labels 与旧 representative-validation 文件都不修改。输出路径相同但内容不同会 fail closed，不能静默覆盖。

Reviewer packet 只允许填写以下值：

| 字段 | 允许值 |
|---|---|
| `identity_label` | `positive` / `negative` / `uncertain` |
| `evidence_type` | `same_controller_direct_identifier` / `public_contact_or_url_noise` / `uncertain_insufficient_evidence` |
| `confidence` | `high` / `medium` / `low` |

Reviewer 必须按看到的原始证据判断，不能迎合候选生成规则。若最终 review 与 occurrence state 冲突，例如 risky-only 候选被判为 positive，或 verified-direct 候选被判为 negative，系统保留该审查诊断但不直接写入监督；必须先修正 parser/context 或补充外部锚点后再进入下一版 freeze。这避免 evidence expert 在训练时接收与其推理状态相矛盾的标签。

### 5.5 Windows 双代理内部审查试运行（历史 pilot）

候选审查本身不依赖 Linux 模型环境，可以在 Windows 对已经生成的 blind packet 执行。正式试运行前修正了一个盲审缺口：早期 `context_preview` 仍带有 `direct/seller_facing/risky/support` parser flags。当前 blind packet 只保留 identifier type/value 与原始 context；带状态标志的预览只保存在 reviewer 不读取的 immutable diagnostic queue，并由契约测试保证这些标志不会再次进入 blind packet。

本次使用两个上下文隔离的代理分别审查 reviewer-A 与 reviewer-B packet。两者不能读取内部队列、模型结果、split 信息或对方输出；发生分歧时只把 candidate UID 和独立 adjudicator packet 交给第三个代理，不透露前两者的答案。该流程必须在论文中披露为 `agent-assisted independent review`，不能表述为人工双人标注，也不能据此报告 human inter-annotator agreement。

实际队列共有 `143` risky-only、`2` mixed-context 和 `1` verified-direct 候选，但经过 Step4/v7 feature readiness 与 component-safe split 过滤后，blind packet 只有 `7` 条，且全部是 train-side risky-only。双代理对 `6/7` 条 identity/evidence 结论完全一致；第三代理将唯一分歧裁决为 uncertain。最终只有五条高置信 public-noise negative 可以物化为 train overlay；一条 uncertain 不进入二分类监督；一条被两代理一致判为 direct-contact positive，但由于与 risky-only parser state 冲突而被 fail-closed 保留为 parser 修复案例。

因此该历史试运行证明审查机制可执行，但当时没有解决 validation 数据不足；pilot readiness 为 valid `4/3/0`、train `5/0/0`，低于 valid `20/20/15` 和 train `20/30/10`。这一诊断随后触发了 5.6 节的 upstream 扩展，不能再当作当前状态。

### 5.6 正式 upstream 扩展、双审与隔离 refreeze

pilot 的结论促使项目向上游扩展，而不是降低门槛：

1. 修复 public URL 候选在审查前被全局并入巨型 component 的错误，允许 feature-unready 候选先接受证据审查，再在正式物化时扩展 Step4/canonical Step7 universe；
2. 对 120 条 score-blind public context 候选执行两名隔离代理独立审查，116 条一致 negative、3 条一致 positive/parser-conflict、1 条经第三审查者裁决为 negative；
3. 从 `market_item.xlsx` 与 `products_data.csv` 中构建 exact platform vendor ID 加 exact inventory overlap 的 cross-snapshot same-account controls；两名独立代理对 361 条 direct persistence 与 36 条 component closure 候选全部给出一致 high-confidence positive；
4. public-noise、cross-snapshot direct 和 component controls 全部被强制标记为 evidence-expert-only，不进入 primary alias benchmark、clean AP、threshold 或内部测试指标；public-noise 是高置信银级上下文噪声控制，不是不同操作者 gold truth；
5. 以固定 SHA256 排名和全局 component-safe 约束选择 public `20/20`、direct `30/20`、component `10/15` 的 train/valid 配额；
6. 生成独立冻结目录 `reports/step16_v8_validation_refreeze/readiness_expansion_v2_20260715/`，不覆盖 canonical Step3–Step7 文件。

最终 occurrence-state-backed readiness 为 valid `24/23/15`、train `20/30/10`。固定测试仍为 200 条，seller 跨 split 重叠为 0，未降低门槛、未读取模型分数。详细证据边界与哈希结果见 `docs/STEP16_V8_READINESS_EXPANSION_PROTOCOL_20260715.zh.md`。

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

primary AP、threshold、template FPR 与模型晋级指标只在 primary representative valid 上计算。数据充分性和 evidence-direction 指标使用 `primary representative valid + isolated evidence-expert valid controls`，controls 的结果单独报告，不能混入 primary AP：

1. selected clean AP 相对 B0 至少 `+0.03`；
2. public-noise FPR 至少降低 `0.20`；
3. direct/component recall 下降不超过 `0.05`；
4. template-clone FPR 不增加；
5. contextual fusion AP 不低于 clean；
6. grouped component bootstrap 的 clean-B0 与 fusion-clean AP delta 95% CI 下界都不得低于 `-0.03`；
7. model selection 从未读取 internal test。

另有数据充分性门槛：representative valid 至少包含 `20` state-backed public-noise negatives、`20` state-backed verified-direct positives、`15` component-anchor positives。

这里不能再按旧 `evidence_type` 名称直接计数：旧 `public_contact_or_url_noise` 中存在“没有共享 identifier、实质是模板/数据包复用”的历史负例。primary rows 必须为明确 benchmark eligible 且不是 `silver_train_only`；额外 controls 必须同时满足 `primary_identity_model_eligible=0`、`evidence_expert_eligible=1`、`evidence_expert_validation_eligible=1`。在此隔离条件下定义：

- public-noise：`review_label=negative`，且 occurrence state 属于 `risky_only_shared / support_only_shared / high_frequency_public`；
- direct positive：`review_label=positive`，且 occurrence state 为 `verified_direct_both_sides`；
- component positive：`review_label=positive`，且 evidence type 为 `same_controller_component_anchor`。

旧 valid 的 `6/18/16` 只是 legacy evidence-type 计数，不视为正式门槛计数。当前隔离 refreeze 的正式计数为 `24/23/15`，Preflight 会在 GPU 方法计算前从 frozen occurrence 与 assignment 重新得到同一结果；任何不一致都应停止，不能降低门槛。

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

当前正式运行只认 `readiness_expansion_v2_20260715`，不再执行历史 queue-v1/refreeze-v1 命令。至少同步：

```text
reports/step15_v8/validation_expansion_queue_v2_20260714/
reports/step15_v8/identity_control_review_20260715/
reports/step16_v8_validation_refreeze/readiness_expansion_v2_20260715/
reports/step15_v7/v2_identifier_redacted_20260714/splits/representative_validation_assignments.csv
reports/step15_v7/v2_identifier_redacted_20260714/splits/representative_validation_manifest.json
reports/step15_v7/v2_identifier_redacted_20260714/clean_embeddings/multilingual_e5_large_identifier_redacted.en_content_train_pool.json
reports/step15_v7/v2_identifier_redacted_20260714/clean_embeddings/multilingual_e5_large_identifier_redacted.en_content_train_pool.npy
reports/step15_v7/v2_identifier_redacted_20260714/clean_embeddings/clean_embedding_manifest.json
reports/step15_v7/v2_identifier_redacted_20260714/clean_embeddings/multilingual_e5_large_identifier_redacted.zh_target_strict.json
reports/step15_v7/v2_identifier_redacted_20260714/clean_embeddings/multilingual_e5_large_identifier_redacted.zh_target_strict.npy

scripts/run_step15_v8_readiness_linux_20260715.sh
scripts/step15_build_v7_clean_embedding_cache.py
scripts/step15_build_v7_inductive_pair_features.py
scripts/step15_v8_common.py
scripts/step15_v8_preflight.py
scripts/step15_build_v8_clean_semantics.py
scripts/step15_run_v8_bridge_audit.py
scripts/step15_train_v8_contextual_evidence.py
scripts/step12_v8_statistical_robustness_audit.py
scripts/step15_v8_downstream_gate.py
scripts/step15_v8_build_sync_manifest.py
scripts/step15_v8_verify_readiness_runtime.py
scripts/step15_v7_common.py
scripts/step16_materialize_v8_reviewed_readiness_freeze.py
scripts/step16_build_v8_context_review_queues.py
scripts/step16_build_v8_identity_control_queues.py
scripts/step16_apply_v8_context_reviews.py
scripts/step16_reconcile_v8_dual_reviews.py
scripts/step16_reconcile_v8_identity_control_reviews.py
tests/test_step15_v8_contextual_evidence_contracts.py
```

同时同步本分支修改过的 policy/schema 和下列文档；若 Linux 状态已经混乱，直接同步整个 `scripts/`、`schema/`、`tests/` 和上述两个 report 根目录更稳妥：

```text
docs/STEP15_V8_CONTEXTUAL_EVIDENCE_FUSION_PLAN_20260714.zh.md
docs/STEP16_V8_READINESS_EXPANSION_PROTOCOL_20260715.zh.md
docs/PROJECT_PROGRESS.md
```

Linux 工作区还必须保留 generated policy 引用的 v6/v7 冻结依赖、原始 Step3/4/5 artifacts、Step7 semantic model policy，以及本地 Multilingual-E5、BGE-M3、LaBSE、GTE reranker 模型目录。正式一键命令为：

其中 transitive identity-control 输入必须同步原文件而不是只同步 review summary，至少包括 `products_data.csv`、`reports/step3_seller_profiles.zh_target_aux.jsonl`、`reports/step5_zh_target_aux_frozen_silver_labels.csv`、原 strict Step3/4/5 文件、原 representative assignment，以及 `schema/step4_silver_candidate_schema.json` 和 `schema/step7_transfer_safe_pair_feature_schema.json`。正式 freeze manifest 会逐一核验这些 SHA-256。

```bash
cd /home/yongpeng/cross-lingual

export CUDA_VISIBLE_DEVICES=0
export V8_RUN_ID=bridge_v8_readiness_20260715
export STEP12_RESAMPLES=5000

bash scripts/run_step15_v8_readiness_linux_20260715.sh
```

runner 的执行顺序固定为：freeze/check-only 与 39 项契约测试；检查 formal runtime 是完全不存在还是完整且哈希一致；首次运行仅重建中文 identifier-redacted E5 cache，并原子重建中英文 v7 feature tables；随后验证完整 runtime chain，再执行 v8 preflight、score-blind context queue snapshot、BGE/LaBSE/reranker 去标识特征、B0–B3 OOF bridge、evidence expert、Step12、return-sync manifest 与 Step20 gate。GPU 只用于 embedding/reranker；LR/linear ranker、evidence expert 和 Step12 主要使用 CPU。V8 preflight 将“CSV 列不存在”“整列在 train 上无可用值”和“可由预注册 fold-train median 插补的局部空值”分开处理；只有最后一种允许继续，且会输出逐特征缺失计数。Stage 8 读取 bridge 时以 artifact 中的全精度验证阈值为权威值，同时核验预测 CSV 的唯一 12 位阈值、summary 的 6 位展示值以及混淆矩阵；summary 展示精度不再被误当作模型阈值精度。

如果同一 `run_id` 已存在，不得删除后覆盖。应使用新路径；已经完整并通过哈希校验的 shared readiness runtime 会复用，不会重复编码：

```bash
export V8_RUN_ID=bridge_v8_readiness_20260715_rerun1
bash scripts/run_step15_v8_readiness_linux_20260715.sh
```

## 10. Windows 同步回传

运行结束后同步整个目录：

```text
reports/step16_v8_validation_refreeze/readiness_expansion_v2_20260715/
reports/step15_v8/<run_id>/
```

前者会新增 Linux 生成的中文 clean-E5 cache、中英文 v7 feature tables、corpus reference 和 v7 manifest；后者包含本次 v8 的全部模型与统计输出。两者缺一不可。

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
