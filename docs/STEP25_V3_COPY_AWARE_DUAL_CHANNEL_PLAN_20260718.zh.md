# Step25-v3 复制感知双通道融合科研计划

## 1. 实验定位

Step25-v3 是 Step25 研究主线的直接延续，不是对 Step25-v1/v2 的覆盖，也不是另起一个无关方法。

Step25-v1 检验了跨 seller、跨 component 的全局模板清洗。该方法没有稳定提高排序性能，也没有充分压低模板负例尾部风险，因此冻结为负结果。

Step25-v2 检验了仅在当前 seller pair 内出现的局部复制文本，并修正了将不可靠清洗后风格写成固定余弦零值的问题。它成功检测了 `109/110` 条中文模板负例中的局部复制内容，并降低了模板负例相对强正例的违规率和公共联系方式噪声风险；但是，把 pair-local-clean style 统一替换 raw style 后，中文总体 AP 从匹配缺失控制的 `0.704847` 降至 `0.670692`，英文 AP 从 `0.468210` 降至 `0.251926`。

因此，Step25-v2 否定的是“统一清洗替换”实现，而不是 pair-local copy signal 本身。Step25-v3 的核心改动是：

> 同时保留原始风格、清洗后风格和复制风险三个信息通道；复制风险只能形成非正向惩罚，不能被模型学成同一操作者的正证据。

## 2. 科研目标

Step25-v3 要回答以下问题：

1. 原始风格信号中是否同时包含真实作者风格和复制模板捷径？
2. 清洗后风格虽然不能独立替代原始风格，是否仍能作为条件残差信号帮助区分真实身份联系和模板复用？
3. 在不删除原始风格的前提下，复制比例、共享片段数量和 raw-clean 差值是否可以压低模板、同主题和公共联系方式噪声？
4. 将复制相关特征的系数约束为非正后，是否能避免当前大量软正例让模型重新把复制内容学成正证据？
5. 将直接 identifier evidence 保留在独立 operational control 中，是否能在不污染 clean scorer 的前提下提升实际图谱发现可靠性？

## 3. 当前证据为什么支持继续 Step25

Step25-v2 的中文结果呈现明显的切片异质性：

| 切片 | P0 raw matched AP | P2 pair-local-clean AP | 差值 |
|---|---:|---:|---:|
| direct/component positive + all negative | 0.486778 | 0.606200 | +0.119421 |
| soft positive + all negative | 0.613258 | 0.486294 | -0.126964 |
| all canonical train | 0.704847 | 0.670692 | -0.034155 |

这说明 pair-local clean representation 不是全局更好或全局更差，而是在不同 evidence type 上方向相反。统一替换必然把两种效应混合。双通道方法的目标正是把这种条件效应显式建模。

同时必须保留以下限制：当前中文 train 为 `573 = 229 positive / 344 negative`，其中 `213/229` positive 是 `silver_train_only`。因此全量 OOF 指标只能作为 D0 内部开发证据；non-silver、direct/component 和 soft-positive 切片必须分别报告，不能用总体 AP 代替论文级确认。

## 4. 数据边界

### 4.1 D0 当前边界

Step25-v3 只读取 Step25-v1/v2 已使用的 canonical train：

- English train：`401 = 116 positive / 285 negative`
- Chinese train：`573 = 229 positive / 344 negative`
- Chinese non-silver positive：`16`
- Chinese silver positive：`213`

D0 已经被 Step24 和 Step25-v1/v2 错误分析消耗。Step25-v3 是 hypothesis-informed retrospective development，允许使用 D0 label 做 grouped OOF 拟合，但：

- 不读取任何 valid/test pair、label、score 或 threshold；
- 不允许在 D0 上搜索 mask、shingle、penalty 或模型超参数；
- 不允许产生 publication promotion；
- 不允许进入 Step11/17；
- 最多只能决定是否值得将完全冻结的 v3 方法复制到未来 D1。

### 4.2 D1 独立复制边界

只有 D0 全部门槛通过后，v3 才能成为 D1 replication candidate。D1 必须满足：

- seller component 与 D0 完全不相交；
- 审查过程看不到任何模型分数；
- 至少 30 条 non-silver direct/component positive；
- 至少 30 条 template-clone negative；
- 至少 20 条 public-contact/URL negative；
- 至少 30 条 semantic-topic negative；
- v3 方法、权重、约束和阈值在进入 D1 前冻结；
- D1 不得用于重新拟合或重新选择 v3 变体。

### 4.3 F1 前瞻性最终边界

F1 必须在模型、特征、阈值和 D1 结论全部冻结后收集，与 D0/D1 component 均不相交，只评估一次。最终论文确认只认 F1。

## 5. 输入与不可变父实验

Step25-v3 复用以下不可变产物：

- Step24 identifier-redacted raw style 和 E5 特征；
- Step25-v1 global-clean style、global boilerplate fractions 和 global reliability；
- Step25-v2 pair-local-clean style、raw fallback、mask fraction、shared shingle count 和 pair-local reliability；
- Step15-v7 factorized evidence weights；
- Step25-v1 occurrence-level identifier state contract。

v3 只读取并校验这些产物，不得写回或覆盖其目录。v3 输出固定到：

```text
reports/step25_template_decontaminated_authorship/
  v3_copy_aware_dual_channel_20260718/
```

## 6. Clean scorer 特征设计

### 6.1 Raw channel

```text
raw_pcm_multilingual_authorship_cosine
raw_mstyledistance_cosine
```

该通道保留 Step24 已发现的跨语言风格能力。v3 不再因清洗后文本不足而丢弃 raw style。

### 6.2 Pair-local-clean channel

```text
pair_local_or_raw_pcm_multilingual_authorship_cosine
pair_local_or_raw_mstyledistance_cosine
```

当 pair-local clean text 可靠时，使用真实 clean cosine；当任一侧剩余有效内容不足时，显式回退到 raw cosine。

因此，不可靠行满足：

```text
clean_or_raw = raw
raw_minus_clean_or_raw = 0
pair_local_style_reliable = 0
```

这避免了 Step25-v2 P0/P2 中 119 条中文不可靠行被中位数替代后完全失去原始风格信息的问题，也不把缺失值伪装成余弦零。

### 6.3 Copy residual channel

```text
pcm_raw_minus_pair_local_or_raw
mstyledistance_raw_minus_pair_local_or_raw
```

如果删除共享复制片段后相似度显著下降，差值会增大，说明原始相似性可能主要由复制内容贡献。两个系数均强制为 nonpositive。

### 6.4 Copy-risk channel

```text
pair_local_maximum_mask_fraction
pair_local_mean_mask_fraction
pair_local_shared_shingle_count_log1p
pair_local_masked_span_count_log1p
pair_local_style_reliable
global_pair_maximum_boilerplate_fraction
global_pair_mean_boilerplate_fraction
global_style_reliable
```

其中 mask fraction、shared shingle、masked span 和 global boilerplate fraction 的系数强制为 nonpositive。reliability indicator 不约束方向，因为它表示测量可用性而不是正负证据。

### 6.5 明确禁止进入 clean scorer 的信息

- Telegram、QQ、微信、email、PGP、wallet 等 identifier；
- candidate_rule_count 或候选生成规则命中；
- review label、evidence type、review note；
- valid/test 拟合统计量；
- 未清洗 seller profile embedding；
- synthetic pair 或 synthetic identity label。

## 7. 模型与对照

所有模型使用相同训练行、factorized evidence weights、component folds、标准化和 L2 强度。不存在 candidate search。

### C0：matched raw-style baseline

仅使用 raw channel。两个相似度系数限制为 nonnegative。这是 v3 的正式匹配基线。

### C1：raw + clean，无 copy penalty

使用 raw、clean、delta 和 reliability，但 delta 不限制方向，不加入 mask/global boilerplate 风险特征。它回答“双表示本身是否有价值”。

### C2：copy-aware dual-channel primary

使用完整 raw、clean、delta 和 copy-risk channel。相似度系数 nonnegative，copy residual 和 copy-risk 系数 nonpositive。这是唯一预注册主模型。

### C3：semantic sensitivity

在 C2 基础上增加 identifier-redacted E5 cosine，仅作敏感性分析，禁止选模。它用于判断语义信号是否补充风格残差，不影响 C2 的晋级决定。

### Frozen parent references

报告中同时列出 Step25-v1 raw-style 和 Step25-v2 P0/P2/P3 的冻结指标，但这些父模型不是 v3 候选，也不会重新解释其原结论。

## 8. 方向约束 LR/L2

v3 使用 projected-gradient logistic regression with L2，而不是树模型或 MLP。每折仅在训练行上：

1. 拟合均值和标准差；
2. 计算 factorized evidence weights；
3. 优化 weighted log loss + L2；
4. 每次更新后把系数投影到预注册方向区间；
5. intercept 始终不约束；
6. 将收敛状态、迭代次数、最终梯度、标准化统计量、权重摘要和系数全部持久化。

约束的科研意义不是保证模型一定提高，而是禁止出现与方法假设相反的捷径：

- 高复制比例不能被学成正身份信号；
- raw-clean 差值不能提升同控制概率；
- 更高的 clean authorship similarity 不能被解释为负证据。

## 9. 分组训练与防泄漏

### 9.1 English grouped OOF

按完整 seller component 做五折。每条 English train pair 恰好获得一次未见其 component 的 OOF 分数。

### 9.2 Source-only transfer

使用全部 English train 拟合 C0-C3，直接给全部 Chinese canonical train 打分。中文 label 不参与训练。

### 9.3 Target grouped OOF

每折训练数据为：

```text
all English train
+ four-fifths Chinese train components
```

held-out Chinese component 的任何 pair 都不能进入训练、标准化或权重拟合。

### 9.4 当前 component 不平衡

当前 573 条中文 train 中最大 component 有 175 条边，165 个 component 只有一条边。必须保留完整 component 隔离，即使折间 prevalence 不均衡。除总体 OOF 外，还要报告每折指标和 component-grouped bootstrap，不能用 row-level bootstrap 掩盖相关性。

## 10. Missingness-only closure control

Step25-v2 的 P1 使用 `global_style_reliable AND pair_local_style_reliable`，没有纯粹隔离 Step25-v1 global-clean fixed-zero 问题。

v3 补做一个只用于关闭该问题的诊断：

- exact frozen Step25-v1 global-clean features；
- reliability 只使用 `global_style_reliable`；
- reference 为 Step25-v1 fixed-zero 结果；
- corrected control 使用 fold-train global-reliable median + indicator；
- 禁止使用 pair-local reliability intersection；
- 结果不得用于选择 C2 或触发晋级。

## 11. Operational identifier control

clean C2 完全不读取 identifier。另行训练 occurrence-level operational control：

- 只使用 English occurrence evidence 和 English grouped-OOF C2 probability；
- verified direct both sides 只能 nonnegative uplift；
- risky-only、support-only、high-frequency public 只能 nonpositive downgrade；
- mixed-context、ambiguous、no-shared-identifier 不改变分数；
- 不使用 Chinese label 训练 expert；
- operational result 与 clean result分别报告；
- operational result 不能帮助选择 clean C2。

## 12. 指标与切片

主排序指标：

- ROC-AUC
- Average Precision

必须报告的切片：

- canonical non-silver；
- direct/component positive + all negative；
- soft positive + all negative；
- pair-local reliable only；
- pair-local unreliable only；
- template-clone negative tail；
- semantic-topic negative tail；
- public-contact/URL negative tail。

负例尾部审计：

- mean score；
- q90/q95；
- top-decile exposure；
- global rank percentile；
- negative-vs-strong-positive violation rate。

所有 AP 差值使用 paired component-grouped bootstrap，固定 5000 次和 seed `20260718`。

## 13. D0 到 D1 晋级门槛

C2 相对 C0 必须同时满足：

1. source-only Chinese AP 差值不低于 `-0.01`；
2. target grouped-OOF AP 至少提高 `0.02`；
3. target grouped-bootstrap 95% CI 下界不低于 `0`；
4. non-silver AP 下降不超过 `0.02`；
5. direct/component AP 下降不超过 `0.02`；
6. soft-positive AP 下降不超过 `0.03`；
7. template violation rate 至少下降 `0.05`；
8. template mean rank 不上升；
9. semantic-topic mean rank 上升不超过 `0.02`；
10. public-noise mean rank 不上升；
11. English grouped-OOF AP 下降不超过 `0.02`。

所有门槛必须通过。通过只表示 `d1_replication_candidate_eligible=true`，不表示论文晋级，也不允许 Step11/17。

## 14. 成功、部分成功和失败的解释

### 全部门槛通过

说明 pair-local copy signal 适合作为受方向约束的辅助通道，值得在新 D1 上做不改配置的独立复制。

### 总体 AP 未提升，但模板/public-noise 风险显著下降

说明复制检测适合作为图谱后置 reliability veto，而不是 pairwise 主分类器。不得宣称提升身份识别，只能进入未来独立图谱审计候选设计。

### soft-positive 下降而 strict/non-silver 提升

说明当前 positive ontology 存在实质冲突。论文主任务必须收缩到有独立身份支持的同控制识别，soft positive 只能作为替代标签敏感性分析。

### English 和 Chinese 均退化

说明冻结 style encoder 的所谓作者风格仍主要依赖内容，Step25 方法线应停止性能优化，转为 evidence-type concept drift 与负结果分析。

## 15. 停止条件

以下任一条件出现，均不得继续在 D0 调参：

- C2 未通过全部 D0 gates；
- 需要修改 shingle length、mask threshold、L2 或方向约束才能通过；
- 需要读取 current valid/test 选择变体；
- 需要把 silver/soft positive 当作 gold 才能得到提升；
- 需要让 copy-risk 系数变为正；
- 需要把 identifier 放入 clean scorer。

## 16. 计划产物

```text
schema/step25_v3_copy_aware_dual_channel_policy.json
scripts/step25_v3_common.py
scripts/step25_v3_build_dual_channel_features.py
scripts/step25_v3_evaluate_copy_aware_fusion.py
scripts/step25_v3_train_operational_identifier_control.py
scripts/step25_v3_build_sync_manifest.py
scripts/run_step25_v3_copy_aware_dual_channel_linux_20260718.sh
tests/test_step25_v3_copy_aware_dual_channel_contracts.py
reports/step25_template_decontaminated_authorship/
  v3_copy_aware_dual_channel_20260718/
```

## 17. 当前科研主张边界

在 D1 和 F1 完成前，Step25-v3 只能支持如下表述：

> Step25-v1/v2 revealed that copied text is both a source of false identity linkage and a component of weak positive supervision. Step25-v3 preregisters a direction-constrained dual-channel model that preserves raw authorship evidence while preventing copy-risk features from becoming positive identity evidence.

不能使用如下表述：

- Step25-v3 已经解决模板污染；
- Step25-v3 已经提高跨语言马甲识别；
- D0 grouped OOF 等于独立 benchmark；
- silver direct/component 等于 proof-level gold；
- operational identifier control 等于 clean cross-lingual model。
