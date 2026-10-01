# Step25-v2：配对局部复制检测与匹配缺失机制诊断

## 1. 文档状态

- 方案版本：`2026-07-17-step25-v2-preregistered-pair-local-copy-missingness-diagnostic`
- Git 分支：`method/step25-v2-pair-local-copy-diagnostic`
- 当前状态：代码与配置实现完成，等待 Linux 数值运行和结果回传
- 实验边界：当前 canonical `train` 上的回顾性机制诊断（D0）
- 论文晋级资格：硬编码为 `false`
- Step11/Step17 图谱入口：硬编码为 `false`
- Step25-v1 状态：保持冻结，不覆盖、不改写、不撤销其负结果

Step25-v2 不是一个为了继续“调到更高分”的新主模型，而是对 Step25-v1 失败原因的隔离诊断。它只回答两个机制问题：

1. Step25-v1 的全局模板检测是否漏掉了只在一对卖家或很小团体之间出现的复制文本？
2. Step25-v1 把清洗后短文本的风格余弦记为 `0`，是否把“不可测”错误编码成了“风格极不相似”？

## 2. 为什么需要 Step25-v2

Step25-v1 使用跨卖家、跨组件的全局重复支持识别通用模板。这个策略科学上较保守：一个文本片段只有在多个独立 seller component 中重复，才被认为是模板并删除。它能够避免同一操作者内部重复文本被模型自我证明为模板，但存在一个明确盲区：

- 卖家 A 复制卖家 B 的文本；
- 该文本只在 A/B 之间出现；
- 或者只在一个很小的局部团体中出现；
- 全局文档频率或跨组件频率未达到 Step25-v1 门槛。

这种 pair-local copy 不会被全局模板目录删除，但它仍可能让风格编码器给出很高相似度，从而把模板复用误认为同一操作者。

第二个问题来自缺失值语义。复制文本被删除后，某些 seller 的剩余文本太短，无法可靠估计风格。Step25-v1 的历史实现使用余弦 `0` 表示这种情况。然而：

- 余弦 `0` 是一个真实数值，表示两个向量近似正交；
- “剩余文本不足，无法计算”是缺失状态；
- 两者不是同一概念。

在线性模型中，把缺失状态写成 `0` 会让模型把数据质量问题学习成身份信号。Step25-v2 因此同时修复复制检测范围和缺失值表达，但通过匹配对照把两种影响分开。

## 3. 科研边界

### 3.1 允许使用的数据

Step25-v2 只读取：

- Step24 冻结的 canonical English `train`；
- Step24 冻结的 canonical Chinese `train`；
- Step24/v7 已经 identifier-redacted 的 seller 文本；
- Step24 的原始风格 pair features；
- Step25-v1 冻结的全局模板清洗 pair features；
- Step5 已冻结的 `train` 标签，仅用于模型训练和 grouped out-of-fold 诊断。

### 3.2 禁止使用的数据

复制检测器和特征构建器不得读取：

- `review_label`；
- `evidence_type`；
- 模型预测分数或错误样本列表；
- `valid` 或 `test`；
- Step11/Step17 图谱结论；
- 当前 D0 指标来调整检测器阈值。

### 3.3 为什么不能晋级

Step25-v2 是在看到 Step25-v1 结果后提出的，因此属于 hypothesis-informed retrospective diagnostic。即使所有机制门槛通过，也只能得出：

> pair-local copy detection 与正确的 missingness treatment 值得在未来独立 D1 数据上预注册验证。

不能得出：

- Step25-v2 是论文主模型；
- Step25-v2 超过现有 baseline；
- 可以进入 Step11/Step17；
- Step25-v1 的负结果已经被推翻。

这些限制在 policy、evaluation summary 和 sync manifest 三处都被硬编码。

## 4. Pair-local copy detector

### 4.1 输入文本

对每条 seller pair，检测器分别取得左右 seller 的 Step15-v7 identifier-redacted clean text。文本由以下内容字段组成：

- `category_concat_top`
- `signature_title_concat`
- `title_concat_top`
- `signature_description_concat`
- `description_concat_top`

以下字段仍被排除：

- seller 原始 ID、alias、market 字段；
- 联系方式拼接字段；
- structured snapshot；
- 未清洗的 profile text。

脚本会重新执行 v7 clean-text replay，并校验语料哈希与冻结的 E5 metadata 一致。这样可以防止 Step25-v2 悄悄换用不同文本边界。

### 4.2 固定检测参数

检测器参数在运行前固定：

| 参数 | 固定值 | 目的 |
| --- | ---: | --- |
| Unicode normalization | `NFKC` | 合并全角/兼容字符变体 |
| case normalization | `casefold` | 消除大小写差异 |
| whitespace | collapse | 消除排版空白差异 |
| character shingle length | `12` | 捕获连续复制内容而非单词巧合 |
| minimum contiguous mask | `24` chars | 至少两个连续 shingle 支持后才删除 |
| minimum reliable remainder | `32` alphanumeric chars | 剩余内容足够才计算 cleaned style |
| maximum mask fraction | `0.95` per side | 防止整篇文本被删除 |
| minimum alphanumeric fraction | `0.5` | 排除主要由标点组成的 shingle |

这些参数不能根据 D0 标签或结果搜索。

### 4.3 检测过程

对每条 pair 单独执行：

1. 分别标准化左右文本。
2. 生成长度为 12 的字符 shingle。
3. 只保留至少一半字符为字母或数字的 shingle。
4. 求左右文本共同 shingle 的交集。
5. 将共同 shingle 覆盖的字符位置合并成连续区间。
6. 只删除长度至少 24 的连续区间。
7. 对左右两侧对称执行删除。
8. 每侧最多删除原文本的 95%。
9. 根据剩余字母数字字符数量生成 `pair_local_style_reliable`。

检测器不要求共同文本在第三个卖家中出现，因此能发现 Step25-v1 全局目录无法发现的 pair-only copy。

### 4.4 持久化纪律

输出保存：

- 清洗后的左右文本；
- 原始文本与清洗文本的 SHA-256；
- mask 字符数、span 数和比例；
- shared shingle 的 SHA-256；
- 左右及 pair-level reliability。

输出不保存原始 shared span 文本，也不保存标签、证据类型或模型分数。

## 5. 风格编码

Step25-v2 沿用 Step24 冻结的两个 authorship encoders：

1. `Blablablab/multilingual-style-representation`
2. `StyleDistance/mstyledistance`

约束如下：

- 使用与 Step24 相同的 repo、revision、local model path 和 embedding dimension；
- 不微调 encoder；
- 不下载新模型；
- 只编码 canonical train pair 的左右文本；
- 不编码 valid/test；
- embedding 进行 L2 normalization；
- metadata 和 matrix 均通过哈希绑定。

由于 pair-local 清洗后的同一 seller 文本会因对手不同而不同，缓存键不再只是 seller UID，而是：

```text
pair_uid::left
pair_uid::right
```

完全相同的清洗文本可以在编码阶段去重以节省计算，但输出仍恢复为 pair-side matrix。

## 6. P0-P4 对照设计

### 6.1 P0：原始风格 + 匹配缺失机制

特征：

- raw PCM style cosine
- raw mStyleDistance cosine
- `pair_local_style_reliable` indicator

当 pair-local 清洗后不可靠时，P0 也把原始 style 视为不可用于匹配比较，并用训练折可靠样本的中位数填补。这样 P0 与 P2 使用完全相同的样本可用性掩码。

P0 是 matched baseline，不是普通 raw-style baseline。它的作用是防止 P2 的差异仅来自“哪些行被判为不可用”。

### 6.2 P1：Step25-v1 全局清洗 + 匹配缺失机制

特征：

- Step25-v1 global-clean PCM cosine
- Step25-v1 global-clean mStyleDistance cosine
- `global_and_pair_local_style_reliable` indicator

P1 使用更严格的交集掩码，因为冻结的 Step25-v1 表征有自己的可靠性边界。P1 只用于诊断全局 detector 与 pair-local detector 的差异，不能选择方法。

### 6.3 P2：pair-local 清洗 + 匹配缺失机制

特征：

- pair-local-clean PCM cosine
- pair-local-clean mStyleDistance cosine
- `pair_local_style_reliable` indicator

P2 是预注册的 primary diagnostic。P0 与 P2 的唯一区别是可靠行使用 raw style 还是 pair-local-clean style；缺失掩码、填补方式、模型、权重、折分和训练边界全部相同。

### 6.4 P3：pair-local 清洗 + raw fallback

可靠 pair 使用 pair-local-clean style；不可靠 pair 回退到 raw style，并追加 reliability indicator。

P3 用于回答“短文本时保留 raw style 是否比填补更合理”，但它不是 matched comparison，因此仅作 sensitivity analysis。

### 6.5 P4：reliable-pair-only

P4 不重新训练模型。它在 `pair_local_style_reliable = 1` 的相同 pair 子集上比较已经拟合的 P0 与 P2。

这可以排除缺失值处理，只检查：

> 在两侧均有足够清洗后文本的 pair 上，删除 pair-local copy 是否改善排序？

P4 不参与模型选择。

## 7. 缺失值机制

### 7.1 禁止固定零

持久化 pair feature table 中，不可靠的 cleaned style 写成 `NaN`，不写成 `0`。

### 7.2 折内中位数填补

每个训练折中：

1. 只使用该折训练数据中 reliability 为 1 的 style 值；
2. 对每个 style 特征计算中位数；
3. 用这个中位数填补训练折和 held-out fold 中的 unreliable rows；
4. 追加 reliability indicator。

中位数不能从 held-out fold、valid 或 test 计算。因此 missingness transform 和 LR/L2 一样，严格在每个 fold 内拟合。

## 8. 训练与评估

### 8.1 固定模型

- 模型：Logistic Regression with L2 regularization
- L2 penalty：`10.0`
- maximum iterations：`400`
- tolerance：`1e-8`
- class weighting：none
- feature standardization：enabled
- evidence weighting：沿用 Step15-v7 factorized evidence weights

不比较其他分类器，不搜索超参数。

### 8.2 三个训练视角

1. English grouped OOF：只在英文 train 内按 seller component 做五折 OOF。
2. Source-only：使用全部英文 train 拟合，给中文 train 打分。
3. Target grouped OOF：每折使用全部英文 train 加其余五分之四的中文 train，给未见中文 seller component 打分。

实际折数固定为五折。所有 held-out fold 都必须同时包含 positive 和 negative。

### 8.3 指标

主要机制指标：

- ROC-AUC
- Average Precision（AP）
- P2 相对 P0 的 AP delta
- component-grouped bootstrap 95% confidence interval
- direct/component-positive slice AP delta
- template-clone negative mean rank percentile delta
- template-clone vs strong-positive pairwise violation-rate delta
- reliable-pair fraction

此外报告：

- canonical non-silver slice；
- soft-positive slice；
- template/topic/public-noise negative tails；
- P4 reliable-only sensitivity。

## 9. 机制门槛

Step25-v2 固定以下门槛：

| 门槛 | 阈值 |
| --- | ---: |
| source-only P2-P0 AP | `>= -0.01` |
| target OOF P2-P0 AP | `>= -0.01` |
| source bootstrap lower bound | `>= -0.02` |
| target bootstrap lower bound | `>= -0.02` |
| target direct/component AP delta | `>= -0.02` |
| target template mean-rank delta | `<= -0.02` |
| target template violation-rate delta | `<= -0.03` |
| Chinese reliable-pair fraction | `>= 0.50` |

全部门槛通过只生成 `mechanism_hypothesis_supported = true`。以下字段无论结果如何都必须为 false：

- `d1_candidate_eligible`
- `publication_promotion_eligible`
- `step11_or_step17_entry_allowed`

## 10. 输出与防覆盖

所有结果写入独立目录：

```text
reports/step25_template_decontaminated_authorship/
  v2_pair_local_diagnostic_20260717/
```

不会覆盖 Step24 或 Step25-v1。主要输出包括：

- `pair_local_texts.en_content_train_pool.jsonl`
- `pair_local_texts.zh_target_strict.jsonl`
- `pair_local_copy_detection_summary.json`
- `embeddings/*.npy`
- `embeddings/*.json`
- `pair_local_style_embedding_manifest.json`
- `pair_features.en_content_train_pool.csv`
- `pair_features.zh_target_strict.csv`
- `pair_feature_summary.json`
- `step25_v2_grouped_oof_predictions.en.csv`
- `step25_v2_grouped_oof_predictions.zh.csv`
- `step25_v2_model_artifacts.json`
- `step25_v2_evaluation_summary.json`
- `step25_v2_sync_manifest.json`

写入函数采用 immutable contract：如果目标文件已存在但内容不同，运行会失败，不会静默覆盖跨代码/数据边界的旧结果。

## 11. Linux 完整运行方式

同步本分支列出的 policy、scripts、test 和本文件后，在 Linux 项目根目录运行：

```bash
cd /home/yongpeng/cross-lingual
bash scripts/run_step25_v2_pair_local_copy_linux_20260717.sh
```

Runner 会依次执行：

1. config-only 验证和契约测试；
2. Step24/Step25-v1/model directory 完整性检查；
3. pair-local copy detection；
4. frozen authorship encoder 推理；
5. P0-P3 pair feature 构建；
6. source-only、English OOF、target grouped OOF、bootstrap；
7. closed sync manifest。

Windows 端只运行静态检查、`py_compile`、config-only 和纯函数单元测试，不运行 encoder 或数值实验。

## 12. 结果解释规则

### 12.1 如果 P2 明显优于 P0

只能说明 pair-local copy removal 在当前 D0 上显示出机制信号。下一步应在新 D1 开发边界上预注册同一 detector 和缺失机制，再验证一次。

### 12.2 如果 P2 与 P0 接近

说明 Step25-v1 的失败不主要来自 pair-only copy 漏检，或者删除复制文本后剩余风格信号仍不足。此时不应继续调 detector 参数。

### 12.3 如果 P2 低于 P0

说明 pair-local detector 很可能同时删除了真实 authorship signal，或者当前文本长度不足以支撑清洗后风格编码。该路线应作为负结果冻结。

### 12.4 如果 P3 优于 P2

可能说明短文本下 raw fallback 比中位数填补更有信息，但 P3 不是 matched comparison，不能直接据此晋级。需要未来独立数据验证。

### 12.5 如果 P4 改善但全量 P2 不改善

说明清洗在“文本充分”的 pair 上有效，但覆盖率或短文本缺失是主要瓶颈。后续工作应改进 occurrence-level 文本收集，而不是调分类器。

## 13. 最终科研定位

Step25-v2 的价值是把一个模糊判断拆成可证伪的机制实验：

- 全局模板检测是否遗漏 pair-local copy；
- cleaned style 的缺失状态是否被错误数值化；
- 在可靠文本子集上，去复制后的 authorship signal 是否仍存在。

它不生成新标签、不扩大数据、不训练 Transformer，也不改变当前论文主线。其结果将决定“pair-local copy-aware authorship”是否值得进入未来独立数据阶段，而不是用当前内部数据继续反复调参。
