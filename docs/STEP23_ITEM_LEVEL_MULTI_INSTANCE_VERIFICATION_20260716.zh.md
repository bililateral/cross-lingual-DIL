# Step23 Item-Level Multi-Instance Seller-Pair Verification

## 1. 当前问题

Step21 和 Step22 已经连续证明两种直接扩充策略无效：

- 在已标注 positive pair 的聚合文本上做确定性文本变化，不能超过等有效权重复制；
- 把同一个真实 seller 的商品集合互斥拆成两个 pseudo profiles，同样不能超过复制对照，反而更容易提高模板复用和同主题 negative 的分数。

Step22 的关键诊断不是“E5 完全没有信息”，而是当前 aggregate-first 表示把 seller 的大量商品压缩成一个向量。这个向量主要描述整个库存的平均主题。对于中文地下市场，平均主题高度相似既可能来自同一操作者，也可能来自模板复用、同类货源或公共数据包，因此它不是可靠的身份不变量。

Step23 不继续调 Step22 的 synthetic weight，也不生成新身份标签。它改变观察粒度：

```text
seller pair = bag(left real items) x bag(right real items)
```

模型不再只看一个聚合 profile cosine，而是观察两个真实 seller 的多组 item-to-item 关系分布。

## 2. 科研假设

两个 seller 即使库存主题相同，其 item-to-item 关系结构仍可能不同：

- 真实同控制 pair 可能在多个商品子集上持续表现出相似的措辞、结构和风格；
- 模板复用 negative 可能只有一两个极高相似的复制商品，其余商品没有一致关系；
- 同主题 negative 可能整体语义中等相似，但不存在稳定的一对一或双向最近邻对应；
- 同一操作者改变部分商品时，最大相似度未必稳定，但 top quantiles、双向 nearest-neighbor coverage 和 style consistency 可能仍保持。

因此 Step23 检验：

> item-to-item relation distribution 是否比 aggregate seller embedding 更能区分 identity continuity 与 topic/template similarity。

该假设失败时，项目应停止从现有文本库存中继续挖身份性能；成功时，才允许进入代表性 validation 和后续 prospective holdout。

## 3. 监督单位没有改变

Step23 的监督单位仍然是已有真实 seller pair：

```text
input: seller_left, seller_right
label: same_controller / different_controller
output: P(same_controller)
```

Item 不是独立 label。一个 positive seller pair 内的所有 item-to-item 组合也不会被分别标为 positive。它们仅用于构成 bag-level distributional features，最终只产生一行 seller-pair feature vector。

Step23 明确不执行：

- 不新增 synthetic seller；
- 不新增 synthetic item；
- 不新增 positive/negative identity label；
- 不把同 seller item pair 当作新马甲真值；
- 不写回 Step5；
- 不进入 Step11/17；
- 不读取 valid/test 指标选择配置。

## 4. 当前可用规模

本地只读接口审计得到：

| Pool | Train pair | Train seller | 与这些 seller 对应的原始 item |
|---|---:|---:|---:|
| English source | 401 | 582 | 62,075 |
| Chinese target | 573 | 676 | 3,439 |

英文 seller 的库存规模差异很大，不能直接编码全部 62,075 条并让大 seller 主导训练。Step23 对每个 train seller 最多选择 12 条 item，理论编码上限约为：

```text
(582 + 676) * 12 = 15,096 items
```

真实数量将在 Linux 根据去重和空文本情况重新计算。

## 5. Train-Only Item Selection

### 5.1 Seller 范围

每个 pool 只从 canonical frozen labels 中读取：

- `split_name=train`；
- `usable_for_supervision=1`；
- `review_label in {positive, negative}`。

Valid/test label 只用于建立排除 seller 集合。任何 train seller 与 valid/test seller 重叠都会 fail closed。Valid/test item 不读取、不编码、不进入 feature cache。

### 5.2 原始来源

当前正式 source/target train seller 只需要两个结构化来源：

- `market_item.xlsx`；
- `2017-12-05-philipjames11-darknetmarketplacedataagora20142015.xlsx`。

`reports/step2_content_item_manifest.csv` 提供 source row 与 seller UID 的规范映射。若正式 train seller 对应到未支持的数据源，builder 直接停止，不猜测 schema。

### 5.3 Identifier Redaction

每条 item 只保留：

- category；
- title；
- description。

明确排除：

- seller alias；
- market name；
- contact；
- structured snapshot；
- seller identifier；
- Step3 解析出的 Telegram、QQ、微信、email、phone、wallet 等 literal。

Redaction 复用 Step15-v7 的固定点规则，直到文本不再变化，并再次检查高精度 identifier residue。删除过程不留下 `[CONTACT_REMOVED]` 一类 presence marker。

### 5.4 Seller-Local Selection

选择过程不读取 pair label，也不使用跨 seller 的 IDF 或 corpus frequency：

1. 在单个 seller 内，按 identifier-redacted category/title/description 生成 exact content signature。
2. 完全相同 signature 只保留 source row 最早的一条。
3. 按 category 建立本地队列。
4. 对 category 做 round-robin，最多选择 12 条。
5. 每个 train seller 至少保留一条，否则 fail closed。

这个过程避免两个问题：

- 大库存 seller 因 item 数量多而主导模型；
- 同一个模板被 seller 重复发布几十次而人为放大。

## 6. Frozen E5 Item Encoding

每条选中 item 使用现有 frozen `intfloat/multilingual-e5-large` 编码。

Embedding cache 必须绑定：

- Step23 policy SHA-256；
- selected-item JSONL SHA-256；
- item-selection summary/manifest SHA-256；
- encoder producer SHA-256；
- Step7 semantic producer SHA-256；
- Step15-v7 redaction producer SHA-256；
- semantic policy SHA-256；
- 本地模型目录 fingerprint；
- item UID 顺序；
- matrix shape。

已有 cache 只有在全部字段完全一致时才能复用。

## 7. Multi-Instance Distributional Features

设左 seller 有 `n` 个 item embedding，右 seller 有 `m` 个：

```text
S_ij = cosine(item_left_i, item_right_j)
S is an n x m relation matrix
```

### 7.1 全矩阵分布

提取：

- mean、standard deviation、minimum、maximum；
- 25%、50%、75%、90%、95% quantiles；
- top-1、top-3、top-5 mean；
- maximum minus median；
- 95% quantile minus median。

最大值反映最强局部匹配；中位数和高分位反映相似性是否广泛存在；两种 concentration gap 用于区分“单个模板复制”与“多 item 持续相似”。

### 7.2 双向最近邻结构

分别计算：

```text
left_nn_i  = max_j S_ij
right_nn_j = max_i S_ij
```

然后对合并后的双向 nearest-neighbor scores 提取 mean、standard deviation、minimum、maximum 和 25/50/75% quantiles，并计算：

- 左右两侧 nearest-neighbor mean 的绝对差；
- mutual top-1 match share。

如果相似只由左侧一个通用模板匹配右侧大量商品造成，双向结构和互为第一邻居比例应与真实多点对应不同。

### 7.3 Identifier-Free Exact Overlap

在 redaction 之后计算：

- title exact intersection 与 Jaccard；
- description exact intersection 与 Jaccard；
- category Jaccard。

这些不是直接身份特征，而是显式描述模板/库存重合，允许 LR 根据已有 hard negatives 学习其正负方向。

### 7.4 Best-Match Style Gaps

对双向 nearest-neighbor 匹配集合计算：

- log text length gap；
- digit ratio gap；
- punctuation ratio gap；
- CJK ratio gap。

每类报告 mean 和 maximum。它用于检验语义相似的 item 是否同时保持形式风格，而不是把 style 先聚合到 seller mean 后丢失局部对应。

### 7.5 对称性

所有特征必须满足：

```text
features(A, B) == features(B, A)
```

契约测试逐字段验证 endpoint swap 不改变结果。

## 8. 模型对照

所有方法使用同一个强 L2 regularized logistic regression、同一 factorized evidence weighting、同一 fold 和同一 train-only preprocessing。

### 8.1 Target-Adapted Grouped OOF

三组表示：

| Model | Features | 目的 |
|---|---|---|
| Aggregate baseline | identifier-redacted seller E5 cosine | 当前低维聚合语义基线 |
| Item multi-instance | 所有 item relation distribution features | 检验 item 分布本身 |
| Aggregate + item | aggregate cosine + item distribution | 检验互补性 |

每个 target fold 的训练数据为：

```text
all English train rows + four-fifths Chinese train components
```

Held-out Chinese component 不出现在该 fold 的训练行中。

### 8.2 Source-Only Controls

同样三组表示只用 English train label 拟合，然后直接预测全部 Chinese train rows：

- source-only aggregate；
- source-only item multi-instance；
- source-only aggregate + item。

这组结果回答 item distribution 是否提高纯跨域迁移能力，而不只是在中文 label 上重新拟合。

## 9. Fold-Local Preprocessing

每个 OOF fold 独立执行：

1. 只用 English train 加当前 Chinese fold-train 拟合 median imputation。
2. 只用相同 train rows 拟合 standardization。
3. 重算 factorized evidence weights。
4. 训练 LR/L2。
5. 预测未见 Chinese seller components。

Item selection 和 frozen E5 encoding不需要标签，也不拟合跨 seller 统计，因此可以在 fold 之前生成。任何 label-aware、IDF、OOV、normalization 或 coefficient fitting 都必须位于 fold 内。

## 10. Selection 与统计门槛

Item-only 和 aggregate+item 只根据 Chinese-train grouped OOF AP 比较。当前 valid/test 不参与方法选择。

相对 aggregate baseline 的晋级门槛全部同时满足：

1. AP gain 至少 `0.02`。
2. 5,000 次 seller-component grouped bootstrap 的 95% CI lower bound 至少 `-0.01`。
3. `template_clone_not_controller` 平均分增加不超过 `0.02`。
4. `semantic_topic_not_controller` 平均分增加不超过 `0.02`。
5. canonical non-silver sensitivity AP 相对 aggregate 的下降不超过 `0.02`。
6. 额外报告 direct/component positives 加全部 negatives 的强证据 sensitivity。
7. `valid_or_test_scores_used=false`。
8. `publication_holdout_untouched=true`。

这只是 development promotion。即便通过，也不能直接形成论文最终性能结论。

## 11. 通过后的链路

只有 Step23 通过 train-OOF 门槛后：

1. 冻结 item selection、feature schema、LR 配置和模型选择结果。
2. 在 representative validation 上只打开一次。
3. 通过后才考虑 Step20 genuinely prospective holdout。
4. 最终 pair scorer 通过 Step12 统计审计后，才进入 Step11/17 explicit allow-list 图谱验证。

如果 Step23 失败：

- 不围绕当前 OOF 调 item cap、quantile 或权重；
- 冻结为第三个负消融；
- 结论转向现有文本库存无法替代真实跨账号身份锚点；
- 不再继续堆叠文本模型。

## 12. 输出

正式输出根目录：

```text
reports/step23_item_multi_instance/v1_20260716/
```

包括：

- selected identifier-redacted item JSONL；
- item selection summary 与 manifest；
- E5 item matrix 与 metadata；
- English/Chinese train pair features；
- feature summary；
- source-only 与 grouped-OOF predictions；
- evaluation summary 与 grouped bootstrap；
- complete sync manifest。

## 13. Linux 运行

```bash
cd /home/yongpeng/cross-lingual
bash scripts/run_step23_item_multi_instance_linux_20260716.sh
```

默认使用 CUDA。CPU 编码仅在必要时使用：

```bash
STEP23_DEVICE=cpu bash scripts/run_step23_item_multi_instance_linux_20260716.sh
```

Windows 只允许 source edit、Python compile、纯契约测试、Bash syntax check 和 Git/sync 管理，不运行 item extraction、E5 encoding、feature generation、LR training 或 numerical bootstrap。
