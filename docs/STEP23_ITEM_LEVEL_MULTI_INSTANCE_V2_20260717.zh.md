# Step23-v2.1 同商品匹配对照的多实例 Seller-Pair 验证

> `v2_20260717` 首次 Linux 结果已作废：跨字段检测把换行规范化误判为 identifier 命中，导致全部商品的 exact-overlap hash 被禁用。v2.1 只修复该实现错误；模型集合、划分、权重、预注册主模型和晋级门槛均保持不变。

## 1. 实验定位

Step21/22 已证明，改写已标注 positive 文本或拆分同一 seller 库存都不能替代新的跨账号身份事实。Step23 不再生成训练标签，而是检验一个更窄的问题：

> 在完全相同的真实商品集合上，item-to-item 关系分布是否比同商品聚合表示提供额外的同控制排序信息？

该实验是 representation audit，不是新的 benchmark，也不是最终论文评估。当前 valid/test、Step16I Dev2 和 Step20 均不得参与配置选择。

## 2. v1 为什么作废

V1 没有执行正式数值实验，但代码审查发现两个不可忽略的问题：

1. 跨字段 identifier redaction 触发后，标题和描述 exact-overlap hash 被置空；旧去重 signature 同时依赖这两个 hash，导致同一 seller 同一类别的不同商品可能被错误合并。
2. V1 aggregate baseline 使用旧 seller-profile 文本的 E5 cosine，而多实例模型使用新选择的真实商品。输入语料和表示容量同时变化，无法把提升归因于 multi-instance distribution。

V2 使用新的输出根目录，不覆盖任何 V1 artifact；v2.1 再使用独立目录，不覆盖已作废的 v2 数值结果。

## 3. 数据边界

只使用 canonical train 中可监督的真实 seller pair：

| Pool | Pair | Seller | 对应原始 item 上限前数量 |
|---|---:|---:|---:|
| English source train | 401 | 582 | 62,075 |
| Chinese target train | 573 | 676 | 3,439 |

每个 seller 最多选择 12 条真实商品。选择过程只在 seller 内完成，不读取 pair label 的正负值，不使用跨 seller 统计，不编码 valid/test seller 的商品。

中文 train 的标签组成必须在解释中保留：

| 标签来源 | Positive | Negative |
|---|---:|---:|
| canonical non-silver | 16 | 186 |
| silver train-only | 213 | 158 |

因此全部 573 行 OOF AP 只能作为 silver-supported development metric，不能替代真实 prospective holdout。

## 4. Identifier redaction 与去重修复

每条商品从 category、title、description 构建文本，并使用 Step3 occurrence literals、seller alias 和高精度 identifier regex 做固定点清洗。

V2.1 将两个概念物理分离：

- `content_signature`：最终清洗、但在编码截断前的完整文本 hash，只负责 seller 内真实内容去重。
- `title_hash/description_hash`：只负责跨 seller exact-overlap。当跨字段清洗触发时置空，并记录 `exact_overlap_eligible=false`。

跨字段清洗是否触发只依据第二遍 redaction 的实际 regex/literal 命中数，不再依据规范化前后文本是否逐字符相等。普通多字段商品必须保留已分别脱敏后的 title/description hash；只有跨字段拼接后确实形成完整 identifier 的商品才禁用 exact-overlap。

因此禁用可能受 identifier 影响的 exact-overlap，不再导致不同商品错误折叠。

## 5. 商品选择

对每个 seller：

1. 按最终 redacted content signature 去重；
2. 按 redacted category 分组；
3. 以确定性 category round-robin 选择；
4. 最多保留 12 条；
5. 不足 1 条则 fail closed。

该过程无随机抽样，不会因运行设备、seed 或文件顺序改变。

## 6. 同商品聚合表示

对 seller 的选中 item embeddings 求均值并重新 L2 normalize。两个 seller 的均值向量 cosine 定义为：

```text
mi_mean_pool_cosine = cosine(mean(E_left), mean(E_right))
```

它与 distribution features 使用完全相同的商品、清洗规则、E5 模型和 item cap，是正式 matched aggregate 的语义核心。

Matched aggregate control 还包含 item 数量、cross-pair 数量、redacted exact-overlap 和 category overlap，用于控制商品可用性与基础结构差异。

## 7. 多实例分布特征

对左侧 `n` 条和右侧 `m` 条 item embeddings 构造 `n x m` cosine matrix，提取：

- cosine mean/std/min/max；
- q25/q50/q75/q90/q95；
- top-1/top-3/top-5 mean；
- max-minus-median 与 q95-minus-median；
- 双向 nearest-neighbor mean/std/min/max 与分位数；
- mutual top-1 share；
- semantic-best-match 的长度、数字、标点和 CJK 比例差异。

所有特征都必须在交换左右 seller 后保持一致。

## 8. 固定模型矩阵

V2 不做候选择优。所有模型均使用同一 LR/L2、折内 median imputation、折内 standardization 和 factorized evidence weights。

| 模型 | 作用 |
|---|---|
| `same_item_mean_pool_cosine_only` | 最简单同商品语义聚合诊断 |
| `item_structure_only` | 检查 item 数量/结构捷径 |
| `same_item_aggregate_control` | 正式 matched baseline |
| `semantic_distribution_no_count` | 检查不含 item count 的分布信号 |
| `aggregate_plus_distribution_primary` | 预注册唯一主模型 |

正式比较固定为：

```text
aggregate_plus_distribution_primary
vs
same_item_aggregate_control
```

其他模型仅用于解释，不参与选择。每个模型都额外训练 English-only source control。

## 9. Grouped OOF

中文 train 根据 Step16I recomputed seller components 分为五折。每折训练数据为全部 English train 加其余四折 Chinese train，预测未见 Chinese components。

每折重新拟合：

- missing-value median；
- feature mean/std；
- factorized weights；
- LR/L2 coefficients。

程序显式断言：

- English/Chinese train seller 无交集；
- English/Chinese component 无交集；
- 每个 Chinese component 只属于一个 fold；
- 每折 held-out 同时包含 positive 和 negative；
- 所有 Chinese train 行恰好获得一次 OOF score。

## 10. 晋级门槛

晋级只表示允许打开一次 representative validation，不是论文性能成立。以下条件必须全部满足：

1. Primary AP 比 matched aggregate 至少高 0.02；
2. 5,000 次 component bootstrap 的 AP delta 95% CI lower bound 不低于 0；
3. canonical non-silver AP 不下降；
4. direct/component positives 加全部 negatives 的 AP 不下降；
5. direct/component positive mean score 下降不超过 0.02；
6. template/topic negative mean score 增幅不超过 0.02；
7. 同切片 q95 增幅不超过 0.02；
8. 同切片 top-decile mean 增幅不超过 0.02；
9. valid/test score 未被读取。

Silver-only 指标单独报告且只能作为 secondary development evidence。

## 11. Artifact 与后续推理

V2 保存：

- 每个 source-only 模型 artifact；
- 每个模型五折 OOF 的 imputation、standardization、coefficients、训练 pair hash 和 held-out components；
- 使用全部 English+Chinese train 拟合的 final artifacts；
- 固定 feature order；
- policy、代码、特征、标签和 component assignment hashes。

后续 valid/Step20 只能先按同一冻结规则构建 pair feature CSV，再调用：

```bash
python3 scripts/step23_score_frozen_pair_features.py \
  --policy schema/step23_item_multi_instance_policy.json \
  --features <frozen_pair_features.csv> \
  --output-csv <scores.csv> \
  --output-manifest <scores.manifest.json>
```

该 scorer 不读取 label、不计算 metric、不选 threshold。

默认情况下，若内部晋级门槛失败，scorer 会拒绝为后续 validation/holdout 生成正式分数。`--allow-ineligible-diagnostic` 只能用于明确标注的负结果诊断，不能用于晋级或论文结果。

## 12. 输出隔离

```text
reports/step23_item_multi_instance/v2_1_20260717/
```

所有已存在但内容不同的输出都会 fail closed；同步 manifest 必须覆盖 policy 声明的全部 artifact，且拒绝未声明文件。

## 13. Linux 正式运行

```bash
cd /home/yongpeng/cross-lingual
export STEP23_DEVICE=cuda
bash scripts/run_step23_item_multi_instance_v2_1_linux_20260717.sh
```

Windows 只执行源码编辑、编译、契约测试、Bash 语法和 Git 管理，不生成正式数据或数值结果。

## 14. 结果解释纪律

- 若未晋级：冻结为真实商品多实例表示的负结果，停止继续增加文本分布特征。
- 若晋级：冻结全部配置，只打开一次 representative valid。
- Representative valid 通过后仍不能形成论文最终结论；最终只认独立收集并冻结的 Step20 prospective holdout。
- 不得把 silver-dominated train OOF 写成中文 benchmark 性能。

## 15. v2.1 Linux 结果与冻结结论

同步 manifest 覆盖 `11` 个 payload（`37,877,331` bytes），本地复核没有缺失、大小差异或 SHA-256 差异。校正后的跨字段逻辑只对 `65,514` 条原始商品中的 `1` 条触发；`6,410` 条选中商品全部来自 train，`synthetic_item_count=0`，valid/test 商品编码数为零，英中 seller/component 交集均为零。

中文 train seller-component grouped OOF 结果：

| 固定方法 | ROC-AUC | AP |
|---|---:|---:|
| same-item mean-pool cosine | 0.694818 | 0.598696 |
| item structure only | 0.611519 | 0.572055 |
| same-item matched aggregate | 0.658754 | 0.593218 |
| semantic distribution no-count | 0.495100 | 0.403124 |
| aggregate plus distribution primary | 0.545775 | 0.458483 |

预注册主模型相对 matched aggregate 的 AP 差为 `-0.134735`，component-grouped bootstrap 95% CI 为 `[-0.193967, 0.001614]`，`P(delta>0)=0.0284`。它在 canonical non-silver 切片上从 `0.177815` 降到 `0.128322`，在 direct/component-positive-plus-all-negatives 切片上从 `0.224848` 降到 `0.217211`。同时，它将 template/topic negatives 的 q95 最多抬高 `0.199726`，top-decile mean 最多抬高 `0.244383`。

因此 `promotion_eligible=false`。该结果不能写成正式中文 benchmark 性能，因为 OOF 标签仍由 silver positives 主导；但它足以在当前开发边界上否定“增加 item-to-item 分布统计会稳定改善身份排序”这一方法假设。Step23-v2.1 冻结为负消融，不进入 Step11/17 或 Step20。后续不再针对同一 OOF 边界挑选分布特征，也不通过生成更多软正例挽救该方法；identifier-redacted mean-pool cosine 只保留为后续真实独立数据上的诊断控制。
