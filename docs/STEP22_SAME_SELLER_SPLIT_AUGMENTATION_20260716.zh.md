# Step22 同源 Seller 互斥 Item 切分增强

## 1. 研究问题

Step21 已证明：围绕少量已标注正例 pair 对聚合文本做 section rotation、segment subsample 和标点归一化，不能超过无增强或等有效权重复制。主轨实际上只有 13 个独立 parent components，生成 48 行并没有增加身份来源。

Step22 改变生成单位。它不再从已存在的 pair 文本派生变体，而是从一个真实中文 seller 的原始 item 集合中构造两个内容互斥的 pseudo profiles：

```text
I_s = I_s_A union I_s_B
I_s_A intersection I_s_B = empty
pseudo_profile_A = profile(I_s_A)
pseudo_profile_B = profile(I_s_B)
```

两个 view 来自同一个已观察 seller 账号，因此可构造 `same_seller_split_positive`。它训练模型适应库存变化、类别变化和描述子集变化，但不声称观察到了两个真实账号之间的 sockpuppet relation。

## 2. 本地数据容量依据

Windows 只做只读容量审计，没有执行模型或数值实验。当前 `step3_seller_profiles.zh_target_strict.jsonl` 包含：

- 5,097 个中文 seller profiles；
- 17,556 条中文 item；
- 排除当前 canonical valid/test seller 后剩余 4,658 个 profiles；
- 至少 4 个 item 且标题、描述各至少两种的 seller 约 1,054 个；
- 至少 6 个 item 且标题、描述各至少三种的 seller 约 628 个；
- 至少 10 个 item 的 seller 约 323 个。

这些是生成前容量估计。最终可用数量由 Linux builder 在 identifier redaction、精确内容闭包和 valid/test portable-alias 排除后重新计算，文档不预写最终生成行数。

## 3. 数据来源与允许的科学表述

所有 pseudo profiles 的 item 必须来自：

```text
reports/step2_content_item_manifest.csv
market_item.xlsx
data_bucket = zh_target_strict
```

允许表述：

> We created train-only pseudo-alias views by partitioning disjoint item subsets from the same observed Chinese seller account.

禁止表述：

> We collected hundreds of new real Chinese sockpuppet pairs.

Step22 的独立来源单位是 source seller。两个 pseudo profiles 是该 seller 的两种 observation views，不是两个新发现的现实账号。

## 4. 泄漏排除顺序

生成前先读取 Step16I-v2 的重算 component assignments 和 permanent exclusion manifest：

1. 收集 canonical `valid/test` 中全部 seller UIDs；
2. 从 item manifest 反向收集这些 seller 的 portable aliases；
3. 排除任何具有相同 seller UID 或 portable alias 的源 item；
4. canonical train seller 继承 Step16I recomputed component；
5. 未进入现有监督 pair 的 seller 获得以真实 seller UID 哈希生成的独立 synthetic component；
6. synthetic child 在折外评估中必须跟随 parent component，不能在 parent component 被 held out 时进入训练。

这不能证明未知别名之间绝无现实身份关系，但能阻止已知 seller、portable alias 和已知 component 穿越当前边界。

## 5. Identifier redaction

每条 item 在分组前先使用 Step3 item identity occurrences 和 Step15-v7 高精度规则删除：

- Telegram、QQ、微信、Wickr、Jabber；
- email、phone、wallet；
- PGP、公钥和 fingerprint；
- seller alias；
- 高精度 seller-facing identifier 变体。

删除后再次执行 residue assertion。任何已知 identifier pattern 残留都会 fail closed。Pseudo profile 不写入原 seller name、market name、source row number、联系方式或 identifier-presence marker。

## 6. Positive view 构造

### 6.1 精确内容连通闭包

仅按 `(title, description)` 联合哈希分组仍可能把“标题相同、描述不同”的记录分到两边。Step22 因此构建 item 内容连通分量：

- 共享精确归一化 title 的 item 必须在同一内容组；
- 共享精确归一化 description 的 item 必须在同一内容组；
- 上述关系按传递闭包合并。

一个内容组只能整体进入 left 或 right。最终再次断言两侧：

- source item UID 不重合；
- 精确 title 不重合；
- 精确 description 不重合。

### 6.2 Category-stratified partition

默认最低条件为：

- parent seller 至少 6 个源 item；
- 至少 6 个互相独立的内容闭包组；
- left/right 各至少 3 个 item；
- left/right 各至少 3 个内容组；
- 每个 parent 最多使用 12 个内容组。

内容组按 category 分层后用固定 seed 分配到两侧。若某类别无法自然分开，则执行确定性全局 rebalance；仍不满足最低条件时丢弃该 seller，不降低门槛。

## 7. Reviewed hard-negative views

不同 seller UID 不能自动视为不同操作者。主负例只从现有 Step5 中文 canonical train 中选择：

- `review_label=negative`；
- `usable_for_supervision=1`；
- `usable_for_core_transfer=1`；
- Step16I recomputed component 无跨 split 泄漏。

左右 seller 各抽取一个 identifier-redacted item view，生成 `reviewed_negative_item_view`。当前只读容量审计显示：344 条 reviewed train negatives 中约 58 条左右双方至少各有 3 个 item，最终数量仍由 Linux 上的精确内容闭包决定。

## 8. Profile 与特征表示

Pseudo profile 保持 Step3 风格的主要字段，但模型可见文本只包括以下三个字段按行直接拼接：

```text
category_concat_top
title_concat_top
description_concat_top
```

不添加 synthetic 专用 section label，并与 v7 clean encoder 的无标签字段拼接方式一致；也不包含 seller、market、contact、structured source marker。冻结 Multilingual-E5 编码 pseudo profiles 后，对每个 pair 构造：

```text
E5 cosine
concat(abs(z_left - z_right), z_left * z_right)
```

后者使用与 Step15-v7 相同的固定 Gaussian Johnson-Lindenstrauss projection 压到 64 维。真实英文/中文 pair 与 synthetic pair 使用相同的 65 维表示和同一个 LR/L2 模型。

## 9. 五个预注册对照

1. `no_augmentation`
2. `equal_weight_duplication_positive_budget`
3. `same_seller_split_positive_only`
4. `equal_weight_duplication_full_budget`
5. `same_seller_split_plus_reviewed_negative_views`

复制对照只复制当前训练折内的真实中文 pair representation，并保持各行原 factorized evidence/component weight 的相对比例，再缩放到与 synthetic 条件相同的 class-level 额外有效权重。因此能够区分：

- 正例权重增加；
- item-disjoint profile representation 提供的新信息。

每折的 synthetic positive 总权重固定为该折真实中文正例有效权重的 0.5；reviewed-negative view 总权重固定为真实中文负例有效权重的 0.25。标准化和缺失值填补只在真实训练行上拟合，所有五个实验共享同一个 reference。

## 10. 评估边界

评估只使用 canonical 中文 train 的五折 seller-component grouped OOF：

- 英文 source train 每折保留；
- 中文 train component 整体 held out；
- synthetic row 跟随 parent seller/component；
- current valid/test、Step16I Dev2 和 Step20 不用于生成规则、权重、模型或阈值选择；
- 主要指标为 AP，同时记录 ROC-AUC；
- 输出每条真实中文 train pair 的五组 OOF scores。

## 11. 晋级条件

正例增强只有同时满足以下条件才视为有效：

```text
AP(same-seller split positive) - AP(no augmentation) >= 0.01
AP(same-seller split positive) - AP(matched positive duplication) >= 0.01
```

完整方法同样必须比无增强和 full-budget duplication 各提高至少 0.01 AP。否则 Step22 冻结为负消融，不进入 Step7/9/15、Step11/17 或 Step20。

即使 OOF 晋级，也只能说明值得冻结后做一次独立真实评估，不能直接支持论文最终有效性结论。

## 12. 输出与 provenance

输出根目录：

```text
reports/step22_same_seller_split/v1_20260716/
```

主要文件：

- `step22_generation_summary.json`
- `step22_generation_manifest.json`
- `pseudo_seller_profiles.jsonl`
- `pseudo_pair_labels.csv`
- `pseudo_pair_lineage.csv`
- `pseudo_item_lineage.csv`
- `pseudo_e5_identifier_redacted.npy/json`
- `step22_grouped_oof_evaluation.json`
- `step22_grouped_oof_predictions.csv`
- `step22_sync_manifest.json`

每条 pair 可追溯到 source seller、source item、source row、parent pair、parent component 和 evidence type。所有 synthetic labels 均强制：

```text
split_name=train
benchmark_eligible=0
usable_for_core_transfer=0
synthetic_train_only=1
```

## 13. Linux 执行

同步代码与 policy 后，在 Linux 项目根目录执行：

```bash
bash scripts/run_step22_same_seller_split_linux_20260716.sh
```

默认使用 CUDA 编码 E5，LR/L2 使用 CPU。CPU 编码方式：

```bash
STEP22_DEVICE=cpu bash scripts/run_step22_same_seller_split_linux_20260716.sh
```

Windows 端不运行原始数据生成、模型编码或数值训练，只进行 Python 静态编译和纯函数契约测试。

## 14. 残余限制

1. 同一 seller 账号的两个 item views 不等价于现实中两个不同账号的马甲关系。
2. 同一 seller 账号可能由多人经营或发生转让；Step22 采用的是“观察账号内 item 共享控制来源”的工作假设。
3. Pseudo profiles 可能仍带有生成器可识别的 item-count 或文本长度分布，因此必须依赖真实 pair OOF，而不能在 synthetic test 上自证。
4. Synthetic 数据可以增加 nuisance variation 和 source-seller diversity，但不能替代真实 cross-account proof positives 或 prospective holdout。
