# Step 15 v5r 加权同域正例插值修复

更新日期：`2026-07-11`

## 1. 修复目的

Step 16G 重跑证明旧 Step 15 v5 的总体指标仍有价值，但进一步审计发现，旧 Phase 4 不能被解释为一个干净的正例数据增强实验。问题不是训练/测试泄漏，而是合成样本构造和权重目标发生了混杂：弱监督父样本被放大为满权重合成样本，中英文正例被直接插值，离散特征被插成不可能的分数值，domain-balanced 版本还把跨域合成样本当成第三个域。

v5r 的目的不是保证指标上升，而是先恢复实验的因果可解释性，使下面的问题可以被真实回答：

> 在固定数据、固定特征、固定模型容量和固定中文测试集条件下，可信的、同域且同证据类型的正例表征插值，是否能在 Phase 3 hard-negative curriculum 之上带来稳定增益？

## 2. 旧版根因

旧 v5 Phase 4 从所有训练正例随机抽两个父样本。当前父样本池中既有英文 full-weight positive，也有中文低权重 silver positive。合成行没有写 `training_sample_weight`，因此下游默认按 `1.0` 计权。三个 seed 中，大多数 synthetic rows 至少包含一个弱父样本，约一半跨英文/中文域，而且绝大多数行在原本离散的共享次数和布尔字段上出现小数。

旧 domain balance 先按原始行数计算域因子，之后才乘 evidence 和 sample quality 权重。Step16G 新增的 115 条中文低权重 negatives 虽然有效质量很小，却按 115 条完整记录改变了整个中文域的平衡系数。跨域合成行又被命名为 `cross_domain_mixup`，导致三域平衡进一步扭曲目标函数。

## 3. v5r 实现

### 3.1 父样本准入

只有同时满足以下条件的训练正例可以成为 mixup parent：

```text
usable_for_core_transfer = 1
core_transfer_eligible = 1
evidence_type_confident = 1
training_sample_weight >= 0.55
```

当前冻结边界下预计有四个可插值组：英文 direct identifier、英文 style/structural soft、中文 direct identifier、中文 style/structural soft。组内必须至少有两个样本。

### 3.2 同域、同证据类型最近邻

锚点父样本先从所有合格正例中采样。第二个父样本只能来自相同 `step15_pool` 和相同 `evidence_type`，并从标准化特征空间中的 5 个最近邻里采样。

这避免了把英文与中文概念直接平均，也避免了把 direct-identifier positive 与 soft structural positive 混成无法解释的中间证据类型。

### 3.3 合成权重

合成样本权重定义为：

```text
w_synthetic = min(w_left, w_right)
```

因此，任何包含 `0.55` 父样本的合成行最多只能获得 `0.55` 权重。合成操作不能提升父证据的可信等级。

### 3.4 特征插值

连续特征仍使用：

```text
z_new = (1 - lambda) * z_left + lambda * z_right
lambda ~ Beta(0.4, 0.4)
```

下列布尔/计数特征不插值，而是从 anchor parent 原样复制：

```text
has_shared_title_clone
has_shared_description_clone
shared_title_count_capped
shared_description_count_capped
shared_category_count_capped
shared_boilerplate_count
shared_low_df_sentence_count
shared_rare_ngram_count
candidate_rule_count_raw
```

这样可防止生成 `0.37 次共享标题` 或 `0.62 个规则命中` 一类脱离真实特征流形的输入。

### 3.5 可追溯 manifest

每个 Phase 4 seed 生成独立 CSV manifest，记录：

```text
synthetic pair_uid
left/right parent pair_uid
left/right domain
left/right evidence_type
left/right parent weight
mixup lambda
inherited synthetic weight
```

manifest 只描述训练期合成行，不写回 Step 5，也不进入 `zh_valid` 或 `zh_test`。

### 3.6 有效权重域平衡

domain-balanced v5r 的权重顺序是：

```text
class-balanced binary weight
-> evidence-type multiplier
-> row training_sample_weight
-> effective domain mass balancing
```

最后一步只允许两个真实域：

```text
en_content_train_pool
zh_target_strict
```

系统按前面所有权重处理后的总质量计算域因子，使两个域的最终 effective weight mass 相等。任何 `cross_domain_mixup` 或未知域都会直接报错。

## 4. 输出隔离

旧 v5 文件不覆盖。新实验和报告均使用 `v5r`：

```text
reports/step15_v5r_weighted_mixup_summary.json
reports/step15_v5r_output_contract_validation.json
reports/step15_v5r_weighted_mixup_slice_level_audit.json
reports/step15_v5r_weighted_mixup_slice_level_audit.csv
reports/step12_v5r_statistical_robustness_zh_test_weighted_mixup_20260711.json
reports/step12_v5r_statistical_robustness_model_metrics_weighted_mixup_20260711.csv
reports/step12_v5r_statistical_robustness_paired_comparisons_weighted_mixup_20260711.csv
```

每个 experiment/phase/seed 的 artifact、prediction 和 mixup manifest 也带完整 `step15_v5r_*` 实验名。

## 5. 评价设计

必须同时检查：

1. `v5r Phase4 - v5r Phase3`：正例插值的直接增量效果。
2. `v5r - legacy v5`：实现修复是否改变旧结论。
3. `v5r - raw E5`：新方法相对原始语义排序基线的整体价值。
4. `v5r domain-balanced - v5r non-domain`：修复后的域平衡是否仍有收益。
5. public-contact、template、topic、direct/component positive 等切片分数。
6. grouped bootstrap 的 ROC-AUC 和 AP 差异及 95% CI。

仅当 Phase 4 相对同配置 Phase 3 有稳定增益时，才能把提升归因于 positive mixup。若 v5r 总体仍强但 Phase 4 不优于 Phase 3，则 Step 15 的主要贡献应归因于 evidence-type hard-negative curriculum，而不是数据增强。

## 6. 当前状态

实现、policy、独立输出命名、Step 12 配对比较和单元测试已完成。三个本地单元测试全部通过；Windows 未运行模型实验。当前状态是等待 Linux 三 seed 重跑并同步结果。

Linux runner 会在统计审计前调用 `scripts/step15_validate_v5r_outputs.py`。该验证器逐一检查 6 个 Phase-4 run 的 artifact 和 parent manifest；一旦发现跨域/跨证据父样本、错误继承权重、manifest 行数不一致或两域有效质量不相等，流水线立即失败，不继续生成 Step 12 结论。
