# Step27：冻结英文源模型的中文半合成残差适配实验计划

版本：2026-07-18，v1 预注册方案
对应配置：`schema/step27_english_pretrained_synthetic_adaptation_policy.json`

## 1. 实验定位

Step27 不改变论文主线。论文主线仍然是：

1. 先在英文地下市场 seller-pair 数据上学习“是否由同一操作者控制”的可迁移规律；
2. 冻结英文源域模型；
3. 使用有限中文 `train` 数据进行目标域适配；
4. 在与训练 seller component 严格隔离的中文数据上判断迁移和适配是否有效；
5. 最终只能在完成新的 Step27-specific 前瞻性冻结后，才考虑用对应 holdout 做一次确认性评估；当前 Step20 只允许准备这套冻结，不允许宣称已经可评估。

Step27 要回答的不是“能否凭空合成更多真实中文马甲”，而是一个更窄、可检验的问题：

> 在英文源评分器完全冻结、中文真实训练边界不变的前提下，来自真实中文父样本的、仅用于训练的半合成视图，是否比等有效权重的简单重复更能改善中文目标域适配？

这个问题保持了“英文训练能力迁移到中文马甲识别”的核心：英文模型始终提供基础 logit；中文训练只学习一个低容量残差。合成数据只参与中文残差训练，不替代英文源训练，也不进入验证集、测试集或最终 holdout。

## 2. 为什么需要 Step27

### 2.1 当前困境

已有项目结果说明：

- 原始多语言语义模型在中文测试上有一定排序能力，但容易受模板复用、同主题商品和公共联系方式噪声影响；
- Step9/Step15 的中文适配能够在部分版本中改善结果，但中文强证据正例仍稀缺；
- 直接扩大模型、LoRA、复杂多任务头、无监督预训练和多轮合成方案均未形成稳定、可确认的提升；
- Step21 的文本增强几乎等同于复制控制，说明“生成更多行”本身不能创造新信息；
- Step24 证明 identifier-redacted E5 单特征英文 LR/L2 可以作为一个定义明确、可冻结和可复现的源评分器。

因此，Step27 不再追求大规模虚构数据，而是进行受控的小样本增强实验。它将“合成内容是否有效”和“只是增加曝光次数/样本权重是否有效”严格拆开。

### 2.2 Step27 能解决什么

如果 M2 稳定超过 M1，说明半合成视图带来的表示变化包含了超出简单重复的训练价值。它可能缓解中文正例少、表示对布局和段落顺序敏感的问题。

### 2.3 Step27 不能解决什么

Step27 不能：

- 把 16 个父正例变成 32 个独立身份关系；
- 证明合成 seller 是真实存在的地下市场操作者；
- 修复错误标签；
- 代替新的强身份正例收集；
- 仅凭当前 `valid` 或 200 条内部 `test` 给出论文确认性结论；
- 证明数据增强一定解决概念漂移。

有效独立样本量仍按真实父 seller component 计算，不能按生成行数计算。

## 3. 冻结的英文源模型

Step27 的基础评分器固定为 Step24 的 source-only E5 LR/L2 control：

```text
artifact file:
reports/step24_content_independent_authorship/v1_20260717/step24_model_artifacts.json

artifact key:
artifacts.source_only.e5_lr_l2_control
```

该模型只使用英文 `train`：

| 项目 | 数值 |
|---|---:|
| 英文训练 pair | 401 |
| Positive | 116 |
| Negative | 285 |
| 输入特征 | `identifier_redacted_e5_cosine` |
| L2 penalty | 10.0 |
| 标准化 mean | 0.911651490484 |
| 标准化 scale | 0.036735755047 |
| intercept | -1.243557552951 |
| coefficient | 0.862429363864 |

Step27 必须按 artifact 文件哈希和 JSON key 加载这组参数。禁止重新训练、校准、微调或根据中文指标改变该模型。

对任一中文 pair，先由冻结源模型得到：

```text
source_logit = frozen_step24_e5_lr_l2(identifier_redacted_e5_cosine)
```

主模型中的 source coefficient 固定为 1，不允许主实验通过学习一个接近 0 的系数绕开英文知识。

## 4. 中文数据边界

Step27 使用当前 canonical `zh_target_strict` 边界，不重新划分：

| Split | Total | Positive | Negative | 用途 |
|---|---:|---:|---:|---|
| `train` | 573 | 229 | 344 | 模型训练和 train-only OOF |
| `valid` | 120 | 30 | 90 | OOF 冻结后才加载为评估目标并生成分数的回顾性开发门槛 |
| `test` | 200 | 50 | 150 | 通过 valid 后才加载为评估目标并生成分数的内部诊断 |

关键证据切片仍然很小。当前 `valid` 仅含 `4` 条 `same_controller_direct_identifier` positive、`0` 条 `same_controller_component_anchor` positive 和 `3` 条 `public_contact_or_url_noise` negative；当前内部 `test` 含 `21` 条 direct positive、`1` 条 component-anchor positive 和 `6` 条 public-noise negative。因此，valid 上的 direct recall/public-noise FPR 条件只是 fail-closed 开发门，不是具有充分统计功效的切片效果证明；test 也只能报告回顾性诊断。Step20 必须在冻结后补充这些关键切片，才能支撑论文中的证据类型结论。

必须继续满足：

- `train`、`valid`、`test` 的 pair 不重叠；
- seller 不跨 split；
- 重新计算的 seller component 不跨 split；
- 不得为了让指标更好重新分配 573/120/200；
- 合成数据不得进入 `valid`、`test`、Step5 freeze 或 Step20。

当前 `valid` 和 `test` 已被历史实验多次观察，因此只能称为 retrospective development/internal diagnostic data，不能再称为全新最终测试集。

这里的“打开”是顺序评估 gate，不表示相关原文或特征在磁盘上完全不存在。构建阶段可以统一清洗各 split 原文、编码冻结 E5，并预计算各 split 的 identifier-redacted pair features。real feature builder 同时保留一个 combined audit 文件，并物理输出 train/valid/test 三个 split 文件。初始训练进程只加载 `train` 的标签和特征；OOF gate 通过后，由单独进程打开 `valid` 标签/特征并生成 valid scores；valid gate 通过后，再由另一个单独进程打开 `test` 标签/特征并生成 test scores。因此，Step27 主张的是进程级 label/feature/score evaluation 被顺序延迟，不主张 valid/test 原文或预计算特征的所有字节完全 unopened。

## 5. 为什么 primary 只使用 16 个 non-silver 正父样本

中文 `train` 虽有 229 个 positive，但其中 213 个是 `silver_train_only=1`，只有 16 个 positive 是 non-silver：

| Review stratum | 父 pair 数 |
|---|---:|
| `identifier_plus_text` | 1 |
| `semantic_structural` | 6 |
| `semantic_only` | 9 |
| 合计 | 16 |

这 16 条按当前重新计算的 seller-component 口径分布在 13 个 component 中。旧 `split_component_id` 口径曾显示约 14 组，但 Step27 的分折、重采样和有效样本量只能认当前 13 组。它们并不全是 proof-level gold positives：只有 1 条属于直接 identifier 支持，其余主要是语义/结构支持。因此，Step27 必须把它们称为“当前冻结监督边界内的 non-silver positive parents”，不能夸大为 16 条直接身份铁证。

Primary track 仅使用这 16 条，是为了避免主结论由大批 silver positive 驱动。每条正父 pair 生成 2 个视图，总计 32 条 synthetic positive views。

## 6. 为什么还要匹配 16 个负父样本

如果只增强 positive，模型可能学到“某种生成痕迹等于 positive”，而不是身份关系。因此必须从真实中文 `train` 中 score-blind 地选 16 个 non-silver reviewed negative parents，并为每条生成同样 2 个视图，总计 32 条 synthetic negative views。

负父样本选择不得读取任何模型分数。匹配变量固定为：

- review stratum family；
- identifier-redacted clean text length bin；
- clean segment count bin；
- clean field missingness pattern。

每个 negative parent 必须从对应 positive parent 的同一个固定 OOF fold 中匹配，二者共享 `matched_set_id`。在同 fold 候选内优先选择 distinct component，并尽量避免重复使用负 component；但实际组件结构包含 giant fold-0 component，若继续绝对排除 positive component，该 fold 的 reviewed-negative 容量可降为 0。因此，当同 fold 的 distinct-component reviewed-negative 容量耗尽时，允许退回同 component 的既有 reviewed-negative，禁止为 fallback 新造标签或把“不同 component”自动推导为 negative。

parent manifest 必须为每个 matched set 记录 `matched_component_relation`，取值为 `distinct_component_preferred` 或 `same_component_fallback`，并按 primary/silver track 报告两类 matched-set 数量。same-component fallback 不是 split 泄漏：component-grouped OOF 会把该 component 的正负父样本及其 descendants 整组放在同一 fold，并在 held fold 时一起排除。但 fallback 也没有增加独立 seller component；计算和报告 parent-component effective sample size 时必须按 component 去重，并单列 fallback 数，不能把一组正负 pair 计成两个独立 component。

因此，每个 seed 的 primary synthetic 上限为：

```text
16 positive parents × 2 variants = 32 synthetic positive rows
16 negative parents × 2 variants = 32 synthetic negative rows
maximum total                    = 64 rows/seed
```

这里的 64 是训练视图数，不是 64 个新身份，也不是 64 条新 gold labels。

## 7. Silver sensitivity 物理隔离

当前中文 `train` 中有 56 条 `silver_direct_or_contact` positive。它们可以用于 sensitivity track，但不得进入 primary track。

Sensitivity 设计为：

- 最多 56 个 silver positive parents，每个 1 个变体；
- 最多匹配 56 个真实 negative parents，每个 1 个变体；
- 最多 112 条 synthetic rows/seed；
- 独立目录、独立 manifest、独立 summary；
- 不与 primary 模型合并；
- 不允许满足任何 primary promotion gate。

这样做是为了回答“若允许 silver direct/contact supervision，增强是否表现不同”，而不是利用银标签人为抬高主结果。

## 8. 半合成视图如何生成

### 8.1 输入文本

只读取以下 clean content fields：

- `category_concat_top`
- `signature_title_concat`
- `title_concat_top`
- `signature_description_concat`
- `description_concat_top`

在任何变换前，必须完整复用 Step15-v7 的 identifier redaction。以下字段禁止进入生成和特征：

- seller 原始账号/别名；
- market/source identifiers；
- contact 字段；
- structured identifier snapshot；
- 未清洗的 `profile_text`。

### 8.2 两个预注册变换

每个父 pair 只生成两个变体：

1. `section_order_rotation`
   - 改变 clean fields/sections 的拼接顺序；
   - 不删除字段内容；
   - 左右 seller 使用由 seed 和 seller UID 决定的独立旋转位置。

2. `segment_order_permutation_with_layout_normalization`
   - 在字段内部重新排列已有 segments；
   - 只规范空白和标点布局；
   - 不加入新词、不跨 seller 拼接、不改变字段归属。

选择这两个非破坏性变换，是因为 15/16 个 non-silver positive 主要依赖软语义/结构证据。激进删除句段或模板 masking 可能直接删掉原标签所依赖的证据，使“标签保持”假设失效。

正负父样本必须使用完全相同的 recipe family 和频率。若变换没有让任一侧文本发生变化，则该变体 fail closed，不允许用原文冒充 synthetic view。

### 8.3 绝对禁止的生成方式

禁止：

- 把同一 seller 随机拆成两个账号并标为跨账号 positive；
- 编造 Telegram、QQ、微信、PGP、邮箱、钱包或 URL；
- 编造市场来源或数据采集 provenance；
- 在不同父 pair 或不同 component 之间拼接内容；
- 把 uncertain、Step11 cluster 或模型高分直接变成标签；
- 因 seller 位于不同 component 就自动标 negative；
- 复制父 pair 的现成 feature row。

## 9. 特征必须重新计算

Step27 不沿用 Step21 的“复制父样本非语义特征，只替换一个 embedding feature”的做法。每条 synthetic view 都要从转换后的 clean text 重新生成 seller representation，再重新计算 pair features。

冻结英文源模型使用：

- `identifier_redacted_e5_cosine`

中文残差仅使用低维、identifier-redacted 特征：

- `clean_token_jaccard`
- `clean_char3_jaccard`
- `clean_title_token_jaccard`
- `clean_description_token_jaccard`
- `clean_category_token_jaccard`
- `clean_text_length_gap_ratio`
- `clean_segment_count_gap_ratio`
- `clean_field_presence_match_fraction`

严禁重新加入：

- `candidate_rule_count_raw`；
- contact/PGP 直接特征；
- 英文 uppercase shortcut；
- 未清洗 profile embedding；
- 由 valid/test 拟合的 IDF、OOV、百分位、缺失值或标准化统计。

所有 imputation、standardization 和权重归一化只能在当前 fold 的训练部分拟合。

## 10. 模型公式

主模型统一为低容量 offset logistic residual：

```text
logit(P_final)
  = logit(P_step24_frozen_english_e5)
  + beta
  + gamma^T x_residual
```

其中：

- `P_step24_frozen_english_e5` 来自冻结英文源评分器；
- source logit 的系数固定为 1；
- `beta` 和 `gamma` 只用中文 fold-train 数据学习；
- residual 使用 L2=10 的 Logistic Regression；
- 不训练 Transformer，不更新 E5，不使用 MLP。

固定 source coefficient 的目的是保证模型确实以英文知识为底座，而不是在中文训练中把英文模型权重学成 0。

另外设置两个 exploratory controls：

1. `step27_m2_learned_source_alpha`：使用与 M2 完全相同的训练行和权重，将冻结英文 source logit 作为一个额外的标准化线性特征学习系数。artifact 同时记录标准化系数和换算后的原始 logit 系数。描述性“接近 0”固定定义为未标准化 `abs(alpha) <= 0.1`；满足该条件会削弱英文知识迁移解释；
2. `step27_m2_target_only_alpha_zero`：使用与 M2 完全相同的训练行和权重，但把 source offset 固定为 0。按同一 split 的十-seed mean real-pair scores 计算 AP；若 `abs(AP(alpha=0)-AP(M2, alpha=1)) <= 0.01`，则描述性地视为 AP 等效，主模型没有证明英文预训练的必要性。

这两个数值界限都只做预注册的 exploratory descriptive diagnosis，不是显著性或正式等效检验，不参与任何 OOF/valid 晋级、模型选择或论文主假设检验。

## 11. M0、M1、M2 的严格对照

### 11.1 M0：真实中文 residual baseline

```text
Frozen English source + all real Chinese train rows
```

M0 不加入 duplicate 或 synthetic row，用来衡量真实中文目标域适配的基础效果。

### 11.2 M1：equal-effective-weight duplication control

```text
M0 + exact duplicates of the selected parent rows
```

M1 使用与 M2 完全相同的父样本、每父行数和总有效权重，但不改变文本或特征。它控制：

- 父样本被模型看见更多次；
- 训练权重增加；
- 训练行数增加；
- 优化器迭代路径变化。

### 11.3 M2：parent-preserving synthetic views

```text
M0 + transformed positive views + matched transformed negative views
```

M2 是 Step27 primary method。

### 11.4 正确的因果比较

Primary comparison 必须是：

```text
M2 vs M1
```

因为只有这个比较能区分“变换内容有价值”和“重复/加权有价值”。

Required secondary comparison 是：

```text
M2 vs M0
```

如果 M2 只超过 M0、却没有超过 M1，结论只能是“增加这些父样本的有效权重可能有用”，不能宣称合成数据有效。

## 12. 权重预算

每个父 pair 的所有 synthetic children 总权重不得超过父行权重的 0.5 倍：

```text
child_weight = parent_training_sample_weight × 0.5 / variants_per_parent
```

Primary 每父 2 个变体，因此每个 child 默认最多为父权重的 0.25。

全体 primary synthetic effective weight 不得超过真实中文 `train` effective weight 的 25%。M1 与 M2 的：

- parent set；
- child count；
- positive/negative weight；
- 每父总权重；
- 每 seed、每 fold 总权重

必须逐项完全一致。

合成行必须继承父样本权重，禁止缺字段时自动按 1.0 计权。禁止跨中英文正例 interpolation。

每次拟合还执行 fold-train 权重归一化。令 `W_real` 为当前拟合中真实中文行的有效权重总和，`W_combined` 为加入 duplicate/synthetic 后的原始总权重，则所有组合行权重统一乘以：

```text
weight_normalization_factor = W_real / W_combined
```

因此归一化后的总有效权重重新等于 `W_real`。OOF 时 `W_real` 只来自当前三个 fold 的真实训练行；全 train artifact 则来自全部真实 `train`。该缩放保持 M1/M2 内部相对权重与二者的严格 parity，同时使固定 L2=10 面对的数据权重尺度与 M0 可比。

## 13. 四折 seller-component OOF

中文 `train` 按重新计算的 seller component 做固定四折：

- fold count：4；
- fold seed：20260718；
- fold manifest 在生成任何 synthetic view 前冻结；
- 父 pair 和全部 descendants 永远属于同一 fold；
- 每个 matched positive/negative parent set 的两条父 pair 必须位于同一 fold；优先 distinct component，容量不足时允许记录为 `same_component_fallback`。无论 relation 如何，留出某一 fold 时相关 component、父样本和 descendants 都会整组排除，不会跨入 held fold；
- 预测某一 held fold 时，只能为其余三个 fold 的父样本生成训练视图；
- held-fold synthetic view 即使预先可计算，也不得进入该 fold 的训练；
- component-disjoint OOF predictions 覆盖全部 573 条真实中文 train rows。

选 4 折而非更多折，是因为 16 个 non-silver positive parents 只覆盖 13 个正 component。四折更容易保证每折有可训练的正负 component，同时避免单折正例过少。

## 14. 十个 seeds 的作用

固定 seeds：

```text
20260320, 20260321, 20260322, 20260323, 20260324,
20260325, 20260326, 20260327, 20260328, 20260329
```

它们只用于：

- 控制变换顺序和左右侧参数；
- 检查优化和生成稳定性；
- 形成每个真实 pair 的 seed-mean score。

禁止：

- 选择表现最好的 seed；
- 把 10 个 seeds 当作 10 个独立数据集；
- 用 seed 间标准差代替 seller-component grouped uncertainty。

统计独立单位始终是真实 seller component。

## 15. 泄漏和捷径审计

训练前必须 fail closed 检查：

1. Step24 artifact 和所有数据文件 hash 完全匹配；
2. 中文边界严格为 573/120/200；
3. train/valid/test 的 pair、seller、component 均无交叉；
4. primary 正父样本恰为 16，按重新计算口径为 13 个 component；
5. 正负父样本均来自合法 train supervision，`matched_component_relation` 取值合法且 primary/silver fallback 数均已报告；
6. 所有 child 与 parent 同 fold，same-component fallback 没有被重复计为额外独立 component；
7. generator 没有读取 valid/test content；
8. clean text 无高精度 identifier residue；
9. 无跨 parent/component 内容拼接；
10. synthetic features 来自重新计算，不是父 feature copy；
11. M1/M2 的 parent 和 effective weight 完全一致。

此外做两个 shortcut classifiers：

- recipe-label predictability AUC 必须不高于 0.60；
- synthetic-vs-real predictability AUC 必须不高于 0.70。

如果生成 recipe 很容易预测 label，说明模型可能只是在识别生成工艺；如果 synthetic 与 real 极易区分，说明增强产生了明显 domain artifact。任一失败都阻断后续评估。

## 16. 晋级门槛

### 16.1 Train OOF 门槛

必须同时满足：

- `AP(M2)-AP(M1) >= 0.02`；
- `AP(M2)-AP(M0) > 0`；
- 10 个 seeds 中至少 8 个 `M2-M1 AP delta > 0`；
- seller-component grouped bootstrap 下界不低于 -0.01；
- direct/component positive recall 下降不超过 0.05；
- template-clone negative FPR 不恶化；
- public-contact/URL negative FPR 不恶化；
- recipe-label AUC 不高于 0.60；
- synthetic-vs-real AUC 不高于 0.70。

这里 bootstrap 下界 -0.01 是 pilot non-degradation guard，不是论文显著性门槛。它不能被解释成已经证明优越。

### 16.2 Single-open valid 门槛

只有 train OOF 全部通过后，才能打开当前 `valid` 一次，并要求：

- `AP(M2)-AP(M1) >= 0.03`；
- `AP(M2)-AP(M0) >= 0.03`；
- direct/component recall 下降不超过 0.05；
- template/public-noise FPR 不恶化。

所有 threshold 在 573 条真实中文 `train` 的十-seed mean OOF predictions 上按 `balanced_accuracy` 预先冻结。打开 valid 后禁止重新选择 threshold。这样 valid 既不参与模型表示选择，也不被用于优化切点；valid 上的 recall/FPR 门槛反映真正的冻结配置表现。

### 16.3 Internal test

只有 valid gate 通过后，才允许在当前 200 条 `test` 上评估一次。不得重新选择模型、threshold 或 seed。该结果只用于内部诊断，不满足论文确认门槛。

### 16.4 Step20

Step20 是未来唯一可能的 confirmatory evaluation，但当前尚未形成可执行的 Step27 evaluation endpoint。当前只允许准备一套新的、版本化的 Step27-specific prospective freeze：

- 必须新建并冻结 Step27-specific policy、代码、M0/M1/M2 model artifacts、OOF thresholds、parent/fold/data manifests 及其 SHA-256；
- 现有面向 Step15-v7 的 Step20 policy 或 freeze manifest 不得授权 Step27 评估；
- prospective 候选的采集/冻结必须发生在 Step27-specific 方法冻结之后，并保持 score-blind；
- 在上述 freeze 完成并通过审计前，不得把 `eligible_for_step20` 或类似字段解释为“已经可以评估”；
- 只有未来被正式授权后才能评估一次，且不允许在 Step20 上重选模型、seed 或 threshold；
- 只有这次未来的 Step27-specific prospective 结果才可能支持正式的外部泛化或方法优越性主张。

## 17. 统计报告

Primary metric：`Average Precision (AP)`。

Secondary metrics：

- `ROC-AUC`
- trapezoidal `PR-AUC`
- `balanced_accuracy`
- `F1`
- `precision`
- `recall`

Primary hypothesis 只有一个：M2 vs M1。M2 vs M0 是按顺序执行的 required secondary gate。learned-alpha、target-only 和 silver sensitivity 都是 exploratory，不参与主晋级，因此不通过挑选其中最好者形成新主结论。

不确定性使用：

- seller-component grouped bootstrap，5000 次，独立固定 base seed `20260718`；
- paired permutation，10000 次，独立固定 base seed `20260719`；
- 每个真实 pair 的 10-seed mean score；
- 同时公开每个 seed 的 delta；
- 报告父 pair/component 数，而不只报告 synthetic row 数；
- 分 track 报告 `distinct_component_preferred` 与 `same_component_fallback` matched-set 数，并按 component 去重计算独立 parent-component ESS。

Bootstrap 与 permutation 必须使用分离的 RNG stream，不得复用同一 base seed；每个输出还必须记录其实际 seed，以便独立重放。

## 18. 输出和可复现性

输出根目录：

```text
reports/step27_english_pretrained_synthetic_adaptation/v1_20260718/
```

实际目录契约如下：

- `parent_manifest/`：输入 hash、573/120/200 canonical pair、四折 component、primary/silver parent manifests，以及分 track 的 `matched_component_relation`/fallback 数；
- `seed_<seed>/primary/` 与 `seed_<seed>/silver_sensitivity/`：生成 lineage、synthetic profiles、synthetic pairs、equal-weight duplication、E5 cache 和完整重算 pair features；
- `pair_features/real/real_pair_features.csv`：包含三个 canonical split 的 combined build/integrity audit 文件，不作为初始训练进程跨 split 加载的入口；
- `pair_features/real/real_pair_features.train.csv`：初始训练和 train-only OOF 唯一加载的真实标签/特征文件；
- `pair_features/real/real_pair_features.valid.csv`：仅在 OOF gate 通过后由独立 valid-scoring 进程打开；
- `pair_features/real/real_pair_features.test.csv`：仅在 valid gate 通过后由独立 internal-test-scoring 进程打开；
- `embeddings/real/`：构建阶段可预计算的 identifier-redacted E5 cache；
- `synthetic_audit/`：leakage、shortcut、lineage 与 effective-weight 合并审计；
- `training/`：仅 `train_oof` 的十 seed predictions、seed-mean predictions、冻结 OOF thresholds、全 train artifacts；
- `statistical_audit/oof_gate/`：打开 valid 前的第一道门；
- `valid_diagnostic/`：只有 OOF gate 通过后才生成的一次性 valid 分数；
- `statistical_audit/valid_gate/`：打开内部 test 前的第二道门；
- `internal_test_diagnostic/` 与 `statistical_audit/final_diagnostic/`：只有 valid gate 通过后才生成，且不参与晋级；
- `manifests/step27_sync_manifest.json`：Linux-to-Windows 的完整文件 allow-list 和 SHA-256。

所有正式输出 immutable。若相同路径已存在但 code/data/policy manifest 不一致，程序必须拒绝覆盖，而不是静默写入新旧混合结果。

## 19. 结果解释矩阵

| 结果 | 允许的结论 | 不允许的结论 |
|---|---|---|
| M2 > M1 且 > M0，并通过 OOF/valid | parent-preserving semi-synthetic views 对当前中文适配有开发价值 | 已在真实新数据上证明泛化 |
| M2 > M0，但 M2 ≈ M1 | 增加父样本曝光/权重可能有用 | 合成变换提供了新信息 |
| M2 < M1 | 变换破坏了有用表示或引入 artifact | 只需生成更多数据即可解决 |
| 未标准化 `abs(learned alpha) <= 0.1` | 描述性证据表明中文目标模型可能绕开英文源模型 | 已证明跨语言迁移 |
| `abs(AP(alpha=0)-AP(M2)) <= 0.01` | 描述性 AP 等效，当前效果未证明依赖英文源知识 | 英文预训练是必要因素或已通过正式等效检验 |
| silver sensitivity 好、primary 不好 | 结果依赖 silver supervision | non-silver 主实验成功 |
| OOF/valid 好、Step20 失败 | 回顾性开发没有外部确认 | 方法具有稳定论文级提升 |

## 20. 预期科研贡献

若 Step27 成功，最合理的贡献表述是：

> 提出并严格评估一种冻结英文源评分器下的 parent-preserving、class-matched、fold-local 中文半合成残差适配方法；通过等有效权重复制控制证明提升来自视图变化而非重复曝光，并在 seller-component 级别控制泄漏和统计依赖。

若 Step27 失败，结果仍有价值：它将进一步说明，在只有 13 个 non-silver positive parent components 的情况下，非破坏性文本视图增强不能替代新的强身份证据。此时应停止扩大合成实验矩阵，把论文重心转向数据可识别性、证据型概念漂移和严格负结果，而不是继续调 threshold 或制造更多相似 synthetic rows。

## 21. 最终原则

Step27 必须始终遵守四条底线：

1. 英文源模型冻结，保证跨语言主线没有被中文 target-only 模型替换；
2. 合成数据只改变训练视图，不创造身份事实；
3. M2 必须超过同权重 M1，才允许声称 augmentation 有效；
4. 当前 valid/test 只做开发诊断；Step20 目前只允许准备 Step27-specific prospective freeze，完成冻结前不得宣称可评估或可确认。
