# Step15-v7 两阶段概念漂移修复与前瞻性评估设计

更新时间：2026-07-14

当前实现版本：`v2_identifier_redacted_20260714`。该版本与早期尚未正式运行的 v1 路径物理隔离，避免不同特征维度产物相互覆盖。

## 1. 文档定位

本文档定义 Step15-v7 的完整科研设计、实现边界、统计检验和前瞻性评估纪律。v7 不是对 Step15-v6 的同一测试集继续调权重，而是一条隔离的新方法链。当前 Windows 工作区只完成代码、数据谱系和契约验证；模型训练、统计重采样和正式结果生成必须在 Linux 服务器执行。

当前不能宣称 v7 有性能提升。v7 的效果只能由 Linux 生成的 Step9-v7、Step15-v7、Step12-v7 结果，以及未来一次性的 Step20 前瞻性留出集结果决定。

Step20 的盲审队列同样 fail-closed：只有达到预注册的 `400` 条 seller-disjoint 候选总量，并同时满足 direct identifier、public-contact noise、template clone、semantic topic 和 ordinary 五类候选配额，才会发布正式双人盲审队列。少量新候选不会被包装成一个统计上不足的 prospective holdout。

## 2. 为什么冻结 Step15-v6

Step15-v6 已完成严格方法审计，但 Step12-v6 的论文晋级条件为 `eligible=false`。这意味着：

1. v6 是应保留的严格负结果，不能删除或用新输出覆盖；
2. 旧的 200 条中文测试边已经多轮参与方法诊断，只能称为 `internal development test`；
3. 不能继续根据这 200 条边的表现修改特征、权重、损失或阈值；
4. v7 的模型选择只能使用重新构建的代表性中文验证集；
5. 论文最终确认必须来自 v7 冻结后新收集、盲审并只评估一次的前瞻性留出集。

冻结配置为 `schema/step15_v6_negative_freeze.json`，冻结程序为 `scripts/step15_freeze_v6_negative_result.py`。程序对 v6 选中模型、Step12 `eligible=false`、所有冻结文件及其 SHA-256 进行核验。相同内容可重复验证，不同内容拒绝覆盖。

## 3. v7 要解决的具体问题

### 3.1 strict-clean 特征仍存在语料参考泄漏和 OOV 语义错误

旧的部分稀有度、模板率和反文档频率特征在完整语料上计算，可能让验证或测试卖家参与参考统计。另一个问题是：两个评估卖家共享一个训练阶段从未出现的字符串时，旧逻辑可能把训练文档频率当成 0，再把它解释成极稀有强证据。这会人为抬高 OOV 克隆边。

### 3.1.1 旧 semantic cache 并不是真正的 clean input

代码复核确认，canonical Step7 的 seller `profile_text` 明确拼接了 `[SELLER]`、`[CONTACTS]` 和结构快照；商品标题或描述中也可能出现 Telegram、邮箱、钱包和卖家别名。因此旧的 E5、BGE-M3、LaBSE、GTE、MPNet 与 reranker 分数虽然未显式读取 pair-level identifier 列，却可能通过文本编码间接携带身份信号。若 Stage A 使用这些分数，再声称 Stage A 与 identifier veto 信息隔离，在方法上不成立。

v2 新建 identifier-redacted Multilingual-E5 cache：只读取类别、标题和描述正文，排除 seller alias、market、contact、structured snapshot 和完整 `profile_text`，再利用 Step3 occurrence literals、高精度联系方式正则以及 seller alias literals 做空格替换。替换不插入 `[CONTACT_REMOVED]` 等标记，避免模型反向利用“存在联系方式”这一事实。旧 semantic scores 保留在 canonical 表中供历史诊断，但不进入 v7 clean feature list。cache manifest 还对本地 E5 模型目录逐文件记录内容指纹；未来 Step20 编码若模型权重、producer 脚本或 v7 policy 任一变化，prospective feature 构建会拒绝运行。

### 3.2 原验证集缺少关键证据切片

原中文验证集中的直接身份正例、连通分量锚点正例和公共联系方式噪声太少，无法支持模型选择。特别是 public-contact/URL noise 在验证侧不足，却曾在测试侧用于诊断，容易形成测试知情调参。

### 3.3 clean 模型无法预测不可见的 identifier evidence type

如果 clean scorer 的输入明确排除了直接联系方式，辅助头却被要求判断“直接身份标识正例/公共联系方式噪声”，任务在信息论上不可辨识。辅助任务可能迫使模型用模板或主题相关特征猜测身份类型，反而污染主任务。

### 3.4 全局证据倍权不区分域、证据类型和置信度

统一的 8 倍权重会把低置信软正例、直接锚点、英文源域样本和中文目标域样本同等放大，容易让少量大连通分量或软标签支配损失。

### 3.5 旧 mixup 不能证明插值本身有效

如果仅增加合成正例而没有等有效权重复制对照，就无法区分：

- 只是正例总权重增加带来的效果；
- 潜在表示插值改变正例局部几何带来的效果。

此外，若同时启用自动 class balancing，增强前后正负类都会再次被自动配平，mixup 的少数类作用会变得难以解释。

### 3.6 当前测试集不能再承担最终论文确认

当前 200 条边在 v2-v6 期间已经被反复观察。即使没有把标签直接放入训练，也存在 test-informed method selection bias。必须新建方法冻结后的 prospective final holdout。

## 4. 当前数据边界

代表性验证构建器根据所有可监督中文 pair 重新计算 seller graph connected components，不信任历史 `split_component_id`。只允许把完整训练连通分量移动到验证集；原测试集成员不移动，只重命名为内部开发测试。

根据当前冻结标签与开发期 split 预演得到的预期规模如下；Linux runner 必须重新生成 manifest 并核验，以下数字不是新的正式运行结果：

| Split | Rows | Positive | Negative | 科研角色 |
|---|---:|---:|---:|---|
| `train` | 523 | 183 | 340 | v7 中文支持训练 |
| `valid` | 170 | 76 | 94 | 唯一模型/阈值选择集 |
| `internal_development_test` | 200 | 50 | 150 | 仅诊断，不允许晋级论文结论 |

代表性 `valid` 证据类型：

| Evidence type | Rows | Distinct seller components |
|---|---:|---:|
| `same_controller_direct_identifier` | 18 | 10 |
| `same_controller_component_anchor` | 16 | 5 |
| `same_controller_style_structural_soft` | 42 | 26 |
| `public_contact_or_url_noise` | 6 | 2 |
| `ordinary_negative` | 46 | 10 |
| `semantic_topic_not_controller` | 16 | 8 |
| `template_clone_not_controller` | 26 | 7 |

构建过程从旧 train 完整转移 12 个 seller components。最终 seller overlap 和 component overlap 都为 0。选择规则不读取任何模型分数。

## 5. strict-clean 20 维特征

### 5.1 保留特征

v7 clean scorer 使用以下 20 个特征：

1. `embedding_cosine_multilingual_e5_large_identifier_redacted`
2. `profile_category_jaccard`
3. `has_shared_title_clone`
4. `has_shared_description_clone`
5. `shared_title_count_capped`
6. `shared_description_count_capped`
7. `shared_category_count_capped`
8. `shared_title_idf_sum`
9. `shared_description_idf_sum`
10. `shared_title_idf_mean`
11. `shared_description_idf_mean`
12. `item_count_percentile_gap_abs`
13. `price_median_percentile_gap_abs`
14. `title_length_median_percentile_gap_abs`
15. `description_length_median_percentile_gap_abs`
16. `digit_ratio_mean_percentile_gap_abs`
17. `punct_ratio_mean_percentile_gap_abs`
18. `repeated_title_share_percentile_gap_abs`
19. `repeated_description_share_percentile_gap_abs`
20. `max_category_share_percentile_gap_abs`

### 5.2 删除特征

以下字段不再进入 strict-clean 主模型：

- `boilerplate_ratio_max`
- `boilerplate_ratio_gap_abs`
- `shared_boilerplate_count`
- `shared_low_df_sentence_count`
- `shared_rare_ngram_count`
- `candidate_rule_count_raw`
- 英文特定 uppercase 特征
- retrieval-only raw lexical/structural 字段
- 任何直接 identifier 字段
- 由旧 `profile_text` 编码得到的 `embedding_cosine_multilingual_e5_large`
- `embedding_cosine_bge_m3 / embedding_cosine_labse / embedding_cosine_gte_multilingual_base`
- `embedding_cosine_paraphrase_multilingual_mpnet / reranker_score_gte_multilingual_reranker_base`

`candidate_rule_count_non_identifier` 仍可作为诊断字段生成，但不进入 20 维主模型。这样不会减少训练样本，只改变输入列。旧语义列也仍保留在输出表中，但 manifest 明确标记为 diagnostic-only。

### 5.3 训练语料参考和 OOV 规则

所有文档频率、模板率、稀有度和市场分位数只在 v7 `train` sellers 上拟合。`valid` 和内部开发测试只应用被冻结的参考量。

共享 OOV signature 的规则为：

```text
effective_df = max(train_df, 2)
idf = log((train_seller_count + 1) / (effective_df + 1)) + 1
```

当 `train_df=0` 时，该 signature 被记录为 OOV diagnostic，但不计入 train-supported rare evidence。数值缺失值使用当前实验 real train rows 的逐列中位数填补；不使用 valid/test 中位数。

feature builder 会在 Linux 上检查 20 个配置特征在 EN+ZH combined train 上均非恒定；任何恒零、全缺失或 OOV-only 配置列都会使运行失败，而不是静默进入模型。

## 6. 潜在 seller-pair 表示

使用冻结的 Multilingual-E5-Large 对 identifier-redacted seller content 生成 embeddings。该 encoder 不做监督微调，也不读取 split label；对每个无序 seller pair 构造：

```text
z_pair_raw = concat(abs(z_left - z_right), z_left * z_right)
```

该形式在交换左右 seller 后保持完全相同。为控制小样本模型维度，使用固定随机种子 `2026071301` 的 Gaussian Johnson-Lindenstrauss projection 投影到 64 维：

```text
z_pair = z_pair_raw @ R
R_ij ~ Normal(0, 1 / sqrt(64))
```

投影矩阵不训练、不看标签、每次可确定性重建。最终 clean scorer 输入为 `20d strict-clean + 64d projected pair latent = 84d`。20 维中的脱敏 E5 cosine 与 64 维 latent 来自同一脱敏缓存，但前者表达全局相似度，后者保留差值/乘积的局部方向信息；LR/L2 决定是否利用后者。

## 7. 因子化样本权重

每个真实训练 pair 的权重为：

```text
w = clip(
    domain_factor
    * evidence_type_factor[domain]
    * confidence_factor
    * component_factor,
    0.1,
    2.5
)
```

其中：

- `domain_factor`：当前 EN、ZH 均为 1.0，不人为制造全局域倍率；
- `evidence_type_factor`：按域和证据类型分别设置，例如 ZH direct positive 1.25、ZH public noise 1.5、ZH soft positive 0.7；
- `confidence_factor`：来自冻结标签的 `training_sample_weight`，裁剪到 `[0.18, 1.0]`；
- `component_factor`：`1/sqrt(component pair count)`，再在每个 `domain × evidence_type` 层内均值归一化；它只抑制同类证据中的重复大分量，不改变该证据类型的整体平均质量；
- 最终权重上限 2.5，代码硬拒绝任何 8 倍权重回归。

证据类型仅用于训练权重和评估切片，不进入模型特征。

## 8. Step9-v7 三组匹配实验

每个 support ratio `0.0 / 0.1 / 0.2 / 0.5 / 1.0`、每个 seed `20260320..20260329` 都运行三组：

1. `no_augmentation`
2. `equal_effective_weight_duplication`
3. `latent_pair_embedding_mixup`

三组使用完全相同的真实 EN train 和抽样 ZH train。`0.0` 是同一 84 维 feature view、只使用英文监督标签的 source-only clean fusion；它是目标域适配的匹配排序对照。support ratio 越高，加入的中文训练支持越多；英文训练集不变。0% 下三组理论上应完全相同，Step12 会检查十个 seed 的 source-only 分数一致性；若不一致说明实现引入了隐藏随机性。需要严格区分：该模型的 ROC-AUC/AP 是 source-label-only ranking；其 threshold-based 指标仍统一使用代表性中文 valid 冻结阈值，因此不能称作“完全不接触目标域验证信息的 strict zero-shot classification”。

中文 support 不是对整个 train 做一次不分层的精确百分比截断，而是在 `review_label × evidence_type` 内使用同一 deterministic rank 分层截取。任意正比例对每个非空层至少保留 1 条，因此小比例下实际总行数可能略高于 `ratio × |ZH train|`；相同 seed 的 `10% ⊆ 20% ⊆ 50% ⊆ 100%`。每个 artifact 必须记录实际中文行数、分层计数及 sampled pair UID 哈希，论文表格使用实际行数，不能把 `10%` 误写成严格等于总训练集的 10%。

### 8.1 正例父样本约束

合成父样本必须：

- 仅来自 ZH train；
- `review_label=positive`；
- `training_sample_weight >= 0.55`；
- 两个父样本属于同一语言域；
- 两个父样本属于同一 evidence type；
- partner 从 projected latent space 中最近的 5 个合格邻居选择；
- 每个 eligible parent 最多生成 5 条 synthetic rows；
- 增强目标只按当前抽样的中文支持集计算，关闭其中 `negative effective weight - positive effective weight` 差额的 50%；英文源域规模不决定中文合成预算，也不用大量合成样本强行完全配平。

### 8.2 潜在表示 mixup

```text
lambda ~ Beta(alpha=0.4, alpha=0.4)
z_new = (1-lambda) * z_anchor + lambda * z_partner
```

只有 64 维 latent 表示插值。20 维 clean 特征从 anchor 原样复制，避免对二元、计数或语料统计特征做无物理意义的线性插值。合成权重从两个父样本因子化权重的较小值开始，最后一条会截断到剩余预算，使 synthetic effective weight 不越过预注册的 50% gap-closure。100% support 主比较若无法满足该预算会 fail-closed；较低 ratio 则如实报告不足，不能描述成已经平衡。

### 8.3 等有效权重 duplication control

duplication 使用与 mixup 完全相同的 anchor、partner、数量和 synthetic weight，但 latent 直接复制 anchor：

```text
z_new = z_anchor
```

代码要求两组 synthetic effective weight 的绝对差不超过 `1e-10`。因此：

- duplication 对 no augmentation 的差异估计“增加正例有效权重”的作用；
- mixup 对 duplication 的差异估计“潜在几何插值”的额外作用。

### 8.4 为什么关闭自动 class balancing

v7 LR/L2 设置 `class_weight=none`。若先自动把正负类配平，再加入正例增强，增强的类别不平衡作用会被第二次配平掩盖。v7 让 duplication/mixup 本身承担正例有效质量补充，并用严格复制对照隔离机制。

LR/L2 配置为 `l2_penalty=10`、`max_iter=200`、标准化开启。每个 ratio/seed 只用真实 EN+ZH train rows 拟合一次 scaler，三组对照共享该 scaler；synthetic rows 不参与均值和方差拟合。三组优化器的总 sample weight 都归一到相同的真实训练行数，从而避免“增加 synthetic rows 同时削弱相对 L2 正则”这一混杂因素。实际合成条数、目标域 gap closure fraction、duplication/mixup 有效权重差和跨域父样本数必须由 Linux 正式 summary 报告，不再引用开发期预演数字。

## 9. Step15-v7 两阶段方法

### 9.1 Stage A：clean cross-lingual ranker

Stage A 从 Step9-v7 的 100% support 三组中选择代表性 `valid` 上十个 seed 的平均 AP 最优者。选择不读取内部开发测试。

Stage A 不存在 evidence-type auxiliary head。它只输出 `P(same_controller)`，负责跨语言候选排序。

### 9.2 Stage B：identifier/reliability veto

Stage B 不是训练标签头，而是只读取推断时可见的 Step3 原始 occurrence context：

- 两侧均为 seller-facing、direct-identity-eligible；
- 非 product/victim-data context；
- 非 support-only；
- token 在 v7 中文训练卖家参考中的 seller frequency 不超过 3；

满足以上条件时记录 `verified_seller_facing_direct`，分数不强制抬高。若共享 token 来自产品数据、客服/支持上下文或高频公共标识符，则记录 `public_or_product_contact_veto`：

```text
P_final = 0.1 * P_clean
```

公共频率参考只由 v7 中文 train sellers 拟合；valid、内部开发测试和未来 prospective sellers 不参与频率统计。未知或混合上下文不改变分数，只输出 ambiguous flag。Stage B 禁止读取：

- `review_label`
- `evidence_type`
- 模型错误信息
- 是否属于 zh_test

这种结构明确分工：clean scorer 负责召回和排序，reliability veto 负责压低 inference-visible 公共联系方式噪声。

## 10. Step12-v7 统计审计

Step12-v7 评估以下模型：

- raw identifier-redacted E5 cosine
- English-label-only source-only clean fusion
- no augmentation
- equal-effective-weight duplication
- latent pair-embedding mixup
- validation-selected clean scorer
- two-stage veto scorer

主要比较：

1. source-only clean fusion vs raw identifier-redacted E5 cosine；
2. validation-selected clean scorer vs source-only clean fusion；
3. mixup vs equal-weight duplication；
4. mixup vs no augmentation；
5. duplication vs no augmentation；
6. validation-selected clean scorer vs raw identifier-redacted E5 cosine；
7. two-stage veto vs selected clean scorer。

Primary metric 为 `average_precision`，secondary metrics 为 `roc_auc` 和真正梯形积分的 `pr_auc`。阈值指标包括 `accuracy / balanced_accuracy / F1 / precision / recall / specificity`，阈值只在代表性 valid 上冻结。

统计方法：

- 按 `v7_component_id` grouped bootstrap，5000 次；
- 相同 seed ID 时做 seed + component 两层 bootstrap；
- 对完整 component score block 做 paired score-swap permutation，5000 次；
- 最多 24 个确定性 CPU workers；
- MAP/MRR 不报告，因为没有预注册 query groups。

内部开发测试的所有结果都标为 diagnostic。Step12-v7 的 publication promotion 固定为 `false`，但会在解封任何 Step20 标签前生成 model/threshold freeze manifest。

## 11. Step20 前瞻性最终留出集

### 11.1 当前现实状态

开发期数据谱系检查显示，现有 strict queue 中：

- 1,016 条已经审核；
- 2,841 条候选在 v7 冻结前已存在；
- 可作为 prospective final holdout 的冻结后新增候选为 0。

因此当前不能伪造“独立最终测试”。必须在 v7 模型与阈值冻结后新增原始卖家/商品数据，并生成 `reports/step20_prospective_holdout/new_data_candidate_queue.csv`。盲审开始前，prospective Step3 seller profiles、Step3 identity signals 和 Step4 candidate pairs 必须齐全并被哈希冻结；它们提供完整的原始身份与文本证据。标签冻结后再接入 canonical Step7 features 和 E5 cache 做密封评分。每条新候选必须符合 `schema/step20_prospective_candidate_schema.json`：语言必须为中文，来源域必须是中文暗网/地下市场，左右原始记录 ID、来源引用和内容 SHA-256 不能为空；同时携带带时区的 `collection_timestamp_utc`，且严格晚于 Step12 模型冻结清单中的 `frozen_at_utc`。配置布尔值或手写市场名不能替代来源与时间证据。

### 11.2 双人盲审

候选准备程序：

- 排除任何已出现在历史 train/valid/internal-test 的 seller；
- 选择的 prospective pairs 彼此 seller-disjoint；
- reviewer 只能看到原始 evidence/context，不得看到模型分数、历史标签或图归属；
- reviewer queue 不显示候选类别和原始 review stratum，避免 `public_noise_candidate` 等名称暗示预期标签；
- reviewer A/B 独立审查；
- 分歧必须由第三位 adjudicator 处理；
- positive 只接受 direct seller identifier 或 component anchor；
- uncertain 不进入二分类评估。

最终冻结最低要求：

| 类型 | 最低数量 |
|---|---:|
| Positive total | 30 |
| Direct/component positive | 20 |
| Public-contact/URL negative | 20 |
| Template-clone negative | 30 |
| Semantic-topic negative | 30 |
| Ordinary negative | 30 |

候选队列目标 400 条是覆盖诊断目标，不是最终冻结的硬门槛；最终证据类型最低数量才是冻结硬门槛。

### 11.3 冻结参考特征变换

新数据必须提供 prospective Step3 profiles/signals、Step4 pairs、canonical Step7 features。标签冻结后，单独运行同一个 identifier-redacted E5 cache builder；`step20_build_prospective_features.py` 只应用 v7 已冻结的训练参考，绝不在 prospective data 上重新拟合 IDF、市场分位数或缺失值策略。它从脱敏缓存生成新 cosine；canonical Step7 的旧 `profile_text` semantic scores 不进入 v7 模型，也不再被当作 clean-cache 一致性基准。

### 11.4 标签密封评分与一次性解封

冻结程序同时发布一个不含标签的 `frozen_pair_universe.csv`。`step20_build_prospective_features.py` 和 `step20_score_prospective_holdout.py` 只读取该 universe 中的 pair UID、seller endpoints 和采集时间，物理上不打开 `frozen_holdout_labels.csv`。打分程序用冻结 artifact 和 valid threshold 生成六个模型的 score 文件；真正的 `review_label/evidence_type` 直到一次性评估锁创建后才解封。

`step20_evaluate_prospective_holdout.py` 在读取标签前以原子 `O_EXCL` 创建：

```text
reports/step20_prospective_holdout/evaluation_v2/EVALUATED_ONCE.lock.json
```

如果执行中断，lock 仍保留，禁止静默重跑。成功后 lock 变为 `evaluation_complete_never_rerun`。评估不做任何模型或阈值选择，只报告 point metrics、seller-disjoint paired bootstrap CI 和 paired permutation test。

## 12. 输出隔离和防覆盖

v7 输出不覆盖 v6、v5 或 canonical Step7/9/15：

- Step15-v7 clean cache/features/splits：`reports/step15_v7/v2_identifier_redacted_20260714/`
- Step9-v7：`reports/step9_v7_latent_mixup/v2_identifier_redacted_20260714/`
- Step15-v7：`reports/step15_v7/two_stage/v2_identifier_redacted_20260714/`
- Step12-v7：`reports/step12_v7/v2_identifier_redacted_20260714/`
- Step20：`reports/step20_prospective_holdout/`，其中 preparation/freeze/features/scores/evaluation 分别进入独立的 `*_v2` 子目录

所有正式 writer 均拒绝不同内容的同路径覆盖。代表性 validation、v7 features、Step9、Step15、Step12 以及 Step20 的 preparation/freeze/features/scores 都先写隐藏的 `.incomplete` staging directory，只有该阶段全部产物成功后才原子发布正式目录；中途失败不会留下可被误读为完整结果的正式目录。Step20 evaluation lock 使用一次性 `O_EXCL` 文件；一旦开始解封标签，即使中断也不能静默重跑。

## 13. Linux 执行流程

运行前要求 Linux 已具备当前项目环境中的 `numpy`、`torch`、`transformers`，并存在本地模型目录 `models/step7/embeddings/multilingual_e5_large/`。只有 identifier-redacted E5 cache 阶段使用 GPU 推理；LR/L2、veto 和 Step12 重采样主要使用 CPU。runner 不联网下载模型，也不会在 Windows 执行。

### 13.1 v7 核心实验

```bash
cd /home/yongpeng/cross-lingual
bash scripts/run_step15_v7_linux_20260714.sh
```

该脚本依次执行：语法/配置检查、v6 negative freeze、代表性 validation、identifier-redacted E5 cache、v7 train-only 20d features、Step9-v7 150 组训练（其中 0% support 提供 source-only sanity/control）、Step15 两阶段评分、Step12-v7 统计审计及 model freeze。

### 13.2 Step20 候选准备

```bash
cd /home/yongpeng/cross-lingual
bash scripts/run_step20_prospective_holdout_linux_20260714.sh prepare
```

完成 A/B 两份 review queue 和必要 adjudication 后：

```bash
bash scripts/run_step20_prospective_holdout_linux_20260714.sh freeze-and-score
```

最终一次性评估必须显式确认：

```bash
export CONFIRM_ONE_TIME_PROSPECTIVE_EVALUATION=YES
bash scripts/run_step20_prospective_holdout_linux_20260714.sh evaluate-once
```

## 14. 成功与失败判定

v7 的结果必须按以下顺序解释：

1. 若 mixup 不优于 equal-weight duplication，则不能声称潜在插值有效；最多说明增加正例权重有效或整个增强无效。
2. 若 two-stage 只降低 public-noise score，却显著损伤 direct/component positive recall，则 veto 不合格。
3. 若内部开发测试提升但 prospective holdout 不提升，论文结论必须以 prospective negative result 为准。
4. 若 Step20 正例证据最低数量不足，停止最终性能声明，不用 soft positive 填补 gold holdout。
5. 只有 prospective AP/ROC-AUC/PR-AUC 的配对区间和置换检验支持，并且关键 evidence slices 没有明显损伤，才允许写入主方法结论。

## 15. 当前完成状态

已完成代码实现和 Windows 端静态审查；Windows 不运行项目脚本，以下契约必须由 Linux runner 在训练前重新验证：

- v6 strict-negative freeze；
- seller-component-safe representative validation；
- identifier-redacted E5 cache builder；
- OOV-safe 20d feature builder；
- symmetric 64d pair latent representation；
- domain × evidence type × confidence × component weighting；
- no-augmentation / duplication / latent mixup 三组对照；
- clean scorer + reliability veto；
- Step12-v7 grouped/bootstrap/permutation/two-level audit；
- Step20 post-freeze 时间溯源、dual review、无标签 pair universe、fail-closed freeze、frozen-reference feature transform、sealed scoring、one-time evaluation；
- Linux runners 和契约测试。

尚未完成：

- Linux v7 模型训练与统计结果；
- v7 冻结后新增中文原始数据；
- Step20 双人审查和最终一次性评估。

在这些结果返回前，v7 是一个实现完成、效果未知的预注册实验方案，不是已证明优于 v6 的方法。
