# 当前实验设计说明

更新日期：`2026-05-22`

本文档说明当前项目的实验设计、设计目的、实现细节、有效数据边界和当前结论边界。它不是历史流水账，而是当前可以复现实验和撰写论文方法部分时应遵循的实验设计说明。

## 1. 研究问题定义

本项目研究的是暗网市场中的 seller-pair verification，也就是给定两个卖家账号，判断它们是否可能由同一操作者控制。

当前实验不是做英文账号与中文账号之间的一对一匹配。当前实际任务是：

1. 在英文市场构造源域监督数据，训练 seller-pair 判别函数。
2. 把这个判别函数零样本迁移到中文市场内部的卖家对。
3. 用少量中文监督数据做目标域 few-shot adaptation。
4. 把 pair-level 打分投影到中文候选图中，生成候选簇。
5. 对候选簇做严格证据审计，区分同控制证据、模板复用、主题相似和不确定候选。

因此，“跨语言”在本项目中指的是从英文源域到中文目标域的判别规律迁移，而不是跨语言实体对齐。

## 2. 总体实验逻辑

项目采用从数据边界到模型再到图审计的流水线：

| Step | 目的 | 核心输出 |
| --- | --- | --- |
| Step 2 | 固定数据域和泄漏排除规则 | `reports/step2_split_summary.json` |
| Step 3 | 构造 seller profile 和 item-level identity signals | `reports/step3_seller_profiles.*.jsonl` |
| Step 4 | 构造候选 seller-pair | `reports/step4_*_silver_candidate_pairs.csv` |
| Step 5 | 人工/规则辅助审查并冻结监督标签 | `reports/step5_*_frozen_silver_labels.csv` |
| Step 7 | 英文源域训练 zero-shot pair verifier，并在中文测试 | `reports/step7_training_summary.json` |
| Step 9 | 中文少样本适配和 calibration 控制实验 | `reports/step9_few_shot_summary.json` |
| Step 11 | 把 scorer 投影成中文候选图和候选簇 | `reports/step11_current_manifest_20260517.json` |
| Step 12 | 固定中文测试集上的统计稳健性审计 | `reports/step12_v2_statistical_robustness_zh_test_20260602.json` |
| Step 13 | 概念漂移与切片诊断 | `reports/step13_concept_drift_audit.json` |

实验设计的核心原则是：

1. 先固定数据边界，再训练模型。
2. Clean scientific line 不使用直接 identifier 特征。
3. Identifier augmented 只能作为 operational control。
4. 中文 `zh_test` 是固定测试集，不能与 train/valid 随机混合。
5. Step 11 只生成候选簇，不把聚类结果当 ground truth。
6. 同控制结论必须有 seller-facing identity proof，例如 Telegram、PGP、Jabber、QQ、微信、电话、钱包等直接证据；商品内容、模板、URL、受害者数据 email 不能直接作为同控制证明。

## 3. 数据域设计

### 3.1 Step 2 数据切分目的

Step 2 的目的不是训练模型，而是定义哪些原始数据属于源域、目标域、辅助域，并排除可能泄漏 benchmark 身份的内容。

当前数据域：

| 数据域 | 用途 | 当前规模 |
| --- | --- | --- |
| `en_content_train_pool` | 英文源域训练池 | `311019` items, `7522` sellers |
| `zh_target_strict` | 严格中文目标域 | `17556` items, `5097` sellers |
| `zh_target_aux` | 中文辅助池，不进入严格 benchmark | `2104` items, `673` sellers |

严格中文目标域只包含两个 market：

| 中文市场 | item 数 |
| --- | ---: |
| `中文暗网交易市场` | `9487` |
| `茶马古道` | `8069` |

英文源域做了严格排除：

- 英文原始 item 数：`459669`
- 英文 eligible item 数：`311019`
- 因 alias overlap 排除的英文 item：`147475`
- 因非目标中文污染排除的英文 item：`1175`
- 英文 post-filter alias overlap：`0`
- 英文 post-filter aux fingerprint overlap：`0`

这样设计的原因是：如果英文训练池中混入 benchmark 或目标域污染，后面的 zero-shot transfer 会变成泄漏实验，而不是迁移实验。

实现文件：

- policy: `schema/step2_split_policy.json`
- runner: `scripts/step2_build_split_manifests.py`
- summary: `reports/step2_split_summary.json`

### 3.2 Step 3 seller profile 设计

Step 3 把 item-level 原始记录压缩为 seller-level profile。每个 profile 包括：

- seller uid
- market/source 信息
- 标题、描述、类目聚合文本
- item 数量、价格统计、文本长度统计
- 风格统计，例如数字比例、标点比例、重复标题比例、重复描述比例
- contact / PGP / Telegram / email / phone / wallet 等 item-level identity signals

当前 profile 数：

| 数据域 | seller profiles | item 数 |
| --- | ---: | ---: |
| `en_content_train_pool` | `7522` | `311019` |
| `zh_target_strict` | `5097` | `17556` |
| `zh_target_aux` | `673` | `2104` |
| total | `13292` | `330679` |

Step 3 同时输出 item-level identity signals。当前总计：

- identity signal 总数：`298775`
- direct-identity-eligible signals：英文 `184650`，中文 strict `1890`，中文 aux `89`
- seller-facing signals：英文 `253065`，中文 strict `3300`，中文 aux `106`
- product-data-risk signals：英文 `67155`，中文 strict `1844`，中文 aux `43`

identity signal 类型包括：

| 类型 | signal 数 | direct-identity-eligible 数 |
| --- | ---: | ---: |
| `wickr` | `112309` | `89746` |
| `phone` | `62616` | `49485` |
| `external_url` | `43661` | 不作为直接身份锚点 |
| `telegram` | `42405` | `37822` |
| `email` | `36652` | `8741` |
| `crypto_wallet` | `305` | `251` |
| `jabber` | `374` | `297` |
| `qq` | `133` | `76` |
| `wechat` | `74` | `40` |
| `pgp_public_key` | `78` | `68` |
| `bat` | `168` | `103` |

重要纪律：Step 3 的 identity signals 只是抽取证据，不是标签。共享 URL、email 或联系方式必须结合上下文判断是否是 seller-facing 身份锚点；如果只是商品样本、受害者数据、外部数据源或 parser noise，不能标为 positive。

实现文件：

- schema: `schema/step3_seller_profile_schema.json`
- runner: `scripts/step3_build_seller_profiles.py`
- summary: `reports/step3_seller_profile_summary.json`

### 3.3 Step 4 候选 pair 设计

Step 4 的目标是从所有 seller 中筛出值得审查或建模的候选卖家对。它不是枚举所有两两组合，而是根据规则生成候选边。

候选生成规则包括：

- profile lexical neighbor
- shared contact exact
- shared description clone
- shared title clone
- structural support
- shared PGP fingerprint via auxiliary alias

`reports/step4_candidate_summary.json` 记录的是 Step 4 初始候选生成结果。后续 Step 5 的 targeted expansion / positive-anchor / item-identity expansion 会向 Step 4 候选表追加或更新少量候选边，因此当前 Step 7/Step 11 使用的 pair-feature 表行数会略高于这里的原始 Step 4 计数。

Step 4 初始候选规模：

| 数据域 | seller 数 | candidate pair 数 | 主要候选 scope |
| --- | ---: | ---: | --- |
| `en_content_train_pool` | `7522` | `6623` | `5553` sockpuppet primary, `1070` same-alias continuity |
| `zh_target_strict` | `5097` | `3793` | `3793` sockpuppet primary |
| `zh_target_aux` | `673` | `580` | `580` sockpuppet primary |

中文 strict 的候选规则命中：

- `profile_lexical_neighbor`: `3487`
- `shared_contact_exact`: `28`
- `shared_description_clone`: `595`
- `shared_title_clone`: `384`
- `structural_support`: `2173`

设计理由：真实同控制 pair 在全图中极稀疏，直接枚举所有 pair 会被海量 easy negatives 淹没。Step 4 先用高召回规则构造 candidate universe，后续 Step 5 扩充只在这个候选体系上追加或更新审查边，Step 7/9/11 使用的是扩充后的当前 pair-feature 表。

当前下游 pair-feature 表规模为：

| 数据域 | 当前 Step 7 pair-feature rows | 说明 |
| --- | ---: | --- |
| `en_content_train_pool` | `6683` | Step 4 初始 `6623` 加后续英文扩充/更新后的当前训练候选表 |
| `zh_target_strict` | `3857` | Step 4 初始 `3793` 加中文 targeted/anchor 扩充后的当前候选表 |
| `zh_target_aux` | `580` | 与 Step 4 初始计数一致 |

Step 11 会进一步只保留 `core_transfer_eligible = 1` 的中文 pair；当前实际进入 Step 11 scorer 的中文候选边为 `3851` 条。

实现文件：

- schema: `schema/step4_silver_candidate_schema.json`
- runner: `scripts/step4_build_silver_candidates.py`
- summary: `reports/step4_candidate_summary.json`

## 4. Step 5 冻结监督标签设计

### 4.1 为什么必须 freeze

Step 5 的目的是把候选 pair 转成可训练、可评估的 frozen supervision。冻结后，模型训练和后续审计不能随意改标签，否则每轮结果都不可比。

Step 5 的核心纪律：

1. `positive` 必须有足够强的同控制证据。
2. `negative` 主要用于模板复用、主题相似、身份噪声等 hard negative。
3. `uncertain` 不进入监督训练和测试。
4. `same_alias_identity_continuity`、closure-derived audit positive 等不能混入主监督。
5. train/valid/test 按 seller component 隔离，避免同一 seller component 同时出现在训练和测试。

### 4.2 当前冻结标签规模

英文源域：

| 指标 | 数值 |
| --- | ---: |
| reviewed rows | `1321` |
| supervision rows | `734` |
| reviewed positive | `226` |
| reviewed negative | `525` |
| reviewed uncertain | `570` |
| primary positive supervision rows | `209` |
| non-identifier positive count | `61` |

英文 split：

| split | rows | positive | negative |
| --- | ---: | ---: | ---: |
| train | `401` | `116` | `285` |
| valid | `152` | `42` | `110` |
| test | `181` | `51` | `130` |

中文严格目标域：

| 指标 | 数值 |
| --- | ---: |
| reviewed rows | `1016` |
| supervision rows | `522` |
| reviewed positive | `102` |
| reviewed negative | `426` |
| reviewed uncertain | `488` |
| primary positive supervision rows | `96` |
| non-identifier positive count | `70` |

中文 split：

| split | rows | positive | negative |
| --- | ---: | ---: | ---: |
| train | `335` | `61` | `274` |
| valid | `81` | `14` | `67` |
| test | `106` | `21` | `85` |

泄漏检查：

| 数据域 | train-valid seller overlap | train-test seller overlap | valid-test seller overlap | alias overlap |
| --- | ---: | ---: | ---: | ---: |
| EN | `0` | `0` | `0` | `0` |
| ZH | `0` | `0` | `0` | `0` |

### 4.3 为什么英文比中文更大

当前论文叙事是“大源域、小目标域”迁移。英文侧承担源域学习，中文侧承担目标域少样本适配和固定评估。中文正样本无法盲目扩充，因为缺少 seller-facing identity anchors 时，把模板相似 pair 标成 positive 会污染 ground truth。

这也是当前实验设计不继续盲扩中文 hard negatives 的原因：中文 hard negatives 已经很多，瓶颈是 proof-level positives 稀缺。

实现文件：

- policy: `schema/step5_freeze_policy.json`
- runner: `scripts/step5_freeze_silver_labels.py`
- summary: `reports/step5_frozen_silver_summary.json`
- EN labels: `reports/step5_en_frozen_silver_labels.csv`
- ZH labels: `reports/step5_zh_target_strict_frozen_silver_labels.csv`

## 5. Step 7 zero-shot 源域训练设计

### 5.1 Step 7 要解决什么问题

Step 7 在英文源域训练 seller-pair verifier，然后直接在中文 strict test 上评估。这一步回答：

> 不看中文训练标签，仅从英文市场学到的 pairwise 判别规律，能否迁移到中文市场？

它是后续 few-shot 的基线。如果 few-shot 要成立，必须和 Step 7 zero-shot、raw semantic baseline 做比较。

### 5.2 特征工程设计

Step 7 的 clean transfer-safe feature view 是 `core_zero_shot_ready_now`。它包含三类特征。

第一类是结构和文本重合特征：

- `same_market_raw_bool`
- `same_source_dataset_bool`
- `profile_category_jaccard`
- `shared_title_count_capped`
- `shared_description_count_capped`
- `shared_category_count_capped`
- `shared_title_idf_sum`
- `shared_description_idf_sum`
- `shared_title_idf_mean`
- `shared_description_idf_mean`
- `boilerplate_ratio_max`
- `boilerplate_ratio_gap_abs`
- `shared_boilerplate_count`
- `shared_low_df_sentence_count`
- `shared_rare_ngram_count`

这些特征用于捕捉标题/描述克隆、类目重合、低频短语和模板污染。IDF 与 boilerplate 设计的目的，是区分“稀有共享内容”和“公共模板复用”。

第二类是 market-relative 风格和规模 gap：

- `item_count_percentile_gap_abs`
- `price_median_percentile_gap_abs`
- `title_length_median_percentile_gap_abs`
- `description_length_median_percentile_gap_abs`
- `digit_ratio_mean_percentile_gap_abs`
- `punct_ratio_mean_percentile_gap_abs`
- `repeated_title_share_percentile_gap_abs`
- `repeated_description_share_percentile_gap_abs`
- `max_category_share_percentile_gap_abs`

这些特征使用市场内 percentile gap，而不是直接使用原始长度、价格或数量。原因是英文市场和中文市场的规模、格式、价格体系不同，直接 raw magnitude 会把市场差异当作身份差异。

第三类是 multilingual semantic features：

| semantic set | 特征 |
| --- | --- |
| default GTE | `embedding_cosine_gte_multilingual_base`, `reranker_score_gte_multilingual_reranker_base` |
| BGE-M3 | `embedding_cosine_bge_m3`, `reranker_score_bge_reranker_v2_m3` |
| E5 | `embedding_cosine_multilingual_e5_large` |
| LaBSE | `embedding_cosine_labse` |
| MPNet | `embedding_cosine_paraphrase_multilingual_mpnet` |

语义特征是必要的，因为同一操作者可能跨账号复用商品主题、描述结构或语义内容。但语义也很危险，因为中文市场存在模板复用和同类商品污染。因此项目保留 raw semantic baseline，同时也训练融合模型来检验结构/风格特征是否能提升迁移。

### 5.3 Clean line 与 identifier control 的区分

Clean scientific line 不使用直接 identifier 特征，例如：

- `has_shared_contact_exact`
- `has_shared_pgp_fingerprint`
- `shared_contact_count_capped`
- `shared_pgp_fingerprint_count_capped`

这些 identifier 特征只进入 `identifier_augmented_default` 和后续 identifier operational control。原因是直接联系方式接近标签证据，如果混入 clean scientific model，会让模型结论变成“抽取器是否发现同联系方式”，而不是“跨语言行为/文本/结构规律是否迁移”。

### 5.4 Step 7 模型和控制实验

Step 7 使用 LightGBM fusion，在英文 train 上训练、英文 valid 上早停和选阈值，然后在中文 strict test 上 zero-shot 评估。

当前 default experiments 共 `17` 个，包括：

- clean default
- BGE-M3 clean
- BGE embedding-only
- E5 embedding-only
- E5 + GTE reranker
- LaBSE embedding-only
- LaBSE + GTE reranker
- paraphrase multilingual MPNet
- MPNet + GTE reranker
- no-reranker
- reranker-only
- no-semantics
- no-style-gap
- no-structural
- raw-style-gap diagnostic control
- identifier augmented operational control
- EN-only ablation

训练稳定性保护：

- `small_validation_guard`: 防止英文 valid 太小时早停不可信。
- `collapse_guard`: 检测 best_iteration 太低或 valid 概率值过少的模型。
- post-train iteration scan: 小样本时避免过早停在退化迭代。

当前有效边界下：

- `small_validation_guard.triggered = false` for all `17` experiments。
- `collapse_guard.triggered = true` for `10/17` experiments。
- 多个 LightGBM fusion 模型仍然是 one-tree 或 shallow solution。

### 5.5 Step 7 当前结果

固定中文 strict test：`106` rows, `21` positive, `85` negative。

关键 Step 7 clean zero-shot 结果：

| experiment | best_iteration | ROC-AUC | AP | balanced accuracy | collapse |
| --- | ---: | ---: | ---: | ---: | --- |
| `core_zero_shot_default` | `1` | `0.588235` | `0.448547` | `0.562465` | true |
| `core_zero_shot_bge_m3` | `1` | `0.601681` | `0.448761` | `0.562465` | true |
| `core_zero_shot_multilingual_e5_large` | `1` | `0.550140` | `0.384493` | `0.538655` | true |
| `core_zero_shot_default_no_structural` | `54` | `0.623529` | `0.287652` | `0.572269` | false |
| `core_zero_shot_paraphrase_multilingual_mpnet_plus_gte_reranker` | `47` | `0.604482` | `0.366364` | `0.524650` | false |
| `identifier_augmented_default` | `1` | `0.606443` | `0.418989` | `0.619888` | true |

Raw semantic baselines 在中文 test 上反而更强：

| raw baseline | ROC-AUC | AP |
| --- | ---: | ---: |
| raw E5 cosine | `0.806723` | `0.520573` |
| raw LaBSE cosine | `0.806162` | `0.518581` |
| raw BGE-M3 cosine | `0.783754` | `0.492048` |

这说明当前源域 LightGBM fusion 不是一个强 zero-shot 模型。它在英文源域学到的结构/模板 shortcut 没有稳定迁移到中文目标域。Raw multilingual semantic ranking 仍然是必须报告的强基线。

实现文件：

- feature schema: `schema/step7_transfer_safe_pair_feature_schema.json`
- semantic policy: `schema/step7_semantic_model_policy.json`
- training policy: `schema/step7_training_policy.json`
- semantic feature runner: `scripts/step7_build_semantic_pair_features.py`
- training runner: `scripts/step7_train_baseline_models.py`
- summary: `reports/step7_training_summary.json`

## 6. Step 9 few-shot adaptation 设计

### 6.1 Step 9 要解决什么问题

Step 9 问的是：

> 在已有英文源域模型和少量中文监督数据的条件下，目标域 few-shot adaptation 是否能改善中文 seller-pair 判别？

这一步不再简单相信 early few-shot 的高 accuracy，因为早期结果曾出现阈值偏移和小测试集幻觉。当前 Step 9 明确以 ROC-AUC、AP 和固定中文 test 为主，避免只看 threshold accuracy。

### 6.2 为什么保留 legacy LightGBM 但不作为主线

早期 few-shot 把少量中文数据和英文数据混在一起继续训练 LightGBM，容易发生：

- 小样本下树模型记忆局部噪声。
- 中文模板复用导致语义特征与标签关系变化。
- valid threshold 偏移造成 accuracy 虚高。
- AUC 下降但 balanced accuracy 暂时上升。

因此当前策略是：

- legacy LightGBM few-shot 保留为 legacy control。
- 主线改为 `logistic_regression_l2` 和 `residual_logistic`。
- 用 hard-boundary sampler，而不是随机抽样。
- 多 ratio 和多 seed 比较。

### 6.3 Hard-boundary support sampling

Step 9 对中文 `zh_train` 做 few-shot support sampling，比例为：

- `0.1`
- `0.2`
- `0.5`
- `1.0`

seed 为：

- `20260320`
- `20260321`
- `20260322`

采样策略不是随机抽 easy examples，而是 hard-boundary：

- hard positive low base score: 基座模型分数低但标签为 positive。
- hard negative high base score: 基座模型分数高但标签为 negative。
- typical/fallback anchors: 维持分布锚点。

这样设计的原因是：目标域 few-shot 的价值应来自边界样本，而不是大量一眼可分的 easy negatives。

### 6.4 当前 Step 9 实验组

当前 policy 中 default experiments 为 `19` 个，主要分为：

| 组别 | 目的 |
| --- | --- |
| clean current baselines | default/BGE/no-structural 的 residual 或 LR/L2 |
| semantic backbone controls | BGE/E5/LaBSE/MPNet embedding-only 或 plus-reranker |
| domain-aware ablations | no-semantics residual/LR |
| minority regularization controls | E5 LR/L2 positive-pair mixup |
| identifier operational controls | identifier augmented LR/L2 |
| legacy controls | old mixed-LightGBM few-shot |

### 6.5 LR/L2 smooth fusion

`logistic_regression_l2` 是当前 clean few-shot 的主要建模方式。它使用源域训练数据加 sampled Chinese support，做全局平滑权重调整。

选择 LR/L2 的原因：

- 在小样本中文支持集下比树模型更不容易记忆局部噪声。
- L2 正则能抑制特征权重爆炸。
- 便于解释 top coefficients。
- 可稳定输出 scorer artifact，供 Step 11 在全量中文候选边上打分。

### 6.6 Residual logistic

`residual_logistic` 冻结 Step 7 基座模型，只学习目标域残差修正。设计逻辑是：

- 英文源域模型提供 base probability。
- 中文 few-shot 只学习如何校正 base score。
- 避免用少量中文数据重写整个判别函数。

这适合检验“目标域少样本是否只需要校准/纠偏，而不需要重训融合层”。

### 6.7 Positive-pair mixup

`core_few_shot_multilingual_e5_large_lr_l2_positive_pair_mixup` 是当前新增的 minority regularization control，灵感来自少数类增强思想（**原论文 :《RABot: Reinforcement-Guided Graph Augmentation for Imbalanced and Noisy Social Bot Detection** 》）。

它要解决的问题不是“增加中文候选边数量”，而是缓解中文目标域监督中的 positive 稀缺和类别不平衡。当前 `zh_train` 只有 `61` 条 positive、`274` 条 negative；如果直接训练 LR/L2，模型容易把高语义 negative、模板复用 negative 和少数 positive 的边界学得不稳定。mixup 的目的就是在不放松 Step 5 标注纪律的前提下，给 positive decision region 增加平滑约束。

它做的不是生成新 seller，也不是生成新文本，更不是写入新标签，而是在 Step 9 训练矩阵中生成 synthetic positive pair representation。每一条 seller-pair 在 Step 7 后已经被表示成一组数值特征，例如：

- 语义特征：`embedding_cosine_multilingual_e5_large`
- 结构特征：`profile_category_jaccard`、`shared_title_count_capped`、`shared_description_count_capped`
- 文本重合特征：`shared_title_idf_mean`、`shared_description_idf_mean`
- 风格/规模差异特征：`digit_ratio_mean_percentile_gap_abs`、`price_median_percentile_gap_abs`、`repeated_title_share_percentile_gap_abs`

mixup 只在这些 pair-level feature vectors 上插值。给定两个合规的中文训练正样本 pair：

```text
x_new = (1 - lambda) * x_positive_i + lambda * x_positive_j
y_new = positive
```

其中 `x_positive_i` 和 `x_positive_j` 是两条真实 positive seller-pair 的特征向量，`x_new` 是仅供训练使用的合成正样本特征向量。这个操作不会反向生成可审查的 seller pair，也不会声称现实中存在一个新的同控制 pair。

当前实验仍然是 source-retained adaptation，而不是 target-only 训练。也就是说，mixup 版本的训练矩阵由三部分组成：

```text
English source train rows
+ sampled Chinese zh_train rows
+ synthetic_train_only positive rows
```

以 `100pct` 为例，实际训练矩阵为：

```text
English source train: 401 rows
Chinese zh_train: 335 rows = 61 positive / 274 negative
Synthetic positive mixup: 122 rows
Final training matrix: 858 rows = 299 positive / 559 negative
```

约束：

- 只在 sampled `zh_train` 内做。
- 只使用 `review_label = positive` 的真实监督行。
- 要求 `usable_for_core_transfer = 1`。
- 要求 `core_transfer_eligible = 1`。
- 排除 `positive_component_closure_audit`。
- 排除 `audit_only`、`audit_only_soft_alias`、`uncertain_holdout`。
- 不写回 Step 5 frozen labels。
- 不进入 `zh_valid` 或 `zh_test`。
- 输出标记为 `synthetic_train_only`。

这些约束的科研含义是：mixup 只能被解释为 training-only minority regularization，不能被解释为新增人工标注、不能用于扩大 benchmark，也不能进入 Step 11 cluster audit 的证据链。

mixup 参数：

- nearest positive neighbor k: `5`
- lambda range: `[0.2, 0.8]`
- target positive-to-negative ratio: `1.0`
- max synthetic per real positive: `2.0`

当前 mixup 实际生成情况：

| ratio | seed | sampled positives | sampled negatives | synthetic rows | zh_test ROC-AUC | zh_test AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 50pct | 20260320 | `59` | `109` | `50` | `0.831373` | `0.558135` |
| 50pct | 20260321 | `59` | `109` | `50` | `0.825210` | `0.557788` |
| 50pct | 20260322 | `59` | `109` | `50` | `0.816246` | `0.530994` |
| 100pct | 20260320 | `61` | `274` | `122` | `0.844818` | `0.593742` |
| 100pct | 20260321 | `61` | `274` | `122` | `0.841457` | `0.589403` |
| 100pct | 20260322 | `61` | `274` | `122` | `0.839216` | `0.579418` |

当前解读：mixup 100pct 是最强 Step 9 点估计，但 Step 12 显示它尚不能稳健声称超过 raw E5。Step 15 v2 进一步提高了 clean 点估计，但 paired grouped bootstrap 仍要求保守表述。

### 6.8 Step 9 当前关键结果

| experiment | ratio | seeds | ROC-AUC | AP | 角色 |
| --- | --- | --- | ---: | ---: | --- |
| `core_few_shot_multilingual_e5_large_lr_l2` | 50pct | 3 seeds | `0.811765` 到 `0.824650` | `0.534180` 到 `0.541473` | clean current candidate |
| `core_few_shot_multilingual_e5_large_lr_l2_positive_pair_mixup` | 100pct | 3 seeds | `0.839216` 到 `0.844818` | `0.579418` 到 `0.593742` | training-only minority regularization |
| `core_few_shot_bge_m3_residual_lr` | 100pct | 3 seeds | `0.817367` | `0.515857` | residual clean control |
| `core_few_shot_labse_lr_l2` | 100pct | 3 seeds | `0.799440` | `0.531286` | semantic control |
| `identifier_augmented_few_shot_default_lr_l2` | 100pct | 3 seeds | `0.783754` | `0.647686` | operational control |

实现文件：

- policy: `schema/step9_training_policy.json`
- runner: `scripts/step9_run_few_shot_adaptation.py`
- summary: `reports/step9_few_shot_summary.json`
- calibration policy: `schema/step9_calibration_policy.json`
- calibration runner: `scripts/step9_run_calibration_adaptation.py`
- calibration summary: `reports/step9_calibration_summary.json`

## 7. Step 9 calibration 设计

Calibration 的目的是在冻结基座模型的情况下校准概率空间，而不是重新学习排序能力。

当前结论：

- Platt scaling 现在数值上可收敛。
- 但当前三个 calibration 实验在固定 `0.5` 阈值下都预测 `0` 个中文正例：`zh_test` confusion 均为 `tp = 0, fp = 0, fn = 21, tn = 85`。
- calibration branch 是 diagnostic/control，不作为当前 Step 11 discovery mainline。

原因是当前核心问题不是单纯概率校准，而是源域 fusion shortcut 和中文目标域 evidence scarcity。校准无法创造新的身份锚点，也无法解决中文模板复用造成的图层噪声。

## 8. Step 11 中文图聚类设计

### 8.1 Step 11 要解决什么问题

Step 11 把 pair-level scorer 作用到中文候选 pair universe，然后构造 seller graph。它回答：

> 当前 scorer 在中文候选空间里会把哪些 seller 连成候选簇？这些簇是否有可审计的身份证据？

Step 11 的输出不是 ground truth。它是 candidate-cluster triage。

### 8.2 当前输入范围

当前只认 manifest：

- `reports/step11_current_manifest_20260517.json`

当前 manifest：

- current summary count: `19`
- referenced CSV count: `75`
- unreferenced Step 11 model-output CSV: `0`

说明：root `reports/` 中仍有 `step11_current_manifest_20260517.csv` 和 `step11_archive_dry_run_20260517.csv` 这类管理 CSV；它们不是 Step 11 scored-pair / cluster model outputs，不参与 manifest-only cluster audit。

规则：后续审计必须读取 manifest 中每个 summary 的 `output_paths`，不能 glob 整个 `reports/`。

### 8.3 当前 Step 11 scorer families

当前 Step 11 包含：

| family | 实验 |
| --- | --- |
| clean E5 LR/L2 | `core_few_shot_multilingual_e5_large_lr_l2_ratio_50pct`, 3 seeds |
| E5 mixup 50pct | `core_few_shot_multilingual_e5_large_lr_l2_positive_pair_mixup_ratio_50pct`, 3 seeds |
| E5 mixup 100pct | `core_few_shot_multilingual_e5_large_lr_l2_positive_pair_mixup_ratio_100pct`, 3 seeds |
| BGE residual | `core_few_shot_bge_m3_residual_lr_ratio_100pct`, 3 seeds |
| LaBSE LR/L2 | `core_few_shot_labse_lr_l2_ratio_100pct`, 3 seeds |
| conservative zero-shot control | `core_zero_shot_bge_m3` |
| operational identifier control | `identifier_augmented_few_shot_default_lr_l2_ratio_100pct`, 3 seeds |

### 8.4 Graph construction

Step 11 graph:

- node: `seller_uid`
- edge: candidate pair
- edge weight: `prob_positive`
- algorithm: connected components
- minimum cluster size: `2`
- graph scope: `zh_target_strict candidate-pair universe only`
- current eligible pair rows scored by Step 11: `3851`

为了减少模板噪声，Step 11 不直接把所有超过阈值的边放进最终图，而是做多层过滤。

过滤顺序：

1. pairwise probability threshold。
2. relation reliability filter。
3. reciprocal top-k filter。
4. iterative shared-neighbor pruning。
5. connected components clustering。

### 8.5 Relation reliability filter

Relation reliability filter 的目的，是把“语义相似”与“身份可靠性”分开。它不是强化学习，而是规则/验证驱动的边可靠性过滤。

正向证据包括：

- shared PGP fingerprint
- shared seller-facing contact
- rare description clone
- rare title clone
- rare ngram support
- structural support
- style consistency

负向惩罚包括：

- boilerplate/template penalty
- semantic-topic-only penalty

当前配置：

- base score: `0.2`
- minimum score: `0.3`
- hard keep direct identity: true
- shared PGP weight: `0.45`
- shared seller contact weight: `0.35`
- boilerplate penalty: `-0.2`
- semantic-topic-only penalty: `-0.25`

注意：Step 11 relation reliability 仍然只是图过滤规则，不等于 proof-level same-controller label。最近严格审计发现，部分 `has_shared_contact_exact` 其实是 product/victim-data URL/email 或 parser noise，所以 cluster audit 必须再接 Step 5 frozen proof-edge 验证。

### 8.6 当前 Step 11 图规模示例

当前 `20260517` manifest 的 primary threshold 图概况：

| scorer | threshold | pre-filter edges | removed by reliability | removed by shared-neighbor | post-filter edges | clusters | largest |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BGE residual 100pct | `0.735` | `896` | `123` | `514` | `221` | `39` | `12` |
| LaBSE LR/L2 100pct | `0.47` | `962` | `166` | `521` | `235` | `44` | `13` |
| E5 mixup 100pct seed 20260320 | `0.720543` | `408` | `76` | `251` | `70` | `11` | `7` |
| E5 mixup 100pct seed 20260321 | `0.792627` | `203` | `31` | `132` | `35` | `7` | `7` |
| E5 mixup 100pct seed 20260322 | `0.720528` | `345` | `78` | `197` | `61` | `12` | `7` |
| E5 LR/L2 50pct seed 20260321 | `0.56` | `440` | `112` | `270` | `51` | `12` | `7` |
| zero-shot BGE-M3 | `0.483444` | `1425` | `260` | `808` | `197` | `43` | `7` |
| identifier LR/L2 100pct | `0.46` | `523` | `91` | `317` | `94` | `17` | `8` |

解读：

- 图过滤确实大量切掉高分但拓扑脆弱的边。
- 但剩余簇仍不能自动解释为同控制团伙。
- 因此 Step 11 后必须做 cluster-level evidence audit。

实现文件：

- policy: `schema/step11_clustering_policy.json`
- runner: `scripts/step11_cluster_chinese_graph.py`
- manifest: `reports/step11_current_manifest_20260517.json`

## 9. Step 11 cluster-level audit 设计

### 9.1 为什么需要严格 cluster audit

模型和图算法只能提出候选簇。暗网 seller 同控制结论必须回到证据。

当前严格审计规则：

- 只审计显式传入的 Step 11 summary。
- 只读取每个 summary 的 `output_paths`。
- 不 glob `reports/`。
- 对所有 current primary graph views 按 seller set 去重。
- 用 Step 5 frozen ZH labels 做 proof-edge audit。
- 只有 frozen positive、`usable_for_core_transfer = 1` 且含 seller-facing direct contact/PGP 的 retained edge，才算 proof-positive edge。

不能作为 proof 的内容：

- external URL
- product/victim-data email
- parser-noise contact
- negative edge
- uncertain edge
- unlabeled edge
- 纯模板克隆
- 纯语义主题相似

### 9.2 当前严格审计结果

当前 audit：

- summary: `reports/step11_cluster_level_audit.current_20260517.json`
- CSV: `reports/step11_cluster_level_audit.current_20260517.csv`
- input summary count: `19`
- primary cluster count total: `441`
- unique cluster set count: `125`
- summary selection mode: `explicit`

决策分布：

| decision | count |
| --- | ---: |
| `same_controller_high_confidence` | `0` |
| `same_controller_core_with_possible_expansion` | `0` |
| `partial_anchor` | `6` |
| `template_clone_not_controller` | `59` |
| `semantic_topic_not_controller` | `52` |
| `uncertain` | `8` |

置信度：

| confidence | count |
| --- | ---: |
| medium | `6` |
| low | `119` |

当前最重要结论：

> 目前没有任何完整 Step 11 cluster 可以被宣称为 high-confidence same-controller cluster。只有 6 个 partial anchors 可以作为 case-study seed，而且只能宣称其中 proof pair/core，不能宣称整个扩展簇都同控制。

实现文件：

- audit runner: `scripts/step11_cluster_level_audit.py`
- strict proof label source: `reports/step5_zh_target_strict_frozen_silver_labels.csv`

## 10. Step 12 统计稳健性设计

### 10.1 为什么不做随机 K-fold 替代主评估

当前 Step 12 保留固定 `zh_test`，不混合 train/valid/test。原因是：

- 当前 Step 5 split 是按 seller component 防泄漏构造的。
- 随机 K-fold 混合 train/valid/test 可能破坏已经建立的测试边界。
- 论文主评估应固定 test split，再用 grouped bootstrap 给置信区间。

Step 12 做的是统计稳健性审计，而不是重定义 benchmark。

### 10.2 Bootstrap 设计

固定测试容器：

- `zh_test` rows: `106`
- positives: `21`
- negatives: `85`
- bootstrap groups: `39`
- largest group size: `14`

Bootstrap 单位是 `split_component_id`，不是单条 edge。原因是同一 seller component 内的 pair 不是独立样本，按 edge 重采样会让置信区间虚窄。

### 10.3 当前 Step 12 结果

关键模型点估计和 95% grouped bootstrap CI：

| model | ROC-AUC | AUC 95% CI | AP | AP 95% CI |
| --- | ---: | --- | ---: | --- |
| raw E5 cosine | `0.806723` | `[0.638524, 0.916667]` | `0.520573` | `[0.220730, 0.745153]` |
| raw LaBSE cosine | `0.806162` | `[0.708636, 0.906667]` | `0.518581` | `[0.296728, 0.711879]` |
| raw BGE-M3 cosine | `0.783754` | `[0.624193, 0.936095]` | `0.492048` | `[0.255147, 0.789091]` |
| Step 7 default fusion | `0.588235` | `[0.410808, 0.819153]` | `0.448547` | `[0.153907, 0.638187]` |
| Step 7 BGE fusion | `0.601681` | `[0.420804, 0.819306]` | `0.448761` | `[0.157892, 0.651601]` |
| E5 LR/L2 50pct seed mean | `0.819048` | `[0.701728, 0.916886]` | `0.540494` | `[0.301265, 0.763798]` |
| E5 mixup 50pct seed mean | `0.826891` | `[0.709282, 0.918753]` | `0.549271` | `[0.312801, 0.767063]` |
| E5 mixup 100pct seed mean | `0.842017` | `[0.727517, 0.926377]` | `0.588995` | `[0.338429, 0.790663]` |
| BGE residual 100pct seed mean | `0.817367` | `[0.726018, 0.914507]` | `0.515857` | `[0.286507, 0.744768]` |
| LaBSE LR/L2 100pct seed mean | `0.799440` | `[0.683225, 0.930634]` | `0.531286` | `[0.306771, 0.786867]` |
| identifier LR/L2 100pct seed mean | `0.783754` | `[0.592593, 0.946429]` | `0.647686` | `[0.347406, 0.866396]` |

关键 paired comparisons：

| comparison | metric | diff | 95% CI | p | supports positive difference |
| --- | --- | ---: | --- | ---: | --- |
| E5 LR/L2 50pct vs raw E5 | ROC-AUC | `+0.012325` | `[-0.108240, 0.147650]` | `0.7368` | false |
| E5 LR/L2 50pct vs raw E5 | AP | `+0.019920` | `[-0.251152, 0.326280]` | `0.8056` | false |
| E5 mixup 100pct vs raw E5 | ROC-AUC | `+0.035294` | `[-0.086161, 0.158168]` | `0.4812` | false |
| E5 mixup 100pct vs raw E5 | AP | `+0.068422` | `[-0.207105, 0.331671]` | `0.5308` | false |
| E5 mixup 100pct vs non-mixup E5 LR/L2 50pct | ROC-AUC | `+0.022969` | `[-0.015896, 0.057428]` | `0.2488` | false |
| E5 mixup 100pct vs Step 7 default fusion | ROC-AUC | `+0.253782` | `[0.018175, 0.410672]` | `0.0332` | true |
| E5 mixup 100pct vs Step 7 default fusion | AP | `+0.140448` | `[-0.080072, 0.447818]` | `0.1336` | false |

当前统计结论：

- Mixup 100pct 是当前最强点估计。
- 它能稳健超过 collapsed Step 7 default fusion 的 AUC。
- 它不能稳健声称超过 raw E5 semantic baseline。
- 因此不能写“few-shot 已统计显著超过 zero-shot/raw semantic baseline”。

实现文件：

- policy: `schema/step12_statistical_robustness_policy.json`
- runner: `scripts/step12_statistical_robustness_audit.py`
- summary: `reports/step12_v2_statistical_robustness_zh_test_20260602.json`

## 11. Step 13 概念漂移审计设计

### 11.1 Step 13 要解决什么问题

Step 13 不训练模型，只读已有 Step 5/7/9/11/12 结果。它回答：

> 为什么英文源域融合模型不能稳定迁移到中文目标域？Few-shot 改善是否是全局改善，还是只在某些切片上改善？

### 11.2 审计内容

Step 13 检查：

- EN vs ZH 的边特征分布漂移。
- positive / negative 条件下的语义、结构、风格、identifier 特征差异。
- 中文高语义负样本比例是否显著升高。
- identifier-present 和 identifier-absent 两个切片性能。
- raw E5 为什么强于 Step 7 fusion。
- few-shot 是否只在某些切片提升。
- Step 11 cluster audit 的证据类型分布。

### 11.3 当前 drift 发现

当前监督数据仍小且不平衡：

- EN: `734` supervision rows, `209` positive, `525` negative
- ZH: `522` supervision rows, `96` positive, `426` negative

最大 EN->ZH 边特征漂移集中在 style gap 和 identifier：

| feature | SMD ZH minus EN |
| --- | ---: |
| `digit_ratio_mean_raw_gap_abs` | `0.854419` |
| `repeated_description_share_percentile_gap_abs` | `-0.852279` |
| `repeated_title_share_percentile_gap_abs` | `-0.849901` |
| `repeated_title_share_raw_gap_abs` | `-0.676433` |
| `punct_ratio_mean_raw_gap_abs` | `0.673458` |
| `has_shared_contact_exact` | `-0.515466` |

Raw E5 与 Step 7 E5 fusion 对比：

- raw E5: AUC `0.806723`, AP `0.520573`
- Step 7 E5 fusion: AUC `0.550140`, AP `0.384493`

Step 13 的解释是：source-domain fusion 特征不是简单“弱”，而是存在 source-domain shortcut。英文中有效的模板/结构/identifier 关联，迁移到中文时被中文模板复用、商品主题相似、身份锚点稀缺共同破坏。

实现文件：

- runner: `scripts/step13_concept_drift_audit.py`
- summary: `reports/step13_concept_drift_audit.json`
- CSV: `reports/step13_concept_drift_audit.csv`
- doc: `docs/STEP13_CONCEPT_DRIFT_AUDIT.md`

## 12. 为什么当前实验这样设计

### 12.1 为什么先扩英文，再谨慎对待中文

英文侧可继续构建 source-domain train/valid/test，并通过 direct identifier、hard negative、template control 增强源域模型。中文侧则不同：目标域正样本 proof-level anchors 稀缺，如果盲目把“看起来相似”的 pair 标成 positive，会让 ground truth 被模板污染。

所以当前设计是：

- 英文侧扩成更扎实的 source domain。
- 中文侧固定 strict test，不轻易新增无证据 positive。
- 中文 few-shot 只使用已冻结监督数据。
- 中文候选簇只能作为 triage，不作为自动标签。

### 12.2 为什么 raw semantic baseline 必须保留

当前 raw E5/LaBSE/BGE 的中文 AUC 明显高于 Step 7 fusion。若只报告 fusion vs few-shot，会夸大 few-shot 效果。

因此论文中必须同时报告：

- raw semantic controls
- Step 7 source-domain fusion
- Step 9 clean few-shot
- Step 9 mixup control
- identifier operational control
- Step 11 strict evidence audit

### 12.3 为什么 clean 和 identifier control 分开

直接 identifier 是强证据，但它和标签定义高度接近。把它放入 clean model 会让模型优势变成“谁能抽到联系方式”，而不是“跨语言行为规律是否迁移”。

因此：

- clean scientific model: 不使用 identifier features。
- operational model: 可以使用 identifier features，用于实战 triage。
- cluster audit: 可以用 Step 5 proof-level identifier 验证候选簇，但不把它混成 clean 模型输入。

### 12.4 为什么 Step 11 不等于发现真团伙

Step 11 的图边来自模型分数和图过滤，不是人工真值。即使图结构密集，也可能只是：

- 模板复用
- 商品主题相似
- 同类黑产术语
- 公共 URL
- product/victim data email
- parser noise

所以 Step 11 只能输出候选簇。真正能写成同控制发现的，必须有 pair-level seller-facing identity proof。

### 12.5 为什么当前不宣称 few-shot 显著超过 raw E5

虽然 E5 mixup 100pct 点估计最高：

- AUC `0.842017`
- AP `0.588995`

但和 raw E5 的 paired bootstrap CI 跨零：

- AUC diff `+0.035294`, CI `[-0.086161, 0.158168]`
- AP diff `+0.068422`, CI `[-0.207105, 0.331671]`

因此当前最严谨的表述是：

> Few-shot LR/L2 和 positive-pair mixup 修复了 collapsed Step 7 fusion baseline，并提供更好的图 triage surface；但在当前固定中文测试集上，尚不能稳健声称 clean few-shot 统计显著超过 raw E5 semantic baseline。

## 13. 当前可写入论文的结论边界

当前可以稳妥写：

1. 本项目构建了一个严格防泄漏的英文源域和中文目标域 seller-pair verification benchmark。
2. 中文目标域存在明显概念漂移，尤其体现在 style gap、模板复用、identifier 稀缺和 source-domain shortcut 失效。
3. Raw multilingual semantic ranking 在中文目标域上仍然很强，尤其 raw E5 和 LaBSE。
4. Step 7 LightGBM fusion 在当前边界下发生浅层 collapse，说明简单源域融合不能稳定迁移。
5. Step 9 LR/L2 和 residual adaptation 比 collapsed fusion 更有用。
6. Positive-pair mixup 是有效的 training-only minority regularization control，提升点估计，但当前统计稳健性不足以宣称超过 raw E5。
7. Step 11 图聚类适合候选 triage，但不能自动产生 same-controller ground truth。
8. 严格 cluster audit 显示当前 `125` 个 unique cluster sets 中没有完整 high-confidence same-controller cluster，只有 `6` 个 partial anchors。

当前不能写：

1. “Few-shot 已经显著超过 zero-shot/raw semantic baseline。”
2. “Step 11 发现的簇都是真实马甲团伙。”
3. “Identifier-augmented model 是 clean scientific model。”
4. “中文模板相似 pair 可以直接标为 positive。”
5. “当前 positive-pair mixup 生成了新的标注数据。”

## 14. 复现实验入口

当前主配置和脚本入口：

| Step | policy/schema | runner |
| --- | --- | --- |
| Step 2 | `schema/step2_split_policy.json` | `scripts/step2_build_split_manifests.py` |
| Step 3 | `schema/step3_seller_profile_schema.json` | `scripts/step3_build_seller_profiles.py` |
| Step 4 | `schema/step4_silver_candidate_schema.json` | `scripts/step4_build_silver_candidates.py` |
| Step 5 | `schema/step5_freeze_policy.json` | `scripts/step5_freeze_silver_labels.py` |
| Step 7 semantic | `schema/step7_semantic_model_policy.json` | `scripts/step7_build_semantic_pair_features.py` |
| Step 7 training | `schema/step7_training_policy.json` | `scripts/step7_train_baseline_models.py` |
| Step 9 few-shot | `schema/step9_training_policy.json` | `scripts/step9_run_few_shot_adaptation.py` |
| Step 9 calibration | `schema/step9_calibration_policy.json` | `scripts/step9_run_calibration_adaptation.py` |
| Step 11 graph | `schema/step11_clustering_policy.json` | `scripts/step11_cluster_chinese_graph.py` |
| Step 11 audit | manifest + Step 5 labels | `scripts/step11_cluster_level_audit.py` |
| Step 12 | `schema/step12_statistical_robustness_policy.json` | `scripts/step12_statistical_robustness_audit.py` |
| Step 13 | current summaries | `scripts/step13_concept_drift_audit.py` |

运行环境纪律：

- Windows workspace 当前主要用于代码修改、文件审计、同步结果核查和文档维护。
- Step 7/9/11 这类依赖模型、LightGBM 或 Linux runtime 的重跑应在 Linux 服务器执行。
- 每次 Linux 重跑同步回 Windows 后，应先核查 summary、manifest、output_paths 和 fixed test container，再解释结果。

## 15. 当前实验设计评价

当前设计总体是合理且严谨的，原因是：

1. 数据域边界清楚，英文源域和中文目标域没有混成一个模糊池。
2. Step 5 冻结标签和 seller component 隔离降低了泄漏风险。
3. Clean scientific model 与 identifier operational control 分离，避免把直接证据当模型泛化能力。
4. Step 7/9/12/13 构成了从点估计到统计稳健性再到概念漂移解释的完整链条。
5. Step 11 严格降级为候选 triage，避免把模型预测反喂成真值。
6. Positive-pair mixup 被限制在 training-only，不污染 frozen labels、valid 或 test。

当前主要限制也很明确：

1. 中文 strict test 只有 `21` 个 positive，统计功效有限。
2. 中文 seller-facing proof positives 稀缺，限制了 few-shot 的可学习空间。
3. Step 7 fusion 在中文上弱于 raw semantic baseline，说明源域融合存在 shortcut。
4. Step 11 目前多为模板/主题候选，proof-level full cluster 尚未出现。
5. Relation reliability filter 仍依赖已有特征和抽取质量，不能替代人工证据审计。

因此，当前项目最可信的科研方向不是简单宣称 few-shot 胜利，而是围绕以下主题组织：

- cross-domain concept drift under scarce identity anchors
- transfer-safe seller-pair verification
- minority regularization under imbalanced target supervision
- graph candidate triage with strict evidence audit
- distinction between semantic similarity and identity reliability in darknet seller linkage

