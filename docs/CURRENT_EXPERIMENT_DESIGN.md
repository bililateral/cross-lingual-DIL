# 当前实验设计说明

更新日期：`2026-07-18`

本文档说明当前项目的实验设计、设计目的、实现细节、有效数据边界和当前结论边界。它不是历史流水账，而是当前可以复现实验和撰写论文方法部分时应遵循的实验设计说明。

> 当前主线更新：Step25-v3.1 已完成 Linux 数值重跑和封闭审计。求解器修复有效，`44/44` artifact 均满足 KKT，但 C2 在 English/source-only Chinese/target OOF 的 AP 分别比 C0 低 `0.058346/0.030238/0.028093`，只有 `2/11` gates 通过。Step25-v3.1 已冻结为严格负结果，不进入 D1、Step11 或 Step17。完整结果见 `docs/STEP25_V3_1_RESULT_AUDIT_20260718.zh.md`。

## 0. 当前方法状态：Step25-v1/v2/v3.1 均已冻结，v3.1 为最终严格负结果

Step24 已证明冻结多语言作者/风格表示提供明显跨语言增量，但同时把复制模板和公共页面排版嵌入为作者风格。其 source-only primary 在中文 D0 上达到 `AP=0.802718`，相对 redacted-E5 提升 `0.158336`，但 template-clone negative 的 mean/q95/top-decile 分数分别增加 `0.028287/0.067389/0.064517`，target grouped-bootstrap CI 也略跨零。Step24 因此 `promotion_eligible=false` 并冻结。

Step25 仍只读取相同 canonical train：English `401 = 116/285`，Chinese `573 = 229/344`。它不读取 label、evidence type 或 score 来找模板，而是在每个语言域内统计跨 seller 的 12 字符 shingle。对任一 seller，完整排除其 seller component；只有至少由三个外部 seller、两个外部 component 支持并连续覆盖至少 24 字符的片段才会删除。目录只保存 shingle hash 与 document frequency。去污染后的文本仍由 Step24 两个冻结 encoder 编码，模型不微调。

主比较固定为 raw style-only LR/L2 对 decontaminated style-only LR/L2；E5+clean-style 只作 secondary，raw/clean/delta/coverage 只作 exploratory。source-only 仍是主迁移路径，target grouped OOF 是次级适配证据。正式模板门槛使用 rank percentile、top-decile exposure 和模板负例排在 direct/component 正例之上的 violation rate，避免 LR 截距变化伪造概率下降。identifier occurrence 仅进入独立 direction-constrained reliability post-scorer，不进入 clean scorer。

当前 D0 已被 Step24 错误分析消耗，即使 Step25 D0 全部通过也只能设置 `d1_candidate_eligible=true`；`publication_promotion_eligible` 永远为 false。实际 Linux 结果中，source-only raw/decontaminated style AP 为 `0.801847/0.799675`，target grouped-OOF raw/decontaminated style AP 为 `0.789848/0.784333`；`d1_candidate_eligible=false`，因此 Step25 已冻结为负结果，不能进入 Step11/17。该结果说明当前跨组件 shingle 规则没有可靠去除真正造成错误的中文模板信号，且去污染未带来排序收益。确认性方法开发仍必须使用新的 score-blind、D0-component-disjoint D1，最终结论只认模型冻结后收集且只评估一次的 F1 prospective holdout。任何 Step21-Step23 合成/派生行仍不得写入 Step5 或冒充真实中文马甲标签。

Step25-v2 随后改为 pair-local copy detector，并修复短文本清洗失败时把 style cosine 写成固定零值的问题。同步回来的 19 文件结果包完整通过哈希审计。它在 `109/110` 条模板负例中检测到局部复制，但将 pair-local-clean style 统一替换 raw style 后，中文 target grouped-OOF AP 从 P0 的 `0.704847` 降至 P2 的 `0.670692`，英文 AP 从 `0.468210` 降至 `0.251926`。P2 虽把模板负例相对强正例的 violation rate 降低 `0.092812`，却提高模板 mean rank percentile `0.001637` 和 top-decile exposure `0.036364`；只有 `3/8` 机制门槛通过。P3 raw fallback 的中文 AP 回升到 `0.737365`。因此 v2 的结论不是“局部复制检测无效”，而是“把清洗表示统一替代原始作者信号会损失过多信息”。

Step25-v3 直接针对这一结果，保留 raw authorship style，同时把 pair-local-clean style、raw-minus-clean residual 和 copy-risk statistics 分成独立低维通道。固定主模型 C2 使用方向约束 LR/L2：raw/clean similarity 只能提供非负身份支持；raw-clean residual 和 copy-risk 只能保持零或降低同控制分数，不能把“复制很多”学成正身份信号。不可靠局部风格使用 raw fallback、残差固定为零并显式记录 reliability。C0、C1、C3 是预注册对照，不允许依据 D0 选择候选；纯 global-clean missingness closure 和英文训练的 operational identifier control 均单独报告，不能影响 C2 或晋级门槛。

原 v3 数值包同步完整，且逐 pair 分数可精确复现 summary。其 C2 target grouped-OOF AP 为 `0.761758`，低于 C0 的 `0.789338`；template-negative mean rank、top-decile exposure 和 violation rate 分别恶化 `0.026907/0.036364/0.047780`，只有 `2/11` gates 通过。但这些分数来自提前停止解：artifact 的 projected-gradient residual 最高为 `0.52`，策略 tolerance 为 `1e-8`。因此不能把它当作最终方法负结果。

旧 v3 输出只保留到 v3.1 完成 feature byte-parity 与新旧分数对照审计，随后已从工作区删除。旧 v3 代码和 policy 仍作为 v3.1 冻结科学矩阵的实现依赖保留；旧的无效预测、artifact 和 summary 不再作为当前结果输入。

Step25-v3.1 只修复数值求解：active-set projected Newton + Armijo backtracking，relative loss 不参与收敛，最终 KKT residual 必须 `<=1e-8`，否则 fail closed。输出写入独立 `v3_1_solverfix_20260718`，旧 v3 不覆盖。实际重跑中最大 residual 为 `2.10e-9`，全部 `44` 个 fit 合格；但 C2 的 target OOF AP 为 `0.761755`，低于 C0 的 `0.789848`，template-clone rank tail 也恶化。原 `11` 个 gate 只有 `2` 个通过，因此该方向正式冻结为负结果，不允许选择 C3、修改阈值或使用 valid/test 继续优化。

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
| Step 11 | 把 scorer 投影成中文候选图和候选簇，并做 explicit allow-list audit | `reports/step11_cluster_level_audit.step16g_imbalance_validation_20260710.json` |
| Step 12 | 固定中文测试集上的 grouped-bootstrap 稳健性审计 | 当前已审计 `step16g_imbalance_20260710`；v5r 待生成 `reports/step12_v5r_statistical_robustness_zh_test_weighted_mixup_20260711.json` |
| Step 13 | 概念漂移与切片诊断 | `reports/step13_concept_drift_audit.step16g_imbalance_validation_20260710.json` |
| Step 15 | evidence-type hard-negative curriculum 与轻量 MLP scorer | 旧 v5 已冻结；v5r 待生成 `reports/step15_v5r_weighted_mixup_summary.json` |
| Step 16G | train-only weak hard-negative support，用于恢复正式 mixup 消融 | `reports/step16g_hard_negative_imbalance_summary.json` |

实验设计的核心原则是：

1. 先固定数据边界，再训练模型。
2. Clean scientific line 不使用直接 identifier 特征。
3. Identifier augmented 只能作为 operational control。
4. 中文 `zh_test` 是固定测试集，不能与 train/valid 随机混合。
5. Step 11 只生成候选簇，不把聚类结果当 ground truth。
6. 同控制结论必须有 seller-facing identity proof，例如 Telegram、PGP、Jabber、QQ、微信、电话、钱包等直接证据；商品内容、模板、URL、受害者数据 email 不能直接作为同控制证明。

### 2.1 Step 15 v5r 修复版状态

旧 `v5` 结果保留为实现修复前的历史对照。当前待 Linux 验证的新分支是 `v5r`，其输出与旧版物理隔离：

- 新 summary：`reports/step15_v5r_weighted_mixup_summary.json`；
- 新 slice audit：`reports/step15_v5r_weighted_mixup_slice_level_audit.json/csv`；
- 新 Step 12：`reports/step12_v5r_statistical_robustness_*_weighted_mixup_20260711.*`；
- 新实验名均以 `step15_v5r_` 开头，不覆盖任何 `step15_v5_` artifact 或 prediction。

`v5r` 只修复已确认的训练目标缺陷，不改变 frozen labels、固定 `zh_valid/zh_test`、特征集或 MLP 容量：

1. Phase 4 正例父样本必须能用于核心迁移、证据类型可信且权重不低于 `0.55`。
2. 只在同一语言域和同一 evidence type 内选择最近邻正例配对，不再做英文/中文跨域向量插值。
3. synthetic row 的 `training_sample_weight` 取两个父样本权重的较小值。
4. 布尔和计数特征从 anchor parent 原样复制，只插值连续特征。
5. domain-balanced 版本在类别、evidence type、row quality 权重全部应用后，再平衡两个真实域的有效权重总量。
6. 每个合成样本写入独立 parent-provenance manifest，便于复核并证明没有进入 validation/test/Step 5。

修复不是预设结果。正式判定依赖 `v5r Phase4 - v5r Phase3`、`v5r - legacy v5` 和 `v5r domain-balanced - v5r non-domain` 的 Step 12 配对 grouped bootstrap。

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

固定中文 strict test：`200` rows, `50` positive, `150` negative。

关键 Step 7 clean zero-shot 结果：

| experiment | best_iteration | ROC-AUC | AP | balanced accuracy | collapse |
| --- | ---: | ---: | ---: | ---: | --- |
| `core_zero_shot_default` | `1` | `0.604000` | `0.490156` | `0.573333` | true |
| `core_zero_shot_bge_m3` | `1` | `0.604200` | `0.494441` | `0.573333` | true |
| `core_zero_shot_multilingual_e5_large` | `1` | `0.623733` | `0.519158` | `0.573333` | true |
| `core_zero_shot_default_no_structural` | `54` | `0.768800` | `0.514122` | `0.603333` | false |
| `core_zero_shot_paraphrase_multilingual_mpnet_plus_gte_reranker` | `47` | `0.662133` | `0.461304` | `0.573333` | false |
| `identifier_augmented_default` | `1` | `0.691200` | `0.599287` | `0.703333` | true |

Raw semantic baselines 在中文 test 上反而更强：

| raw baseline | ROC-AUC | AP |
| --- | ---: | ---: |
| raw E5 cosine | `0.748000` | `0.542839` |
| raw LaBSE cosine | `0.813333` | `0.608477` |
| raw BGE-M3 cosine | `0.748933` | `0.526585` |

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

它要解决的问题不是“增加中文候选边数量”，而是缓解中文目标域监督中的 positive 稀缺和类别不平衡。Step 16G 当前训练边界为 `229` 条 positive、`344` 条 negative。这里必须区分证据强度：229 条 positive 中只有 `16` 条原始满权重训练正例，另有 `213` 条 Step 16B/D 低权重 silver positive；344 条 negative 中新增的 `115` 条 Step 16G 行同样是低权重、train-only weak supervision。mixup 的目的，是在不改动 validation/test 和不伪造新标签的前提下，对较高可信 positive decision region 施加平滑约束。

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

以 Step 16G 边界下的 `100pct` 为例，预期训练矩阵为：

```text
English source train: 401 rows = 116 positive / 285 negative
Chinese zh_train: 573 rows = 229 positive / 344 negative
Synthetic positive mixup: 115 rows
Final training matrix: 1089 rows = 460 positive / 629 negative
```

约束：

- 只在 sampled `zh_train` 内做。
- 只使用 `review_label = positive` 的真实监督行。
- 要求 `usable_for_core_transfer = 1`。
- 要求 `core_transfer_eligible = 1`。
- 要求来源行 `training_sample_weight >= 0.55`，排除低可信 silver positive。
- 合成行权重取两个父样本权重的较小值，不得把弱标签放大为满权重。
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
- minimum source training weight: `0.55`
- synthetic weight mode: `minimum_parent_weight`
- `100pct` contract: `synthetic_row_count` 必须大于 `0`，否则运行直接失败。

Step 16G 应用后的直接 augmentation smoke test：

| ratio | real positive | real negative | eligible positive sources | synthetic rows | synthetic weight range |
| --- | ---: | ---: | ---: | ---: | ---: |
| 100pct | `229` | `344` | `72` | `115` | `[0.55, 1.0]` |

此前 Step 16C/E 平衡训练边界下的 Step 9 artifacts 记录 `synthetic_row_count = 0`，因此那一批结果不是有效 mixup 结果。Step 16G Linux 重跑已经确认 100pct 三个 seed 均生成 `115` 条 synthetic rows，但同支持比例的正式消融没有证明收益：mixup 相对 non-mixup 的 AUC 差值为 `-0.006000`，AP 差值为 `+0.001205`，两项 grouped-bootstrap CI 均跨 `0`。

Step 12 的正式消融现已改为相同中文支持比例：`mixup 100pct` 对 `non-mixup E5 LR/L2 100pct`。旧的 `mixup 100pct` 对 `non-mixup 50pct` 同时改变了 support ratio 和 augmentation，不能用于隔离 mixup 的因果效果。

### 6.8 Step 9 当前 Step16G 结果

当前固定测试边界为 `200 = 50 positive / 150 negative`。下表的 seed-mean 指先对三个 seed 的同一 pair 分数求均值，再计算排序指标；它不等于三个单 seed 指标的算术平均。

| experiment | ratio | ROC-AUC | AP | 角色 |
| --- | ---: | ---: | ---: | --- |
| raw E5 cosine | n/a | `0.748000` | `0.542839` | raw semantic control |
| `core_few_shot_multilingual_e5_large_lr_l2` | 100pct | `0.762667` | `0.556429` | same-ratio non-mixup control |
| `core_few_shot_multilingual_e5_large_lr_l2_positive_pair_mixup` | 100pct | `0.756667` | `0.557633` | training-only mixup control |
| `core_few_shot_default_no_structural_lr_l2` | 10pct | `0.825867` | `0.616743` | exploratory clean candidate; not yet in Step12 paired audit |
| `identifier_augmented_few_shot_default_lr_l2` | 50pct | `0.874444` | `0.800381` | operational identifier control, not clean mainline |

100pct same-ratio paired comparison：

| comparison | metric | difference | 95% grouped-bootstrap CI | supports positive difference |
| --- | --- | ---: | --- | --- |
| mixup minus non-mixup | ROC-AUC | `-0.006000` | `[-0.031289, 0.021427]` | no |
| mixup minus non-mixup | AP | `+0.001205` | `[-0.049545, 0.053483]` | no |

当前 mixup 无效的根因不是“没有生成合成行”，而是：

- 训练器是标准化线性 LR/L2，`class_weight = balanced`、`l2_penalty = 5.0`；同类样本的凸组合不产生新的非线性可分方向。
- 类别平衡权重会随增广后的类别计数重新计算，抵消单纯 oversampling。seed `20260320` 中，加入 mixup 后的有效正类权重占比反而从 `0.439745` 轻微降至 `0.436759`。
- 115 条合成行中每个 seed 有 `102-106` 条权重仅为 `0.55`；它们主要重复已有弱正例邻域，而不是增加新的 proof-level identity variation。
- seed `20260320` 的 non-mixup/mixup 系数余弦相似度为 `0.976459`，中文测试分数 Pearson 相关为 `0.959852`。mixup 主要移动分数尺度，没有稳定改善排序。
- 当前 25 维向量混合了连续 embedding、binary flags 和 capped counts。seed `20260320` 有 `15/115` 条合成行在原本离散的特征上产生小数值，存在 off-manifold 风险。

因此该实验应保留为负面 minority-regularization control，不能写成“mixup 提升了中文马甲识别”。

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

说明：root `reports/` 中保留 `step11_current_manifest_20260517.csv` 作为历史管理清单；`step11_archive_dry_run_20260517.csv/json` 已于 `2026-07-14` 在确认不参与当前 manifest-only audit 后删除。管理清单不是 Step 11 scored-pair / cluster model output，不能作为模型结果输入。

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

当前 Step16G explicit six-summary validation 的 primary threshold 图概况：

| scorer | threshold | pre-filter edges | removed by reliability | removed by shared-neighbor | post-filter edges | clusters | largest |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Step15 v5 non-domain seed mean | `0.376301` | `1218` | `101` | `706` | `336` | `58` | `11` |
| Step15 v5 domain-balanced seed mean | `0.456284` | `1783` | `189` | `882` | `524` | `84` | `13` |
| E5 mixup 100pct seed 20260320 | `0.403304` | `1836` | `161` | `968` | `528` | `96` | `9` |
| E5 mixup 100pct seed 20260321 | `0.450980` | `1549` | `142` | `835` | `458` | `81` | `10` |
| E5 mixup 100pct seed 20260322 | `0.406027` | `1834` | `176` | `992` | `498` | `92` | `9` |
| zero-shot BGE-M3 | `0.483444` | `1425` | `260` | `808` | `197` | `43` | `7` |

解读：

- 图过滤确实大量切掉高分但拓扑脆弱的边。
- 但剩余簇仍不能自动解释为同控制团伙。
- 因此 Step 11 后必须做 cluster-level evidence audit。

实现文件：

- policy: `schema/step11_clustering_policy.json`
- runner: `scripts/step11_cluster_chinese_graph.py`
- current audit input mode: six explicit summaries; no `reports/` globbing

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

- summary: `reports/step11_cluster_level_audit.step16g_imbalance_validation_20260710.json`
- CSV: `reports/step11_cluster_level_audit.step16g_imbalance_validation_20260710.csv`
- input summary count: `6`
- primary cluster appearances: `454`
- unique cluster set count: `212`
- summary selection mode: `explicit`

决策分布：

| decision | count |
| --- | ---: |
| `same_controller_high_confidence` | `0` |
| `same_controller_core_with_possible_expansion` | `2` |
| `partial_anchor` | `8` |
| `template_clone_not_controller` | `78` |
| `semantic_topic_not_controller` | `111` |
| `uncertain` | `13` |

置信度：

| confidence | count |
| --- | ---: |
| high | `2` |
| medium | `8` |
| low | `202` |

当前最重要结论：

> 目前仍没有任何完整 Step 11 cluster 可以被宣称为 high-confidence same-controller cluster。两个 high-confidence anchored cores 是同一底层 seller core 的重叠子集，不是两个独立新发现；189/212 个唯一 seller sets 被判为 template/topic non-controller。

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

- `zh_test` rows: `200`
- positives: `50`
- negatives: `150`
- bootstrap groups: `126`
- largest group size: `14`

Bootstrap 单位是 `split_component_id`，不是单条 edge。原因是同一 seller component 内的 pair 不是独立样本，按 edge 重采样会让置信区间虚窄。

### 10.3 当前 Step 12 结果

关键模型点估计和 95% grouped bootstrap CI：

| model | ROC-AUC | AUC 95% CI | AP | AP 95% CI |
| --- | ---: | --- | ---: | --- |
| raw E5 cosine | `0.748000` | `[0.643574, 0.841912]` | `0.542839` | `[0.388123, 0.679719]` |
| E5 LR/L2 100pct seed mean | `0.762667` | `[0.672478, 0.845673]` | `0.556429` | `[0.409951, 0.693599]` |
| E5 mixup 100pct seed mean | `0.756667` | `[0.665910, 0.838327]` | `0.557633` | `[0.410763, 0.694825]` |
| Step15 v5 non-domain seed mean | `0.866533` | `[0.799393, 0.919161]` | `0.725220` | `[0.572913, 0.837135]` |
| Step15 v5 domain-balanced seed mean | `0.865333` | `[0.797979, 0.925327]` | `0.644989` | `[0.490746, 0.796990]` |

关键 paired comparisons：

| comparison | metric | diff | 95% CI | p | supports positive difference |
| --- | --- | ---: | --- | ---: | --- |
| E5 mixup 100pct vs E5 non-mixup 100pct | ROC-AUC | `-0.006000` | `[-0.031289, 0.021427]` | `0.6464` | false |
| E5 mixup 100pct vs E5 non-mixup 100pct | AP | `+0.001205` | `[-0.049545, 0.053483]` | `0.9860` | false |
| Step15 non-domain vs raw E5 | ROC-AUC | `+0.118533` | `[0.012272, 0.223274]` | `0.0276` | true |
| Step15 non-domain vs raw E5 | AP | `+0.182381` | `[0.019890, 0.324430]` | `0.0304` | true |
| Step15 domain-balanced vs raw E5 | ROC-AUC | `+0.117333` | `[0.013792, 0.228072]` | `0.0284` | true |
| Step15 domain-balanced vs raw E5 | AP | `+0.102150` | `[-0.066863, 0.284338]` | `0.2412` | false |
| Step15 domain-balanced vs non-domain | AP | `-0.080231` | `[-0.189542, 0.023211]` | `0.1476` | false |

当前统计结论：

- Step9 100pct mixup 已真实执行，但不能改善同支持比例 non-mixup；这是负面消融结论。
- Step15 v5 non-domain 是当前 clean fixed-test 最强点估计，并在 paired grouped bootstrap 中超过 raw E5 的 AUC 和 AP。
- Step15 domain-balanced 的 AUC 与 non-domain 接近，但 AP 低 `0.080231`；该差值本身 CI 仍跨 `0`。
- 当前 Step15 Phase4 存在弱父样本放大和跨域合成行权重缺陷，因此上述 Step15 提升只能称为 promising internal result，不能直接归因于 mixup，也不能替代 prospective holdout。

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
- ZH: `893` supervision rows, `309` positive, `584` negative

最大 EN->ZH 边特征漂移集中在 style gap 和 identifier：

| feature | SMD ZH minus EN |
| --- | ---: |
| `repeated_description_share_percentile_gap_abs` | `-0.853096` |
| `repeated_title_share_percentile_gap_abs` | `-0.832624` |
| `digit_ratio_mean_raw_gap_abs` | `0.704300` |
| `repeated_title_share_raw_gap_abs` | `-0.661037` |
| `shared_category_count_capped` | `-0.551895` |

Raw E5 与 Step 7 E5 fusion 对比：

- raw E5: AUC `0.748000`, AP `0.542839`
- Step 7 E5 fusion: AUC `0.623733`, AP `0.519158`

Step 13 的解释是：source-domain fusion 特征不是简单“弱”，而是存在 source-domain shortcut。英文中有效的模板/结构关联迁移到中文后发生条件分布变化。当前审计同时否定了“中文所有语义负例都更高”这种过度简化：在英文负例 q90 阈值下，中文高语义负例率反而更低，真正漂移集中在重复率、风格 gap、结构计数和证据类型组合。

当前自动生成的 Step13 findings 还有两处必须按数值修正：

- 当前最强 Step15 clean point estimate 是 non-domain-balanced (`0.866533 / 0.725220`)，不是 domain-balanced (`0.865333 / 0.644989`)。
- domain-balanced 相对 raw E5 的 ROC-AUC 差异已有统计支持；没有统计支持的是 AP 差异。

实现文件：

- runner: `scripts/step13_concept_drift_audit.py`
- summary: `reports/step13_concept_drift_audit.step16g_imbalance_validation_20260710.json`
- CSV: `reports/step13_concept_drift_audit.step16g_imbalance_validation_20260710.csv`
- generated doc expected from Linux: `docs/STEP13_CONCEPT_DRIFT_AUDIT_STEP16G_IMBALANCE_VALIDATION_20260710.md` (not yet synchronized locally)

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

### 12.5 为什么当前不宣称 Step 9 mixup 有效

Step16G 已让 E5 mixup 100pct 真正生成 `115` 条 synthetic rows，但相对相同 100pct support 的 non-mixup：

- AUC diff `-0.006000`, CI `[-0.031289, 0.021427]`
- AP diff `+0.001205`, CI `[-0.049545, 0.053483]`

因此当前最严谨的表述是：

> 当前数据不支持 Step 9 positive-pair mixup 的正向效果。Step15 v5 non-domain scorer 在当前固定测试集上显著超过 raw E5，但其 Phase4 mixup 存在弱父样本放大和跨域插值混杂，因此不能把 Step15 的提升归因于 mixup。

## 13. 当前可写入论文的结论边界

当前可以稳妥写：

1. 本项目构建了一个严格防泄漏的英文源域和中文目标域 seller-pair verification benchmark。
2. 中文目标域存在明显概念漂移，尤其体现在 style gap、模板复用、identifier 稀缺和 source-domain shortcut 失效。
3. Raw multilingual semantic ranking 在中文目标域上仍然很强，尤其 raw E5 和 LaBSE。
4. Step 7 LightGBM fusion 在当前边界下发生浅层 collapse，说明简单源域融合不能稳定迁移。
5. Step 9 LR/L2 和 residual adaptation 比 collapsed fusion 更有用。
6. Step 9 positive-pair mixup 是一个已执行但结果为负面的 training-only minority-regularization control。
7. Step15 v5 non-domain scorer 在当前 fixed test 上获得 `0.866533` AUC 和 `0.725220` AP，并在 paired grouped bootstrap 中超过 raw E5；这是 promising internal result，不是 prospective holdout 结论。
8. Step 11 图聚类适合候选 triage，但不能自动产生 same-controller ground truth。
9. 严格 cluster audit 显示当前 `212` 个 unique cluster sets 中没有完整 high-confidence same-controller cluster；两个 anchored cores 是同一底层 core 的重叠子集，另有 `8` 个 partial anchors。

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
| Step 15 | `schema/step15_evidence_type_policy.json` | `scripts/step15_train_incremental_hard_negative.py` |
| Step 16G | `schema/step16g_hard_negative_imbalance_policy.json` | `scripts/step16g_expand_hard_negative_train.py` |

运行环境纪律：

- Windows workspace 当前主要用于代码修改、文件审计、同步结果核查和文档维护。
- Step 7/9/11 这类依赖模型、LightGBM 或 Linux runtime 的重跑应在 Linux 服务器执行。
- 每次 Linux 重跑同步回 Windows 后，应先核查 summary、manifest、output_paths 和 fixed test container，再解释结果。

### 14.1 Step25-v2 配对局部复制机制诊断

Step25-v1 的全局模板清洗已经冻结为负结果。Step25-v2 不修改该结论，而是独立检查两个可能的实现瓶颈：只在当前 seller pair 内出现的复制文本是否被全局目录漏检，以及清洗后短文本是否被错误写成余弦 `0`。

Step25-v2 固定使用四个全量模型和一个可靠切片敏感性分析：

- `P0`: raw style + pair-local reliability mask + fold-train median imputation + reliability indicator
- `P1`: Step25-v1 global-clean style + global/local reliability intersection
- `P2`: pair-local-clean style + 与 P0 完全相同的 reliability mask 和 missingness transform
- `P3`: pair-local-clean style，缺失时显式回退 raw style
- `P4`: 在 pair-local reliable rows 上比较已拟合的 P0/P2，不重新训练

复制检测仅使用 identifier-redacted canonical-train pair text，以固定 12-character shingles 和 24-character minimum contiguous run 对左右文本对称去复制；不读取 label、evidence type、score、valid 或 test。模型仍是固定强正则 LR/L2，评估包括 English grouped OOF、source-only Chinese scoring、English+Chinese target grouped OOF 和 component-grouped bootstrap。

这是一个 D0 retrospective mechanism diagnostic。无论结果如何，它都不能选择论文主模型、不能进入 Step11/17，也不能撤销 Step25-v1。完整实现和解释边界见 `docs/STEP25_V2_PAIR_LOCAL_COPY_MISSINGNESS_DIAGNOSTIC_20260717.zh.md`。

Step25-v2 已完成，不再是待运行设计。其结果为中文 P0/P2/P3 target grouped-OOF AP `0.704847/0.670692/0.737365`，英文 P0/P2 grouped-OOF AP `0.468210/0.251926`，仅 `3/8` 机制门槛通过。该实验保留为“局部复制检测有效、统一清洗替换无效”的混合机制结果。

### 14.2 Step25-v3 复制感知双通道延续

Step25-v3 不继续把 clean representation 当作 raw representation 的替代品。它使用四个固定实验回答不同因果问题：

- `C0 matched raw style`：给 v3 新求解器和相同样本权重提供匹配 raw baseline。
- `C1 raw + clean, no copy penalty`：检验 clean style 是否能在保留 raw style 后提供增量，而不把效果归因于惩罚项。
- `C2 copy-aware dual-channel primary`：加入 raw-clean residual 和 copy-risk；相似度系数非负，copy residual/risk 系数非正。
- `C3 redacted-E5 sensitivity`：检查语义通道是否改变结论，只作 sensitivity，不可被选择为主模型。

输入仍为 Step25-v2 的 canonical train：英文 `401 = 116/285`、中文 `573 = 229/344`。英文 grouped OOF、英文 source-only 到中文、英文加中文的 target grouped OOF 都固定使用五个 seller-component folds，并要求与 v2 fold assignment 完全一致。中文 target OOF 是已被 v1/v2 消耗的 retrospective D0，只能支持 D1 replication candidate，不能支持论文晋级。

Step25-v3 修复了两个归因问题。第一，不可靠 pair-local-clean style 直接回退 raw style，raw-clean residual 为零，避免把缺失编码为中位数或余弦零。第二，额外运行纯 Step25-v1 global-clean missingness closure，单独比较冻结 fixed-zero 与 fold-train median plus indicator；该 closure 不参与 C2、任何 gate 或模型选择。

Clean scorer 明确禁止 direct identifier、candidate-rule、review label 和 evidence type。Identifier occurrence 只进入单独 operational control：用 English actionable occurrence rows 和 English C2 component-OOF probability 训练小型方向约束 offset expert，再对 Chinese source-only C2 做敏感性报告；中文标签不参与 expert 拟合，结果不能改变 clean model 晋级资格。

原 v3 已完成 Linux 运行和同步，产物内部一致，但求解器终止无效，不能用于最终解释。v3.1 的 11 项契约测试、四个 config-only preflight、Linux runner、`44/44` KKT audit 和两份 feature byte-parity audit 均已通过。正确收敛后的得分仅发生微小变化，gate 仍为 `2/11`，从而把旧 v3 的定性失败转化为可正式冻结的严格负结果。完整修复边界见 `docs/STEP25_V3_1_SOLVER_CONVERGENCE_REPAIR_20260718.zh.md`，结果审计见 `docs/STEP25_V3_1_RESULT_AUDIT_20260718.zh.md`。

## 15. 当前实验设计评价

当前设计总体是合理且严谨的，原因是：

1. 数据域边界清楚，英文源域和中文目标域没有混成一个模糊池。
2. Step 5 冻结标签和 seller component 隔离降低了泄漏风险。
3. Clean scientific model 与 identifier operational control 分离，避免把直接证据当模型泛化能力。
4. Step 7/9/12/13 构成了从点估计到统计稳健性再到概念漂移解释的完整链条。
5. Step 11 严格降级为候选 triage，避免把模型预测反喂成真值。
6. Positive-pair mixup 被限制在 training-only，不污染 frozen labels、valid 或 test；但 Step15 当前仍需修复父样本权重继承和跨域配对约束。

当前主要限制也很明确：

1. 中文 strict test 已扩为 `50` 个 positive / `150` 个 negative，但其中只有 `22` 个 direct/component primary positives，软证据标签仍占较大比例。
2. 中文 seller-facing proof positives 稀缺，限制了 few-shot 的可学习空间。
3. Step 7 fusion 在中文上弱于 raw semantic baseline，说明源域融合存在 shortcut。
4. Step 11 目前多为模板/主题候选，proof-level full cluster 尚未出现。
5. Relation reliability filter 仍依赖已有特征和抽取质量，不能替代人工证据审计。
6. Step15 Phase4 让低权重父样本生成默认满权重 synthetic positives，并把跨中英文合成行当成独立 domain；当前 domain-balanced 结果因此存在目标函数混杂。

因此，当前项目最可信的科研方向不是简单宣称 few-shot 胜利，而是围绕以下主题组织：

- cross-domain concept drift under scarce identity anchors
- transfer-safe seller-pair verification
- minority regularization under imbalanced target supervision
- graph candidate triage with strict evidence audit
- distinction between semantic similarity and identity reliability in darknet seller linkage
