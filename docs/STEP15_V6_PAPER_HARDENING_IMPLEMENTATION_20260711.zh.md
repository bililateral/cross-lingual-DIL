# Step15 v6 论文强化实施说明

更新时间：2026-07-12

## 1. 实施状态

Step15 v6.4 已在 Windows 工作区完成代码、policy、严格归纳式特征血缘、输出隔离和静态测试，但尚未在 Windows 上执行任何模型训练。真正的模型训练、重采样审计和图谱构建必须在已经配置模型与依赖的 Linux 服务器上完成。

当前方法分支：`method/step15-v6-paper-hardening`。

当前 v5r 已冻结为内部开发基线 `internal-dev-v5r-20260711`：

- JSON manifest：`reports/manifests/step15_internal_dev_v5r_20260711.json`
- CSV manifest：`reports/manifests/step15_internal_dev_v5r_20260711.csv`
- manifest 覆盖 v5r summary、policy 及 summary 中全部 output paths；缺失任何 artifact、valid prediction、test prediction 或 mixup manifest 都会拒绝冻结。
- 当前中文 `test = 200` 只被定义为 `fixed_internal_development_test_not_prospective_final_holdout`，不再表述为最终独立测试集。

## 2. 为什么不能继续直接调 v5r

v5r 在当前内部测试边界上的非 domain-balanced seed-mean 点估计为：

- ROC-AUC：`0.866533`
- AP：`0.725220`

该结果证明 evidence-type hard-negative curriculum 具有研究价值，但尚不等于论文级结论，原因包括：

1. 当前 200 条中文 test 是经过多轮方法开发后形成的内部开发边界，不是方法冻结后前瞻收集的 holdout。
2. 50 个 test positive 中只有 18 个属于 strict direct/component positive，6 个属于 soft-primary，另外 26 个只能作为 secondary/sensitivity positive。
3. v5r clean 特征仍包含 `candidate_rule_count_raw`、未做域内标准化的 retrieval-only lexical/structural 字段，以及英文特定 uppercase gap，存在 source-domain shortcut 和间接 identifier 泄漏解释风险。
4. v5r 的 phase 主要用于数据递增诊断；需要一个同架构、同特征、同训练边界的 all-at-once baseline，才能把提升归因到 curriculum，而不是网络结构或输入变化。
5. 三个 seed 不足以证明优化稳定性，seed-mean 还可能掩盖单 seed 崩溃。
6. 旧 AP 在并列分数时受 CSV 行顺序影响；旧 PR-AUC 实际只是 AP 的重复字段，MAP/MRR 在没有 query groups 时也不成立。

因此 v6 的目标不是继续围绕 test 调权重，而是修复特征血缘、方法归因、指标语义、随机性和输出可追溯性。

## 3. Step4/Step7 特征血缘与归纳式参考统计修复

### 3.1 候选样本数量不变

Step4 代码定义了新的派生字段：

`candidate_rule_count_non_identifier`

它等于全部 candidate rule hits 减去：

- `shared_contact_exact`
- `shared_pgp_fingerprint`
- `shared_pgp_fingerprint_via_aux_alias`

原字段 `candidate_rule_count`、候选规则、排序键、候选阈值和 `pair_uid` 均未修改，因此该改动只增加解释字段，不增加或删除训练样本。

为了保护既有 Step4/Step7/v5/v5r 结果，v6 Linux runner 不再重建 canonical Step4。它先用已经冻结的 manifest 校验三个 pool 的 pair count 与排序后 `pair_uid` SHA-256。当前冻结值是 EN `6683`、ZH strict `3857`、ZH aux `580`。canonical CSV 尚未物化该新列时，v6 隔离构建器直接从冻结的 `candidate_rule_hits` 删除 contact/PGP 规则后精确派生，并在 manifest 中报告派生来源计数。

### 3.2 不重新计算 embeddings，也不覆盖 canonical Step7

本轮 seller profiles 和 pair universe 不变，没有理由重新计算大模型 embedding。v6 不再调用 canonical preview rebuild 或 nonsemantic refresh；`step15_build_v6_inductive_pair_features.py` 读取现有 semantic-enriched pair feature 表，原样复制全部 embedding/reranker 分数，只在 `reports/step15_v6/features/` 写入隔离特征文件，并对语义列计算前后哈希。

验收条件：

- pair universe 完全一致；
- 所有 semantic column hashes 完全一致；
- v6 隔离文件中的 `candidate_rule_count_non_identifier` 在每行均有值；
- semantic scores 不重新计算。

### 3.3 修复完整数据池统计造成的传导式评估

旧 pair feature preview 中的 IDF、boilerplate/rare signature 和 market percentile 是按整个语言池计算的。它没有使用 test 标签，不是标签泄漏，但 valid/test seller 的协变量参与了参考分布，严格来说属于 transductive feature construction。

v6.4 保留既定 30 维定义，但重新计算其中 18 个 corpus-relative 字段：

- 英文参考只使用 EN frozen train pairs 涉及的 `582` 个 seller；
- 中文参考只使用 ZH frozen train pairs 涉及的 `676` 个 seller；
- reference fitting 只使用 split membership 选择训练 seller，不读取标签值决定统计量；
- valid、内部 test 和 Step11 候选统一应用冻结的 train-seller IDF、boilerplate、rarity 和 percentile reference；
- reference bundle、producer、policy、全部输入和输出均记录 SHA-256；
- 现有全量内存验证确认 EN `6683`、ZH `3857` pair 不变，所有 semantic hashes 前后一致。

这使 v6 主方法成为严格的 inductive evaluation。旧 Step7/Step9 结果仍可作为历史诊断，但不得与 v6.4 的归纳式方法表混合解释。

## 4. v6 两个特征视图

### 4.1 strict-clean 30D 主视图

主方法从 v5r 34 维 clean 视图中移除：

- `candidate_rule_count_raw`
- `sparse_lexical_similarity_raw`
- `structural_support_score_raw`
- `uppercase_ratio_mean_percentile_gap_abs`

保留 30 个 semantic、reranker、content-structural 和跨市场相对 style-gap 特征。其 corpus-relative 值全部来自前述 frozen train-seller reference。它不包含 direct contact、PGP、contact count、原始 candidate rule count 或英文专属 uppercase 字段。

目的：让 clean scientific model 的分数只能由跨语言可比较的内容、结构与风格信号解释，不能通过候选生成规则间接读取 identifier 存在性。

### 4.2 normalized-retrieval 33D 消融视图

在 strict-clean 30D 基础上加入：

- `candidate_rule_count_non_identifier`
- `sparse_lexical_similarity_raw`
- `structural_support_score_raw`

其中 lexical 和 structural retrieval 字段只使用 final Phase3 train rows 分别在英文、中文域内拟合 mean/std；未知域直接报错。随后所有特征再进入统一全局 scaler。

该视图是 retrieval feature ablation，不是默认主方法。只有它稳定优于 strict-clean，才能说明 retrieval strength 在排除 identifier 并做域内标准化后仍有可迁移价值。

## 5. 统一 train-only scaler

所有使用同一 feature set 的 v6 模型共享一个 scaler。scaler 只在 `phase3_add_contact_url_noise` 的最终 train-only 数据上拟合，并在 Phase0 至 Phase4、M0 至 M5、valid 和 test 上复用。

这样做解决两个问题：

1. 不同 phase 不再因为各自 mean/std 不同而改变特征坐标系；warm-start 参数真正位于同一个空间。
2. M0-M5 的差异不再混入 scaler 差异，方法增量可以被单独解释。

artifact 保存：

- feature names；
- global means/stds；
- domain-standardized feature names；
- 每个域的 means/stds 和 fit row count；
- standardizer SHA-256。

## 6. seller-component-aware 样本权重

同一 seller component 可能产生多条 pair edge。若每条边等权，一个大 component 会被重复计算，造成伪样本量和置信区间过窄。

v6 对每条真实 train edge 先施加：

`component_weight = 1 / sqrt(component_train_edge_count)`

再把 component weight 归一化为全体均值 1。可信 mixup 行继承两个 parent component factor 中较小者。

v6 明确的权重流水线为：

1. component inverse-sqrt weight；
2. row-quality / `training_sample_weight`；
3. 按前两步所得 effective weight mass 计算的 class-balance factor；
4. evidence-type multiplier；
5. 可选 post-quality domain mass balance。

class balance 不再按正负原始行数计算。它把 component 与 row-quality 权重后的正类、负类 effective mass 分别拉到总质量的一半，再保持总体均值不变。所有阶段均输出平衡前后正负 effective mass、min/mean/max、component count、最大 component edge count 和 domain effective mass，便于复核。

## 7. M0-M5 递进对照

所有主实验固定：

- 输入：strict-clean 30D；
- hidden dimension：16；
- optimizer、L2、patience、最大 epoch 相同；
- 英文 source train + 中文 target train；
- 中文 valid 只用于 early stopping、lambda/最终候选选择和 threshold；
- 中文 test 不参与任何选择。
- M0、M1、M2 各固定 `1000` 次 optimizer update；M3 固定 `4 × 250` 次，三者总更新预算一致。

### M0：all-at-once binary baseline

一次性使用 Phase3 全部 evidence types，从随机初始化训练 binary identity head：

- 不使用 evidence-type multiplier；
- 不做 domain balance；
- 不做 mixup；
- `lambda_evidence = 0`。
- 单阶段固定 `1000` 次 optimizer update，并从这些 checkpoint 中只按 `zh_valid AP` 保留最佳参数。

它是最关键的同架构、同特征 baseline。M3 如果不能稳定超过 M0，就不能声称 curriculum 本身有效。

### M1：M0 + evidence-type weighting

仍是 all-at-once、固定 1000 次更新，仅加入 train-only evidence weights：

- public contact/URL noise：8.0
- template clone negative：1.5
- semantic topic negative：1.5

M1 vs M0 只检验困难负样本权重是否有价值。

### M2：M1 + domain balance

在 row quality 与 evidence weights 后，将英文和中文真实域的 effective weight mass 拉平。M2 同样固定 1000 次更新。M2 vs M1 检验 domain balance；不再把 synthetic pseudo-domain 当成第三域。

### M2b：M3 的同训练预算非课程对照

M2b 与 M3 使用完全相同的四个连续训练阶段、同一 seed、同一 scaler、每阶段固定 250 次 optimizer update、相同的 250 个 validation checkpoint 选择机会和 warm-start 方式，但四个阶段从一开始都使用 Phase3 的完整 evidence-type 集合。它不逐步引入困难负样本。固定预算仍从 250 个 checkpoint 中选择 zh_valid AP 最佳参数，不因 patience 提前终止。

M2 现在也使用与 M3 相同的总更新数，因此 `M3 vs M2` 不再混入总训练预算差异；但它仍混入“单阶段连续优化”和“四阶段重启 optimizer moment”的轨迹差异。M2b 进一步复现 M3 的四阶段优化轨迹而始终暴露完整数据。因此课程效应只有在 `M3 vs M2` 与 `M3 vs M2b` 两个 AP 区间下界都大于 0 时才允许声明。

### M3：真正 warm-start curriculum

M3 使用同一 seed、同一 scaler 和上一 phase 的 best parameters，依次训练：

1. direct/component positive + ordinary negative；
2. 加入 soft positive 与 semantic-topic negative；
3. 加入 template-clone negative；
4. 加入 public-contact/URL noise。

每一阶段重置 optimizer moment，但不重置模型参数。正式 M3/M2b、M4/M4c、M5 及 M3 ablations 均固定 250 次 optimizer update；这里匹配的是更新次数与 checkpoint 选择机会，阶段数据组成差异就是被检验的处理。命令若跳过前置 phase 或打乱 phase，会在任何训练和文件写入之前失败，不能再生成伪造的 canonical Phase3 endpoint。

### M4：M3 + trusted positive-pair mixup

只有 Phase4 增加 mixup：

- 同真实语言域；
- 同 evidence type；
- parent `training_sample_weight >= 0.55`；
- usable/core-transfer/evidence-confidence gates 全部通过；
- 从五个最近邻正例中选择 partner；
- synthetic weight 取两个 parent 的较小值；
- binary/count 特征复制 anchor，只插值连续特征；
- synthetic evidence label 被 mask；
- synthetic rows 永不进入 Step5、valid 或 test。

### M4c：M4 的同训练预算无 mixup 对照

M4c 完整复现 M4 的五阶段训练和 Phase4 的 250 次继续更新，但显式禁止生成 synthetic positive。`M4 vs M3` 检查加入 Phase4 后的总体增益，`M4 vs M4c` 排除“多一次完整优化和 checkpoint 选择机会”本身带来的变化。两者样本数差异是 mixup 处理的定义组成，而不是额外优化次数。只有这两个 AP 区间下界都大于 0，mixup claim 才成立。

### M5：M3 + auxiliary evidence head

M5 不使用 Phase4 mixup，在 M3 基础上增加 evidence-type auxiliary head。只预注册：

- `lambda_evidence = 0.1`
- `lambda_evidence = 0.3`

两个候选的选择只看十个 seed 预测均值形成的 ensemble zh_valid AP；同时记录 per-seed valid AP 均值作为稳定性诊断。test 指标不参与 lambda 选择。

实现上两个 lambda 候选在训练阶段都只写 artifact 和 `zh_valid` prediction。完成十个 seed 后，脚本先按 valid ensemble AP 冻结选择，数值完全相同时按简单性顺序优先 `lambda = 0.1`；随后只从已冻结 artifact 物化被选候选的 `zh_test` prediction，绝不重新训练。未选 lambda 不生成 test。这样避免了“两个 lambda 都先看 test，再声称只按 valid 选择”的隐性选择风险。

## 8. 标签来源消融

对 M3 额外运行：

1. `gold_only`
2. `gold_plus_high_confidence_silver`
3. `gold_plus_all_silver`

high-confidence silver 仅预注册 `silver_direct_or_contact`（当前权重 `0.55`）；低权重 component closure、template structural 和 relaxed tiers 不被包装成 high confidence。该定义不按 test 表现临时改动。该消融用于回答当前提升究竟来自方法，还是来自 213 个 weak positive 和 158 个 weak negative 的数量扩张。

## 9. source-label-only LR/L2 baseline

新增 `step15_v6_source_only_lr_l2_strict_clean`：

- 只用英文 train；
- 只用英文 valid 早停和选择 threshold；
- 英文 test 报告 source-domain 表现；
- 中文 test 仅做一次迁移评估；
- 使用同一 strict-clean 30D；
- 使用 L2 正则和 component-aware weights。

该模型的身份监督和参数训练只使用英文标签，但中文 18 个 corpus-relative 字段使用中文 frozen train sellers 的无标签协变量参考。因此准确名称是 `source-label-only with unlabeled target-reference preprocessing`，不是“不接触任何目标协变量”的纯 zero-shot。该 baseline 用于区分“Step7 LightGBM collapse”与“英文身份监督本身不可迁移”；raw E5/LaBSE/BGE cosine 才是无需目标身份标签、也无需该融合层训练的纯语义迁移控制。

## 10. 指标语义 v2

统一版本：`2026-07-v2-tie-aware`。

### 10.1 AP

相同分数的样本作为一个完整 threshold group 一次性更新 precision/recall。AP 不再依赖 CSV 中并列样本的行顺序。

### 10.2 PR-AUC

PR-AUC 使用 tie-grouped precision-recall curve 的梯形积分，不再复制 AP 数值。

### 10.3 MAP/MRR

当前任务没有预注册 query groups，因此 global pair ranking 不再输出伪 MAP/MRR；字段值为 null，并记录 `not_applicable_without_preregistered_query_groups`。

### 10.4 threshold metrics

ACC、balanced accuracy、F1、precision、recall、specificity 均使用 valid-frozen threshold，并在 Step12 做 component-grouped bootstrap。

## 11. Step12 v6 统计设计

固定比较：

- M1 vs M0
- M2 vs M1
- M3 vs M2
- M3 vs M2b matched-budget full-data replay
- M4 vs M3
- M4 vs M4c matched-continuation without mixup
- validation-selected M5 vs M3
- M3/M4 vs strongest clean Step9 control
- validation-selected final v6 vs M0
- final v6 vs strongest clean Step9
- final v6 vs raw E5

主指标 AP，次指标 ROC-AUC 和 PR-AUC。主分析先冻结每个方法的十种子 seed-mean scorer，只按 `split_component_id` 做 paired grouped bootstrap；模型表和切片表中的无后缀 `*_ci_low/high` 均指这个主区间。补充分析再同时重采样 model seeds 与 test components，输出 `*_two_level_ci_low/high`，用于量化训练随机性。模型间 comparison 表同时保留 `primary_paired_grouped_component_bootstrap` 和 `supplemental_two_level_seed_and_component_bootstrap` 两行，promotion 与方法贡献判断只读取前者。相同 seed ID 的补充分析共享 seed 重采样索引。Step9 正式控制也扩为相同的十个 seed，避免三 seed 对十 seed 的 ensemble-size 混杂。

显著性检验不再把普通 bootstrap 中差值符号比例冒充 p-value。v6.4 对每次 permutation 按 `split_component_id` 整组交换 candidate/baseline score vector，重新计算非可加的 AP/ROC-AUC/PR-AUC，并使用双侧 `(1 + extreme_count) / (B + 1)`。Holm-Bonferroni 只校正该 paired randomization p-value，且在“analysis mode × metric”内把 all-test 与各 evidence slice 的全部比较放在同一 family 中。

promotion rule：

1. final v6 相对 M0 的 AP difference CI 下界大于 0；
2. final v6 相对 strongest clean Step9 的 AP difference CI 下界大于 0；
3. final v6 相对 M0 和 strongest clean Step9 的两个配对比较中，都必须至少有 8/10 个同编号 seed 的 AP 差值为正。
4. 上述两个 required-baseline AP comparisons 的 Holm-adjusted p-value 均不大于 0.05。
5. 在 `strict + soft primary positive vs all negatives` 切片上，相对 M0 和 valid-selected strongest clean Step9 的 grouped AP difference CI 下界也都必须大于 0；18 个 strict-only positive 继续作为小样本诊断，不单独决定 promotion。

未满足时，Step11 publication validation 默认被阻止；只能显式使用 diagnostic override，且输出必须标记为 non-promoted diagnostic。

## 12. 五个中文 test evidence slices

Step12 强制校验 Step16F test positive：

- strict direct/component：18
- soft primary：6
- secondary/sensitivity：26

报告：

1. all test：50 positive / 150 negative
2. strict vs all negatives：18 / 150
3. strict + soft vs all negatives：24 / 150
4. soft only vs all negatives：6 / 150，明确标记不稳定
5. secondary only vs all negatives：26 / 150

禁止只报告全体 50 positive 的单一 AP，而隐去证据强度差异。

固定测试边界不仅检查 `200 / 50 / 150` 聚合计数，还预注册排序无关的 SHA-256：一份覆盖 `pair_uid + review_label + split_component_id`，另一份覆盖 50 个 test positive 的 `pair_uid + paper_evidence_tier`。任何同计数换样本或证据层级漂移都会 fail closed。

## 13. Step13 provenance-aware drift

Step13 新增以下独立 cohort：

- raw Step4 candidate universe
- gold train
- silver train only
- fixed valid gold
- internal development test gold

silver train 的分布变化明确标记为 active-sampling distribution，不能当作自然 EN-vs-ZH concept drift。核心 Step15/Step12 runner 不再提前生成 Step13；只有 Step12 promotion 通过并完成 Step11 显式图审计后，后置 runner 才同时显式传入隔离后的 `--step7-summary`、`--step9-summary`、`--step12-v6-summary`、`--step11-manifest` 和 `--step11-audit` 生成当前最终版 Step13，且不自动寻找最新文件。这样不会把“尚未经过图验证的漂移审计”误记成最终审计。

## 14. Step11 v6 验证

只有 Step12 promotion 通过后，`step11_build_v6_runtime_policy.py` 才生成 runtime policy。runtime policy：

- 禁用 auto selector；
- 读取 validation-selected Step15 v6 experiment/phase；
- 使用 10-seed ensemble；
- clean allow-list 固定为 validation-selected Step15 v6、matched M0、validation-selected strongest clean Step9 和 `raw_bge_m3_cosine`；旧 Step7 LightGBM BGE 不再作为该图对照，因为它是在旧 corpus-relative 特征尺度上训练，直接作用于 v6 inductive features 会形成 preprocessing mismatch；
- raw BGE 直接读取 v6 pair table 的 `embedding_cosine_bge_m3`，其图阈值只取 Step12 `raw_bge_m3_cosine` 在 `zh_valid` 上冻结的 `mean_zh_valid_scores` threshold，不读取 test 指标选阈值；
- runtime policy 同时逐文件复核 Step15 active manifest 和 Step12 completion manifest，把模型、预测、v6 特征、Step12 summary/model-metrics/policy 全部绑定到 SHA-256；
- publication runtime 除关闭 auto selector 外，还拒绝任何不在固定 scorer-family/scorer-token roster 中的显式 CLI 请求；
- 所有输出写入 `reports/step11_v6/`，不覆盖旧 Step11；
- manifest builder 校验 validation mode、runtime-policy hash、summary hash、全部 output-path hash，并要求 scorer token 集合与预注册 roster 完全相等；主图阈值高于分数上限、主阈值下无候选边或图过滤后无边，任一情况都会拒绝进入 publication manifest；
- publication manifest 同时绑定自己的 CSV 路径、SHA-256 和行数；cluster audit JSON 记录 canonical self-hash，并绑定 audit CSV 的 SHA-256、行数、decision counts 和 per-scorer counts；Step13 在使用图审计前重新核对全部字段与实际 CSV；
- runtime policy、scored-pair CSV、cluster CSV、summary、manifest、cluster audit 和最终 Step13 都采用不可变路径写入：同路径同字节允许幂等重放，同路径异内容直接失败并要求新 run ID 或输出路径，不再静默覆盖；
- cluster audit 只读取该 manifest，不再直接扫描或 glob summary；
- 每个 summary 使用自己记录的 runtime policy 回放过滤过程。

`clean_topology` 与 `identifier_assisted_operational` 必须分开运行和报告：

- clean mode 把 shared contact/PGP 权重置零，关闭 direct-identity context 和 direct hard keep；身份字段只能用于过滤完成后的 proof-edge retention 审计，不能改变入图结果；
- operational mode 才允许 direct identity 加权与 hard keep，并保证 direct proof edge 通过 reliability、reciprocal、shared-neighbor 和 triangle stages；
- 每个阶段都报告 reviewed proof-positive retention、reviewed negative removal、isolated two-seller proof-pair retention、seller/component/edge 数量。冻结标签只用于后验审计，不参与过滤决策。

## 15. 标签隐藏、分数盲的独立复核

第一版队列虽隐藏了 evidence tier，却错误保留 `original_review_notes`，其中部分文字直接写有旧 positive 或 Step11 retained 结论，因此该版审查已作废且未用于任何统计。修正版 Step16H 将 80 个 valid/test positive candidates 与从同一冻结 valid/test 边界确定性抽取的 80 个 negative controls 混合，分别生成 reviewer A/B 两份不同顺序的 160 行队列。队列隐藏：

- 旧 evidence tier
- recommended use
- 旧 confidence/rationale
- 原 `review_label`、`reviewer_id` 和全部 conclusion-bearing `review_notes`
- 所有 Step7/9/11/15/17 分数和图谱状态

审查者只能看到 Step4 原始左右商品/profile 预览、共享 contact/PGP/alias、标题/描述/类别重合证据，不知道单行原标签。80 个负控制按 `valid = 30 / test = 50` 固定配额和预注册哈希种子抽取，用于检查审查者是否把模板或公共数据误判成同控制。

允许结论：

- `strict_same_controller`
- `soft_same_controller`
- `different_controller`
- `uncertain`

v2 曾完成一轮 AI-assisted sensitivity review，但其 completion manifest 绑定的是较早 producer-script bytes，不能和当前代码组成 hash-closed publication bundle，因此已被 v3 完全取代且不进入当前提交或正式统计。

v3 evidence-complete 盲审已经完成：reviewer 只看到 opaque `blind_id`、Step4 原始 pair evidence、Step3 每侧原始 contact occurrences 及独立重建的 component candidate path；`pair_uid`、旧标签、parser eligibility、mapping 和模型/图分数均隐藏。共享工作区只保证程序性遮蔽，不声称文件系统访问控制隔离。

v3 两名独立 AI reviewer 的 160 行 exact agreement 为 `0.875`，Cohen's kappa 为 `0.7992`，nominal Krippendorff's alpha 为 `0.7980`。positive-candidate 子集 agreement 为 `0.8875`，negative-control 子集为 `0.8625`；20 个冲突已由第三 reviewer 裁决。最终旧 positive candidates 为 `18 strict / 48 soft / 11 different / 3 uncertain`；80 个 negative controls 为 `0 strict / 24 soft / 55 different / 1 uncertain`。这比 v2 稳定，但仍是 AI-assisted sensitivity audit，不是人类 gold annotation，也不写回 Step5/Step16F。当前 test 仍只能作为内部开发边界。

## 16. 输出隔离和 summary 合并

v6 全部结果位于：

- `reports/step15_v6/`
- `reports/step12_v6/`
- `reports/step11_v6/`

Step4、Step7 和 Step15 evidence labels 都作为只读冻结输入，不由 v6 runner 重建。Step7 不重新训练或覆盖 canonical summary；`step15_v6_refresh_step7_control.py` 只读旧预测并把 metric-v2 control 写入 `reports/step15_v6/baselines/`。Step9 使用 `--output-root reports/step15_v6/baselines/step9`，并显式读取 v6 归纳式 EN/ZH pair features；所有模型、预测和 summary 均写入隔离目录。Step15 的 intermediate phases 只写 artifact 与 `zh_valid` prediction；部分 warm-start phase prefix 永远不能被当作 endpoint。除 M5 valid-selected 特例外，每个实验只有完整预注册 endpoint phase 才能读取并输出 `zh_test` prediction/metrics。核心 runner 共 11 个阶段，第一阶段在训练前编译脚本并运行全部 85 项契约测试，最终终止于 Step12；Step13 只存在于 promotion 后的 Step11 runner 最后阶段。

Step15 v6 summary 的写入模式是：

`merge_by_experiment_phase_seed_same_input_manifest_only`

因此核心 M0-M5 和 ablations 可以分两条命令运行，不会相互覆盖。只有 policy version、代码/输入文件 hash 完全相同时才允许按 `(experiment, phase, seed)` 合并；任何输入边界变化都会拒绝合并并要求新版本输出路径。

兼容性检查发生在训练和任何 v6 文件写入之前；首次创建新子目录由原子写函数负责。训练后先由 `step15_validate_v6_outputs.py` 证明完整 experiment/phase/seed coverage、固定更新数、M4/M4c Phase0-3 valid predictions 和参数完全一致、M4-only Phase4 synthetic rows、M5 双候选十种子 valid 覆盖、仅 selected M5 test 输出和 source-only 十种子，再允许生成 manifest。活动 manifest 同时冻结归纳式 feature reference/manifest、该验证报告、主 Step15 summary、source-only summary、隔离的 Step7/Step9 summaries、五组 Step9 十种子所引用的 artifacts、Step3 profiles、Step4 candidates、Step5 labels、canonical/inductive Step7 features、Step15 evidence labels、policy、producer scripts 和所有被引用输出的 SHA-256，并复算 manifest 自身哈希。Step12 逐文件复核上述绑定，拒绝中断重跑留下的旧预测。canonical Step12 policy v5 输出固定使用 5000 次 bootstrap 与 5000 次 paired randomization，并写入 `reports/step12_v6/method_audit_v4_inductive_20260712/`；目标文件任一已存在即 fail closed，不能静默覆盖。

## 17. 预测分数序列化完整性

第一次 Linux v6 执行到 Step12 输入校验时，`step15_v6_m0/seed=20260325/valid` 的 summary ROC-AUC 为 `0.766667`，但 Step12 从 prediction CSV 复算得到 `0.766852`。原因不是统计实现不同，而是训练脚本用全精度内存概率计算 summary，却把 CSV 分数舍入到六位小数；接近分数被量化为并列后，ROC-AUC、AP 和 PR-AUC 的排序关系可能改变。

修复后，Step15 主模型、source-only 对照以及被 Step9 复用的 Step7 prediction writer 均输出 Python 可往返的完整浮点表示。Step12 的严格复算容差不放宽；新增测试专门验证小于 `1e-6` 的正负分数差不会在落盘时消失。由于 producer-script 和预测文件均受活动 manifest 哈希约束，修复前的未完成 Linux bundle 必须归档，不能与修复后的结果混用或作为论文指标来源。

第一次修复后的 Linux resume 又在 M5 test 物化处发现一个旧消费者契约：代码仍对 artifact threshold 先做六位舍入，再与全精度 CSV threshold 比较。现在物化前会严格绑定 artifact、run record 和所有 validation prediction rows 的完整精度 threshold，检查数值有限且位于 `[0,1]`，并逐行验证 `pred_positive == (prob_positive >= threshold)`。端到端 M5 测试使用 `0.5000003456789012`，另有负测试确认 threshold drift 和 prediction drift 都会 fail closed。Step15-v6/Step12-v6 相关路径已不存在旧式 rounded-threshold equality check。

### 17.1 Step12 确定性 CPU 并行优化

第一次 canonical Step12 统计运行只有一个 Python 进程持续占用约一个逻辑核。它并未卡死，但旧实现对每个 grouped-bootstrap 样本分别为 13 个指标重复调用完整评估函数：fixed seed-mean 与 two-level 各一次，因此同一标签/分数向量最多被完整排序和计算 26 次。模型审计、5 个 evidence slices 以及 19 个 comparison scopes 也全部串行。5000 次 bootstrap 和 5000 次 randomization 的科研设计本身没有问题，问题是执行计划重复计算且没有利用服务器的 24 个物理核。

优化版不改变统计定义。每个重采样分数向量只调用一次 `evaluate_probabilities`，然后从同一返回对象提取 AP、ROC-AUC、PR-AUC、阈值指标和 confusion counts；metric-specific paired randomization 只计算当次需要的排名指标。并行任务按三类固定边界划分：24 个 model/alias、5 x 24 个 evidence-slice/model、以及按 policy 原顺序生成的 19 个 comparison-scope。policy 上限固定为 24 个 worker，每个 worker 固定 1 个 native thread，避免 24 个 Python 进程各自再次启动多线程库造成过度订阅。

随机性仍由原公式唯一决定。每个 model task 使用原 bootstrap seed 与 `bootstrap_seed + 1000003 * seed_count`；每个 slice task 使用原 `base_seed + 200003 * (slice_offset + 1)` 及 seed-count 偏移；每个 comparison scope 保留原 comparison、scope、two-level 和 metric-specific randomization 偏移。worker 不接收共享 RNG，也不依赖领取任务顺序。`ProcessPoolExecutor.map` 按输入顺序返回，最终 CSV 的 model、slice、comparison、analysis-mode 与 metric 行顺序和旧版一致。

等价性验证分三层：所有 24 个原有 Step12 单元测试继续通过；新增测试证明 `workers=1` 与 `workers=2` 的 model rows、slice rows 和 comparison rows 完全相同；并直接从 commit `0fced64` 动态加载优化前实现，在相同 ties、components、seed scores 和随机种子下逐字段比较 model metrics、slice metrics、primary grouped bootstrap/randomization 与 supplemental two-level 输出，结果均为 exact match。小型旧/新基准中，消除重复评估本身得到 `7.19x` model-loop 加速，尚未计入 24 进程并行收益。

该优化不使用 GPU。Step12 操作的是每次约 150-200 行的小数组，并反复做稳定排序、组件索引和 Python 控制流；迁移 CUDA 会增加传输、kernel launch、tie/rounding 重实现和结果一致性风险。RTX 5090 应用于需要大模型前向或训练的步骤，而 Step12 最合理的硬件路径是当前双路 CPU 的 24 个物理核。运行过程中会分别打印 model bootstrap、slice bootstrap 和 paired comparison 三阶段完成时间，避免长时间无输出被误判为卡死。

## 18. 当前允许与禁止的论文结论

在 Linux v6、Step12、盲审和 prospective holdout 完成前，只允许说：

> 已实现并预注册一个 evidence-type incremental hard-negative curriculum 的论文强化评估框架，用于检验它是否缓解中文目标域中模板复用、主题相似和公共联系方式噪声造成的 hard-negative concept drift。

当前禁止说：

- v6 已经优于 v5r；
- curriculum 已被统计显著证明；
- 50 个 test positives 全部是 gold same-controller；
- Step11 clusters 是真实团伙；
- 当前内部 test 是独立 final holdout。

最终论文主张仍需要方法冻结后构建的 prospective Chinese holdout，只评估一次，且 seller component 与现有 train/valid/internal-test 完全隔离。

## 19. 当前验证与 Linux 执行入口

Windows 本地已通过：

- Python compile；
- 两份 Bash runner 已通过 Git Bash `bash -n`；
- Step15、source-only、Step12 config-only；
- 全量本地测试为 `85/85` 通过，覆盖特征参考、M5 frozen-artifact test 物化、exact frozen-threshold binding、sub-micro prediction score 序列化、paired component randomization、Holm family、strict+soft promotion gate、串行/进程池确定性等价、manifest/hash、raw-BGE threshold provenance、publication scorer allow-list、空主图拒收、不可变写入、cluster-audit CSV 完整性、Step13 provenance 和图验证合同；
- Step4 candidate universe：EN `6683`、ZH strict `3857`、ZH aux `580`，修改前后 pair hash 完全一致。

整个仓库同步到 Linux 后，从仓库根目录运行：

```bash
chmod +x scripts/run_step15_v6_linux_20260711.sh \
         scripts/run_step11_v6_after_promotion_20260711.sh

bash scripts/run_step15_v6_linux_20260711.sh
```

第一条 runner 完成后必须先审查 `reports/step12_v6/method_audit_v4_inductive_20260712/step12_v6_statistical_robustness.json` 的 `promotion` 与 `method_claims`。只有 `promotion.eligible = true` 时才运行：

```bash
bash scripts/run_step11_v6_after_promotion_20260711.sh
```

若 promotion 失败，Step11 publication validation 应停止；不能通过调 internal test、降低阈值或启用 auto selector 绕过。
