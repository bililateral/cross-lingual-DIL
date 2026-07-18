# Step25 之后的科研推进计划

更新日期：`2026-07-18`

## 1. 当前决策

Step25-v1、v2 和正确收敛的 v3.1 已共同回答同一个问题：依赖全局或 pair-local 文本去污染、显式 copy-risk penalty 和方向约束，不能在当前中文监督上稳定改善同控制 seller-pair 排序。继续 Step25-v4、调 penalty、换阈值或把 C3 事后改成主模型都会继续消费 D0，并产生 model-selection bias。

因此立即冻结 Step25 方法线。下一步不再以“再训练一个模型”为起点，而以“论文证据是否成立”为起点。

## 2. 当前最关键的事实

- English canonical train：`401 = 116 positive / 285 negative`。
- Chinese canonical train：`573 = 229 positive / 344 negative`。
- Chinese positive 中 `213/229` 是 train-only silver，canonical non-silver positive 只有 `16`。
- 当前 D0 已被 Step24/25 多轮机制实验消耗，不能继续作为新方法选择边界。
- C3 redacted-E5 sensitivity 虽有小幅点估计改善，但 source-only 和 target grouped-bootstrap CI 均跨零；non-silver slice 只有 `16` 个正例，不能支撑新主线。
- Operational identifier control 只覆盖 `3` 条 verified-direct Chinese pair，不能证明一般化的 evidence fusion。

当前主要瓶颈不是优化器、模型容量或阈值，而是独立高置信中文身份证据不足，以及 silver positive 与 copy/template risk 的混杂。

## 3. 第一阶段：Step26 论文级证据审计，不训练

建立一个只读审计步骤，统一整理当前仍有效的结果，不再读取已删除的旧 v3 artifact。审计对象只保留：

1. Raw E5/BGE/LaBSE semantic controls；
2. 一个冻结的 Step7 source-only baseline；
3. Step15-v7 clean LR/L2 no-augmentation；
4. equal-weight duplication 和 latent mixup 负消融；
5. Step24 frozen authorship representation；
6. Step25-v1/v2/v3.1 机制负结果；
7. Step12 grouped bootstrap；
8. 现有 Step11/17 explicit allow-list 图谱审计，只作为后验 operational evidence。

Step26 必须输出：每个模型的训练域、特征来源、是否含 identifier、是否使用 silver、选择边界、有效 evaluation slice、AP/ROC-AUC、component-grouped CI、hard-negative tail 和当前科学状态。它的目标是形成论文主表和消融表，而不是选新模型。

## 4. 第二阶段：建设新的 D1 复制集

D1 必须在当前模型和协议冻结后收集，并满足：

- pair 和完整 seller component 与当前 train/D0/valid/test 不重叠；
- reviewer 看原始 seller/item/identifier evidence，但看不到任何模型分数；
- positive 只接受 seller-facing direct identifier、明确迁移声明或可闭合 component evidence；
- template similarity、同主题商品、公共联系方式、victim/product data contact 不能标 positive；
- uncertain 保留，但不进入二分类主指标；
- 生成 pair UID manifest、seller-component manifest 和文件 SHA-256；
- 两名独立 reviewer 加冲突裁决，不能把模型或聚类预测当 ground truth。

最低 readiness 目标：

| Evidence slice | Minimum |
|---|---:|
| direct/component positive | 30 |
| template-clone negative | 30 |
| semantic-topic negative | 30 |
| public-contact/URL negative | 20 |
| ordinary negative | 30 |

如果 direct/component positive 达不到 `30`，不启动新模型比较；应把证据稀缺作为研究结论，而不是放宽 positive 定义。

## 5. 第三阶段：仅允许一个预注册方法假设

只有 D1 readiness 通过后，才允许预注册一个与 Step25 不同的新假设：

`identifier-redacted semantic + frozen raw authorship, followed by occurrence-level evidence states`

其科学结构为：

- clean scorer：只使用 identifier-redacted semantic 和冻结 raw authorship，不使用 copy-risk、identifier、review label 或 evidence type；
- evidence layer：只根据 occurrence-level `verified_direct / mixed / risky-only / public / none` 状态进行方向受限的后处理；
- clean scorer 的权重只从 English source 或 train component-OOF 获得；
- evidence layer 若中文 direct/public 训练样本不足，则固定为规则敏感性，不训练参数；
- 不训练 Transformer、不做 LoRA、不做 MLP、不再做 synthetic text generation；
- 正式对照只保留 raw semantic、source-only、Step15-v7、Step25 C0 和新方法。

C3 当前结果只能用于提出这个假设，不能提供模型参数、阈值或 D1 选择依据。

## 6. D1 晋级门槛

在 D1 上一次性评估，primary metric 为 AP，并同时要求：

- 相对 strongest clean baseline 的 AP delta `>=0.03`；
- component-grouped bootstrap lower bound `>=0`；
- direct/component recall 下降不超过 `0.05`；
- template-clone FPR 至少下降 `0.10`；
- public-contact/URL FPR 至少下降 `0.10`；
- non-silver/gold slice 不发生 `>0.02 AP` 的退化；
- 不在 D1 上重新选模型、权重或阈值。

全部通过才允许后续 Step11/17 explicit allow-list 图谱验证。任何一项失败都冻结为负结果。

## 7. 如果无法获得 D1

如果在预定采集周期内仍无法获得至少 `30` 条 component-disjoint 强身份证据 positive，应停止追求“新模型显著提升”的论文叙事，转向数据与负结果论文：

`Evidence-type concept drift in cross-lingual underground-market identity linkage`

可成立的贡献包括：

- 英文到中文 seller-pair identity linkage 的证据型概念漂移定义；
- template clone、semantic topic、public contact 与 direct identity 的分层 benchmark；
- 对 source-only、target adaptation、mixup、LoRA、item distribution、authorship decontamination 和 copy-aware penalty 的统一严格负结果；
- seller-component 分组评估、silver/non-silver sensitivity 和 leakage discipline；
- 解释为什么高语义或高 copy overlap 不能等价于 same controller。

这比继续在当前 D0 上调模型更可信，也更符合现有结果能够支持的结论边界。

## 8. 立即执行顺序

1. 保持 Step25-v3.1 和现有 canonical split 冻结；
2. 实现 Step26 只读论文证据审计；
3. 生成尚未与当前 component 重叠的 score-blind D1 candidate queue；
4. 完成人类独立审核和 component exclusion；
5. D1 readiness 通过后再预注册唯一新方法；
6. D1 不足则直接进入数据/负结果论文写作，不再增加模型实验。
