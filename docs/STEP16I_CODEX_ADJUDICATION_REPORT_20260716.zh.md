# Step16I Codex 裁决报告

日期：2026-07-16
分支：`data/step16i-integrity-dev2`
用途：回顾性内部开发与敏感性分析，不是 prospective final holdout

## 1. 裁决来源

本轮 160 条候选先由两个相互隔离的 AI reviewer 在不知道 `pair_uid`、模型分数、图聚类结果和历史标签的条件下独立审查。两路结果完成后，项目负责人授权 Codex 代行第三方裁决。

准确的 provenance 是：

```text
two_ai_blind_reviews_plus_codex_adjudication_owner_authorized
```

这表示项目负责人采纳裁决结论，但不表示项目负责人逐条阅读了 160 条记录，也不等于两名人类 reviewer 独立标注。除非以后由人类逐条复核并签署，否则论文中不得写成 `human_verified`、`human double annotation` 或人工金标准。

## 2. 裁决纪律

裁决阶段只读取两份盲审队列中已经向 reviewer 展示的证据和两路审查理由。在最终 blind-ID 决定冻结前，不读取 `blind_mapping.csv`。

采用以下保守标准：

1. `same_controller` 必须有可见的 seller-facing direct identifier、明确自称/迁移/备用号声明，或可以独立闭合的 component anchor。
2. 同商品、同主题、相同模板、相同履约话术和长段文案复制都不能单独证明同一控制者。
3. 证据支持模板或主题噪声时可判 `different_controller`。
4. 文案高度独特但仍缺少身份闭合证据时保留 `uncertain`，不为了扩正例强行二分类。
5. 裁决结果不写回 Step5，不进入当前 train/valid/test，不具备 prospective final eligibility。

## 3. 最终结果

| 裁决 | 数量 |
|---|---:|
| `same_controller` | 1 |
| `different_controller` | 108 |
| `uncertain` | 51 |
| 总计 | 160 |

证据类型分布：

| Evidence type | 数量 |
|---|---:|
| `same_controller_direct_identifier` | 1 |
| `same_controller_style_structural_soft` | 28 |
| `template_clone_not_controller` | 29 |
| `semantic_topic_not_controller` | 61 |
| `ordinary_negative` | 36 |
| `uncertain_insufficient_evidence` | 5 |

去除 51 条 `uncertain` 后，回顾性二分类压力集只有 109 条，其中 1 个 positive、108 个 negative，正例率约为 `0.917%`。因此它适合测试模型是否会把软相似性误报成同控制，不适合作为平衡的主评估集，也没有解决中文 proof-positive 稀缺问题。

## 4. 唯一同控制裁决

唯一 `same_controller` 是 blind ID：

```text
retdev2_69bc5acce819b33653f3
```

解盲后对应：

```text
market_item.xlsx|中文暗网交易市场|seller_raw:26064
market_item.xlsx|中文暗网交易市场|seller_raw:27020
```

两侧商品标题/正文都以罕见自称“独孤信”署名，并与贷款/股票数据销售叙述绑定。它被裁决为：

```text
same_controller
same_controller_direct_identifier
high confidence
```

另一个棋牌后台候选虽然复用了非常独特的第一人称控制叙述，但没有共同账号标识、迁移声明或独立闭合 component anchor，因此最终保留为 `uncertain / same_controller_style_structural_soft`，没有被强行扩成正例。

## 5. 产物与约束

盲 ID 裁决：

```text
reports/step16i_retrospective_dev2/codex_adjudication_v1_20260716/
```

受控解盲后的独立 Dev2 标签表：

```text
reports/step16i_retrospective_dev2/codex_adjudicated_dev2_v1_20260716/
```

关键约束：

- `dataset_owner_authorized=true`
- `human_verified_per_row=false`
- `retrospective_development_only=true`
- `prospective_final_eligible=false`
- `step5_supervision_eligible=false`
- `paper_primary_benchmark_eligible=false`

160 个 pair、160 个 blind ID 和 160 个候选 component 均唯一；与 Step16I v2 永久排除清单中的历史 pair 重叠为 0。输出 CSV 的 SHA-256 已写入各自 summary。

## 6. 科研结论

这次裁决的主要价值不是扩充 positive，而是确认了历史未使用候选池主要由主题相似、模板复制和普通负例构成。它为 false-positive 压力测试提供了 108 条回顾性困难负例，并保留 51 条证据不足样本用于不确定性分析。

它同时给出一个明确的负面结论：当前本地历史候选池不能提供论文规模的中文 proof-level positive。最终论文若需要独立、可信的身份识别主评估，仍必须依赖模型冻结之后的新原始快照、外部可核验证据和真正逐条完成的人类复核。
