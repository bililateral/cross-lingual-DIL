# Step16I 数据完整性审计与回顾性 Dev2 建设计划

日期：2026-07-16
状态：v1 完整性审计已完成；保守 component 合并规则已纠正，等待 v2 重跑并生成 Dev2
适用分支：`data/step16i-integrity-dev2`

## 1. 文档目的

Step16I 不是新的模型训练步骤，也不是继续调整 Step15-v8 的手段。它只解决两个已经被当前审计明确暴露的数据问题：

1. 后续 train-only 银标扩充后，原始 Step5 中保存的 `split_component_id` 已经不能完整表示当前训练边构成的 seller 连通分量。
2. 当前 valid、test 和历史候选池已经被多轮模型分析、图审计或辅助标注过程使用，不能继续被描述为独立的 prospective final benchmark。

Step16I 的目标是：

- 重新计算并冻结真实的 seller-connected components；
- 验证所有 grouped OOF、grouped bootstrap 和 split 隔离是否使用了正确的 component；
- 将评估标签按证据强度分层，而不是把所有 positive 当成同等强度的 gold；
- 建立一个严格标明为 retrospective internal development 的 `zh_dev2`；
- 生成永久排除清单，为未来真正 prospective 中文 holdout 留出清晰边界。

Step16I 不得覆盖当前 Step5 frozen labels，不得重写 V8 结果，也不得把旧数据重新命名为 prospective 数据。

## 2. 当前审计结论

### 2.1 Step15-v8 是严格负结果

最新完整 V8 运行已经通过工程和数据 readiness 检查，但没有通过预注册的方法晋级门槛。关键结果如下：

| 检查项 | 结果 |
|---|---:|
| Selected clean 相对 B0 的 valid `Delta AP` | `-0.025691` |
| Grouped bootstrap mean `Delta AP` | `-0.027194` |
| Grouped bootstrap 95% CI | `[-0.123717, 0.055103]` |
| Contextual fusion 相对 clean 的 mean `Delta AP` | `+0.025268` |
| Contextual fusion 95% CI | `[-0.003016, 0.095010]` |
| Public-noise FPR reduction | `0.0` |
| `validation_data_readiness_met` | `true` |
| `method_gates_met` | `false` |
| `promotion_eligible` | `false` |

因此，V8 的正确结论是：

> V8 完成了预注册评估，但 clean bridge 没有稳定优于 B0，contextual evidence fusion 只表现出正向趋势，并且没有降低 public-noise FPR。V8 应冻结为严格负结果，不进入 Step11/17 publication validation，也不能再依据当前 valid/test 调参。

这不是程序运行失败，也不能通过修改阈值、降低晋级门槛或重新选择 seed 来改写。

### 2.2 原始 Step5 train component 字段陈旧

当前中文 Step5 主监督边界为：

| Split | Pair | Positive | Negative |
|---|---:|---:|---:|
| train | 573 | 229 | 344 |
| valid | 120 | 30 | 90 |
| internal development test | 200 | 50 | 150 |

现有主 split 之间没有重复 `pair_uid`，也没有 seller overlap。问题出现在 train 内部的 component 编号：

- 表内名义 `split_component_id` 数量为 434；
- 按当前全部 train pair 的左右 seller 重新做并查集，只得到约 222 个真实 seller-connected components；
- 最大 train 连通分量包含约 175 条 pair、110 个 seller，但被旧字段拆成约 41 个不同 component ID；
- 56 条 `silver_direct_or_contact` 实际只来自约 15 个连通分量；
- 29 条 `silver_component_closure` 实际只来自约 7 个连通分量。

原因是 Step16B/D/E/G 等 train-only 扩充改变了训练图结构，而原始 Step5 行上的历史 component 字段没有在每次扩充后全图重算。

如果后续代码直接使用这一陈旧字段进行 GroupKFold、OOF 或 grouped bootstrap，同一个 seller 图可能进入多个 fold，产生分组泄漏并高估独立样本量。

### 2.3 最新 V8 未被陈旧 Step5 component 字段污染

必须区分历史 Step5 字段问题和最新 V8 的实际实现：

- 最新 V8 readiness freeze 对当前 seller 图重新计算了 canonical components；
- V8 bridge 使用 readiness 生成的 component assignment，而不是盲信原始 Step5 的历史 `split_component_id`；
- V8 的 train OOF 按重算后的 component 分组；
- primary train、representative valid 和 internal development test 之间没有 pair、seller 或 component overlap；
- V8 的负结论不能归因于本节所述的陈旧 component 字段。

Step16I 仍需独立重算并固化 component，目的是让 Step5 到后续所有审计共享同一份可复核的 component overlay，而不是推翻已经完成的 V8。

### 2.4 Train positive 的名义数量高于强证据数量

中文 train 有 229 条 positive，其中：

- 213 条为 `silver_train_only`；
- 仅约 16 条属于非银标的原始或历史审查 positive。

银标 positive 的主要来源包括：

| 银标来源 | 数量 |
|---|---:|
| Template/structural silver | 85 |
| Direct/contact silver | 56 |
| Component closure silver | 29 |
| Relaxed rank/structural | 30 |
| Relaxed template | 11 |
| Relaxed high similarity | 2 |

因此，229 条 positive 不能在论文中被描述为 229 条彼此独立的 proof-level gold positives。它们可以作为带权重的弱监督训练支持，但必须同时报告：

- silver 行数；
- 非银标行数；
- 真实 seller component 数；
- 各证据层级数量；
- closure 和模板规则带来的相关性。

### 2.5 评估 positive 的证据层级不一致

Step16F 对 valid/test 的 80 条 positive 做了证据层级复核：

| 论文证据桶 | 数量 |
|---|---:|
| Direct/component primary | 22 |
| Soft primary or separately reported slice | 14 |
| Secondary or sensitivity only | 44 |

Step16H-v3 的独立 AI-assisted sensitivity review 对同一批 80 条当前 positive 得到：

| 决策 | 数量 |
|---|---:|
| `strict_same_controller` | 18 |
| `soft_same_controller` | 48 |
| `different_controller` | 11 |
| `uncertain` | 3 |

该复核不能自动改写 Step5，因为它由独立 AI agents 完成，而不是论文意义上的两名人类标注者。它证明的是标签敏感性风险，不是新的 gold annotation。

后续报告必须至少分成：

1. `strict_gold`：seller-facing direct identifier 或独立 component anchor 支持；
2. `soft_primary`：较强结构、风格或多证据闭合，但没有 proof-level direct anchor；
3. `sensitivity_only`：弱 component/semantic、历史闭包、证据冲突、AI review 不一致或需复核；
4. `uncertain`：证据不足，不进入二分类主评估。

### 2.6 存在预测辅助标注和确认偏差风险

现有 Step5 没有发现把具体 `prob_positive`、AUC 或 threshold 直接写入标签的证据，但部分历史 review notes 使用了图结果或 cluster 结果作为辅助判断：

- 25 条记录明确提到 Step11，其中 2 条 positive 进入监督，分别位于 train 和 test；
- 260 条监督 negative 的说明包含类似 `not retained by graph support` 的依据，其中约 145 条位于 valid/test；
- 约 9 条 valid/test positive 使用了已有 positive cluster 或 component closure 类描述。

这不等同于训练时读取 test label，但意味着当前 benchmark 不能声称完全 prediction-blind。Step16I 必须给这些行加上可审计风险标志，例如：

- `step11_assisted_review`；
- `graph_nonretention_cited`；
- `cluster_or_component_closure_cited`；
- `prediction_blind_gold_eligible=false`。

这些行可以保留在回顾性开发或 sensitivity analysis 中，但不能直接进入未来 prospective final holdout。

### 2.7 当前本地新增强证据已经耗尽

对当前 Step3、Step4、Step16 和 V8 context queues 的联合审计结果是：

- 新增 proof-level 双侧 direct positive：0；
- seller/component 独立的 prospective public-noise candidate：0；
- 当前 Step3 direct pair 已全部进入 Step5；
- Step16 proof-positive queue 当前为 0 行；
- V8 尚未正式审查的少量 risky-only pair 只来自少数 component，并且与现有监督或 controls 共享 seller；
- Step4 仍有隔离的 soft candidates，但没有新的共享 direct identity anchor。

当前本地旧池仍可提供有限的 template、semantic-topic 和 uncertain candidates，用于 retrospective dev2。它不能产生新的 prospective proof-positive 或独立 public-noise benchmark。

### 2.8 Step16I v1 实际审计结果

Linux v1 审计读取了中文 `893` 条 primary rows 和英文 `734` 条 primary rows，生成
`1,627` 条 component assignment 与 `7,991` 条永久排除记录。EN、ZH 和跨语言范围的
pair、完整 seller UID、portable seller alias、重算 seller component 均为零跨 split
重叠。中文旧 component 字段把 `595` 个旧 ID 映射到 `383` 个真实连通组件，英文旧字段
则有 9 个 ID 过度合并互不连接的子图；它们是历史字段质量警告，不是当前 split 泄漏。

v1 唯一阻断项来自 V8 readiness：3 个持久化 component 各合并两个互不连接、但处于同一
split 的 seller 子图。原契约错误地要求持久化分区与 seller 图连通分区完全等价。该要求
过强，因为保守合并只会减少有效独立组数，不会把同一 seller 带入不同 split。修正规则为：
保守合并记 warning；真实连通组件被拆到多个持久化 ID、跨 split seller/alias/component、
重复 pair 或未知 split 仍然 fail closed。v1 失败记录永久保留，修正后必须使用新 run ID。

## 3. Step16I 的科研边界

### 3.1 可以做的事

- 修复 component 统计口径；
- 验证 grouped OOF 和 bootstrap 的分组完整性；
- 将历史评估标签分层；
- 为旧候选构建 score-blind retrospective review queue；
- 建立独立于现有 train/valid/test 的内部 `zh_dev2`；
- 生成永久排除清单和数据 lineage manifest；
- 估计未来 prospective holdout 所需的独立 component 数量。

### 3.2 不能做的事

- 不能覆盖 `reports/step5_zh_target_strict_frozen_silver_labels.csv`；
- 不能把 silver positive 改名为 gold positive；
- 不能把 AI agent review 宣称为人类双标 gold；
- 不能把 Step11 cluster、模型高分或规则命中直接转换为 identity label；
- 不能把旧 Step4/Step16/V8 candidates 称为 prospective；
- 不能为了满足目标数量降低 seller-facing evidence 标准；
- 不能把 uncertain 强制二值化；
- 不能使用当前 valid/internal test 选择模型、权重或 threshold；
- 不能因为 component 重算改变，就重解释或选择性删除 V8 的严格负结果；
- 不能在没有冻结后新时间快照的情况下启动 Step20 final evaluation。

## 4. 工具一：Component Integrity Audit

当前实现文件名为：

```text
scripts/step16i_audit_data_integrity.py
```

对应 policy 文件名为：

```text
schema/step16i_data_integrity_policy.json
```

### 4.1 目的

工具一只读现有标签和 seller pair，重新建立 canonical seller graph，并输出不可覆盖的审计 overlay。它不修改 Step5 原文件。

必须完成：

1. 标准化左右 `seller_uid`；
2. 对每个监督 split 的所有 pair endpoint 做 union-find；
3. 生成稳定、与行顺序无关的 `canonical_component_id_step16i`；
4. 对比历史 `split_component_id`，统计 stale、split 和 merge；
5. 检查 train/valid/test 的 pair、seller、alias 和 component overlap；
6. 检查重复 canonical pair 和左右反转重复；
7. 验证 V8 readiness component assignments 是否与重算图一致；
8. 验证 V8 readiness assignment 的持久化 component partition 与重新计算结果一致；
9. 为所有已消费数据生成永久排除清单。

V8 各 OOF fold 和 grouped bootstrap 的 component 使用情况由既有 V8 fold manifest、Step12-v8 与代码契约单独审计。当前 Step16I 工具不读取这两类 artifact，因此不能把它描述为已经自动复验 OOF/Bootstrap。

### 4.2 输入

当前实现直接读取：

```text
reports/step5_zh_target_strict_frozen_silver_labels.csv
reports/step5_en_frozen_silver_labels.csv
reports/step16_v8_validation_refreeze/
readiness_expansion_v3_reprofix_20260716_112833_31791/
representative_validation_assignments.v8_readiness.csv
```

工具从 Step5 `review_notes` 中匹配 Step11、graph、cluster/component 等辅助审查痕迹。当前版本不直接读取 Step16F、Step16H 或 Step3 occurrence；这些证据层级结果在本文中作为解释依据，但不应被误写为工具一已经自动融合的输入。所有实际输入和 producer code 均记录 SHA-256。

### 4.3 输出

建议输出到独立 run root：

```text
reports/step16i_data_integrity/<run_id>/
```

当前实现生成：

```text
component_assignments.csv
permanent_exclusion_manifest.csv
summary.json
```

永久排除实体按类型分开保存：历史 `pair_uid`、完整 `seller_uid`、经过 NFKC 与
case-fold 标准化的可携带跨市场 seller alias，以及重算的 seller component。纯数字
账号和 `/shop/数字` 属于市场内部编号，只由完整 UID 隔离，不会被误当成跨市场 alias；
真正可携带的文本 alias 即使在不同市场对应不同完整 UID，也不能重新进入 `dev2`。

输出要求：

- 每个输出带 `run_id`、policy version 和输入哈希；
- 已存在的 run root直接拒绝运行，不能静默覆盖，也不提供 identical replay 模式；
- summary 必须同时报告 pair count 和 independent component count；
- `summary.json` 中的 `v8_readiness_assignment_check` 必须由实际 archive 验证结果计算。持久化 component 在同一 split 内保守合并多个不连通子图只记 warning；真实 seller-connected component 被拆到多个持久化 ID、任何跨 split 重叠或重复 pair 才 fail closed；
- 若发现真实跨 split seller/component overlap，工具必须 fail closed；
- 历史 component stale 本身应被报告并生成 overlay，不应通过覆盖 Step5 来隐藏。

## 5. 工具二：Retrospective Dev2 Builder

当前实现文件名为：

```text
scripts/step16i_prepare_retrospective_dev2.py
```

对应 policy 文件名为：

```text
schema/step16i_retrospective_dev2_policy.json
```

当前工具二只负责准备候选 universe、盲映射和两份独立 reviewer queue。它不读取 reviewer 决定、不做 reconcile、不做 adjudication，也不 materialize 二分类 dev2。这样可以防止在审查协议和人工角色尚未落实时自动生成标签。

### 5.1 目的

工具二从旧候选池中构建一个明确标记为 retrospective 的内部开发扩展集。它优先补：

- template-clone negatives；
- semantic-topic negatives；
- public/support noise candidates，如果存在独立 component；
- evidence-insufficient uncertain cases；
- 仅在出现新的 proof-level seller-facing evidence 时接收 strict positive。

工具不得假定 `zh_dev2` 一定能达到预设规模。若当前本地数据无法提供独立 public-noise 或 proof positive，正确输出是 `evidence_scarcity`，不是放宽规则。

### 5.2 输入

当前实现读取：

```text
reports/step4_zh_target_strict_silver_candidate_pairs.csv
reports/step3_item_identity_signals.zh_target_strict.csv
reports/step16i_data_integrity/<integrity_run_id>/permanent_exclusion_manifest.csv
```

永久排除 manifest 已覆盖 Step5 全部历史 split，并在 V8 archive 可用时纳入 readiness assignments。工具二不读取 Step11 分数或 cluster decision。候选抽样使用完整 eligible seller graph 的 connected components，每个被选 component 只取一个确定性代表 pair。

### 5.3 `prepare` 输出

```text
reports/step16i_retrospective_dev2/preparation_v1_20260716/
  blind_mapping.csv
  reviewer_a_queue.csv
  reviewer_b_queue.csv
  preparation_manifest.json
```

其中 `blind_mapping.csv` 只供后续受控 reconciliation 使用，不能交给 reviewer。两份 reviewer queue 隐藏 `pair_uid`、seller UID、候选规则、历史 review、模型、Step11、cluster、graph 和 threshold 字段，并使用不同的确定性顺序。

### 5.4 当前工具二之后的人工流程

当前实现到 review queue preparation 为止。后续必须由单独、经过审查的 reconciliation/materialization 工具或受控人工流程完成，且必须满足：

- A/B reviewer 独立完成，不能看到对方决定；
- 不一致或任一方为 `uncertain` 时必须进入第三方 adjudication；
- adjudicator 不能看到模型分数；
- AI-assisted review 必须标记为 sensitivity review，不能转成论文 human gold；
- 只有真实双人独立人工审查加仲裁，才能设置 `human_gold_eligible=true`；
- 不与现有 train/valid/test、V8 controls、Step16 reviewed controls 共享 pair；
- 默认不与上述集合共享 seller 或 canonical component；
- 若未来 materialize dev2，`strict_gold` 和 `soft_primary` 必须分开保存、分开报告；
- `uncertain` 不进入二分类指标；
- 所有最终行必须带 `retrospective_only=true`；
- 所有最终行必须带 `prospective_final_eligible=false`；
- 若 strict positive 或独立 public-noise 数量为 0，summary 必须如实记录 0。

## 6. 盲审规则

### 6.1 Reviewer 可以看到

- 左右 seller profile 原文；
- item title、description 和必要上下文；
- seller-facing identifier occurrence 原文及前后文；
- identifier 类型和值；
- product-data、support-only、victim-data 和 public URL context；
- market/source/time 元数据；
- 与身份判断有关的 candidate rule evidence，但不能包含模型输出。

### 6.2 Reviewer 不能看到

- Step7、Step9、Step15、Step17 的 score；
- `prob_positive`；
- selected threshold；
- 当前模型预测标签；
- Step11 cluster membership、retained edge 或 cluster decision；
- 当前 Step5 label；
- 另一名 reviewer 的决定；
- 候选是否属于模型 error case；
- `strict_gold`、`soft_primary` 等预期目标桶。

### 6.3 Identity 决策标准

`strict_same_controller`：

- 双侧 seller-facing direct identifier 一致；或
- 独立、可复核的 component anchor 闭合；或
- 明确的账号迁移、备用号、同店声明等直接控制证据。

`soft_same_controller`：

- 存在多项相互支持的结构、风格和内容证据；
- 但缺少 proof-level seller-facing identity anchor；
- 只能进入 `soft_primary` 或 sensitivity analysis，不能混入 strict gold。

`different_controller`：

- 共享 token 仅属于公共客服、公共 URL、商品/受害者数据或中转 context；或
- 模板、主题、商品结构相似，但原始证据支持独立 seller；或
- 有明确冲突身份信息。

`uncertain`：

- 只有相似性，没有足够身份支持；
- direct/risky context 无法区分；
- 原始证据缺失或相互冲突。

不允许为了提高样本量把 `soft_same_controller` 或 `uncertain` 自动改成 strict positive。

## 7. Linux 执行顺序

以下顺序只在两个 Step16I 工具和两个 policy 完成 Windows 静态检查并同步到 Linux 后执行。本文自身不运行数据管道。

### 7.1 同步文件

至少同步：

```text
scripts/step16i_audit_data_integrity.py
scripts/step16i_prepare_retrospective_dev2.py
scripts/run_step16i_integrity_dev2_linux_20260716.sh
schema/step16i_data_integrity_policy.json
schema/step16i_retrospective_dev2_policy.json
tests/test_step16i_data_integrity_contracts.py
docs/STEP16I_DATA_INTEGRITY_AND_RETROSPECTIVE_DEV2_PLAN_20260716.zh.md
```

同时补同步最新 V8 readiness root，保证复现链完整：

```text
reports/step16_v8_validation_refreeze/
readiness_expansion_v3_reprofix_20260716_112833_31791/
```

### 7.2 静态检查

推荐直接执行一键 runner；它会先运行契约测试，再执行工具一、检查 integrity/readiness gate，最后才生成 dev2 盲审队列：

```bash
cd /home/yongpeng/cross-lingual
bash scripts/run_step16i_integrity_dev2_linux_20260716.sh
```

下列分步命令用于排错或审计 runner 的具体动作。

```bash
cd /home/yongpeng/cross-lingual

python3 -m py_compile \
  scripts/step16i_audit_data_integrity.py \
  scripts/step16i_prepare_retrospective_dev2.py

python3 -m unittest tests.test_step16i_data_integrity_contracts

python3 -c 'import json; [json.load(open(p, encoding="utf-8")) for p in [
  "schema/step16i_data_integrity_policy.json",
  "schema/step16i_retrospective_dev2_policy.json"
]]; print("Step16I JSON policies: OK")'
```

### 7.3 运行工具一

```bash
INTEGRITY_RUN_ID="step16i_integrity_20260716_v1"

python3 scripts/step16i_audit_data_integrity.py \
  --policy schema/step16i_data_integrity_policy.json \
  --run-id "$INTEGRITY_RUN_ID"
```

运行后必须先检查：

```text
datasets.*.leakage.detected = false
v8_readiness_assignment_check.status = pass
datasets.*.recomputed_component_count 已重新计算
```

如任一跨 split overlap 非 0，应停止后续流程。不能直接运行 dev2 builder。

### 7.4 准备 retrospective dev2 盲审包

```bash
python3 scripts/step16i_prepare_retrospective_dev2.py \
  --policy schema/step16i_retrospective_dev2_policy.json \
  --permanent-exclusion-manifest \
    "reports/step16i_data_integrity/$INTEGRITY_RUN_ID/permanent_exclusion_manifest.csv"
```

随后只把 `reviewer_a_queue.csv` 和 `reviewer_b_queue.csv` 分别交给两名独立 reviewer。不得把 `blind_mapping.csv`、两个 queue 或参考标签一起交给同一 reviewer。当前两个工具到此结束，不执行尚未实现的 reconcile/materialize 命令。

## 8. Prospective final holdout 的启动条件

真正的 `zh_final_holdout` 必须来自最终模型冻结之后的新时间快照或新原始记录。至少满足：

1. 原始数据记录时间晚于模型、policy 和 threshold 的冻结时间；
2. pair、seller、alias 和 canonical component 均不在永久排除清单；
3. 先冻结 pair universe，再进行标签审查；
4. 两名人类 reviewer 分数盲审；
5. 冲突由第三方仲裁；
6. `uncertain` 不进入二分类主评估；
7. 模型和阈值在标签解封前冻结；
8. 只进行一次 final evaluation；
9. 数据、标签、代码、模型和评估输出分别保存 SHA-256；
10. 样本量以独立 component 为单位做功效分析，不能只统计 pair 行数。

当前 35 个 valid components 对小幅 `Delta AP` 的统计功效不足。以当前 bootstrap 方差作粗略估计，若要检测 `Delta AP=0.03`，可能需要约 200 到 300 个独立 components，保守情形可能更多。最终数量必须由预注册的 component-level Monte Carlo/bootstrap power analysis 决定，不能把固定 pair 数当作充分性的证明。

## 9. Step16I 完成标准

Step16I 只有同时满足以下条件才算完成：

- canonical component overlay 已生成且哈希冻结；
- train component stale 问题已量化，而不是覆盖原文件后消失；
- V8 readiness assignment 的 component partition 已验证；
- grouped OOF/bootstrap 的 component 来源由既有 V8/Step12 artifact 单独复核，不冒充 Step16I 自动检查；
- V8 未受陈旧字段污染的结论已由工具复核；
- 预测辅助标注风险行已显式标记；
- Step16F/Step16H 已有 evidence-tier 结果被保留为审计依据；当前 Step16I 工具没有自动改写或重新裁决这些标签；
- permanent exclusion manifest 已生成；
- retrospective dev2 双盲队列已生成；双审、仲裁和 materialization 是后续人工阶段，当前工具不伪造已完成；
- dev2 与现有监督及 controls component-disjoint；
- dev2 明确标记 `prospective_final_eligible=false`；
- 本地 proof direct 和独立 prospective public-noise 为 0 时，报告 evidence scarcity，不降低标准；
- 未启动任何新的模型训练或 Step20 final evaluation。

## 10. 最终科研表述

完成 Step16I 后，可以严谨地表述：

> 当前项目已经获得一个经过 seller/component 隔离和多轮完整性审计的回顾性开发基准，并明确区分 strict identity evidence、soft evidence 与 sensitivity-only labels。Step15-v8 在该开发框架下得到严格负结果。现有本地快照中的新增 proof-level direct positives 和独立 public-noise prospective candidates 已经耗尽，因此最终泛化结论必须等待模型冻结之后的新中文时间快照和一次性 prospective evaluation。

不能表述为：

> 当前旧候选池已经构成独立 final holdout，或者 V8 因 component 字段错误而失败。

前者违反 prospective 定义，后者与最新 V8 readiness 的实际 component 重算机制不符。
