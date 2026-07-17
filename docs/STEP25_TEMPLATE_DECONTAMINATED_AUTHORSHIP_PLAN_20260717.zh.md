# Step25 跨卖家模板去污染的作者风格迁移实验

## 1. 当前结论与立项原因

Step24 已完成工程与数值验证，但没有通过预注册晋级门槛。它证明了两个事实：

1. 冻结的多语言作者/风格表示相对 identifier-redacted E5 提供了明显的跨语言增量；
2. 这些表示同时会把复制模板、公共联系方式页面和广告排版一致性识别为作者风格一致。

Step24 在中文 canonical train 上的关键现象是：

| 类型 | E5 cosine | PCM authorship cosine | mStyleDistance cosine |
|---|---:|---:|---:|
| direct-identifier positive | 0.903 | 0.889 | 0.992 |
| component-anchor positive | 0.931 | 0.951 | 0.993 |
| template-clone negative | 0.903 | 0.870 | 0.977 |
| public-contact/URL negative | 0.927 | 0.932 | 0.998 |

因此失败原因不是编码器完全没有信号，而是输入给编码器的 seller 聚合文本仍含大量跨 seller 复制内容。identifier redaction 去除了联系方式捷径，却没有去除 boilerplate/template shortcut。只用三个正向 cosine 的 LR/L2 无法判断高相似来自同一作者还是复制模板。

Step25 检验一个更具体、可证伪的问题：

> 仅用训练语料、完全不读取标签和模型分数，识别跨 seller 高频重复片段并在风格编码前删除，能否降低模板复制负例的排序位置，同时保留 direct/component positive 的作者风格信号？

## 2. Step25 不是什么

Step25 不是：

- 给 Step24 调阈值；
- 根据 Step24 false positive 手写 seller 黑名单；
- 用 `template_clone_not_controller` 标签拟合模板词典；
- 用 valid/test 文本计算重复频率；
- 对两个外部 style encoder 做本地微调；
- 再次生成 synthetic positive；
- 把 occurrence identifier 特征混入 clean scorer；
- 在当前 D0 上得到高分后直接进入 Step11 或形成论文主结论。

Step24-v1 的 policy、结果和模型 artifacts 保持冻结。Step25 写入独立目录：

```text
reports/step25_template_decontaminated_authorship/v1_20260717/
```

## 3. D0、D1、F1 三道数据边界

### 3.1 D0：当前 canonical train

D0 是当前 Step24 已经分析过的英文/中文 canonical train：

- English：`401 = 116 positive / 285 negative`；
- Chinese：`573 = 229 positive / 344 negative`；
- Chinese seller components：`222`；
- Chinese positive 中 `213/229` 为 train-only silver。

Step25 的假设来自 D0 上的 Step24 错误，因此 D0 已被消耗。D0 只允许：

- 验证代码和模板去污染机制是否按预期工作；
- 估计该方向是否值得建设新开发数据；
- 形成探索性消融和负结果。

无论 D0 指标多高，`publication_promotion_eligible` 永远是 `false`。

### 3.2 D1：未来独立开发批次

D1 才能用于 Step25 方法开发与选择。最低要求：

- 与 D0 seller component 完全不重叠；
- 审查时不可见 Step24/Step25 模型分数；
- direct/component positive 至少 `30`；
- template-clone negative 至少 `30`；
- public-contact/URL negative 至少 `20`；
- uncertain 不进入二分类模型选择。

D0 continuation gate 全部通过，只代表可以投入资源构建 D1，不代表方法成立。

### 3.3 F1：未来前瞻性最终 holdout

F1 必须在模型、模板检测规则、LR 参数和 reliability expert 全部冻结后才收集：

- 与 D0/D1 seller component 均不重叠；
- reviewer 不可见任何模型分数；
- 只评估一次；
- 不在 F1 上重新拟合 threshold；
- 结果无论正负都必须报告。

## 4. 输入文本与身份清除

Step25 完整重放 Step15-v7/Step24 的 identifier-redacted seller 文本，字段固定为：

```text
category_concat_top
signature_title_concat
title_concat_top
signature_description_concat
description_concat_top
```

明确排除：

```text
source_seller_raw
alias_normalized
source_market_raw
source_seller_id_raw
contact_concat_top
structured_snapshot_concat_top
profile_text
```

每个 seller 的重放文本 SHA-256 必须和 Step24 原始 embedding metadata 绑定的语料 hash 一致。Step25 不重新定义 identifier redaction，也不恢复被删除的联系方式。

## 5. 无标签模板检测算法

### 5.1 文本规范化

对 identifier-redacted 文本执行：

1. Unicode `NFKC`；
2. `casefold`；
3. 连续空白折叠为一个空格。

该过程不读取 review label、evidence type、模型分数或 split 以外的信息。

### 5.2 字符 shingle

固定生成长度为 `12` 的字符 shingle。一个 shingle 至少一半字符必须是字母或数字，以避免仅由分隔符组成的片段成为模板证据。

目录只保存：

```text
shingle_sha256
seller_document_frequency
component_document_frequency
character_length
```

禁止将原始 shingle 文本写入 artifact。

### 5.3 seller component 交叉拟合

对 seller `s` 去污染时，完整排除 `s` 所在 seller component 的所有 seller。一个 shingle 只有同时满足以下条件才可支持删除：

- 在当前 component 之外至少出现于 `3` 个 seller；
- 在当前 component 之外至少覆盖 `2` 个独立 component。

这样可以防止同一身份组件内部的重复文本反过来证明自己是公共模板，也防止同一 pair 的两侧互相支持删除。

### 5.4 连续片段删除

单个 12 字符 shingle 命中不直接删除。只有重叠命中合并成至少 `24` 个连续字符的片段后才删除。

该机制对“近重复”采用局部覆盖定义：即使模板中有少量插入、删除或替换，只要仍保留多个连续的精确 shingle，公共片段仍会被覆盖；它不声称解决任意语义改写。

每个 seller 最多删除原文本的 `95%`。若去污染后有效字母/数字少于 `32`，该 seller 被标记为 `decontaminated_text_reliable=0`。任何 pair 只要一侧不可靠，其 decontaminated style cosine 固定为 `0.0`，表示“没有可靠风格支持”，而不是把统一 fallback 文本的 cosine=1 当作正证据。

## 6. 冻结风格编码器

Step25 复用 Step24 已固定的两个模型路径和 revision：

```text
models/step24/authorship/multilingual_style_representation/
models/step24/authorship/mstyledistance/
```

Windows 本地可以没有这两个目录。config-only validation 不检查模型；Linux 数值运行时必须存在模型、`config.json` 和 `step24_model_provenance.json`。

编码器参数不更新，不做 LoRA，不做本地 fine-tuning。

## 7. Pair feature

Step25 固定生成十项 train-only pair feature：

```text
identifier_redacted_e5_cosine
raw_pcm_multilingual_authorship_cosine
raw_mstyledistance_cosine
decontaminated_pcm_multilingual_authorship_cosine
decontaminated_mstyledistance_cosine
pcm_raw_minus_decontaminated
mstyledistance_raw_minus_decontaminated
pair_maximum_boilerplate_fraction
pair_mean_boilerplate_fraction
decontaminated_pair_reliable
```

其中 raw feature 从冻结的 Step24 pair feature 读取；decontaminated feature 来自 Step25 新 embedding。difference 和 coverage 不参与 primary 选择，仅用于探索性诊断。

## 8. 固定模型矩阵

### 8.1 Matched baseline

```text
raw_style_lr_l2_control
```

输入：raw PCM + raw mStyleDistance。

### 8.2 Primary

```text
decontaminated_style_lr_l2_primary
```

输入：decontaminated PCM + decontaminated mStyleDistance。

这是唯一 primary，直接检验表示去污染是否有效。

### 8.3 Secondary

```text
decontaminated_semantic_style_lr_l2_secondary
```

输入：redacted E5 + 两项 decontaminated style。它用于检查 E5 是否仍污染 direct/component slice，不参与模型选择。

### 8.4 Exploratory

```text
raw_clean_delta_exploratory
```

输入 raw、clean、difference、coverage 和 reliability flag。该模型用于理解 raw-clean 差值是否含信息，不能成为 D0 主模型，也不能依据其 D0 结果替换 primary。

所有模型固定使用：

- LR/L2；
- `l2_penalty=10.0`；
- 无 class weight；
- fold 内标准化；
- 既有 factorized evidence weights；
- 不读取 valid/test。

## 9. 三种训练/评估路径

### 9.1 Source-only 主迁移

仅使用英文标签训练 LR/L2，在中文 D0 上打分。中文文本参与无标签、component-cross-fitted 模板识别，但中文标签不参与 source-only 模型训练。因此准确名称是：

```text
source supervision + target-unlabeled template preprocessing
```

不能把它描述成完全不接触中文文本的 strict zero-shot。

### 9.2 English grouped OOF

英文按 seller component 做五折 OOF，生成无泄漏 clean probability。该分数只供 occurrence reliability expert 作为 offset，不能使用英文 in-sample score训练 expert。

### 9.3 Target grouped OOF

每折训练集为全部英文 train 加四折中文 train，完整排除 held-out 中文 seller component。该结果仍是 D0 次级内部开发证据，不是正式中文 benchmark。

## 10. 排序稳健性与模板尾部审计

两个独立 LR 的概率截距可能不同，因此不能只比较模板负例的原始概率均值。Step25 同时输出：

- ROC-AUC；
- AP；
- component-grouped paired bootstrap；
- non-silver AP；
- direct/component positive + all negatives AP；
- soft-positive slice；
- 每个 fold 的独立指标；
- 原始 probability mean/q95/top-decile mean，仅作诊断；
- 全局 rank percentile mean/q95；
- template negative 的 top-decile exposure；
- template negative 排在 direct/component positive 之上的 pairwise violation rate。

正式 D0 continuation gate 使用 rank-based 指标，不使用可被截距平移伪造的原始概率下降。

## 11. D0 continuation gate

必须同时满足：

1. source-only AP 相对 raw style 下降不超过 `0.01`；
2. source bootstrap lower bound 不低于 `-0.01`；
3. target OOF AP 下降不超过 `0.01`；
4. target bootstrap lower bound 不低于 `-0.02`；
5. source/target non-silver AP 下降不超过 `0.02`；
6. source/target direct-component AP 下降不超过 `0.02`；
7. template mean rank percentile 至少降低 `0.02`；
8. template q95 rank percentile 至少降低 `0.02`；
9. template top-decile exposure 至少降低 `0.02`；
10. template-vs-strong-positive violation rate 至少降低 `0.03`；
11. public-noise 与 semantic-topic mean rank percentile 不恶化；
12. 至少 `80%` 的中文 pair 两侧仍有足够去污染文本。

全部通过只产生 `d1_candidate_eligible=true`，仍然强制：

```text
publication_promotion_eligible=false
```

## 12. 独立 occurrence reliability expert

clean scorer 不使用 identifier。Step25 另建独立 post-scorer，输入来自 Step3 occurrence context：

- 双侧 seller-facing direct token 数；
- risky-only token 数；
- support-only token 数；
- mixed-context token 数；
- 高频公共 token 数；
- shared token、item、market 数；
- URL/domain、Telegram、email、phone/wallet 类型标志。

训练只使用英文 actionable occurrence rows，clean probability 必须来自英文 seller-component grouped OOF。目标中文标签不参与 expert 训练。

方向约束为：

- `verified_direct_both_sides`：只允许增加 logit；
- `risky_only_shared`、`support_only_shared`、`high_frequency_public`：只允许降低 logit；
- `direct_with_mixed_context`、`ambiguous`、`no_shared_identifier`：保持 clean score。

review label、evidence type、model error、split membership 均不得成为 expert feature。输出中的 label/evidence type 只用于事后切片审计。

## 13. 防泄漏与不可覆盖规则

1. Step25 输出目录与 Step24 完全隔离。
2. 所有 JSON/CSV/NPY 使用 immutable writer；路径存在但内容不同即报错。
3. 模板检测器的函数输入不含 label、evidence type 或 score。
4. 每个 seller 只使用自身 component 之外的 seller 支持模板删除。
5. valid/test seller、文本、标签、分数和 pair feature 均不读取。
6. secondary/exploratory 模型不能替换 preregistered primary。
7. 同步必须返回整个 Step25 目录，并用 `step25_sync_manifest.json` 校验所有 `25` 个 payload。
8. 同步 manifest 还绑定 Step9/Step15-v7/Step24 的直接公共代码与上游 policy；模板 summary 绑定 component assignment，reliability summary 绑定英文/中文 clean prediction 与两域 Step3 occurrence 输入，防止 Linux 依赖版本错位。

## 14. Linux 一键运行

需要同步代码、policy、测试和文档，但不需要把 Windows 上不存在的模型重新下载。Linux 保留原模型路径即可。

```bash
cd /home/yongpeng/cross-lingual
bash scripts/run_step25_template_decontaminated_authorship_linux_20260717.sh
```

运行结束后同步整个目录：

```text
reports/step25_template_decontaminated_authorship/v1_20260717/
```

不能只同步 evaluation summary。

## 15. 结果解释纪律

如果 D0 gate 失败：冻结 Step25 为负结果，不改 shingle 长度、频率门槛、mask 长度或 LR 参数来追当前 D0 分数。

如果 D0 gate 通过：只能说明该表示方向值得建设 D1。完成 D1 开发、冻结所有配置并收集 F1 后，才可能形成确认性方法结论。

Step25 的论文问题不是“能否把当前 AP 调高”，而是：

> 在严格去 identifier 的跨语言地下市场身份链接中，无标签跨 seller 模板去污染能否让作者风格表示更接近控制者信号，而不再奖励复制广告文本？
