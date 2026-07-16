# Step21 中文正例训练专用合成增强方案

更新时间：2026-07-16

## 1. 为什么新增 Step21

当前项目的核心瓶颈不是中文负例不足，而是可信中文同控制正例稀缺。外部公开数据很难同时满足中文、地下市场、seller-pair、可核验同控制标签和可公开复现这五项条件。因此，Step21 检验一个更窄的问题：在不伪造新真值、不修改现有 benchmark 的前提下，能否通过训练专用文本变换提高中文正例表征的稳定性。

Step21 不是新数据采集步骤，也不声称产生新的真实中文马甲。它属于 `training-only augmentation`。合成行的独立样本数始终为零；有效独立样本数仍等于真实父 seller component 数。

## 2. 当前可用父样本及其限制

canonical 中文冻结集当前为：

| Split | Positive | Negative | Binary total |
|---|---:|---:|---:|
| train | 229 | 344 | 573 |
| valid | 30 | 90 | 120 |
| internal development test | 50 | 150 | 200 |

229 个中文 train positive 中，213 个是 `silver_train_only`，只有 16 个非 silver。Step21 不再复用后来已被降级的 v7 representative-validation overlay，而是严格读取 canonical `split_name=train`，并使用 Step16I v2 完整性审计重新计算的 seller component。因此主轨允许 16 个父 pair，silver direct/component 敏感性轨允许 85 个父 pair。

主轨 16 个父 pair 的证据构成为：

- 1 个 `same_controller_direct_identifier`；
- 15 个 `same_controller_style_structural_soft`。

这意味着主轨不是“16 个直接身份 gold positive”。文中必须原样披露其证据组成。

## 3. 两条物理隔离的合成轨道

### 3.1 `primary_non_silver`

只使用 canonical 中文 train 中的非 silver positive。每个父 pair 生成 3 个变体，预计得到 48 个 synthetic pair。该轨道是主要方法对照，但不能把 48 写成新增 48 个真实马甲样本。

### 3.2 `sensitivity_silver_anchor`

只使用 canonical 中文 train 中、证据类型为 direct identifier 或 component anchor 的 silver positive。每个父 pair 生成 2 个变体，预计得到 170 个 synthetic pair。该轨道仅用于敏感性分析，不能与主轨合并后报告一个更大的“真实数据规模”。

两个轨道拥有不同目录、manifest、lineage 和评估结果，防止误合并。

## 4. 合成前的身份信息处理

每个父 seller profile 只读取以下内容字段：

- `category_concat_top`；
- `signature_title_concat`；
- `title_concat_top`；
- `signature_description_concat`；
- `description_concat_top`。

然后复用 Step15-v7 的高精度 redaction：

- 删除 Step3 occurrence 中的 seller alias、Telegram、QQ、微信、email、Jabber、电话、网址、钱包、PGP 等 literals；
- 再应用同一组通用高精度正则；
- 反复执行到 fixed point；
- 只要仍有已知 identifier residue 就 fail closed。

合成器不会创造新联系方式，不会把真实联系方式改写成另一个虚构联系方式，也不会保留 identifier-presence marker。其目的是增强 clean content representation，而不是制造更容易分类的 identifier shortcut。

## 5. 三种确定性文本变换

### 5.1 `section_rotation`

改变类别、标题、描述等 profile section 在编码文本中的排列顺序，但不把一个字段的值写到另一个字段。它模拟卖家调整广告区块布局。

### 5.2 `segment_subsample`

当聚合字段包含多个 `||` 或换行 segment 时，使用固定 seed 选择一半左右的 segment，至少保留一个。它模拟同一操作者在不同账号只发布部分库存或缩短描述。

### 5.3 `layout_punctuation_normalization`

统一中英文逗号、分号、问号、感叹号及空白布局，不改写核心语义。它模拟输入法和排版差异。

所有变换由 `global_seed=20260716`、track、parent pair、variant 和 side 共同派生确定性随机流。同一输入和 policy 必须生成相同内容。

V1 generation audit 发现主轨 48 条中有 12 条在原定变换后两侧文本均未变化。V2 对这种 no-op 使用确定性的双侧 section rotation fallback；若 fallback 后仍然两侧都不变，生成过程直接失败。该修正只使用 train 文本结构，不读取任何 valid/test 分数。

## 6. 输出格式与 provenance

每条轨道输出：

- `synthetic_market_items.csv`：保持 vendor/title/description/category/market 等 item 风格字段；
- `synthetic_seller_profiles.jsonl`：保持 Step3 seller-profile 主要结构；
- `synthetic_pair_labels.step5_compatible.csv`：字段集合与当前 Step5 frozen labels 一致，但只服务隔离训练器；
- `equal_weight_duplication_control.step5_compatible.csv`：相同父样本和权重预算的复制对照；
- `synthetic_pair_lineage.csv`：记录 synthetic pair、parent pair、parent component、证据类型、label tier、变换及权重；
- E5 embedding cache、五折 OOF predictions 和 evaluation summary。
- `step21_sync_manifest.json`：覆盖最终目录中全部生成、缓存和评估文件的大小与 SHA-256，用于回传完整性核验。

所有 active-v2 synthetic UID 以 `synthetic://step21/v2/` 开头，market 标记为 `SYNTHETIC_TRAIN_ONLY`。synthetic label 强制设置：

- `split_name=train`；
- `benchmark_eligible=0`；
- `silver_train_only=1`；
- `usable_for_core_transfer=0`。

因此这些文件无法被常规 Step5/Step7 core-transfer 路径误当成 benchmark 真值。

## 7. 权重设计

如果父样本权重为 `w`、生成 `k` 个变体，则每个 synthetic child 的原始权重为：

```text
w_child = w / k
```

该父样本全部 synthetic children 的总权重为 `w`。训练时原父样本仍保留，因此增强相当于额外增加一份父样本质量预算，而不是让生成行数任意放大权重。

评估器重新使用 factorized effective parent weight，并在每个训练折内除以该父样本的变体数。文本增强和 duplication control 的 synthetic effective weight 严格相同。

## 8. 为什么必须有等权复制对照

只比较“无增强”和“文本增强”不能证明文本变换有效。提升可能仅来自正类总权重增加。因此 Step21 同时比较：

1. `no_augmentation`；
2. `equal_effective_weight_duplication`；
3. `identifier_redacted_text_augmentation`。

只有文本增强的 OOF AP 同时高于无增强和等权复制，才支持“变换后的表示提供了额外信息”。如果它只高于无增强但不高于复制，则结论是 minority reweighting，而不是 data-generation gain。

## 9. 评估协议

Step21 不读取当前 valid/test 选择方法。它在最终 v7 中文 train seller components 上做 5-fold grouped OOF：

1. 每折保留全部英文 source train；
2. 中文训练 component 整体进入 train 或 held-out fold，不能拆开；
3. synthetic child 永远跟随父 component，只能在父 component 位于训练侧时加入；
4. imputation、standardization、factorized weights 和 LR/L2 都在折内训练数据拟合；
5. 汇总所有中文 train OOF score 后计算 ROC-AUC 和 AP；
6. valid/test、Step16I Dev2 和 Step20 完全不参与方法选择。

Step21 的正面结果仍不能替代 Step20。论文最终有效性必须在真实、前瞻性、模型冻结后收集的数据上评估一次。

### 9.1 v1 分折缺陷及 v2 修正

首次 Linux v1 运行完整同步，但 OOF fold 数量为 `326/46/54/74/73`，其中一个 46 行 fold 全为正例。根因不是同步或模型，而是旧贪心分组器优先平衡累计正例，未联合约束负例和总行数。该 OOF 拼接会引入严重的 fold-specific probability-base-rate drift，因此 v1 数值仅保留为无效诊断。

中文 train 存在一个不可拆分的 175 行 seller component（11 positive / 164 negative），完全等大的五折在数学上不可实现。v2 按 component 的总数、positive 和 negative 三维归一化误差做确定性贪心分配，预期 fold 总数约为 `194/95/95/94/95`，positive 约为 `30/50/50/49/50`，且每折必须同时含正负标签。v2 写入独立路径，不覆盖 v1。

## 10. Linux 执行

同步 policy、三个 Python 文件、runner 和测试后，在 Linux 项目根目录执行：

```bash
bash scripts/run_step21_synthetic_train_only_linux_20260716.sh
```

GPU 只用于 frozen Multilingual-E5 编码；LR/L2 五折评估使用 CPU。若必须使用 CPU 编码，可执行：

```bash
STEP21_DEVICE=cpu bash scripts/run_step21_synthetic_train_only_linux_20260716.sh
```

## 11. 论文中允许和禁止的表述

允许：

> We evaluated deterministic, identifier-redacted, train-only positive-pair augmentation with parent-component grouping and an equal-effective-weight duplication control.

禁止：

> We collected 152 new Chinese sockpuppet positives.

48/170 是生成行数，不是独立真实正例数。Step21 只能缓解有限样本下的表示方差，不能解决 ground-truth scarcity、标签可信度或 prospective evaluation 缺失。

## 12. Linux v2 同步与数值结果

### 12.1 同步完整性

有效结果根目录为：

```text
reports/step21_synthetic_train_only/v2_balanced_grouped_oof_20260716/
```

`step21_sync_manifest.json` 声明 `21` 个 payload 文件、总计 `9,139,042` bytes。Windows 回传审计重新计算了每个文件的大小和 SHA-256，结果为：

- missing files：`0`；
- size mismatches：`0`；
- SHA-256 mismatches：`0`。

因此本节数值不是由漏传、截断或混入 v1 文件造成的。

### 12.2 生成结果

主轨 `primary_non_silver` 使用 `16` 个真实中文 train parent pairs、`13` 个 seller components，生成 `48` 个 synthetic pairs 和 `96` 个 synthetic seller profiles。父样本由 `1` 个 direct-identifier positive 和 `15` 个 style/structural soft positives 构成。父样本总权重与 synthetic 总权重均为 `16.0`。文本变化审计为 `41` 个 both-changed 和 `7` 个 one-side-changed，没有 no-op synthetic pair。

敏感性轨 `sensitivity_silver_anchor` 使用 `85` 个 silver parent pairs、`19` 个 seller components，生成 `170` 个 synthetic pairs 和 `340` 个 profiles。父样本包含 `56` 个 direct-identifier positives 和 `29` 个 component-anchor positives；父样本与 synthetic 总权重均约为 `38.05`。文本变化审计为 `156` 个 both-changed 和 `14` 个 one-side-changed，同样没有 no-op。

两轨都明确记录：

```text
new_real_positive_count = 0
new_independent_identity_count = 0
may_be_used_for_validation_or_test = false
```

### 12.3 分组折外边界

OOF 评估包含 `573` 条中文 train rows、`229` positives、`344` negatives 和 `222` 个重算 seller components。五个 held-out folds 为：

| Fold | Rows | Positive | Negative |
|---:|---:|---:|---:|
| 0 | 194 | 30 | 164 |
| 1 | 95 | 50 | 45 |
| 2 | 95 | 50 | 45 |
| 3 | 94 | 49 | 45 |
| 4 | 95 | 50 | 45 |

每折均有正负标签。Fold 0 的不平衡来自一个不可拆分的 `175` 行 seller component，不是随机拆分失败。与 v1 相比，v2 已修复单类 fold 和极端 base-rate 漂移问题，因此只有 v2 可以用于 Step21 方法判断。

### 12.4 主轨结果

| Method | ROC-AUC | AP |
|---|---:|---:|
| No augmentation | 0.765195 | 0.654591 |
| Equal-effective-weight duplication | 0.765220 | 0.653481 |
| Identifier-redacted text augmentation | 0.763075 | 0.652302 |

差值为：

- text augmentation minus no augmentation AP：`-0.002289`；
- text augmentation minus equal-weight duplication AP：`-0.001179`。

文本增强没有超过任一必要对照。差异虽小，但方向为负，不能声称 augmentation gain。

### 12.5 Silver-anchor 敏感性结果

| Method | ROC-AUC | AP |
|---|---:|---:|
| No augmentation | 0.765195 | 0.654591 |
| Equal-effective-weight duplication | 0.770044 | 0.654724 |
| Identifier-redacted text augmentation | 0.764471 | 0.648531 |

差值为：

- text augmentation minus no augmentation AP：`-0.006060`；
- text augmentation minus equal-weight duplication AP：`-0.006192`。

即使允许更多 direct/component silver parents，文本变换仍没有获得 representation gain。复制对照的 ROC-AUC 略升，但 AP 几乎不变，说明增加少数类有效权重最多轻微改变全局排序，并未改善正例优先检索质量。

### 12.6 科研结论与后续处置

Step21-v2 是协议有效、文件完整、结果为负的消融实验。它支持以下结论：

1. 失败不是由 v1 分折缺陷、文件漏传或文本变换 no-op 造成的；这些问题在 v2 均已修复。
2. 对现有少量 parent identities 做确定性、identifier-redacted 文本扰动，没有创造新的操作者证据，也没有提高 grouped OOF AP。
3. 主轨只有 `13` 个独立 parent components，且 `15/16` parents 是 soft positives；生成 `48` 行不能把有效独立样本量变成 `48`。
4. Silver sensitivity 也失败，表明问题不只是主轨 parent 数量小，而是当前变换没有增加可泛化的 identity information。
5. Step21 不进入 Step7/9/15 主训练，不进入 Step11/17 图谱验证，也不触发 prospective holdout。它保留为论文中的 negative augmentation ablation，或作为“样本行数增加不能替代独立身份锚点”的实证证据。

后续不应继续围绕同一 OOF 边界调 section rotation、segment ratio 或权重。真正能改变结论的输入是新的、独立的真实 controller identities 和模型冻结后的 prospective evaluation，而不是继续派生同一批父样本。
