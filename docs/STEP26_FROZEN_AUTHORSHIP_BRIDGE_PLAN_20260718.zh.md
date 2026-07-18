# Step26 冻结作者风格同边界桥接实验计划

更新日期：`2026-07-18`

## 1. 为什么现在必须先做 Step26

当前最容易被误读的结果来自 Step24 和 Step15-v8：

- Step24 的英文源域 semantic-plus-style 模型在中文 canonical `train` 上取得 `AP=0.802718`，而 matched identifier-redacted E5 为 `AP=0.644383`；
- Step15-v8 在 corrected `120` 条 representative valid 和 `200` 条 internal development test 上的表现明显较低，contextual fusion 在 internal test 上为 `AP=0.620525`；
- 这两个结果不是同一个 evaluation boundary。Step24 从未编码或评分中文 valid/test seller，Step15-v8 也没有使用 PCM/mStyleDistance。因此不能把 `0.802718` 与 `0.620525` 直接比较，更不能据此判断作者风格表示一定优于或劣于 v8。

Step26 的首要目的不是再训练模型，而是填补这个缺失的同边界实验：将已经冻结的 Step24 编码器和英文 source-only LR/L2 系数，原封不动地应用到 corrected Step15-v8 的同一批 `120 + 200` pair。只有这一步完成，才能判断 Step24 的高分是可迁移的作者风格信号，还是主要由中文 train silver positive/evidence composition 造成的开发集假象。

## 2. 核心科研问题

Step26A 只回答一个预注册问题：

> 在不使用中文 valid/test 拟合编码器、模型系数、标准化参数、阈值或候选配置的前提下，Step24 冻结的 identifier-redacted 作者风格表示，能否在 corrected representative valid 上稳定提高 seller-pair identity ranking，并且不增加 template clone 或 public-contact/URL noise 在顶部候选中的暴露？

该问题同时检验三个可能根因：

1. **同边界缺失**：Step24 的好结果是否能离开 canonical train；
2. **silver/composition confounding**：Step24 的高 AP 是否依赖 `213/229` 个 train-only silver positive；
3. **style contamination**：PCM/mStyleDistance 是否仍把模板、公共广告格式或高频文案当成作者风格。

## 3. 数据边界

### 3.1 代表性验证集

- 精确继承 corrected Step15-v8 的 `120` 个 `pair_uid`；
- `30 positive / 90 negative`；
- 仅该 split 可以决定是否允许一个 Step26B 机制实验；
- 不重新抽样、不移动 pair、不读取旧 170-row invalidated validation overlay。

### 3.2 内部开发测试集

- 精确继承 corrected Step15-v8 的 `200` 个 `pair_uid`；
- `50 positive / 150 negative`；
- 仅作为机制诊断，不能满足任何 promotion gate；
- 由于该 test 已在历史多轮分析中被查看，不能承担论文 final holdout 角色。

### 3.3 隔离规则

- 当前中文 canonical train seller 与 valid/test seller 完全不重叠；
- valid 与 internal test seller 完全不重叠；
- component ID 使用 corrected Step15-v8 冻结预测中的 `v7_component_id`；
- `silver_train_only=1` 的 pair 禁止进入 Step26 evaluation；
- Step26 不新增、不移动、不修改 Step5 标签。

## 4. 盲编码顺序

Step26 严格区分“生成冻结分数”和“连接评估标签”：

1. 从三个 Step15-v8 冻结 comparator 文件取得精确 `pair_uid` allow-list；
2. 直接从 canonical `pair_uid` 解析左右 seller UID，不读取 label/evidence；
3. 使用 Step15-v7 的完整 identifier-redacted E5 cache 校验 seller 存在性；
4. 重放 v7 的相同文本字段、seller-specific identity literal redaction 和高精度通用 identifier redaction；
5. 校验完整中文 clean-text corpus hash 与 v7 E5 metadata 完全一致；
6. 使用冻结 PCM 与 mStyleDistance 模型对 evaluation seller 编码；
7. 生成三个 cosine 特征，并应用冻结 English source-only LR/L2 artifact；
8. 所有 Step24 source 分数生成后，才连接 frozen label、evidence type 与 Step15-v8 comparator 分数。

这保证中文 valid/test 不参与 feature fitting、standardization、coefficient fitting、model selection 或 threshold selection。

## 5. 冻结表示与模型

### 5.1 三个 pair feature

1. `identifier_redacted_e5_cosine`
2. `pcm_multilingual_authorship_cosine`
3. `mstyledistance_cosine`

PCM 与 mStyleDistance 使用 Step24 policy 中已经 pin 住的 revision、路径、最大长度、维度和 normalization。禁止本地微调，禁止更换 checkpoint，禁止根据 Step26 结果修改文本清洗。

### 5.2 三个 English source-only artifact

1. `e5_lr_l2_control`：只用 redacted E5；
2. `style_only_lr_l2_control`：只用 PCM 与 mStyleDistance；
3. `semantic_style_lr_l2_primary`：使用全部三维特征。

三个 artifact 均来自 Step24 的 `401 = 116 positive / 285 negative` 英文 canonical train。Step26 只读取其冻结的 feature order、standardization mean/scale、intercept 和 coefficients，不执行任何 `.fit()`。

### 5.3 同 pair 对照

Step26 对每个 pair 同时保存：

- raw redacted E5 cosine；
- raw PCM cosine；
- raw mStyleDistance cosine；
- 三个 frozen English source-only LR/L2 分数；
- Step15-v8 B0 分数；
- Step15-v8 selected clean 分数；
- Step15-v8 contextual fusion 分数。

所有模型必须在完全相同的 pair 顺序上比较。

## 6. 评价指标

### 6.1 主指标

- `average_precision`，即 AP。

当前正例率分别为 `25%`，AP 比 Accuracy 更能反映顶部 positive ranking 质量。

### 6.2 次指标

- `roc_auc`；
- `pr_auc`，按 precision-recall 曲线梯形积分。

Step24 English source-only artifact 没有冻结一个可合法迁移的 source validation threshold，因此 Step26 不为它事后选择中文阈值，也不将 ACC/F1 作为该桥接实验的主结论。

### 6.3 证据切片

必须单独报告：

- direct/component positive 加全部 negatives；
- soft positive 加全部 negatives；
- ordinary negative；
- public-contact/URL noise；
- semantic-topic negative；
- template-clone negative。

负例切片报告 mean、q95、maximum score，以及进入 `top-k` 的数量，其中 `k` 等于该 split 的真实 positive 数量。top-positive-budget intrusion 是 scale-invariant 排序诊断，不会因不同模型概率刻度不同而失真。

## 7. 统计比较

Step26 使用 paired seller-component grouped bootstrap：

- resamples：`5000`；
- seed：`2026071801`；
- 每次以完整 `v7_component_id` 为单位有放回抽样；
- 计算 frozen Step24 primary 相对 Step15-v8 clean 的 AP delta；
- 输出 point delta、mean delta、95% CI 与 `P(delta > 0)`。

不能用逐 pair 独立 bootstrap，因为同一 seller component 内的 pair 不独立。

## 8. Step26A 晋级门槛

只使用 representative valid 检查全部门槛：

1. primary AP 相对 v8 clean 至少 `+0.03`；
2. paired grouped-bootstrap AP-delta 95% CI lower bound `>= 0`；
3. direct/component slice AP delta `>= -0.03`；
4. soft-positive slice AP delta `>= -0.03`；
5. top-positive-budget 中 template negative intrusion 不增加；
6. top-positive-budget 中 public-noise intrusion 不增加。

六项必须全部通过。internal test 无论表现多高都不能补救 valid gate 失败。通过仅允许进入一个 Step26B 机制实验，不允许直接形成论文性能结论。

## 9. Step26B 的条件式后续

Step26B 不是本轮自动执行内容。只有 Step26A 通过后才允许实现：

- clean scorer 继续负责无 identifier 的候选排序；
- direct-support expert 只处理 bilateral seller-facing direct evidence，并只允许 uplift；
- public/template-noise expert 只处理 risky/support/high-frequency public evidence，并只允许 downgrade；
- mixed/ambiguous/no-shared-identifier 默认保持 clean score；
- clean probability 必须来自 component-grouped OOF 或冻结 source scorer；
- 禁止统一乘 `0.1`、禁止全局 copy penalty、禁止八分类辅助 head。

Step26A 不通过时，不得通过调整 gate、换 primary 或查看 internal test 后重写 Step26B。

## 10. 结果决策树

### 10.1 Step26A 通过

说明 Step24 表示至少有一部分可跨边界复现的 authorship signal。随后只做一次预注册 Step26B conditional evidence gate，再冻结配置并进入 Step20 prospective holdout。

### 10.2 Step26A 未通过

说明 Step24 的高 D0 分数无法在 corrected valid 上复现，或其收益由 template/public noise 抵消。此时停止 D0 模型搜索，优先建设 seller-component-disjoint D1 与 Step20；若强身份证据仍不足，论文转为 evidence-type concept drift、数据集与严格负结果分析。

## 11. 实现与输出

实现文件：

- `schema/step26_frozen_authorship_bridge_policy.json`
- `scripts/step26_common.py`
- `scripts/step26_build_frozen_style_cache.py`
- `scripts/step26_evaluate_frozen_authorship_bridge.py`
- `scripts/step26_build_sync_manifest.py`
- `scripts/run_step26_frozen_authorship_bridge_linux_20260718.sh`
- `tests/test_step26_frozen_authorship_bridge_contracts.py`

Linux 输出隔离在：

`reports/step26_frozen_authorship_bridge/v1_20260718/`

该目录包含 clean-text/embedding manifest、pair manifest、pair features、long-format predictions、model/slice metrics、bootstrap comparison、summary 和 sync manifest。所有写入为 immutable identical-replay；不同内容不能覆盖同一路径。

## 12. 当前状态

Windows 端只完成了 policy、代码、测试和 config-only 校验，没有运行模型编码、训练或真实数值评估。PCM/mStyleDistance 模型仍只要求在 Linux 原路径存在。真实结论必须等待 Linux 运行结果同步回来后再审计。
