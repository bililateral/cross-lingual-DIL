# Step27-v1.1 来源契约修复与诊断方案

版本：2026-07-19
分支：`method/step27-english-pretrained-synthetic-adaptation`
Policy：`schema/step27_v1_1_exact_replay_policy.json`
Linux runner：`scripts/run_step27_v1_1_exact_replay_linux_20260719.sh`
输出根目录：`reports/step27_english_pretrained_synthetic_adaptation/v1_1_20260719/`

## 1. 修复对象

Step27 的科学问题保持不变：冻结只在英文 pair 上训练的 Step24 E5 LR/L2 来源评分器，再比较中文真实训练、等有效权重复制和父样本保持型半合成视图是否改善目标域残差学习。

原 v1 不能作为该问题的有效答案。代码审计发现：

1. Step15-v7/Step24 的 E5 输入是五个允许字段非空值按固定顺序用换行连接，文本中没有字段标题。
2. Step27-v1 给字段插入了类似 `[CATEGORIES]` 的人工标题，因此每个真实 seller 的 E5 输入都发生变化。
3. v1 又把真实 seller 与合成 seller 一起重新编码，没有复用已经冻结并被 Step24 哈希绑定的真实 E5 cache。
4. 因而 v1 中所谓“冻结英文来源评分器”只冻结了 LR 参数，没有冻结 LR 实际接收到的 E5 特征分布。

这会把输入契约漂移混入合成方法效果。v1 的失败既不能证明 M2 无效，也不能证明它有效，必须作为工程无效运行冻结。

## 2. v1.1 的精确重放

### 2.1 真实中文文本

真实 seller 文本严格使用 Step15-v7 的规则：

1. 按冻结的五字段顺序读取原始 seller profile。
2. 只拼接非空字段值，以换行分隔。
3. 对整段文本运行相同 identifier redaction，空文本回退为 `content unavailable`。
4. 重新构造的中文 train seller 文本语料哈希必须等于 Step24 clean-text manifest 中的冻结哈希。

字段级清洗副本只用于 Step27 明确列出的八个低维 residual features，不再替代 E5 输入。

### 2.2 真实中文 embedding 与 pair feature

真实 seller 不再重新编码。运行时从 Step15-v7 的冻结 `float32` E5 matrix 中按 `seller_uid` 精确取子集，并验证：

- cache metadata SHA-256；
- cache matrix SHA-256；
- seller UID 完整性与顺序；
- 向量逐元素精确重放；
- real pair 的 `identifier_redacted_e5_cosine` 与 Step24 pair feature 在绝对误差 `5e-13` 内一致。

任一条件失败即停止，不允许降级为重新编码。

Step24 的 reference chain 不是从可变 summary 动态推断，而是由 v1.1 policy 固定以下独立 SHA-256：Step24 policy、sync manifest、clean-text manifest、pair-feature summary、中文 pair-feature CSV 和 source artifact。运行时还要求 sync manifest 的文件记录、pair summary 的 E5 cache 记录和 clean-text manifest 的 cache 记录相互一致。因此不能通过同时替换 cache、summary 和 reference CSV 让漂移后的文件互相“证明”。

### 2.3 合成中文文本

合成 seller 仍需用冻结 Multilingual-E5 编码，因为它们没有历史 cache。其处理约束为：

- 变换前父文本必须能逐字重放 Step15-v7 的 values-only 清洗文本；
- 变换只允许段落/字段顺序和布局标点变化，不增删身份内容；
- 变换后再次执行 identifier redaction；
- semantic E5 文本保留变换后的字段顺序；
- residual lexical/structural features 按命名字段组计算，因此不会把字段轮换误当成字段语义变化；
- 每个 cache 记录未截断 token 长度、超过 `max_length` 的 seller 数量、比例和被截 token 总数。该审计不改变 encoder 参数。

在生成父样本或编码任何 synthetic text 前，Linux runner 会先运行只计算目录哈希的模型契约 preflight。当前 E5 模型目录的文件指纹必须与冻结 v7 metadata 中记录的模型指纹完全一致；该 preflight 不加载模型，也不执行数值实验。若模型目录不同，运行立即停止，必须恢复生成 v7 cache 时使用的精确模型快照；禁止降低检查强度，或把一个模型空间中的真实 cache 与另一个模型空间中的 synthetic embedding 混合。

除模型目录外，v1.1 policy 还固定 semantic model policy、v7 identifier-redaction policy、semantic encoder producer 和 v7 redaction producer 四个文件的 SHA-256。运行时同时检查实际 import 的模块路径、冻结 v7 metadata 的 producer hash 和这些固定值，防止只修改 `max_length`、text prefix、pooling 或编码实现而模型目录保持不变时产生静默表示漂移。

## 3. 合成样本与版本隔离

所有 synthetic seller/pair/duplication UID 必须从 policy 的 `synthetic_uid_prefix` 生成。v1.1 固定为：

```text
synthetic://step27/v1_1
```

禁止代码硬编码旧 `v1` 前缀。v1 与 v1.1 使用不同 policy、runner、输出根目录和 manifest，不允许跨版本覆盖或复用不匹配 artifact。

合成数据审计还要求每个 seed/track 至少存在一个相对父 pair 发生变化的重新计算特征；若所有 synthetic features 与 parent 完全相同，M2 退化成复制控制并立即失败。

Policy 的 `fail_closed_on_no_op=true` 被作为硬契约执行：每个 matched set 的每个预定 variant 都必须生成，最终行数必须等于固定 child budget，而不是只检查“不超过上限”。

## 4. 新增 S0 来源基线

v1.1 必须报告：

```text
step27_s0_frozen_english_source_only
```

S0：

- 直接加载 Step24 `artifacts.source_only.e5_lr_l2_control`；
- 不训练中文 residual；
- 不使用中文标签拟合任何参数；
- 十个 seed 重复同一组确定性来源分数，仅为表格和配对统计兼容，不能当作十个独立样本；
- ROC-AUC/AP 是纯冻结英文 scorer 的中文排序结果；
- 依赖中文 OOF 阈值的 F1、ACC 等只属于 threshold diagnostic，不能描述为 source-only 参数学习。

在 canonical train universe 上，S0 必须精确复现 Step24：

| Metric | Expected |
|---|---:|
| ROC-AUC | `0.7550015233065909` |
| AP | `0.6443826343928266` |

## 5. 修复后的比较与门槛

正式方法角色：

| ID | 作用 |
|---|---|
| S0 | 冻结英文来源模型，无中文参数拟合 |
| M0 | 真实中文 train residual baseline |
| M1 | 与 M2 父样本和有效权重匹配的 duplication control |
| M2 | 父样本保持型 synthetic-view residual，主要方法候选 |

M2 首先必须在 component-grouped train OOF 同时满足：

1. 相对 M1 的 AP 增量达到预注册门槛；
2. 相对 M0 的 AP 增量为正；
3. 相对 S0 的观察 AP 差与 component bootstrap 下界均不低于 `-0.01`；该门槛是在 v1.1 repair replay 运行前冻结的工程诊断门槛，不是原 v1 的预注册门槛；
4. 十 seed 方向、直接/component positive recall、template/public-noise FPR 和 synthetic shortcut 审计全部通过。

## 6. 为什么 v1.1 不打开旧 valid/test

来源重放缺陷是在查看过既有开发结果后发现的。即使修复只改工程契约，继续在同一 valid/test 上判断修复是否成功仍会产生事后选择偏差。因此 policy 明确冻结：

- `existing_valid_open_authorized = false`；
- `existing_internal_test_open_authorized = false`。

运行器只产生 train component-grouped OOF 和 Step12 技术门槛结果。底层评分 CLI 在读取对应 split 特征和标签之前再次检查 policy authorization，且只接受 canonical run-scoped gate path；Step12 自身也在构造输入 manifest 或读取任何工件前拒绝 post-hoc `valid_gate/final_diagnostic`；sync manifest 会拒绝任何出现在 v1.1 root 中的 valid/test 或后续 gate artifact。`technical_oof_gate_pass=true` 只表示实现值得在新的、冻结的开发批次复现，不授权打开旧 valid/test，更不构成论文性能结论。

## 7. 验证状态与 Linux 执行

Windows 仅完成不依赖模型数值运行的检查：

- 修改脚本 Python compilation：通过；
- Step27 contract tests：`47/47` 通过；
- 旧 v1 policy/output namespace 与 v1.1 物理隔离：通过。

正式数值编码、训练、统计与 manifest 生成只在 Linux 运行：

```bash
cd /home/yongpeng/cross-lingual
bash scripts/run_step27_v1_1_exact_replay_linux_20260719.sh
```

应同步回 Windows 的唯一结果根目录是：

```text
reports/step27_english_pretrained_synthetic_adaptation/v1_1_20260719/
```

结果解释顺序必须先检查 exact replay、S0 historical reproduction、tokenizer truncation 与 synthetic displacement，再解释 M2/M1/M0 的 OOF 差值。任何重放失败都属于工程失败，不得解释成科学负结果。
