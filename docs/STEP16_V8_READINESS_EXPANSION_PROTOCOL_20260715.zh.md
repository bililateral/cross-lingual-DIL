# Step16-v8 证据切片扩充与隔离 Refreeze 协议

## 1. 目的

Step15-v8 的预注册验证门槛要求：

| Split | Public-noise negative | Verified-direct positive | Component-anchor positive |
|---|---:|---:|---:|
| `valid` | 20 | 20 | 15 |
| `train` | 20 | 30 | 10 |

初始数据不满足上述门槛。此次扩充的目的不是提高模型分数，也不是把模型预测回填为标签，而是建立足够且可审计的 occurrence-level 证据切片，使 public-noise veto、direct uplift 和 component non-regression 可以在不读取固定内部测试集的情况下训练和验证。

## 2. 不可变约束

1. 固定 200 条中文内部开发测试 pair 及其哈希不变。
2. 任何 seller connected component 不得跨 `train`、`valid` 和内部开发测试。
3. 候选生成和 split 分配不得读取模型分数、模型错误或测试指标。
4. 两名审查者独立读取相同的 source-evidence packet；packet 隐藏 queue kind、split、旧标签和模型信息。
5. 两人只有在 identity label、evidence type 和 confidence 全部一致且均为 high 时，候选才可直接进入物化阶段；否则必须由第三名独立审查者裁决。
6. 规则命中只能生成候选，不能直接生成 identity ground truth。
7. 预注册的 `20/20/15` 与 `20/30/10` 门槛不得降低。

## 3. Public-noise negative 扩充

候选来自 Step3 item-level identity occurrence，同一 identifier 在两个 seller 中共享，但 occurrence context 显示为 product data、support-only、victim data、公共网址或高频公共标识符。

候选构建已修复一个历史错误：未审候选不能先按共享公共 URL 全局连成 component。否则一个高频公共网址会把大量无关 seller 合成巨型 component，并错误地把大多数候选判为跨 split。审查前仅使用 provisional pair component；审查接受后才按完整 seller graph 重新闭包。

正式 120 条盲审候选的双审结果为：

- 116 条：两人一致判为 high-confidence `public_contact_or_url_noise` negative；
- 3 条：两人一致判为 high-confidence direct positive，但与初始 parser state 冲突，只能进入 parser-conflict 诊断，不直接物化为 public negative；
- 1 条：两人分歧，经第三人独立裁决为 high-confidence public URL negative。

因此可用于后续选择的 public-noise negatives 共 117 条。正式 refreeze 只从不接触内部测试、且不桥接原 train/valid component 的候选中，以固定 SHA256 排名选择所需配额。

## 4. Verified-direct 与 component-anchor 控制扩充

### 4.1 数据来源与可验证关系

`products_data.csv` 与 `market_item.xlsx` 的严格中文市场 profile 共享 431 个完全相同的 platform vendor ID。进一步检查显示，其中 403 个 vendor 同时具有至少一条逐字精确共享的 title 或 description。严格侧这些 profile 全部来自 `中文暗网交易市场`；辅助侧缺失 market 字段，但相同 vendor ID 加精确库存重合为同一市场跨快照账号持续性提供了直接来源证据。

这类关系必须命名为：

> cross-snapshot same-account identity-persistence control

它不是新发现的 sockpuppet alias，也不得用于夸大跨账号马甲识别的数据规模。

### 4.2 Direct persistence control

Direct 候选两端分别为：

- 严格中文市场中的 vendor profile；
- `products_data.csv` 中具有相同 platform vendor ID 的辅助 profile。

候选还必须具有精确 title 或 description 重合。原始 vendor ID 只允许出现在审查证据和 occurrence-level evidence provenance 中；clean semantic text、clean pair features、B0-B3 clean scorer 输入均不得包含该 ID。

### 4.3 Component closure control

对 auxiliary item 数量不少于 4 的 vendor，按固定哈希将商品划成两个不重叠 cohort，每侧至少 2 条。两个 cohort 通过相同 strict-profile -> auxiliary-profile platform ID 路径闭合，因此其 pair 可作为 same-controller component closure control。

该控制只验证 component anchor non-regression，不代表两个不同账号的别名关系。

### 4.4 主基准隔离

所有 direct/component cross-snapshot controls 必须同时满足：

```text
primary_identity_model_eligible = 0
evidence_expert_eligible = 1
evidence_expert_validation_eligible = 1
benchmark_eligible = 0
```

因此：

- 不进入 B0-B3 clean scorer 的训练；
- 不参与 clean representation 或 model family 选择；
- 不参与 primary valid AP、threshold 选择或内部测试指标；
- 只为 occurrence-level evidence expert 提供 train/valid 证据覆盖，并单独报告。

## 5. 独立审查结果

跨快照队列共 397 条：

- direct persistence：361 条；
- component closure：36 条。

两名独立盲审者对全部 397 条均给出相同 high-confidence positive 结论；direct/component evidence type 与候选 provenance 一致，没有分歧，因此不需要第三人裁决。该一致性来源于候选本身使用 exact platform ID、exact inventory overlap 和 hash-bound source records，而不是模型分数。

## 6. Refreeze 与特征物化

隔离物化已经完成，执行内容为：

1. 选择至少 20/20 个 valid/train public negatives；
2. 选择至少 20/30 个 valid/train direct controls；
3. 选择至少 15/10 个 valid/train component controls；
4. 将获准 strict public negatives 扩展到 Step4 和 canonical Step7 pair universe；
5. 为 selected auxiliary profiles 和 cohort profiles 构建 Step3 profiles；
6. direct controls 两侧追加 hash-bound `platform_vendor_id` occurrence signal，使 occurrence state 可复算为 `verified_direct_both_sides`；
7. component controls 不向 pair 两端注入共享直接 identifier，防止把 component control 错算成 direct；
8. Linux 重新生成 identifier-redacted E5 cache 和 v7 train-reference pair features；
9. 在完整 seller/component graph 上重算 split component，并核验固定内部测试哈希；
10. 输出 manifest、producer SHA、输入输出哈希、候选 UID 列表和 readiness check。

正式冻结目录为：

```text
reports/step16_v8_validation_refreeze/readiness_expansion_v2_20260715/
```

其中 `v2` 修复了首版生成 policy 中英文/中文 v7 feature 输出不在同一原子发布目录的问题。首版未用于模型训练或结果选择；正式运行只认 `v2` 的 policy、manifest 和输出路径。

物化后的选入配额为：

| 角色 | `train` | `valid` | 是否进入 primary alias benchmark |
|---|---:|---:|---|
| reviewed public-noise negative control | 20 | 20 | 否，仅 evidence expert |
| cross-snapshot direct control | 30 | 20 | 否，仅 evidence expert |
| component-closure control | 10 | 15 | 否，仅 evidence expert |

与原有 occurrence-state-backed 行合并后，正式 readiness 为：

| Split | Public-noise negative | Verified-direct positive | Component-anchor positive |
|---|---:|---:|---:|
| `valid` | 24 | 23 | 15 |
| `train` | 20 | 30 | 10 |

其他冻结结果：

- primary `train` 为 924 行，即英文 source `train=401` 加中文 primary `train=523`；primary `valid` 为 170 行；英文 `valid/test` 不进入 v8 训练；
- evidence-expert-only controls 为 `train=60`、`valid=55`，由 public `20/20`、direct `30/20`、component `10/15` 组成；
- 固定内部开发测试仍为 200 行，pair UID 哈希为 `ea9e5f46b742cb017e3122b00536e7741208adebb5d293ff31237e634d226ef5`；
- seller 跨 split 重叠为 0；
- Step3 profiles 为 5,197，item identity signals 为 4,530；
- Step4 与 canonical Step7 pair universe 均为 3,964，原有 3,857 行逐字段保持不变；
- 仅 6 条原 `uncertain + unusable_for_supervision` 行在隔离控制 overlay 中获得 public-noise negative 审查结论；没有覆盖任何原有可用 binary supervision，也没有把这些行提升为 primary benchmark；
- public-noise negative 的证据足以支持“共享标识符处于公共/商品/支持上下文”，但不足以把不同操作者身份当作 gold truth，因此固定为 `high_confidence_silver_agent_reviewed_public_noise_control`；
- 所有 115 条扩充行均设置 `usable_for_supervision=0`、`usable_for_core_transfer=0`、`primary_identity_model_eligible=0`、`benchmark_eligible=0`，只通过显式 v8 evidence-control 路径进入专家训练或充分性审计；
- manifest 中 `thresholds_lowered=false`、`model_scores_read=false`；11 个正式输出、23 个直接/传递输入哈希及三个 JSON 自哈希均已复核闭合；
- `python -m unittest tests.test_step15_v8_contextual_evidence_contracts` 为 `33/33` 通过；再次运行 materializer `--check-only` 得到相同 readiness。测试还独立约束 summary 的 train 计数只能使用英文 `train`，不能误计英文 `valid/test`。

Windows 仅完成数据物化、无模型契约检查和 identifier redaction 检查。模型编码、v7 feature 数值重建与 v8 方法评估仍必须在 Linux 运行。Linux runner 会在首轮生成 runtime；后续更换 `V8_RUN_ID` 时只复用通过完整哈希链验证的 runtime，任何部分存在或哈希不一致都会 fail closed。当前 Linux 一键入口为：

```bash
bash scripts/run_step15_v8_readiness_linux_20260715.sh
```

## 7. 论文解释边界

此次扩充能解决的是 evidence expert 的数据充分性和验证缺口，不能单独证明跨账号中文马甲识别已经解决。论文必须分别报告：

1. primary cross-account alias benchmark；
2. public-noise hard-negative slice；
3. cross-snapshot same-account direct control；
4. component-closure control；
5. prospective Step20 holdout。

只有 primary benchmark 与 prospective holdout 上的结果才能支撑马甲识别性能结论；cross-snapshot controls 只能支撑证据融合机制是否按预期工作。

## 8. 当前状态

候选生成、双人独立审查、第三人分歧裁决、审查对账、隔离 refreeze、Step4/canonical Step7 扩展、manifest、自哈希、固定测试集和 component-disjoint 检查均已完成。数据充分性门槛已经通过，但这只表示 Step15-v8 可以开始正式 Linux 实验，不表示方法已通过 promotion gate。只有 Linux 完成 B0–B3、contextual evidence expert 与 Step12-v8 后，才能根据预注册门槛决定是否进入 Step20；内部 200 条测试集仍不能用于模型选择。
