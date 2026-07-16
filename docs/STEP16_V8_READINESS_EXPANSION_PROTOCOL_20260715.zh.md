# Step16-v8 证据切片扩充与隔离 Refreeze 协议

> **2026-07-15 修正通知**：首次 Linux V2 运行完成后，审计发现旧 representative-validation assignment 将 45 条 `silver_train_only` 行错误保留在 primary valid。V2 数值结果已失效。当前只认尚待 Linux 生成的 V3 canonical-split freeze；完整根因与修正见 `docs/STEP15_V8_V2_POSTRUN_AUDIT_AND_V3_CORRECTION_20260715.zh.md`。

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

V2 隔离物化曾成功运行，但其 primary split 继承了旧 V7 promotion assignment，因此不得继续使用。V3 物化代码已完成，必须在 Linux 重新执行以下内容：

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

V3 正式冻结目录为：

```text
reports/step16_v8_validation_refreeze/readiness_expansion_v3_20260715/
```

V3 不覆盖 V2。它继续保留 V2 的原子英文/中文 feature 发布修复，同时新增 canonical split 恢复、primary eval fail-closed、动态 component-safe control 分配、补充 URL 控制双审链和逐字节 identical-replay 验证。

补充 URL 审查的 V3 派生目录为 `reports/step15_v8/profile_url_control_review_v3_20260715/`。输入 reviewer lane 与输出 summary/CSV 物理分离，避免旧轮次产物因候选域变化而被覆盖。

物化后的选入配额为：

| 角色 | `train` | `valid` | 是否进入 primary alias benchmark |
|---|---:|---:|---|
| reviewed public-noise negative control | 20 | 20 | 否，仅 evidence expert |
| cross-snapshot direct control | 30 | 20 | 否，仅 evidence expert |
| component-closure control | 10 | 15 | 否，仅 evidence expert |

与 canonical valid/train 中原有 occurrence-state-backed 行合并后，V3 预期 readiness 为：

| Split | Public-noise negative | Verified-direct positive | Component-anchor positive |
|---|---:|---:|---:|
| `valid` | 24 | 23 | 15 |
| `train` | 20 | 30 | 10 |

V3 必须在 Linux 重现并由 manifest 绑定以下边界，否则运行失败：

- primary `train` 为 974 行，即英文 source `train=401` 加中文 canonical `train=573`；primary `valid` 为 120 行；英文 `valid/test` 不进入 v8 训练；
- evidence-expert-only controls 为 `train=60`、`valid=48`。Train canonical baseline 为零，因此新增 public/direct/component `20/30/10`；valid canonical baseline 已有 public `4`、direct `3`、component `0`，因此只新增 `16/17/15`，与 baseline 合计后仍严格达到预注册的 `20/20/15`。
- 固定内部开发测试仍为 200 行，pair UID 哈希为 `ea9e5f46b742cb017e3122b00536e7741208adebb5d293ff31237e634d226ef5`；
- seller 跨 split 重叠为 0；
- 所有 371 条 `silver_train_only` 行仍在 train，primary valid/test 中该字段计数必须为 0；
- Step3、Step4 与 canonical Step7 扩展后的精确行数由 V3 freeze manifest 重新记录，不能引用 V2 行数代替；
- 真实冻结输入测试确认，原 context 队列有 12 条 canonical-valid-compatible public controls；两轮 score/split-blind URL 审查的最终候选域为 10 条，其中 8 条获两位不同 reviewer 的 high-confidence 一致 negative 结论，但 `5kqp0.com` 与 `jnqp.com` 各自重复了 context 队列中的同一 seller pair。按 canonical `pair_uid + URL token` 合并后，补充池只新增 6 个唯一 pair，两个池合计 18 个唯一候选；canonical valid 已自带 4 条 occurrence-backed public-noise negative，因此只需从候选中确定性选择 16 条，最终达到不变的总门槛 20。另有 2 条不确定或分歧候选被排除，1 个更早的 DeepMix 提案因两侧 `.onion` 域名不同而在进入最终候选域前被 source-literal contract 拒绝；
- 没有覆盖任何原有可用 binary supervision，也没有把 evidence controls 提升为 primary benchmark；
- public-noise negative 的证据足以支持“共享标识符处于公共/商品/支持上下文”，但不足以把不同操作者身份当作 gold truth，因此固定为 `high_confidence_silver_agent_reviewed_public_noise_control`；
- 本次仅补足 canonical freeze 相对预注册门槛的缺口，共物化 108 条控制行（`train=60`、`valid=48`）；它们均设置 `usable_for_supervision=0`、`usable_for_core_transfer=0`、`primary_identity_model_eligible=0`、`benchmark_eligible=0`，只通过显式 v8 evidence-control 路径进入专家训练或充分性审计；
- manifest 必须记录 `thresholds_lowered=false`、`model_scores_read=false`，并闭合 supplemental candidate/reviewer/profile 传递哈希；
- 每条 supplemental URL control 必须在 pair 两侧物化相同的 `external_url` 风险 occurrence，并复算为 `risky_only_shared`、`support_only_shared` 或 `high_frequency_public`；只选中行数但 occurrence state 不成立时必须失败；
- Step20 新 manifest 的 self-hash 必须覆盖 assignment CSV hash；新建 evidence controls 的 scope 必须为 `evidence_expert_control`；
- 本地 Step15-v8 共发现 50 项契约测试，43 项静态/真实冻结输入测试通过，7 项 V3 artifact 测试因 Linux 尚未物化而明确跳过；Step15-v6/v7/v8 合并为 99 项、92 项通过、同 7 项延后。其中真实冻结输入配额测试已验证 public `valid/train=20/20`、direct `20/30`、component `15/10`，且 seller/component 隔离成立。Linux runner 在物化后设置 `STEP15_V8_READINESS_ROOT` 并重跑全部 50 项；V3 数值产物、行数和哈希尚待 Linux 生成，文档不预先宣称其已完成。

Windows 仅完成源记录静态核对、审查决定落盘、Python 语法检查和不执行模型的单元契约测试；没有运行 V3 数据物化、模型编码、特征数值重建、训练或评估。上述科研流水线仍必须在 Linux 运行。Linux runner 会在首轮生成 runtime；后续更换 `V8_RUN_ID` 时只复用通过完整哈希链验证的 runtime，任何部分存在或哈希不一致都会 fail closed。当前 Linux 一键入口为：

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

V2 结果已因 primary-valid silver 污染而失效；V3 源码、补充双审输入和本地契约检查已完成，但 V3 freeze 和数值结果尚未在 Linux 生成。下一步只能运行 V3 一键 runner。只有 V3 完成 B0-B3、contextual evidence expert 与 Step12-v8 后，才能根据预注册门槛决定是否进入 Step20；内部 200 条测试集仍不能用于模型选择。
