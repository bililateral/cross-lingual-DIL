# Step25-v3.1 求解器修复结果审计

更新日期：`2026-07-18`

## 1. 审计结论

Step25-v3.1 的 Linux 返回包同步完整，求解器修复有效，但预注册的 C2 copy-aware dual-channel 方法没有获得支持。v3.1 应冻结为严格负结果，不能进入 D1、Step11 或 Step17，也不能继续在当前 D0 上调 copy penalty、方向约束或阈值。

旧 v3 在完成 v3.1 feature byte-parity 和结果对照审计后已从工作区删除；其无效指标仅保留在进度记录中，不再作为可消费 artifact。v3.1 是唯一可用于解释该方法的正式结果。

## 2. 完整性与数值正确性

- 目录包含 `10` 个文件；manifest 绑定其中 `9` 个 payload，总计 `1,527,262` bytes。
- 全部 payload、全部 `26` 个 producer、payload 聚合哈希、producer 聚合哈希和 manifest 自身哈希均由本地重新计算通过。
- `44/44` 个 C0-C3 constrained LR/L2 artifact 完整存在。
- 所有 artifact 的 termination reason 均为 `projected_gradient_kkt_tolerance`。
- 最大 final projected-gradient residual 为 `2.101590901304462e-09`，低于 `1e-8` tolerance。
- 英文和中文 pair-feature CSV 均与旧 v3 逐字节一致。
- 英文 `401` 行和中文 `573` 行均无重复 pair，无 seller component 跨 fold，无系数方向违规。
- CSV 独立重算的全部 `12` 组 ROC-AUC/AP 与 summary 完全一致。

因此，本次结果变化只能归因于求解器正确收敛，而不是特征、数据、fold、权重或模型矩阵变化。

## 3. 主要结果

| Evaluation | C0 ROC-AUC | C0 AP | C2 ROC-AUC | C2 AP | C2-C0 AP |
|---|---:|---:|---:|---:|---:|
| English grouped OOF | 0.658772 | 0.474662 | 0.676800 | 0.416316 | -0.058346 |
| Source-only on Chinese train | 0.870697 | 0.801847 | 0.851706 | 0.771609 | -0.030238 |
| Target grouped OOF | 0.865327 | 0.789848 | 0.853115 | 0.761755 | -0.028093 |

中文 target grouped bootstrap 的 C2-C0 AP 95% CI 为 `[-0.092698, 0.020856]`，`P(delta>0)=0.1462`。中文 source-only CI 为 `[-0.079772, 0.011120]`。两者都不支持 C2 改善，点估计均明确为负。

## 4. 关键切片

- canonical non-silver：C0 AP `0.351898`，C2 AP `0.325518`，差值 `-0.026380`。
- direct/component positives + all negatives：C0 AP `0.519074`，C2 AP `0.467395`，差值 `-0.051679`。
- soft positives + all negatives：C0 AP `0.745925`，C2 AP `0.714695`，差值 `-0.031231`。
- pair-local reliable：C0 AP `0.765023`，C2 AP `0.750084`。
- pair-local unreliable：C0 AP `0.914419`，C2 AP `0.818547`。

C2 没有通过任何主要正例稳健性要求，尤其明显伤害 direct/component slice。

## 5. 负例尾部

Target grouped OOF 中：

- template clone mean-rank delta `+0.026923`；
- template clone q95-rank delta `+0.067832`；
- template clone top-decile exposure delta `+0.036364`；
- template clone versus strong-positive violation delta `+0.048097`；
- semantic-topic mean-rank delta `+0.003064`，仍在预注册上限内；
- public-noise mean-rank delta `-0.072334`，top-decile exposure delta `-0.25`。

这说明 copy-aware 机制只对很小的 public-noise slice 有预期方向的效果，却把核心 template-clone slice 排得更高，不能视为整体去模板污染成功。

## 6. 为什么 C2 失败

正确收敛后的 11 个 C2 fit 中，两个 raw-minus-clean residual 系数始终为零；pair-local mean mask、shared-shingle、masked-span 和 global mean boilerplate 等多数 copy-risk 系数也始终为零。只有 pair-local maximum mask 和 global maximum boilerplate 在少数 fold 中获得小的负系数。

训练数据中 copy-risk 与大量 silver positive 同时出现。方向约束阻止模型把 copy-risk 学成正身份证据，但无法凭空创造稳定的负向鉴别力。加入 clean/reliability 通道后，模型重新分配 raw style 系数，却没有获得足够有效的 copy penalty，因此 template-clone 排名反而上升，direct/component positive 排名下降。

## 7. C3 与 operational control

C3 redacted-E5 sensitivity 在三个边界上均高于 C0：English AP `0.495430`、source-only Chinese AP `0.811496`、target OOF AP `0.795782`。但 C3 是预注册 sensitivity，不是 primary，也没有经过 C2 的完整晋级门槛，不能事后改成主模型。它只能支持一个未来独立假设：identifier-redacted semantic signal 可能比显式 copy penalty 更有价值。

删除旧 v3 前完成的只读 grouped-bootstrap 进一步表明，C3-C0 的 source-only AP CI 为 `[-0.025570, 0.046289]`，target OOF CI 为 `[-0.038852, 0.047708]`，均明显跨零。Non-silver target slice 虽有 `+0.053387` 点估计，但只有 `16` 个正例，CI 为 `[-0.092597, 0.226176]`。因此 C3 也不足以直接启动新模型主线。

Operational identifier control 将 Chinese AP 从 `0.771609` 提升至 `0.778196`，但只提升了 `3` 条 verified-direct positive，`8` 条 public-noise negative 没有发生分数变化。它是窄覆盖 sensitivity，不是 public-noise 问题的解决方案。

## 8. 最终门槛与后续边界

原 `11` 个 gate 只有 semantic mean-rank bounded 和 public mean-rank nonincrease 两项通过，即 `2/11`。因此：

- `d1_replication_candidate_eligible=false`；
- `publication_promotion_eligible=false`；
- `step11_or_step17_entry_allowed=false`；
- 不再继续 Step25-v4 或在 D0 上调 copy-risk 权重；
- 将 Step25-v1/v2/v3.1 作为一组有机制诊断价值的负结果保存；
- 后续主线应回到更可靠的独立身份证据建设，或在新的、完全冻结的实验中预注册 redacted-semantic 假设，而不是继续消费当前 D0。
