# Step28-v13 full-378 模型训练就绪状态与启动说明

日期：2026-08-02
状态：`V1_5_CONTRACT_AND_SEEDS_FROZEN_OPEN_SPLITS_NOT_GENERATED`

## 1. 当前结论

v1.3 和 v1.4 均已失败关闭；当前正式后继是 v1.5。v1.5 的设计 policy、prelock 与四份一次性 seed 承诺已冻结，尚未生成可发布数据，也尚未训练模型。

- 旧 `v13_training_ready_v1_2_order_repair_20260731` 字节保留，但因公开候选捷径和 Audit custody 破坏而失去正式训练资格。
- 失败版本 `v13_training_ready_v1_3_full378_fresh_20260801` 的 policy、prelock、四份 seed 承诺和失败标记原样保留；其 train/development/Audit split 均未发布，禁止修补后重用。
- 失败版本 `v13_training_ready_v1_4_full378_scope_repair_20260802` 完成了 train 500/500 world 的内存构造，但在写盘复核阶段失败；train 未发布，development/Audit 未启动，四份 seed 与失败记录禁止重用。
- 新正式目标为 `v13_training_ready_v1_5_20260802`。v1.5 identity policy/prelock 与四份 seed 承诺已冻结；split、M0 分数、训练模型和效果报告仍为 0。
- 当前仍为 **NO-GO**；不得把配置预检或单元测试表述成数据成功或模型成功。

## 2. v1.3/v1.4 失败根因与 v1.5 修复边界

v1.3 的 `mechanism_slot_uid` 是每个 world 复用的机制模板槽名，但最终校验器错误地要求它在整个 split 全局唯一，因此第二个 world 起必然碰撞。修复只把唯一性作用域改为 `(world_uid, mechanism_slot_uid)`；seller、item、controller、identity asset/value 等真正实体仍保持跨 split 全局约束。

v1.4 修复了机制槽作用域，但暴露出两项独立的持久化合同错误。第一，原子写入使用 Windows 长路径接口，而紧随其后的 `Path.read_bytes()` 没有；264 字符的投影文件实际已写出，却被误报为不存在。第二，生成器声明 split manifest v2，而 train 的后续消费者仍要求 v1；即使绕过第一处，后续也必然失败。v1.5 统一长路径读取、遍历和 stat，并把 manifest/execution-lock 版本改成单一权威常量。

v1.5 当前正式合同：

- `schema/step28_v13_full378_v1_5_identity_transfer_policy.json`
- identity design 正式 self-hash：`d3515e7791f164b7a6a2ac55345c0306cec2fc82dcce2800a7649c0d58c67be5`
- 模型实现源码闭包：76 个文件
- `schema/step28_v13_full378_v1_5_fresh_release_preceremony_lock.json`
- 正式 prelock self-hash：`04636dcf45bf295111d434aadf240425af451c268501073bee9eea094b21c589`
- 数据生成源码闭包：51 个文件

v1.5 正式捷径 policy self-hash 为 `05ab8fa726a240055b33792c2554ee92c37931bd874e00879de8cdf3559e9530`；正式近似链接 policy self-hash 为 `f7173066ef285f688ee699dc709cb11648c49f1effecb34c248e9add6767b577`。

两-world 真实生成预检已通过：2 个完整 K28 world、756 个 pair 通过最终聚合和二次重放，回执 self-hash 为 `48ab5983aebe4e80f373b06839f66f5659279b7515b07e25c257bd1a6bca3f6e`。新增端到端测试还把两个 world 完整写入临时 public/private stage，再由正式 manifest 消费者逐文件验证。v1.3 与 v1.4 共八个 master commitment 均进入禁用集合；v1.4 失败 train 的 42,000 个身份值也以哈希加入 v1.5 排除库，与原 283,496 个哈希零交集，合计 325,496 个禁用哈希。v1.5 四份 seed 已按固定顺序一次性提交，train/development/Audit A/Audit B 的公开回执 SHA-256 分别为 `94be3d06…ba38`、`90c5952e…fdb`、`f154bc8e…ad73`、`cf2f74ca…3ff8`；原始密钥未公开。

## 3. 新数据边界

四个 split 均为 500 个互不连通的 K28 world。每个 world 使用全部 378 个无序 seller pair，其中 20 个正例、358 个负例。每个 split 因此有 189,000 pair、10,000 正例和 179,000 负例；自然 AP 基线为 `20/378`。

train/development 标签可公开用于规定用途。Audit A/B 的标签、qrel、controller、机制、身份资产和生成私有证据只允许存在于 Git 忽略的 `private_custody/`。C40 不得进入训练、调参、阈值或主评估。

Audit 文本生成另需一次性授权：必须先完成 train/development、五份 M1、M1 独立性门和公开捷径审计。Audit B 文本只能在 Audit A 文本完成后生成；这不授权打开任一 Audit 真值。

## 4. 模型与训练矩阵

- **M0**：冻结英文 `LightGBM + legacy18 + LaBSE`，不使用任何合成中文标签重新拟合。
- **C0**：冻结英文 `LightGBM + legacy18`，只作敏感性对照。
- **M1-r01…r05**：五个 train-only 无信息对照。每份将完整 identity33 行做端点不重合双射；五个拟合进程各自只能挂载一份矩阵。
- **M2**：与 M1 使用完全相同的 p0、样本、权重、变换、L2 和求解器，只把错配 identity33 换为正确 identity33。
- **M3-base/M3-joint**：只用 fresh synthetic train 标签训练的直接中文 LightGBM 强对照；M3-joint 额外读取正确 identity33。它们不是迁移模块。

共享 L2 只在 fresh train 的 5 折 world-grouped OOF 上选择。development 只冻结各角色的 world-equal F1 阈值并执行五份 M1 对 M0 的 AP 等价门；不允许重新选择 L2、模型结构或概率校准。

## 5. 指标和重放要求

分类必须报告 ROC-AUC、AP、梯形 PR-AUC、Precision、Recall、F1、Specificity、Balanced Accuracy、MCC、Brier、Log Loss、Recall@FPR=1% 和原始/世界等权混淆矩阵。

检索对每个 world 的 28 个 seller 全部轮流查询，报告 MRR、MAP、Recall@1/3/5/10 和 NDCG@1/3/5/10。置信区间与置乱/Bootstrap 的独立单位始终是 world，不能把 189,000 个 pair 当作独立样本。

以下三层必须生成不可变深度重放回执：

1. development：从冻结模型逐行重算预测、阈值、全部指标、M1 等价门及 9,999 次 AP bootstrap；
2. Audit A/B 盲预测：在真值打开前逐行重算分类概率、双向检索分数和全部模型 lineage；
3. Audit 评估：重新打开已承诺真值，重算分类、检索、私有诊断和每个 bootstrap 数组。

文件哈希相同但无法通过计算重放，仍然必须停止。

## 6. 正式执行顺序

1. 冻结 identity design policy；冻结 fresh pre-ceremony lock。
2. 按 train、development、audit_a、audit_b 顺序各提交一次 master seed；禁止重抽。
3. 生成并审计 train/development；物化五份 M1，执行 M1 标签独立性与公开捷径门。
4. 签发 Audit 生成授权，依次生成 Audit A、Audit B。
5. 执行跨版本精确交集、冻结近似链接、可见文本诊断和独立 finalizer；只有根状态 PASS 才进入模型阶段。
6. 建立 label-free M0 execution lock；Windows 生成公共投影，Linux 运行 `scripts/run_step28_v13_m0_linux_20260801.sh`。
7. 同步 Linux bundle 后，在 Windows 运行 `scripts/run_step28_v13_post_gpu_windows_20260801.ps1 -Phase PostGpu`。
8. development M1 等价门失败则 Audit A/B 保持密封并停止。
9. 显式运行 `-Phase AuditA`；只有 A 通过且深度重放通过才授权 `-Phase AuditB`。

数据侧 Windows 总入口为 `scripts/run_step28_v13_full378_dataset_windows_20260801.ps1`。任何 runner 的恢复运行只能验证和跳过已完成阶段，不能重新训练、换 seed 或覆盖产物。

## 7. 当前验证与清理

v1.5 正式 Windows `Validate` 实际运行 480 项合同测试：473 项通过、7 项既有声明跳过、0 项失败，用时 843.816 秒。另有 joblib 在 Windows 无法探测物理核心数后回退逻辑核心数的非致命环境警告；未改变测试结果或科研产物。

重型回归包括两-world 真实生成与二次重放、完整临时写盘和 manifest 消费、Windows 超 260 字符路径、合成数据合同、训练就绪构建器及 Step7/Step28 后半段专项。当前没有生成 v1.5 正式 seed、正式 split 或模型。

旧 v2 执行树及诊断派生物已按依赖边界清理；冻结 v1.2、历史文档和仍被哈希引用的输入保留。当前未发现还能安全删除的中间临时目录。后续每个正式阶段结束后仍须复查临时工作区和磁盘占用。

## 8. 结论边界

当前只允许说“v1.3/v1.4 已失败关闭；v1.5 正式合同与四份 seed 承诺已经冻结，split 生成尚未开始”。不能说中文合成数据集已经生成，不能说 M0/M1/M2/M3 已训练，也不能报告任何效果。即使 synthetic Audit 最终通过，也只能支持合成身份机制上的内部因果证据；真实中文地下市场外部有效性仍需新收集、从未参与开发的真实 ground truth。
