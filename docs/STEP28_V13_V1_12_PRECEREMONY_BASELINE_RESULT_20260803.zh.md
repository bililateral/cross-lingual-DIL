# Step28-v13 v1.12 干净重启预仪式基线结果

> **历史关闭声明（2026-08-30）**：本文只证明当时的设计预检，不是最终数据结论。V1.12 后续永久质量失败，专属运行代码已清理；不得把本结果解释为当前训练资格。

日期：2026-08-03

## 结论

v1.12 已通过**设计阶段预仪式基线**，但尚未取得正式数据生成资格。当前结果只证明：在不读取正式密钥、不生成正式 split、不训练模型的条件下，新实现能够读取压缩失败历史、重放两套完整 K28 world，并拦截 v1.3–v1.11 已知故障。

它不能证明正式中文合成数据集已经生成，也不能支持任何 M0/M1/M2/M3 科研效果结论。

## 已冻结的实现边界

- 合同：`docs/STEP28_V13_V1_12_CLEANROOM_PRECEREMONY_CONTRACT_20260803.zh.md`
- 策略：`schema/step28_v13_v1_12_cleanroom_preceremony_policy.json`
- 预仪式实现：`scripts/step28_v13_v1_12_preceremony.py`
- 契约测试：`tests/test_step28_v13_v1_12_preceremony_contracts.py`
- 两世界回执：`reports/step28_synthetic_chinese_dataset/design_preflights/v1_12_cleanroom_20260803/two_world_preceremony_receipt.json`

策略文件 SHA-256 为 `ad4bfc710cfda3e61c67c76da20d47a94c07a7e6484b78a6893100976d777112`，canonical self-hash 为 `1f748568d766a23f9b2ee1022d779fc1afa51f64ae513c49cd6fab10de897755`。策略固定了合同、实现、测试、15 个 v1.2 纯净源文件、失败身份排除归档和清理清单的字节边界。

## 两世界预检结果

回执 SHA-256 为 `d21964a248e1138e65a654262e026c8c1457f8500e4915dbf5b83cdaba09d243`，canonical self-hash 为 `4f7763d838959c463c7c430f018367d8566707b2e18b4cee7894a5f87aef7f4d`。结果为：

- 2 个 K28 world、756 条 pair；40 正、716 负；
- 168 个身份资产、756 行 identity33；
- 5 份 M1 对照在两个 world 上共 10 张映射，均为端点不动点为零的双射，并保持完整 identity33 行多重集；
- 人工触发 1 次候选身份碰撞，固定逐资产计数器从 `0` 前进到 `1` 后成功，不重抽 master seed；
- 24 条机制槽记录以 `(world_uid, mechanism_slot_uid)` 全部唯一，跨 world 的 12 个模板槽复用按设计存在，world 内重复为 0；
- 915,996 个历史失败身份哈希和 90 个禁用 master commitment 均完成校验；
- 正式数据行数为 0，正式密钥/种子访问为 false，科学指标和模型训练均为 false。

## 已覆盖的历史故障

契约测试覆盖 v1.3 的机制槽作用域、v1.4 的 Windows 长路径与单一 manifest 版本、v1.5 的逐资产碰撞处理、v1.6 的最终正文 self-hash、v1.7 的优化器失败关闭语义、v1.8 的标题保持与描述登记后缀投影、v1.9 的自然语言与 join UID 分域、v1.10 的单一成员合同，以及 v1.11 生成/发布共用复合键审计器。

第一次非正式设计调用曾因强制碰撞分支漏导入 `itertools` 而在内存阶段终止；未访问正式密钥、未生成回执或数据。修复后端到端回归已强制执行该分支，失败缓存未保留。这不是一次正式实验运行。

## 验证结果

- `python -m py_compile`：通过；
- v1.12 定向契约测试：15/15 通过；
- 全仓 `python -m unittest discover -s tests`：396 项，389 通过、7 项既有声明跳过、0 失败，用时 809.034 秒；
- `git diff --check`：通过；
- 两世界回执精确重跑：复用相同冻结字节，不允许异字节覆盖；
- 测试生成的 `__pycache__` 已清理。

## 尚未完成的开跑条件

当前实现**故意不包含**正式 seed ceremony、四 split 正式生成器、正式 release tree/custody router 或模型训练。优化器部分目前验证的是失败关闭合同，不是使用未来完整正式输入执行的真实数值收敛证明。

因此下一阶段必须先实现并验证上述正式链路，完成真实优化器预检和新的冻结源码闭包；只有新的正式策略把授权由 false 改为 true 后，才能一次性生成四个新 seed commitment 并开始 train。不得修改本回执或把它提升为正式数据资格。
