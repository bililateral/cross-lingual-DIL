# Step 28-v13 正式中文合成数据集 C40 行序修复合同

更新日期：2026-07-31  
当前状态：`FROZEN_READY_FOR_GENERATION / PARENT_KEYS_REUSED`  
修复版本：`v13_training_ready_v1_2_order_repair_20260731`

## 1. 修复原因

`v13_training_ready_v1_20260729` 的 C40 成员集合、标签公式、文本、身份33维
和四分区隔离均通过原发布门，但发布后逐行审查发现一个未被原门覆盖的严重
捷径：实现把“选择/补齐候选”的 HMAC 排名同时用于最终文件行序。由于正负
样本来自大小不同的分层池，正例和负例在文件中的位置明显不同，train 和
development 的行号可高准确率预测标签。原 v1 因此不得用于 M1/M2 训练、
调参或评估；原发布清单只作为失效历史证据保存。

第一次修复产物 v1.1 已生成四个 split，但最终封版器把 train/development
的公开统计捷径审计路径错误套用于 Audit A/B；Audit A/B 的对应文件按合同
封存在 `sealed_supervision/`，因此封版器在 Audit A 拒绝继续，且从未生成
正式 `release_manifest.json`。修正比较器后对 v1.1 做只读复核，四个 split
均仅有 `candidate_pairs.csv`、`candidate_sampling_audit.csv` 和
`world_generation_audit.jsonl` 三个顺序相关文件改变，语义等价证明通过。
但 v1.1 已绑定旧封版工具哈希，不得事后补签，只作为未发布中间产物；v1.2
必须在新实现合同下重新预检、重新生成和重新封版。

## 2. 唯一允许的修复

每个世界先按原规则得到完全相同的 C40 成员集合，再按独立随机域排序：

```text
HMAC-SHA256(candidate_key,
            world_uid,
            "selected_global_rank",
            canonical_pair_uid)
```

若 HMAC 相同，以 `canonical_pair_uid` 的 UTF-8 字节升序打破。该排序不读取
controller、label、机制、文本、身份33维或模型分数。

本修复必须同时满足：

- 复用 v1 的四把 split-private 结构密钥，不举行新密钥仪式；
- 数据生成 `run_id` 仍为 `v13_training_ready_v1_20260729`；
- world、seller、item、文本、C40 成员集合、controller、身份值、身份33维
  和 `pair_uid -> label` 映射与 v1 完全相同；
- 不删行、不换 key、不重抽 world/C40、不降低任何门槛；
- 只允许 C40 序列化顺序及依赖该顺序的表、回执和 manifest 改变。

## 3. 三重顺序门

1. C40 生成器使用独立 `selected_global_rank` 随机域；
2. split 构建器在写盘前逐世界重算 40 条顺序，验证世界块连续且顺序精确；
3. 最终发布器从磁盘独立重算，不信任构建器回执。任一世界不一致即拒绝
   发布。

发布后还必须运行逐行捷径审计。train/development 可以使用标签检查行号、
UID、文本统计和可见元数据代理；Audit A/B 只做无标签残留与精确行序重放，
不得为诊断打开封存标签。

## 4. 父子版本等价证明

在删除失效 v1 数据字节前，必须生成机器可读比较报告并验证：

- 所有非顺序依赖文件逐字节相同；
- C40、标签和顺序依赖表按主键排序后语义相同；
- 每个 split 的 UID、身份值和 label 映射完全相同；
- 差异文件集合不超出预登记允许清单；
- v1 四 split 与 v1.2 四 split 的结构密钥 commitment 分别相同。

比较报告必须固定父发布清单文件哈希
`6924dadb669bf056302418ac012e2f027b1bd3e9e00cf0c0e5e515258a3d3ce0`
及其自哈希
`f3e0b9a2de4b89613d008b7bc8ee11e4a87c0475fdd572ddea832ef543ead7aa`。

## 5. 科研边界与发布资格

M0/M1/M2 定义、四 split 各 500 worlds、C40 正负配额、身份33维、五份 M1
置乱、检索任务、指标和推断边界全部继承 2026-07-29 父发布合同。修复不
提高科研主张：它仍是机制分层合成 case-control 数据集，不代表真实地下
市场自然分布；同一 Windows 用户下的封存也不构成人员盲法。

只有以下条件全部通过，v1.2 才可标记
`PASS_DATASET_ONLY_READY_FOR_M0_M1_M2`：四个 500-world 精确预检、三重
顺序门、原 14 项统计捷径门、标签/qrels 独立重放、五份 M1 重放、跨 split
零交集、父子等价比较、发布后逐行审计和完整回归测试。

## 6. 冻结精确预检

同一 v5/v1.2 科研实现合同下，四个 500-world v3 精确预检及登记器全量
重放均通过。所有 split 的独立 `selected_global_rank` 行序精确，身份33维
无全零列；登记器从 checkpoint 重新训练三种审计模型，并逐元素核对全部
OOF 与 9,999 次 bootstrap 统计量。

| split | 最大对称 AUC | 95% 上界 | 身份33维秩 | 报告 SHA-256 |
|---|---:|---:|---:|---|
| train | 0.509703 | 0.516557 | 31 | `aecac7e846a49b30053e06e057c7c66bd0bfc72ccc4ed0aaa2e2dd7a1b0551f6` |
| development | 0.508037 | 0.515723 | 29 | `9e51567bacd0dc0a4b471a3a8a4562e03cda73f508d98de3d07ee4e36d2fea53` |
| audit_a | 0.509954 | 0.517763 | 30 | `222835593b6f4f01076eb88556f2c41f29ffb49b7df1eada449bce9d6bf7f4b3` |
| audit_b | 0.505748 | 0.513693 | 32 | `ac810bf8999cec28da6ad0a3503dc9f0200800640b32cc9f60c35915b3faec43` |

旧 v4/v1.1 的 v2 预检数值相同且当时登记器重放通过，但它们绑定包含错误
Audit 路径合同的旧实现哈希，不能登记给 v1.2。更早的 v1 预检还把
checkpoint 放入额外子目录，不符合同目录 `<报告名>.checkpoint.*` 约定。
两轮都只保留为失效历史诊断；没有放宽登记器，v3 已按正确路径完整重跑。

## 7. 清理纪律

v1.2 获得正式资格并完成父子等价证明后，失效 v1 和未发布 v1.1 的未跟踪
数据目录可以清理；必须保留 Git 中的父策略/合同、发布哈希、失败审计报告
和交接文档记录。私钥保管目录不是垃圾，不得删除或提交。`__pycache__`、
`.pyc`、中断 staging 目录和确认无引用的临时报告可在最终全项目审计后清理。
