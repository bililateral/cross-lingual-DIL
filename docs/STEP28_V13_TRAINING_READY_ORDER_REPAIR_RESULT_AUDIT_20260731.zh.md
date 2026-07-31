# Step 28-v13 正式中文合成数据集行序修复结果审计

更新日期：2026-07-31
正式版本：`v13_training_ready_v1_2_order_repair_20260731`
结论：`PASS_DATASET_ONLY_READY_FOR_M0_M1_M2`

## 1. 为什么需要 v1.2

原 v1 把 C40 的选择/补齐 HMAC 排名继续用于最终文件行序。由于正负样本池
大小不同，文件位置能够预测标签，v1 不得用于训练或评估。v1.1 已修复行序，
但封版比较器把 train/development 的公开审计路径套到 Audit A/B，因而没有
生成正式发布清单。v1.2 在新实现合同下重新完成四个 500-world 精确预检、
四分区生成和最终封版，没有覆盖或补签旧产物。

## 2. 正式发布证据

- 实现合同：`ee4d249aaf421d4e6a6603e9e0ae779d9b3c384cef266e5234b0b0918800a831`
- 发布清单文件 SHA-256：
  `81ca7d9d2040d500b3bcb2ffc9af6aeb72c581754dbd075b94dd6cf8904b8275`
- 发布清单自哈希：
  `59001459bc9b3a908ab0efa1f9f46a6c821bf6078ba3dc5f3308f910d0c5e00b`
- 父子等价报告文件 SHA-256：
  `3b0d1e29a713ce0402a14170d68936bd9a5b25f745321a5b988b1071817d2998`
- 父子等价报告自哈希：
  `db9757deec76529ecfe61e4a16fab2f90351c892400c1f32f89300e68d944d5d`

四个 split 都仅有
`observed/candidate_pairs.csv`、
`private_audit/candidate_sampling_audit.csv` 和
`private_audit/world_generation_audit.jsonl` 三个行序相关文件与 v1
字节不同；按主键比较后语义完全相同，越界变化为 0。结构 key commitment
逐 split 相同，没有换 key、重抽 world/C40、删行或事后降门槛。

| split | world | seller | item | 完整 pair | C40 | 正例 | 体积 GiB | split manifest SHA-256 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| train | 500 | 14,000 | 50,273 | 189,000 | 20,000 | 8,000 | 1.974 | `b178f7869014be6ba7130cd5294e31aa232d6a52ad589547217810b0caaa9f9c` |
| development | 500 | 14,000 | 50,750 | 189,000 | 20,000 | 5,000 | 0.925 | `902307ab9a8bc17cbf4e8564d14c4c8b4bc2bb17c70e48ba8de13696160da088` |
| audit_a | 500 | 14,000 | 50,603 | 189,000 | 20,000 | 5,000 | 0.946 | `ebcd400fa07aa3cfe4a6d3270ff791e57fc69825c6e1d857f56035db8e503d02` |
| audit_b | 500 | 14,000 | 50,445 | 189,000 | 20,000 | 5,000 | 0.979 | `a166567ea2b41ddd69d83df17be6799e3a73236653af333614e64f7383cffe3f` |

所有 world、seller、item、pair、controller、identity UID/value、query 和
relation 的跨 split 交集均为 0。Audit A/B 各有 2,000 query、54,000
relation 和 54,000 封存 qrel。train 的五份 M1 各有 189,000 行，整 33 维
向量 multiset 保持、端点不重合且写盘后重放精确。M1 只保证 pair 端点
不重合，不保证源/目标 pair 的 controller 集合不重合；因此它是保守的
整向量置乱对照，不能写成 controller-disjoint 或严格无身份关联的 null。

## 3. 逐行捷径审计

发布后报告
`formal_order_repair_v1_2_row_shortcut_audit_v1_20260731.json` 的文件
SHA-256 为
`62bcc36cdfaadc5a257f7bfaaf0915b50b09b3e7902cc14efd349cf47c22390e`，
自哈希为
`f42da98621ca507063709c013fe51712e8d0f223f297691b826f62ded8860956`。

审计逐条打开四个 split 的全部可见商品、卖家档案和计划身份槽，但标签相关
诊断仅限 train/development；没有打开 Audit A/B 标签、qrel 或 controller
membership。结果如下：

| split | 可见 item | seller profile | 计划身份槽 | 内部标记 | 原 surface 残留 | 规范值残留 | 行序错误 world |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 50,273 | 14,000 | 82,124 | 0 | 0 | 0 | 0 |
| development | 50,750 | 14,000 | 82,182 | 0 | 0 | 0 | 0 |
| audit_a | 50,603 | 14,000 | 82,052 | 0 | 0 | 0 | 0 |
| audit_b | 50,445 | 14,000 | 88,658 | 0 | 0 | 0 | 0 |

分世界 OOF 中，UID/文件位置组的最大对称 AUC 为 train `0.508649`、
development `0.512257`；冻结无关元数据组为 train `0.505178`、
development `0.502243`，均处于随机附近。正式哈希报告中的
`visible_m0_proxy` 共 24 项（5 项 nuisance 加 19 项可见文本/档案代理）；
gradient tree/logistic 的 train AUC 为 `0.570414/0.572021`，
development AUC 为 `0.572561/0.577379`。这是构建合同允许 M0 利用的
共享作者风格/文本基线诊断；每个原型在每世界分给三个 controller，不是
控制者专属暗号。私有 AST 推导的 `same_inferred_base_style` 和 oracle
style 只作生成机制诊断，模型挂载合同明确禁止读取，不能当成 M0 的可见
成绩或真实跨语言作者风格迁移证据。

相比之下，预期身份33信号的分世界 OOF 梯度树 AUC/AP 为 train
`0.856478/0.857001`、development `0.845714/0.772310`。这只证明合成器、
解析器和身份特征链包含可学习信号，不代表 M2 已训练或优于 M0/M1。

原逐行审计的职责是残留 surface、可见代理和 train/development 标签诊断；
它自身没有把 500 个 world 的精确集合、跨 world 连续块和全部正式文件
哈希重新绑定为一个门。终审后新增的独立加固审计不导入生成器、预检器、
封版器或原逐行审计，实现上重新完成下列检查：

需要同时明确：v3 精确预检报告里的 `candidate_output_order_audit` 是
`build_split_in_memory()` 返回的构建器收据，预检器只按固定字典核对，
并没有第三套自身的 HMAC 排序实现。因此“预检行序精确”只表示构建器门在
精确预检规模上通过，不能称为预检器独立重放。冻结修复合同本身已被正式
release 哈希固定，不作事后改写；真正独立于构建器的当前证据是封版器从
磁盘重算，以及下面的独立发布树加固审计。

- 哈希核对正式树全部 160 个文件及四份 split manifest 的成员集合、大小；
- 核对四个 split 各 500 个 world 的精确集合和冻结顺序；
- 核对全部 80,000 条 C40 均为连续的 40-row world 块、pair 唯一且端点
  公式精确；
- 用标准库 HMAC-SHA256 独立重算每个 world 的
  `selected_global_rank` 顺序；
- 只哈希封存文件字节，不解析 label、qrel、controller、身份资产或 M1。

加固报告
`formal_order_repair_v1_2_release_tree_hardening_audit_v1_20260731.json`
状态为 `PASS_INDEPENDENT_FORMAL_RELEASE_TREE_HARDENING_AUDIT`，文件
SHA-256 为
`a053f71101b320868c8a52658bf78ae18bc3d67e84f71064f536a28d594fa3bd`，
自哈希为
`fbfe00737077f6422d24030e65840c447f9e1f314597ba3664171515f16580a7`。

## 4. 提交前验证

相关脚本的 `py_compile` 和 `git diff --check` 均通过。首次全仓回归暴露的唯一
错误来自测试夹具：5-world 设计期迷你构造沿用了正式 overlay 的
`generation_enabled=true`，因而被生产门正确拒绝。修复仅在测试夹具中显式
设为 `false`，没有放宽生成器门槛，也没有改变正式数据或实现合同。随后该
测试文件 18 项全部通过。

最终执行 `python -m unittest discover -s tests`，共发现 381 项测试：
374 项通过、7 项按既有声明跳过、0 失败，用时 798.904 秒。跳过项中 6 项
属于已撤销 Step28-v11 测试类，1 项属于已被 training-ready 测试替代的
`dataset_smoke_v3` 不可执行旧锁。

## 5. 结论边界

本结果授予的是“数据字节可供冻结 M0、五个 M1 和 M2 使用”，不是模型效果
成功。数据是机制分层合成 case-control 总体，不代表真实地下市场 prevalence、
自然校准或真实中文泛化。Audit A/B 是同一 Windows 用户下的逻辑封存，
`blind_custody_attested=false`；任何真实跨语言结论仍需新的独立真实中文
标签数据。

清理合同允许在父子等价证明完成后删除失效 v1 完整数据。当前工作区因此
只能验证已封存等价报告和九份父 manifest 的哈希，不能在不恢复 v1 全部
原字节的前提下重新执行父子逐文件语义比较。该限制不改变当前 v1.2 的
160 文件独立哈希/行序验证，但父子等价证明必须准确称为清理前完成并
哈希封存的历史仪式，而不是当前工作区可随时重放的检查。
